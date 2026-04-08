"""Integrated analysis services for synthetic persona investigation.

This module provides the final-stage orchestration that connects:
Data acquisition -> preprocessing -> feature extraction -> detection -> clustering
and returns normalized outputs for API, dashboard, and reporting layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics.pairwise import cosine_similarity

from app.attribution_clustering import AttributionClusteringConfig, run_attribution_clustering_pipeline
from app.config import DATASET_PATH, PROCESSED_DATA_DIR
from app.data_acquisition.preprocessing import preprocess_pipeline
from app.data_acquisition.twitter_scraper import fetch_tweets
from app.feature_extraction import FeatureExtractionConfig, FeatureExtractor
from app.persona_detection import load_model, predict_usernames_from_feature_frame
from app.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

DEFAULT_MODEL_PATH = PROCESSED_DATA_DIR / "best_persona_model.pkl"
HISTORY_PATH = PROCESSED_DATA_DIR / "analysis_history.jsonl"
DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime controls for the integrated analysis flow."""

    max_posts_from_api: int = 100
    linked_persona_similarity_threshold: float = 0.75
    linked_persona_limit: int = 10


def generate_risk_score(probability: float) -> str:
    """Convert synthetic probability into risk bands."""

    score = float(np.clip(probability, 0.0, 1.0))
    if score < 0.4:
        return "Low"
    if score <= 0.7:
        return "Medium"
    return "High"


def _canonical_username(username: str) -> str:
    return username.strip().lower().lstrip("@")


@lru_cache(maxsize=1)
def _load_raw_dataset_cached() -> pd.DataFrame:
    """Load local dataset once per process for low-latency lookups."""

    if not DATASET_PATH.exists():
        return pd.DataFrame(columns=["username", "text", "timestamp", "label"])

    try:
        frame = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        logger.warning("Failed reading local dataset: %s", exc)
        return pd.DataFrame(columns=["username", "text", "timestamp", "label"])

    for required in ["username", "text", "timestamp"]:
        if required not in frame.columns:
            logger.warning("Dataset missing required column: %s", required)
            return pd.DataFrame(columns=["username", "text", "timestamp", "label"])

    frame["username"] = frame["username"].astype(str).map(_canonical_username)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _load_model_bundle_cached() -> dict[str, Any] | None:
    """Load the Phase 4 model bundle once and reuse it across requests."""

    if not DEFAULT_MODEL_PATH.exists():
        logger.warning("Model bundle not found at %s. Falling back to heuristic scoring.", DEFAULT_MODEL_PATH)
        return None

    try:
        return load_model(DEFAULT_MODEL_PATH)
    except Exception as exc:
        logger.warning("Failed loading model bundle. Falling back to heuristics: %s", exc)
        return None


def _extract_user_posts(username: str, max_posts_from_api: int) -> pd.DataFrame:
    """Fetch account posts from local dataset first, then X API scraper."""

    canonical = _canonical_username(username)
    dataset = _load_raw_dataset_cached()
    from_dataset = dataset[dataset["username"] == canonical].copy()

    if not from_dataset.empty:
        return from_dataset.rename(columns={"text": "post_text"})[["username", "post_text", "timestamp"]]

    scraped = fetch_tweets(username=canonical, max_tweets=max_posts_from_api)
    if not scraped:
        return pd.DataFrame(columns=["username", "post_text", "timestamp"])

    frame = pd.DataFrame(scraped)
    if frame.empty:
        return pd.DataFrame(columns=["username", "post_text", "timestamp"])

    frame = frame.rename(columns={"text": "post_text"})
    frame["username"] = frame["username"].astype(str).map(_canonical_username)
    return frame[["username", "post_text", "timestamp"]]


def _heuristic_probability(feature_row: pd.Series) -> float:
    """Fallback score when no trained model is available."""

    posts_per_day = float(feature_row.get("posts_per_day", 0.0))
    night_ratio = float(feature_row.get("night_activity_ratio", 0.0))
    vocab = float(feature_row.get("vocabulary_richness", 0.0))
    punctuation = float(feature_row.get("punctuation_usage", 0.0))
    grammar = float(feature_row.get("grammar_consistency", 0.0))

    score = (
        0.22 * min(posts_per_day / 20.0, 1.0)
        + 0.22 * night_ratio
        + 0.16 * (1.0 - min(vocab, 1.0))
        + 0.20 * min(punctuation * 8.0, 1.0)
        + 0.20 * (1.0 - min(grammar, 1.0))
    )
    return float(np.clip(score, 0.0, 1.0))


def _merge_feature_frames(preprocessed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build reusable user-level feature outputs for downstream layers."""

    extractor = FeatureExtractor(FeatureExtractionConfig(tfidf_max_features=400))

    working = preprocessed_df.copy()
    working["label"] = 0

    stylometric = extractor.extract_stylometric_features(working)
    behavioral = extractor.extract_behavioral_features(working)
    network = extractor.extract_network_features(working)
    tfidf = extractor.vectorize_text(working)

    merged = stylometric.merge(behavioral, on="username", how="outer")
    merged = merged.merge(network, on="username", how="left")
    merged = merged.merge(tfidf, on="username", how="left")
    merged = merged.fillna(0.0)

    return merged.reset_index(drop=True), stylometric, behavioral


def _build_reference_feature_set(current_preprocessed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a combined user-level feature frame for clustering context."""

    dataset = _load_raw_dataset_cached()

    if dataset.empty:
        return _merge_feature_frames(current_preprocessed)

    historical_input = dataset.rename(columns={"text": "post_text"})[["username", "post_text", "timestamp"]]
    historical_preprocessed = preprocess_pipeline(historical_input)

    if historical_preprocessed.empty:
        return _merge_feature_frames(current_preprocessed)

    combined = pd.concat([historical_preprocessed, current_preprocessed], ignore_index=True)
    combined = combined.drop_duplicates(subset=["username", "original_text", "timestamp"], keep="last")
    return _merge_feature_frames(combined)


def _build_timeline(preprocessed: pd.DataFrame) -> dict[str, Any]:
    """Generate hour/day distributions and posting frequency summaries."""

    if preprocessed.empty:
        return {
            "hour_distribution": {str(i): 0 for i in range(24)},
            "day_distribution": {day: 0 for day in DAY_ORDER},
            "posts_per_day": {},
            "word_count_series": [],
            "avg_word_length": 0.0,
            "vocabulary_richness": 0.0,
        }

    working = preprocessed.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"])

    hour_counts = working["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    day_counts = (
        working["timestamp"]
        .dt.day_name()
        .value_counts()
        .reindex(DAY_ORDER, fill_value=0)
    )

    posts_per_day = (
        working.assign(date=working["timestamp"].dt.date)
        .groupby("date")
        .size()
        .astype(int)
        .to_dict()
    )

    word_series = working["clean_text"].fillna("").astype(str).str.count(r"\S+").tolist()
    tokens = [token for text in working["clean_text"].fillna("").astype(str) for token in text.split()]

    avg_word_length = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
    vocab_richness = float(len(set(tokens)) / len(tokens)) if tokens else 0.0

    return {
        "hour_distribution": {str(key): int(value) for key, value in hour_counts.to_dict().items()},
        "day_distribution": {str(key): int(value) for key, value in day_counts.to_dict().items()},
        "posts_per_day": {str(k): int(v) for k, v in posts_per_day.items()},
        "word_count_series": [int(v) for v in word_series],
        "avg_word_length": avg_word_length,
        "vocabulary_richness": vocab_richness,
    }


def _linked_personas_from_similarity(
    username: str,
    feature_frame: pd.DataFrame,
    threshold: float,
    limit: int,
) -> list[str]:
    """Find nearest personas by cosine similarity for extra attribution context."""

    if feature_frame.empty or len(feature_frame) < 2:
        return []

    matrix = feature_frame.drop(columns=["username"]).to_numpy(dtype=float, copy=False)
    similarity = cosine_similarity(matrix)

    usernames = feature_frame["username"].astype(str).tolist()
    canonical = _canonical_username(username)

    if canonical not in usernames:
        return []

    idx = usernames.index(canonical)
    scores = similarity[idx]

    ranked = [
        (usernames[i], float(score))
        for i, score in enumerate(scores)
        if i != idx and float(score) >= threshold
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:limit]]


def format_api_response(analysis_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize full analysis output to the contract required by the API."""

    return {
        "username": analysis_data["username"],
        "prediction": analysis_data["prediction"],
        "synthetic_score": float(analysis_data["synthetic_score"]),
        "behavioral_features": analysis_data["behavioral_features"],
        "stylometric_features": analysis_data["stylometric_features"],
        "cluster_id": int(analysis_data["cluster_id"]),
        "linked_personas": list(analysis_data["linked_personas"]),
        "risk_level": analysis_data["risk_level"],
    }


def _append_history(entry: dict[str, Any]) -> None:
    """Persist lightweight history of analyses for future comparisons."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": entry.get("username"),
        "prediction": entry.get("prediction"),
        "synthetic_score": entry.get("synthetic_score"),
        "risk_level": entry.get("risk_level"),
        "cluster_id": entry.get("cluster_id"),
    }

    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def read_analysis_history(limit: int = 100, username: str | None = None) -> list[dict[str, Any]]:
    """Read persisted analysis history from JSONL storage."""

    if limit <= 0:
        return []

    if not HISTORY_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    target = _canonical_username(username) if username else None

    with HISTORY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_username = _canonical_username(str(item.get("username", "")))
            if target and item_username != target:
                continue
            rows.append(item)

    rows.sort(key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return rows[:limit]


@lru_cache(maxsize=128)
def analyze_user(username: str, config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Run the complete end-to-end analysis pipeline for one username."""

    started = datetime.now(timezone.utc)
    cfg = config or AnalysisConfig()
    canonical = _canonical_username(username)

    raw_posts = _extract_user_posts(canonical, max_posts_from_api=cfg.max_posts_from_api)

    if raw_posts.empty:
        empty = {
            "username": canonical,
            "prediction": "Human",
            "synthetic_score": 0.0,
            "behavioral_features": {},
            "stylometric_features": {},
            "cluster_id": -1,
            "linked_personas": [],
            "risk_level": "Low",
            "timeline": _build_timeline(pd.DataFrame()),
            "network_graph": {"nodes": [], "links": []},
            "generated_at": started.isoformat(),
            "response_time_seconds": 0.0,
        }
        _append_history(empty)
        return empty

    preprocessed = preprocess_pipeline(raw_posts)
    if preprocessed.empty:
        raise ValueError("No usable posts after preprocessing. Try a different username.")

    feature_frame, stylometric_df, behavioral_df = _build_reference_feature_set(preprocessed)

    model_bundle = _load_model_bundle_cached()
    if model_bundle is not None:
        predictions = predict_usernames_from_feature_frame(
            model_bundle=model_bundle,
            feature_frame=feature_frame,
            username_col="username",
            threshold=0.5,
        )
    else:
        heuristic_scores = feature_frame.apply(_heuristic_probability, axis=1)
        predicted_labels = (heuristic_scores >= 0.5).astype(int)
        predictions = pd.DataFrame(
            {
                "username": feature_frame["username"],
                "predicted_label": predicted_labels,
                "classification": np.where(predicted_labels == 1, "AI", "Human"),
                "synthetic_score": heuristic_scores,
                "model_name": "HeuristicFallback",
            }
        )

    phase5_result = run_attribution_clustering_pipeline(
        X=feature_frame.drop(columns=["username"]),
        predictions=predictions,
        usernames=feature_frame["username"].tolist(),
        config=AttributionClusteringConfig(eps=0.5, min_samples=2, edge_threshold=0.7),
        add_louvain_communities=True,
    )

    user_mask = predictions["username"].astype(str).map(_canonical_username) == canonical
    if not user_mask.any():
        raise ValueError(f"Unable to find prediction row for user '{canonical}'.")

    prediction_row = predictions[user_mask].iloc[0]
    synthetic_score = float(prediction_row["synthetic_score"])
    prediction_label = str(prediction_row["classification"])

    cluster_assignments = phase5_result.get("cluster_assignments", {})
    cluster_id = int(cluster_assignments.get(canonical, -1))
    clusters = phase5_result.get("clusters", {})

    linked_personas = []
    if cluster_id in clusters and cluster_id != -1:
        linked_personas = [u for u in clusters[cluster_id] if _canonical_username(u) != canonical]

    if not linked_personas:
        linked_personas = _linked_personas_from_similarity(
            username=canonical,
            feature_frame=feature_frame,
            threshold=cfg.linked_persona_similarity_threshold,
            limit=cfg.linked_persona_limit,
        )

    style_row = stylometric_df[stylometric_df["username"] == canonical]
    behavior_row = behavioral_df[behavioral_df["username"] == canonical]

    stylometric_features = (
        {}
        if style_row.empty
        else {
            "word_count": float(style_row.iloc[0].get("word_count", 0.0)),
            "avg_word_length": float(style_row.iloc[0].get("avg_word_length", 0.0)),
            "avg_sentence_length": float(style_row.iloc[0].get("avg_sentence_length", 0.0)),
            "vocabulary_richness": float(style_row.iloc[0].get("vocabulary_richness", 0.0)),
            "punctuation_usage": float(style_row.iloc[0].get("punctuation_usage", 0.0)),
            "grammar_consistency": float(style_row.iloc[0].get("grammar_consistency", 0.0)),
        }
    )

    behavioral_features = (
        {}
        if behavior_row.empty
        else {
            "posts_per_day": float(behavior_row.iloc[0].get("posts_per_day", 0.0)),
            "time_between_posts": float(behavior_row.iloc[0].get("time_between_posts", 0.0)),
            "night_activity_ratio": float(behavior_row.iloc[0].get("night_activity_ratio", 0.0)),
            "coordination_score": float(behavior_row.iloc[0].get("coordination_score", 0.0)),
            "coordination_flag": int(behavior_row.iloc[0].get("coordination_flag", 0)),
        }
    )

    timeline = _build_timeline(preprocessed)

    response = {
        "username": canonical,
        "prediction": "AI" if prediction_label.lower() == "ai" else "Human",
        "synthetic_score": synthetic_score,
        "behavioral_features": behavioral_features,
        "stylometric_features": stylometric_features,
        "cluster_id": cluster_id,
        "linked_personas": linked_personas,
        "risk_level": generate_risk_score(synthetic_score),
        "timeline": timeline,
        "network_graph": phase5_result.get("api_response", {}).get("graph", {"nodes": [], "links": []}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    response["response_time_seconds"] = float(round(elapsed, 4))

    _append_history(response)
    return jsonable_encoder(response)


def _timeline_chart_image(analysis_data: dict[str, Any]) -> io.BytesIO:
    """Render behavioral timeline chart into an in-memory PNG."""

    timeline = analysis_data.get("timeline", {})
    hours = timeline.get("hour_distribution", {})

    fig, ax = plt.subplots(figsize=(7, 2.8))
    x_values = list(range(24))
    y_values = [int(hours.get(str(i), 0)) for i in x_values]
    ax.bar(x_values, y_values, color="#2f5597")
    ax.set_title("Activity by Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Posts")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _network_chart_image(analysis_data: dict[str, Any]) -> io.BytesIO:
    """Render attribution network graph into an in-memory PNG."""

    graph_data = analysis_data.get("network_graph", {"nodes": [], "links": []})
    graph = nx.Graph()

    for node in graph_data.get("nodes", []):
        graph.add_node(node.get("id", "unknown"))

    for edge in graph_data.get("links", []):
        graph.add_edge(edge.get("source"), edge.get("target"), weight=edge.get("weight", 0.0))

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network links detected", ha="center", va="center")
        ax.axis("off")
    else:
        position = nx.spring_layout(graph, seed=42)
        nx.draw_networkx(
            graph,
            pos=position,
            ax=ax,
            with_labels=True,
            node_color="#9ecae1",
            edge_color="#3182bd",
            font_size=8,
            node_size=700,
        )
        ax.set_title("Suspicious Persona Network")
        ax.axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def generate_report(data: dict[str, Any], output_path: str | Path | None = None) -> bytes:
    """Generate PDF report bytes for one analysis result."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story: list[Any] = []
    story.append(Paragraph("Synthetic Persona Investigation Report", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    summary_rows = [
        ["Username", str(data.get("username", ""))],
        ["Detection Result", str(data.get("prediction", ""))],
        ["Synthetic Score", f"{float(data.get('synthetic_score', 0.0)):.4f}"],
        ["Risk Level", str(data.get("risk_level", ""))],
        ["Cluster ID", str(data.get("cluster_id", -1))],
        ["Linked Personas", ", ".join(data.get("linked_personas", [])) or "None"],
    ]

    summary_table = Table(summary_rows, colWidths=[2.0 * inch, 4.8 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    behavioral = data.get("behavioral_features", {})
    stylometric = data.get("stylometric_features", {})

    story.append(Paragraph("Behavioral Insights", styles["Heading2"]))
    story.append(
        Paragraph(
            (
                f"Posts/day: {behavioral.get('posts_per_day', 0.0):.3f} | "
                f"Avg interval (sec): {behavioral.get('time_between_posts', 0.0):.2f} | "
                f"Night ratio: {behavioral.get('night_activity_ratio', 0.0):.3f}"
            ),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Stylometric Analysis", styles["Heading2"]))
    story.append(
        Paragraph(
            (
                f"Word count: {stylometric.get('word_count', 0.0):.2f} | "
                f"Avg word length: {stylometric.get('avg_word_length', 0.0):.2f} | "
                f"Vocabulary richness: {stylometric.get('vocabulary_richness', 0.0):.3f}"
            ),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    timeline_image = _timeline_chart_image(data)
    story.append(Paragraph("Behavioral Timeline", styles["Heading2"]))
    story.append(Image(timeline_image, width=6.3 * inch, height=2.4 * inch))
    story.append(Spacer(1, 0.12 * inch))

    network_image = _network_chart_image(data)
    story.append(Paragraph("Cluster Network Visualization", styles["Heading2"]))
    story.append(Image(network_image, width=6.1 * inch, height=3.0 * inch))

    doc.build(story)
    pdf_bytes = buffer.getvalue()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes)

    return pdf_bytes
