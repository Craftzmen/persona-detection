"""Phase 5 Attribution and Clustering for AI persona relationship analysis.

This module consumes Phase 3 features and Phase 4 predictions to:
1. Filter AI-labeled personas
2. Compute weighted similarity (stylometric + behavioral)
3. Cluster AI personas using DBSCAN
4. Analyze cluster composition and cohesion
5. Build a similarity network graph
6. Optionally visualize/export graph and cluster outputs
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.logging_utils import setup_logging


logger = setup_logging(__name__)


@dataclass(frozen=True)
class AttributionClusteringConfig:
    """Configuration for Phase 5 attribution + clustering pipeline."""

    eps: float = 0.5
    min_samples: int = 3
    stylometric_weight: float = 0.6
    behavioral_weight: float = 0.4
    edge_threshold: float = 0.7
    low_similarity_threshold: float = 0.3
    use_precomputed_distance: bool = False
    include_similarity_matrix: bool = True
    include_cluster_stats: bool = True


DEFAULT_STYLOMETRIC_PREFIXES = (
    "tfidf_",
    "word_count",
    "avg_word_",
    "avg_sentence_",
    "punctuation_",
    "grammar_",
    "vocabulary_",
    "unique_word_count",
)

DEFAULT_BEHAVIORAL_PREFIXES = (
    "posts_per_day",
    "time_between_posts",
    "night_activity_ratio",
    "hour_",
    "likes_",
    "replies_",
    "retweets_",
    "in_degree",
    "out_degree",
    "betweenness",
    "pagerank",
    "clustering",
    "community_",
    "coordination_",
)


def _ensure_numeric_dataframe(X: pd.DataFrame | np.ndarray | sparse.spmatrix) -> pd.DataFrame:
    """Convert supported matrix inputs into a numeric DataFrame."""

    if sparse.issparse(X):
        dense = X.toarray().astype(float)
        return pd.DataFrame(dense, columns=[f"f_{idx}" for idx in range(dense.shape[1])])

    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        numeric = np.nan_to_num(X.astype(float), copy=False)
        return pd.DataFrame(numeric, columns=[f"f_{idx}" for idx in range(numeric.shape[1])])

    if isinstance(X, pd.DataFrame):
        numeric = X.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return numeric

    raise TypeError("X must be a pandas DataFrame, numpy array, or scipy sparse matrix.")


def _normalize_ai_mask(predictions: pd.DataFrame, ai_label: str = "AI") -> np.ndarray:
    """Create a boolean mask selecting AI personas from Phase 4 outputs."""

    if "classification" in predictions.columns:
        return (
            predictions["classification"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq(ai_label.lower())
            .to_numpy()
        )

    if "predicted_label" in predictions.columns:
        labels = pd.to_numeric(predictions["predicted_label"], errors="coerce").fillna(0).astype(int)
        return labels.eq(1).to_numpy()

    raise ValueError("predictions must include either 'classification' or 'predicted_label'.")


def _pick_feature_subsets(
    X: pd.DataFrame,
    stylometric_columns: list[str] | None,
    behavioral_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve stylometric and behavioral feature column lists."""

    all_columns = X.columns.astype(str).tolist()

    if stylometric_columns is None:
        stylometric_columns = [
            col for col in all_columns if col.startswith(DEFAULT_STYLOMETRIC_PREFIXES)
        ]
    else:
        stylometric_columns = [col for col in stylometric_columns if col in all_columns]

    if behavioral_columns is None:
        behavioral_columns = [
            col for col in all_columns if col.startswith(DEFAULT_BEHAVIORAL_PREFIXES)
        ]
    else:
        behavioral_columns = [col for col in behavioral_columns if col in all_columns]

    if not stylometric_columns and not behavioral_columns:
        # Fallback to all columns when no semantic split is available.
        stylometric_columns = all_columns

    if not behavioral_columns:
        behavioral_columns = [col for col in all_columns if col not in stylometric_columns]

    return stylometric_columns, behavioral_columns


def filter_ai_accounts(
    X: pd.DataFrame | np.ndarray | sparse.spmatrix,
    predictions: pd.DataFrame,
    usernames: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Filter inputs down to AI personas only.

    Args:
        X: Phase 3 feature matrix.
        predictions: Phase 4 prediction output DataFrame.
        usernames: Optional usernames list (uses predictions['username'] if omitted).

    Returns:
        A tuple of ``(X_ai, usernames_ai, synthetic_scores_ai, ai_indices)``.
    """

    X_df = _ensure_numeric_dataframe(X)

    if len(X_df) != len(predictions):
        raise ValueError("X and predictions must have the same row count.")

    if usernames is None:
        if "username" not in predictions.columns:
            raise ValueError("Provide usernames or include 'username' in predictions.")
        usernames_arr = predictions["username"].astype(str).to_numpy()
    else:
        usernames_arr = np.asarray(list(usernames), dtype=str)

    if len(usernames_arr) != len(X_df):
        raise ValueError("usernames length must match number of rows in X.")

    ai_mask = _normalize_ai_mask(predictions)
    ai_indices = np.where(ai_mask)[0]

    if ai_indices.size == 0:
        logger.warning("No AI personas found in predictions; returning empty AI subset.")
        return X_df.iloc[0:0].copy(), [], np.array([], dtype=float), ai_indices

    scores = pd.to_numeric(predictions.get("synthetic_score", 0.0), errors="coerce").fillna(0.0).to_numpy()

    X_ai = X_df.iloc[ai_indices].reset_index(drop=True)
    usernames_ai = usernames_arr[ai_indices].tolist()
    synthetic_scores_ai = scores[ai_indices]
    return X_ai, usernames_ai, synthetic_scores_ai, ai_indices


def compute_similarity(
    X_ai: pd.DataFrame | np.ndarray | sparse.spmatrix,
    stylometric_columns: list[str] | None = None,
    behavioral_columns: list[str] | None = None,
    stylometric_weight: float = 0.6,
    behavioral_weight: float = 0.4,
    low_similarity_threshold: float = 0.3,
) -> dict[str, np.ndarray]:
    """Compute stylometric, behavioral, and weighted cosine similarities."""

    X_df = _ensure_numeric_dataframe(X_ai)
    if X_df.empty:
        empty = np.empty((0, 0), dtype=float)
        return {
            "stylometric_similarity": empty,
            "behavioral_similarity": empty,
            "similarity_matrix": empty,
        }

    n_accounts = len(X_df)
    if n_accounts > 5000:
        logger.warning(
            "Computing full similarity for %d accounts may be memory intensive.",
            n_accounts,
        )

    style_cols, beh_cols = _pick_feature_subsets(X_df, stylometric_columns, behavioral_columns)

    style_sim = np.zeros((n_accounts, n_accounts), dtype=float)
    beh_sim = np.zeros((n_accounts, n_accounts), dtype=float)

    if style_cols:
        style_sim = cosine_similarity(X_df[style_cols].to_numpy(dtype=float, copy=False))
    if beh_cols:
        beh_sim = cosine_similarity(X_df[beh_cols].to_numpy(dtype=float, copy=False))

    if style_cols and beh_cols:
        total_weight = max(stylometric_weight + behavioral_weight, 1e-12)
        weighted = (stylometric_weight * style_sim + behavioral_weight * beh_sim) / total_weight
    elif style_cols:
        weighted = style_sim
    else:
        weighted = beh_sim

    np.fill_diagonal(weighted, 1.0)
    if n_accounts > 1:
        tri_upper = weighted[np.triu_indices(n_accounts, k=1)]
        avg_pair_similarity = float(tri_upper.mean()) if tri_upper.size else 0.0
        if avg_pair_similarity < low_similarity_threshold:
            logger.info(
                "Low overall pairwise similarity observed (avg=%.4f < %.4f).",
                avg_pair_similarity,
                low_similarity_threshold,
            )

    return {
        "stylometric_similarity": style_sim,
        "behavioral_similarity": beh_sim,
        "similarity_matrix": weighted,
    }


def run_dbscan_clustering(
    X_ai: pd.DataFrame | np.ndarray | sparse.spmatrix,
    eps: float = 0.5,
    min_samples: int = 3,
    use_precomputed_distance: bool = False,
    similarity_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster AI personas using DBSCAN.

    If ``use_precomputed_distance`` is True, this function expects
    ``similarity_matrix`` and runs DBSCAN on distance = 1 - similarity.
    """

    X_df = _ensure_numeric_dataframe(X_ai)
    if X_df.empty:
        logger.warning("AI subset is empty; DBSCAN will return no labels.")
        return np.array([], dtype=int)

    if len(X_df) == 1:
        return np.array([-1], dtype=int)

    try:
        if use_precomputed_distance:
            if similarity_matrix is None:
                raise ValueError("similarity_matrix is required when use_precomputed_distance=True.")
            distances = 1.0 - np.clip(similarity_matrix, 0.0, 1.0)
            model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = model.fit_predict(distances)
        else:
            # DBSCAN on vectors directly with cosine metric follows the requested behavior.
            model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            labels = model.fit_predict(X_df.to_numpy(dtype=float, copy=False))
    except Exception as exc:
        logger.exception("DBSCAN clustering failed: %s", exc)
        return np.full(shape=len(X_df), fill_value=-1, dtype=int)

    return labels.astype(int)


def analyze_clusters(
    usernames_ai: list[str],
    cluster_labels: np.ndarray,
    synthetic_scores_ai: np.ndarray,
    similarity_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Analyze cluster composition and compute per-cluster statistics."""

    if len(usernames_ai) != len(cluster_labels):
        raise ValueError("usernames_ai and cluster_labels must have identical length.")

    if len(synthetic_scores_ai) != len(cluster_labels):
        raise ValueError("synthetic_scores_ai and cluster_labels must have identical length.")

    cluster_assignments = {
        username: int(label)
        for username, label in zip(usernames_ai, cluster_labels, strict=True)
    }

    clusters: dict[int, list[str]] = {}
    for username, label in cluster_assignments.items():
        clusters.setdefault(int(label), []).append(username)

    isolated = clusters.get(-1, []).copy()

    cluster_stats: dict[int, dict[str, float | int]] = {}
    for cluster_id, members in clusters.items():
        member_indices = [idx for idx, label in enumerate(cluster_labels) if int(label) == int(cluster_id)]
        cluster_size = len(member_indices)
        avg_score = float(np.mean(synthetic_scores_ai[member_indices])) if member_indices else 0.0

        if similarity_matrix is not None and cluster_size > 1:
            sub = similarity_matrix[np.ix_(member_indices, member_indices)]
            tri = sub[np.triu_indices(cluster_size, k=1)]
            avg_sim = float(tri.mean()) if tri.size else 0.0
        elif cluster_size == 1:
            avg_sim = 1.0
        else:
            avg_sim = 0.0

        cluster_stats[int(cluster_id)] = {
            "cluster_size": cluster_size,
            "average_similarity": avg_sim,
            "average_synthetic_score": avg_score,
        }

    return {
        "cluster_assignments": cluster_assignments,
        "clusters": {int(k): v for k, v in clusters.items()},
        "isolated_personas": isolated,
        "cluster_statistics": cluster_stats,
    }


def build_network_graph(
    usernames_ai: list[str],
    similarity_matrix: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    synthetic_scores_ai: np.ndarray | None = None,
    similarity_threshold: float = 0.7,
) -> nx.Graph:
    """Build a weighted similarity graph where nodes are AI personas."""

    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be a square 2D matrix.")

    if len(usernames_ai) != similarity_matrix.shape[0]:
        raise ValueError("usernames_ai length must match similarity matrix size.")

    G = nx.Graph()

    for idx, username in enumerate(usernames_ai):
        node_attrs: dict[str, Any] = {}
        if cluster_labels is not None:
            node_attrs["cluster_id"] = int(cluster_labels[idx])
        if synthetic_scores_ai is not None:
            node_attrs["synthetic_score"] = float(synthetic_scores_ai[idx])
        G.add_node(username, **node_attrs)

    n = len(usernames_ai)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim > similarity_threshold:
                G.add_edge(usernames_ai[i], usernames_ai[j], weight=sim)
                edge_count += 1

    if edge_count == 0 and n > 1:
        logger.info(
            "No edges passed threshold %.2f; consider lowering threshold for sparse graphs.",
            similarity_threshold,
        )

    return G


def visualize_graph(
    G: nx.Graph,
    output_path: str | Path | None = None,
    interactive: bool = False,
    title: str = "AI Persona Attribution Network",
) -> None:
    """Visualize the network with cluster-based colors and similarity-weighted edges."""

    if G.number_of_nodes() == 0:
        logger.warning("Graph has no nodes; skipping visualization.")
        return

    if interactive:
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed; falling back to matplotlib graph visualization.")
            interactive = False

    pos = nx.spring_layout(G, seed=42, weight="weight")

    if interactive:
        import plotly.graph_objects as go

        edge_traces = []
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = float(data.get("weight", 0.0))
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line={"width": max(1.0, 4.0 * w), "color": "rgba(120,120,120,0.45)"},
                    hoverinfo="none",
                )
            )

        node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
        for node, attrs in G.nodes(data=True):
            x, y = pos[node]
            cid = int(attrs.get("cluster_id", -1))
            score = float(attrs.get("synthetic_score", 0.5))
            node_x.append(x)
            node_y.append(y)
            node_color.append(cid)
            node_size.append(10.0 + 30.0 * score)
            node_text.append(f"{node}<br>cluster={cid}<br>score={score:.3f}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker={
                "size": node_size,
                "color": node_color,
                "colorscale": "Viridis",
                "showscale": True,
                "line": {"width": 1, "color": "#1f1f1f"},
            },
            hovertext=node_text,
            hoverinfo="text",
        )

        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                xaxis={"showgrid": False, "zeroline": False, "visible": False},
                yaxis={"showgrid": False, "zeroline": False, "visible": False},
            ),
        )

        if output_path is None:
            fig.show()
        else:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out)
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; cannot render graph visualization.")
        return

    plt.figure(figsize=(12, 8))

    node_colors = [int(G.nodes[node].get("cluster_id", -1)) for node in G.nodes()]
    node_sizes = [140.0 + 460.0 * float(G.nodes[node].get("synthetic_score", 0.5)) for node in G.nodes()]
    edge_widths = [0.5 + 4.0 * float(data.get("weight", 0.0)) for _, _, data in G.edges(data=True)]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.tab20,
        alpha=0.9,
    )
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.35, edge_color="#666666")
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=140)
        plt.close()


def tune_similarity_threshold(
    similarity_matrix: np.ndarray,
    quantile: float = 0.9,
    lower_bound: float = 0.5,
    upper_bound: float = 0.95,
) -> float:
    """Dynamically tune edge threshold based on similarity distribution."""

    if similarity_matrix.size == 0 or similarity_matrix.shape[0] <= 1:
        return float(lower_bound)

    tri = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]
    if tri.size == 0:
        return float(lower_bound)

    q_value = float(np.quantile(tri, quantile))
    return float(np.clip(q_value, lower_bound, upper_bound))


def export_graph_to_json(G: nx.Graph) -> dict[str, Any]:
    """Export graph into a frontend-friendly node/link JSON structure."""

    return {
        "nodes": [
            {
                "id": node,
                **{k: v for k, v in attrs.items()},
            }
            for node, attrs in G.nodes(data=True)
        ],
        "links": [
            {
                "source": u,
                "target": v,
                **{k: v_attr for k, v_attr in attrs.items()},
            }
            for u, v, attrs in G.edges(data=True)
        ],
    }


def save_cluster_assignments_csv(
    cluster_assignments: dict[str, int],
    output_csv: str | Path,
) -> Path:
    """Save username-to-cluster mapping to CSV."""

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "username": list(cluster_assignments.keys()),
            "cluster_id": list(cluster_assignments.values()),
        }
    )
    frame.sort_values(["cluster_id", "username"]).to_csv(out, index=False)
    return out


def detect_communities_louvain(G: nx.Graph) -> dict[str, int]:
    """Optional Louvain community detection on the graph.

    Returns a mapping ``username -> community_id``. If unavailable, returns {}.
    """

    try:
        communities = nx.community.louvain_communities(G, weight="weight", seed=42)
    except Exception as exc:
        logger.info("Louvain community detection unavailable or failed: %s", exc)
        return {}

    community_map: dict[str, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            community_map[str(node)] = int(cid)
    return community_map


def run_attribution_clustering_pipeline(
    X: pd.DataFrame | np.ndarray | sparse.spmatrix,
    predictions: pd.DataFrame,
    usernames: Iterable[str] | None = None,
    config: AttributionClusteringConfig | None = None,
    stylometric_columns: list[str] | None = None,
    behavioral_columns: list[str] | None = None,
    graph_output_path: str | Path | None = None,
    graph_interactive: bool = False,
    save_clusters_csv_path: str | Path | None = None,
    export_graph_json_path: str | Path | None = None,
    add_louvain_communities: bool = False,
) -> dict[str, Any]:
    """End-to-end Phase 5 orchestration.

    This function is designed to plug directly into Phase 4 prediction outputs.
    """

    cfg = config or AttributionClusteringConfig()

    X_ai, usernames_ai, synthetic_scores_ai, _ = filter_ai_accounts(
        X=X,
        predictions=predictions,
        usernames=usernames,
    )

    if X_ai.empty:
        empty_graph = nx.Graph()
        return {
            "cluster_assignments": {},
            "clusters": {},
            "isolated_personas": [],
            "graph": empty_graph,
            "similarity_matrix": np.empty((0, 0), dtype=float),
            "cluster_statistics": {},
            "api_response": {
                "cluster_assignments": {},
                "clusters": {},
                "isolated_personas": [],
                "graph": {"nodes": [], "links": []},
            },
        }

    similarities = compute_similarity(
        X_ai=X_ai,
        stylometric_columns=stylometric_columns,
        behavioral_columns=behavioral_columns,
        stylometric_weight=cfg.stylometric_weight,
        behavioral_weight=cfg.behavioral_weight,
        low_similarity_threshold=cfg.low_similarity_threshold,
    )
    similarity_matrix = similarities["similarity_matrix"]

    cluster_labels = run_dbscan_clustering(
        X_ai=X_ai,
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        use_precomputed_distance=cfg.use_precomputed_distance,
        similarity_matrix=similarity_matrix,
    )

    analysis = analyze_clusters(
        usernames_ai=usernames_ai,
        cluster_labels=cluster_labels,
        synthetic_scores_ai=synthetic_scores_ai,
        similarity_matrix=similarity_matrix,
    )

    threshold = cfg.edge_threshold
    if threshold <= 0:
        threshold = tune_similarity_threshold(similarity_matrix)

    graph = build_network_graph(
        usernames_ai=usernames_ai,
        similarity_matrix=similarity_matrix,
        cluster_labels=cluster_labels,
        synthetic_scores_ai=synthetic_scores_ai,
        similarity_threshold=threshold,
    )

    if graph_output_path is not None:
        visualize_graph(
            G=graph,
            output_path=graph_output_path,
            interactive=graph_interactive,
        )

    if save_clusters_csv_path is not None:
        save_cluster_assignments_csv(analysis["cluster_assignments"], save_clusters_csv_path)

    graph_json = export_graph_to_json(graph)
    if add_louvain_communities:
        community_map = detect_communities_louvain(graph)
        if community_map:
            for node in graph_json["nodes"]:
                node["louvain_community"] = community_map.get(node["id"], -1)

    if export_graph_json_path is not None:
        out_json = Path(export_graph_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(graph_json, indent=2), encoding="utf-8")

    result = {
        "cluster_assignments": analysis["cluster_assignments"],
        "clusters": analysis["clusters"],
        "isolated_personas": analysis["isolated_personas"],
        "graph": graph,
    }

    if cfg.include_similarity_matrix:
        result["similarity_matrix"] = similarity_matrix
        result["stylometric_similarity"] = similarities["stylometric_similarity"]
        result["behavioral_similarity"] = similarities["behavioral_similarity"]

    if cfg.include_cluster_stats:
        result["cluster_statistics"] = analysis["cluster_statistics"]

    result["api_response"] = {
        "cluster_assignments": result["cluster_assignments"],
        "clusters": result["clusters"],
        "isolated_personas": result["isolated_personas"],
        "cluster_statistics": analysis["cluster_statistics"],
        "graph": graph_json,
    }

    return result
