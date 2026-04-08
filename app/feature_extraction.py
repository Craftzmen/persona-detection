"""Phase 3 feature extraction for AI persona detection.

This module transforms a preprocessed post-level dataset into user-level feature
matrices suitable for machine learning. It includes:

1. Stylometric features
2. Behavioral/temporal features
3. Interaction-network features
4. TF-IDF text vectors

The primary entry point is ``FeatureExtractor.extract_features``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils.logging_utils import setup_logging


logger = setup_logging(__name__)


REQUIRED_COLUMNS = {"username", "clean_text", "timestamp", "label"}
OPTIONAL_INTERACTION_COLUMNS = ("likes", "replies", "retweets")
OPTIONAL_NETWORK_COLUMNS = (
    "interacted_with",
    "reply_to",
    "mentions",
    "retweet_of",
    "quoted_user",
)

SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
TOKEN_PATTERN = re.compile(r"\b\w+\b")
PUNCT_PATTERN = re.compile(r"[.!?,;:]")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")
REPEATED_PUNCT_PATTERN = re.compile(r"[!?.,]{2,}")
NUMBERS_PATTERN = re.compile(r"\d+")


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for feature extraction behavior.

    Attributes:
        tfidf_max_features: Maximum TF-IDF feature count.
        tfidf_ngram_range: N-gram range for TF-IDF.
        min_posts_per_user: Minimum posts required to keep a user.
        coordination_threshold: Correlation threshold for coordinated activity.
        include_visualizations: If True, plotting helpers may be called.
    """

    tfidf_max_features: int = 400
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    min_posts_per_user: int = 1
    coordination_threshold: float = 0.85
    include_visualizations: bool = False


def _safe_div(numerator: float, denominator: float) -> float:
    """Return a safe division result or 0.0 when denominator is zero."""

    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_label(value: Any) -> int:
    """Normalize mixed label formats into binary integer labels.

    Supported values include {0, 1, "human", "ai", "synthetic", "bot"}.
    """

    if pd.isna(value):
        return -1

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)
        return -1

    text = str(value).strip().lower()
    if text in {"0", "human", "real", "organic"}:
        return 0
    if text in {"1", "ai", "synthetic", "bot", "generated"}:
        return 1

    return -1


def _parse_targets(value: Any) -> list[str]:
    """Convert an interaction target field into a list of usernames."""

    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip().lower() for item in value]
    else:
        raw_items = [chunk.strip().lower() for chunk in str(value).split(",")]

    return [item.lstrip("@") for item in raw_items if item and item != "nan"]


class FeatureExtractor:
    """Extract Phase 3 features from a preprocessed social dataset."""

    def __init__(self, config: FeatureExtractionConfig | None = None) -> None:
        self.config = config or FeatureExtractionConfig()
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            lowercase=True,
        )

    def extract_features(
        self,
        dataset: pd.DataFrame,
        save_debug_csv: bool = False,
        debug_output_dir: str | Path | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a model-ready feature matrix X and labels y.

        Args:
            dataset: Preprocessed post-level DataFrame with required columns:
                username, clean_text, timestamp, label.
            save_debug_csv: If True, save merged feature matrix and labels.
            debug_output_dir: Optional output folder for debug exports.

        Returns:
            A tuple ``(X, y)`` where X is a user-level numeric feature matrix and
            y is the corresponding binary label series.
        """

        df = self._validate_and_prepare(dataset)
        if df.empty:
            logger.warning("Input dataset is empty after preparation.")
            return pd.DataFrame(), pd.Series(dtype="int8", name="label")

        stylometric_df = self.extract_stylometric_features(df)
        behavioral_df = self.extract_behavioral_features(df)
        network_df = self.extract_network_features(df)
        tfidf_df = self.vectorize_text(df)

        labels = self._build_user_labels(df)

        merged = stylometric_df.merge(behavioral_df, on="username", how="outer")
        merged = merged.merge(network_df, on="username", how="left")
        merged = merged.merge(tfidf_df, on="username", how="left")
        merged = merged.merge(labels, on="username", how="inner")

        merged = merged.fillna(0.0)
        y = merged["label"].astype("int8")
        X = merged.drop(columns=["label", "username"])

        if save_debug_csv:
            self._save_debug_outputs(merged, X, y, debug_output_dir)

        return X, y

    def summarize_per_user(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptive summary statistics for each user.

        This function is useful for quick diagnostics and dataset exploration.
        """

        df = self._validate_and_prepare(dataset)
        if df.empty:
            return pd.DataFrame()

        grouped = df.groupby("username", observed=True)

        summary = grouped.agg(
            post_count=("clean_text", "count"),
            avg_post_length_chars=("clean_text", lambda s: s.str.len().mean()),
            avg_word_count=("clean_text", lambda s: s.str.count(r"\S+").mean()),
            first_post=("timestamp", "min"),
            last_post=("timestamp", "max"),
        ).reset_index()

        active_days = grouped["timestamp"].agg(
            lambda s: max((s.max().date() - s.min().date()).days + 1, 1)
        )
        summary = summary.merge(
            active_days.rename("active_days").reset_index(),
            on="username",
            how="left",
        )
        summary["posts_per_day"] = summary["post_count"] / summary["active_days"].clip(lower=1)
        return summary

    def extract_stylometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract user-level stylometric features from post text."""

        working = df[["username", "clean_text"]].copy()
        source_text = self._best_text_column(df)

        working["tokens"] = working["clean_text"].str.findall(TOKEN_PATTERN)
        working["word_count"] = working["tokens"].str.len().astype("int32")
        working["char_count"] = working["tokens"].map(lambda t: sum(len(x) for x in t)).astype("int32")
        working["avg_word_length_post"] = (
            working["char_count"] / working["word_count"].replace(0, np.nan)
        ).fillna(0.0)

        sentence_parts = source_text.fillna("").astype(str).str.split(SENTENCE_SPLIT_PATTERN)
        sentence_lengths = sentence_parts.map(
            lambda parts: np.mean([len(TOKEN_PATTERN.findall(p)) for p in parts if p.strip()])
            if any(p.strip() for p in parts)
            else 0.0
        )
        punct_counts = source_text.fillna("").astype(str).str.count(PUNCT_PATTERN)
        text_lengths = source_text.fillna("").astype(str).str.len()

        working["avg_sentence_length_post"] = sentence_lengths.astype(float)
        working["punctuation_usage_post"] = (punct_counts / text_lengths.replace(0, np.nan)).fillna(0.0)
        working["grammar_consistency_post"] = self._grammar_consistency_score(source_text)

        grouped = working.groupby("username", observed=True)
        aggregated = grouped.agg(
            word_count=("word_count", "mean"),
            avg_word_length=("avg_word_length_post", "mean"),
            avg_sentence_length=("avg_sentence_length_post", "mean"),
            punctuation_usage=("punctuation_usage_post", "mean"),
            grammar_consistency=("grammar_consistency_post", "mean"),
        ).reset_index()

        vocab = grouped["tokens"].agg(lambda lists: [tok for tokens in lists for tok in tokens])
        vocab_stats = vocab.map(
            lambda tokens: pd.Series(
                {
                    "unique_word_count": len(set(tokens)),
                    "total_word_count": len(tokens),
                    "vocabulary_richness": _safe_div(len(set(tokens)), len(tokens)),
                }
            )
        )
        vocab_df = vocab_stats.reset_index()

        result = aggregated.merge(vocab_df, on="username", how="left")
        return result.drop(columns=["total_word_count"], errors="ignore")

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and interaction behavior features per user."""

        working = df[["username", "timestamp", "clean_text"]].copy()
        grouped = working.groupby("username", observed=True)

        date_counts = (
            working.assign(post_date=working["timestamp"].dt.date)
            .groupby(["username", "post_date"], observed=True)
            .size()
            .rename("posts_on_date")
            .reset_index()
        )
        posts_per_day = date_counts.groupby("username", observed=True)["posts_on_date"].mean()

        sorted_df = working.sort_values(["username", "timestamp"])
        sorted_df["delta_seconds"] = (
            sorted_df.groupby("username", observed=True)["timestamp"].diff().dt.total_seconds()
        )
        interval_feature = sorted_df.groupby("username", observed=True)["delta_seconds"].mean().fillna(0.0)

        hour_dist = pd.crosstab(working["username"], working["timestamp"].dt.hour, normalize="index")
        hour_dist = hour_dist.reindex(columns=range(24), fill_value=0.0)
        hour_dist.columns = [f"hour_{hour:02d}_ratio" for hour in hour_dist.columns]

        night_ratio = grouped["timestamp"].agg(lambda s: s.dt.hour.between(0, 5).mean()).rename(
            "night_activity_ratio"
        )

        features = pd.DataFrame(
            {
                "posts_per_day": posts_per_day,
                "time_between_posts": interval_feature,
                "night_activity_ratio": night_ratio,
            }
        ).reset_index()

        features = features.merge(hour_dist.reset_index(), on="username", how="left")

        interaction_df = self._extract_interaction_behavior(df)
        automation_df = self._extract_automation_patterns(df)

        features = features.merge(interaction_df, on="username", how="left")
        features = features.merge(automation_df, on="username", how="left")

        return features.fillna(0.0)

    def extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction graph features and community/coordinated signals."""

        graph = self._build_interaction_graph(df)
        users = sorted(df["username"].unique())

        if graph.number_of_nodes() == 0:
            logger.warning(
                "No interaction columns available; network features default to zeros."
            )
            return pd.DataFrame(
                {
                    "username": users,
                    "in_degree": 0.0,
                    "out_degree": 0.0,
                    "betweenness": 0.0,
                    "pagerank": 0.0,
                    "clustering": 0.0,
                    "community_id": -1,
                    "community_size": 0,
                    "coordination_score": 0.0,
                    "coordination_flag": 0,
                }
            )

        graph.add_nodes_from(users)

        in_degree = dict(graph.in_degree(weight="weight"))
        out_degree = dict(graph.out_degree(weight="weight"))
        undirected = graph.to_undirected()

        betweenness = nx.betweenness_centrality(undirected, normalized=True, weight="weight")
        pagerank = nx.pagerank(graph, weight="weight")
        clustering = nx.clustering(undirected, weight="weight")

        community_map, community_size = self._community_detection(undirected)
        coordination = self._coordination_features(df)

        rows = []
        for user in users:
            rows.append(
                {
                    "username": user,
                    "in_degree": float(in_degree.get(user, 0.0)),
                    "out_degree": float(out_degree.get(user, 0.0)),
                    "betweenness": float(betweenness.get(user, 0.0)),
                    "pagerank": float(pagerank.get(user, 0.0)),
                    "clustering": float(clustering.get(user, 0.0)),
                    "community_id": int(community_map.get(user, -1)),
                    "community_size": int(community_size.get(user, 0)),
                    "coordination_score": float(coordination.get(user, {}).get("score", 0.0)),
                    "coordination_flag": int(coordination.get(user, {}).get("flag", 0)),
                }
            )

        return pd.DataFrame(rows)

    def vectorize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate user-level TF-IDF vectors from clean text."""

        user_documents = (
            df.groupby("username", observed=True)["clean_text"]
            .agg(lambda posts: " ".join(posts.astype(str)))
            .reset_index()
        )

        if user_documents.empty:
            return pd.DataFrame(columns=["username"])

        tfidf_matrix = self.vectorizer.fit_transform(user_documents["clean_text"])
        tfidf_names = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_names, index=user_documents.index)
        tfidf_df.insert(0, "username", user_documents["username"].values)
        return tfidf_df

    def visualize_distributions(
        self,
        feature_frame: pd.DataFrame,
        output_dir: str | Path | None = None,
    ) -> None:
        """Optionally plot selected feature distributions for diagnostics."""

        if feature_frame.empty:
            logger.warning("Feature frame is empty; skipping visualization.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed; skipping visualization.")
            return

        plots = [
            "word_count",
            "vocabulary_richness",
            "posts_per_day",
            "night_activity_ratio",
            "coordination_score",
        ]
        available = [feature for feature in plots if feature in feature_frame.columns]
        if not available:
            logger.warning("No target features available for visualization.")
            return

        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = None

        for feature in available:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(feature_frame[feature], bins=30)
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")
            fig.tight_layout()

            if out_dir is not None:
                fig.savefig(out_dir / f"{feature}_distribution.png", dpi=140)
                plt.close(fig)
            else:
                plt.show()

    def _validate_and_prepare(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and prepare canonical dataframe types."""

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame")

        missing = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        df = dataset.copy()
        df["username"] = df["username"].fillna("").astype(str).str.strip().str.lower()
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["label"] = df["label"].map(_normalize_label)

        missing_ts = int(df["timestamp"].isna().sum())
        invalid_labels = int((df["label"] == -1).sum())

        if missing_ts:
            logger.warning("Dropping %d rows with invalid timestamps.", missing_ts)
        if invalid_labels:
            logger.warning("Dropping %d rows with unsupported labels.", invalid_labels)

        df = df.dropna(subset=["timestamp"])
        df = df[df["label"] != -1]
        df = df[df["username"] != ""]

        if self.config.min_posts_per_user > 1:
            counts = df["username"].value_counts()
            keep_users = counts[counts >= self.config.min_posts_per_user].index
            dropped_users = len(counts) - len(keep_users)
            if dropped_users > 0:
                logger.info(
                    "Dropping %d users with fewer than %d posts.",
                    dropped_users,
                    self.config.min_posts_per_user,
                )
            df = df[df["username"].isin(keep_users)]

        return df.reset_index(drop=True)

    def _grammar_consistency_score(self, source_text: pd.Series) -> pd.Series:
        """Compute a lightweight grammar consistency score per post.

        Rule-based signals:
        - Penalize repeated punctuation (e.g., "!!!")
        - Penalize multiple consecutive spaces
        - Reward sentence-ending punctuation where available
        """

        text = source_text.fillna("").astype(str)
        repeated_punct = text.str.contains(REPEATED_PUNCT_PATTERN, regex=True).astype(float)
        multi_space = text.str.contains(MULTISPACE_PATTERN, regex=True).astype(float)
        ends_with_terminal_punct = text.str.strip().str.endswith((".", "!", "?")).astype(float)

        score = 1.0 - (0.4 * repeated_punct + 0.3 * multi_space) + 0.3 * ends_with_terminal_punct
        return score.clip(lower=0.0, upper=1.0)

    def _best_text_column(self, df: pd.DataFrame) -> pd.Series:
        """Select the richest text column available for style heuristics."""

        for candidate in ("original_text", "post_text", "text", "clean_text"):
            if candidate in df.columns:
                return df[candidate]
        return df["clean_text"]

    def _extract_interaction_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate likes/replies/retweets if those columns exist."""

        available = [col for col in OPTIONAL_INTERACTION_COLUMNS if col in df.columns]
        if not available:
            logger.info("Interaction metrics not found; defaulting interaction features to zero.")
            users = sorted(df["username"].unique())
            return pd.DataFrame(
                {
                    "username": users,
                    "avg_likes": 0.0,
                    "avg_replies": 0.0,
                    "avg_retweets": 0.0,
                    "total_interactions": 0.0,
                }
            )

        numeric = df[["username", *available]].copy()
        for col in available:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0.0)

        grouped = numeric.groupby("username", observed=True)

        result = pd.DataFrame({"username": sorted(df["username"].unique())})
        for col in OPTIONAL_INTERACTION_COLUMNS:
            feature_name = f"avg_{col}"
            if col in available:
                feature = grouped[col].mean().rename(feature_name).reset_index()
                result = result.merge(feature, on="username", how="left")
            else:
                result[feature_name] = 0.0

        totals = np.zeros(len(result), dtype=float)
        for col in available:
            tmp = grouped[col].sum().reset_index().rename(columns={col: "tmp_sum"})
            result = result.merge(tmp, on="username", how="left")
            totals += result["tmp_sum"].fillna(0.0).to_numpy()
            result = result.drop(columns=["tmp_sum"])
        result["total_interactions"] = totals

        return result.fillna(0.0)

    def _extract_automation_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate automation likelihood via repeated/templated text usage."""

        working = df[["username", "clean_text"]].copy()
        working["normalized_template"] = (
            working["clean_text"].str.lower().str.replace(NUMBERS_PATTERN, "<num>", regex=True)
        )

        grouped = working.groupby("username", observed=True)

        duplicated_ratio = grouped["clean_text"].agg(
            lambda s: _safe_div(s.duplicated(keep=False).sum(), len(s))
        )
        template_ratio = grouped["normalized_template"].agg(
            lambda s: _safe_div(s.duplicated(keep=False).sum(), len(s))
        )
        short_post_ratio = grouped["clean_text"].agg(lambda s: (s.str.count(r"\S+") <= 4).mean())

        result = pd.DataFrame(
            {
                "username": duplicated_ratio.index,
                "repeated_post_ratio": duplicated_ratio.values,
                "templated_post_ratio": template_ratio.values,
                "short_post_ratio": short_post_ratio.values,
            }
        )
        result["automation_pattern_score"] = (
            0.45 * result["repeated_post_ratio"]
            + 0.45 * result["templated_post_ratio"]
            + 0.10 * result["short_post_ratio"]
        )
        return result

    def _build_interaction_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Create a directed graph from available interaction columns."""

        graph = nx.DiGraph()

        available_cols = [col for col in OPTIONAL_NETWORK_COLUMNS if col in df.columns]
        if not available_cols:
            return graph

        for _, row in df.iterrows():
            user = str(row["username"]).strip().lower()
            if not user:
                continue

            for col in available_cols:
                for target in _parse_targets(row[col]):
                    if not target or target == user:
                        continue

                    if graph.has_edge(user, target):
                        graph[user][target]["weight"] += 1.0
                    else:
                        graph.add_edge(user, target, weight=1.0)

        return graph

    def _community_detection(self, graph: nx.Graph) -> tuple[dict[str, int], dict[str, int]]:
        """Assign each user to a graph community and return community sizes."""

        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            return {}, {}

        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
        community_map: dict[str, int] = {}
        community_size: dict[str, int] = {}

        for idx, members in enumerate(communities):
            size = len(members)
            for user in members:
                community_map[str(user)] = idx
                community_size[str(user)] = size

        return community_map, community_size

    def _coordination_features(self, df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        """Detect coordinated behavior from correlated hourly posting patterns."""

        user_hour = (
            df.assign(hour_bin=df["timestamp"].dt.floor("h"))
            .groupby(["username", "hour_bin"], observed=True)
            .size()
            .rename("post_count")
            .reset_index()
        )

        if user_hour.empty:
            return {}

        matrix = user_hour.pivot(index="hour_bin", columns="username", values="post_count").fillna(0.0)
        if matrix.shape[1] <= 1:
            user = matrix.columns[0]
            return {str(user): {"score": 0.0, "flag": 0}}

        corr = matrix.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        features: dict[str, dict[str, float | int]] = {}
        for user in corr.columns:
            peers = corr.loc[user].drop(index=user)
            if peers.empty:
                features[str(user)] = {"score": 0.0, "flag": 0}
                continue

            score = float(peers.mean())
            high_corr = bool((peers >= self.config.coordination_threshold).any())
            features[str(user)] = {"score": score, "flag": int(high_corr)}

        return features

    def _build_user_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create one label per user; mixed users default to majority label."""

        grouped = df.groupby("username", observed=True)["label"]
        label_stats = grouped.agg(["mean", "nunique", "count"]).reset_index()

        mixed = label_stats[label_stats["nunique"] > 1]
        if not mixed.empty:
            logger.warning(
                "Detected %d users with mixed labels; using majority label.",
                len(mixed),
            )

        label_stats["label"] = (label_stats["mean"] >= 0.5).astype("int8")
        return label_stats[["username", "label"]]

    def _save_debug_outputs(
        self,
        merged: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        debug_output_dir: str | Path | None,
    ) -> None:
        """Persist debug CSV outputs for feature inspection."""

        out_dir = Path(debug_output_dir) if debug_output_dir else Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        merged_path = out_dir / "feature_matrix_with_labels.csv"
        X_path = out_dir / "X_features.csv"
        y_path = out_dir / "y_labels.csv"

        merged.to_csv(merged_path, index=False)
        X.to_csv(X_path, index=False)
        y.to_frame(name="label").to_csv(y_path, index=False)

        logger.info("Saved debug feature outputs to %s", out_dir.resolve())


def build_feature_matrix(
    dataset: pd.DataFrame,
    config: FeatureExtractionConfig | None = None,
    save_debug_csv: bool = False,
    debug_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience function for one-shot feature extraction."""

    extractor = FeatureExtractor(config=config)
    return extractor.extract_features(
        dataset=dataset,
        save_debug_csv=save_debug_csv,
        debug_output_dir=debug_output_dir,
    )
