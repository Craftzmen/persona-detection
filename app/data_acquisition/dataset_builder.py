"""Build canonical datasets for persona detection.

By default this module uses pre-built benchmark datasets. Live scraping can be
enabled by setting DATA_SOURCE_MODE=live for future flexibility.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import (
    DATA_SOURCE_MODE,
    DATA_SOURCE_MODES,
    DEFAULT_AI_POST_COUNT,
    DEFAULT_TWEET_COUNT,
    DATASET_PATH,
    PREBUILT_DATASET_MERGED_PATH,
    PREPROCESSED_DATASET_PATH,
    ensure_directories,
)
from app.data_acquisition.ai_generator import generate_ai_persona
from app.data_acquisition.prebuilt_datasets import load_all_prebuilt_datasets, merge_prebuilt_datasets
from app.data_acquisition.preprocessing import preprocess_pipeline
from app.data_acquisition.twitter_scraper import fetch_tweets
from app.utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def build_dataset(
    username: str | None = None,
    max_human_tweets: int = DEFAULT_TWEET_COUNT,
    num_ai_posts: int = DEFAULT_AI_POST_COUNT,
) -> pd.DataFrame:
    """Build, shuffle, and save the combined dataset.

    Args:
        username: Optional username used only in live mode.
        max_human_tweets: Maximum number of human tweets to scrape in live mode.
        num_ai_posts: Number of AI posts to generate in live mode.

    Returns:
        A shuffled pandas DataFrame with columns:
        username, tweet_text, timestamp, label, dataset_name.

    Side effects:
        Saves both:
        - Raw combined dataset to DATASET_PATH and PREBUILT_DATASET_MERGED_PATH
        - Phase 2 preprocessed dataset to PREPROCESSED_DATASET_PATH
    """

    ensure_directories()

    if DATA_SOURCE_MODE not in DATA_SOURCE_MODES:
        raise ValueError(
            f"Unsupported DATA_SOURCE_MODE='{DATA_SOURCE_MODE}'. Expected one of {sorted(DATA_SOURCE_MODES)}"
        )

    if DATA_SOURCE_MODE == "dataset":
        datasets = load_all_prebuilt_datasets()
        dataset = merge_prebuilt_datasets(datasets)
        if dataset.empty:
            raise ValueError(
                "No rows loaded from pre-built datasets. Configure TWIBOT22_SOURCE, TWIBOT20_SOURCE, "
                "CRESCI_SOURCE, and PAN2019_SOURCE with valid files."
            )
    else:
        if not username:
            raise ValueError("username is required when DATA_SOURCE_MODE=live")

        human_tweets = fetch_tweets(username=username, max_tweets=max_human_tweets)
        ai_tweets = generate_ai_persona(username=username, num_posts=num_ai_posts)

        human_rows: list[dict[str, Any]] = [
            {
                "username": tweet["username"],
                "tweet_text": tweet["text"],
                "timestamp": tweet["timestamp"],
                "label": "human",
                "dataset_name": "live_human",
            }
            for tweet in human_tweets
        ]
        ai_rows: list[dict[str, Any]] = [
            {
                "username": tweet["username"],
                "tweet_text": tweet["text"],
                "timestamp": tweet["timestamp"],
                "label": "ai",
                "dataset_name": "live_ai",
            }
            for tweet in ai_tweets
        ]

        if not human_rows:
            logger.warning("No human tweets were collected for %s.", username)
        if not ai_rows:
            logger.warning("No AI tweets were generated for %s.", username)

        dataset = pd.DataFrame(
            human_rows + ai_rows,
            columns=["username", "tweet_text", "timestamp", "label", "dataset_name"],
        )

        if dataset.empty:
            raise ValueError("Unable to build dataset because both human and AI rows are empty.")

    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dataset.to_csv(PREBUILT_DATASET_MERGED_PATH, index=False)
    dataset.to_csv(DATASET_PATH, index=False)

    # Build Phase 2-ready preprocessing output from the same merged raw rows.
    preprocessing_input = dataset[["username", "tweet_text", "timestamp"]].rename(
        columns={"tweet_text": "post_text"}
    )
    preprocessed = preprocess_pipeline(preprocessing_input)

    # Re-attach labels after preprocessing by matching canonical row keys.
    if not preprocessed.empty:
        label_map_source = dataset.rename(columns={"tweet_text": "original_text"})[
            ["username", "original_text", "timestamp", "label", "dataset_name"]
        ].copy()
        label_map_source["timestamp"] = pd.to_datetime(label_map_source["timestamp"], utc=True, errors="coerce")

        preprocessed = preprocessed.merge(
            label_map_source,
            how="left",
            on=["username", "original_text", "timestamp"],
        )

    preprocessed.to_csv(PREPROCESSED_DATASET_PATH, index=False)

    logger.info("Saved dataset with %d rows to %s", len(dataset), DATASET_PATH)
    logger.info("Saved merged prebuilt dataset with %d rows to %s", len(dataset), PREBUILT_DATASET_MERGED_PATH)
    logger.info("Saved preprocessed dataset with %d rows to %s", len(preprocessed), PREPROCESSED_DATASET_PATH)
    return dataset
