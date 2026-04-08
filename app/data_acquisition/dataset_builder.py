"""Build a combined human/AI dataset for persona detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import (
    DEFAULT_AI_POST_COUNT,
    DEFAULT_TWEET_COUNT,
    DATASET_PATH,
    PREPROCESSED_DATASET_PATH,
    ensure_directories,
)
from app.data_acquisition.ai_generator import generate_ai_persona
from app.data_acquisition.preprocessing import preprocess_pipeline
from app.data_acquisition.twitter_scraper import fetch_tweets
from app.utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def build_dataset(
    username: str,
    max_human_tweets: int = DEFAULT_TWEET_COUNT,
    num_ai_posts: int = DEFAULT_AI_POST_COUNT,
) -> pd.DataFrame:
    """Build, shuffle, and save the combined dataset.

    Args:
        username: Username to scrape for human tweets and anchor the AI persona.
        max_human_tweets: Maximum number of human tweets to scrape.
        num_ai_posts: Number of AI posts to generate.

    Returns:
        A shuffled pandas DataFrame with columns username, text, timestamp, label.

    Side effects:
        Saves both:
        - Raw combined dataset to DATASET_PATH
        - Phase 2 preprocessed dataset to PREPROCESSED_DATASET_PATH
    """

    ensure_directories()

    human_tweets = fetch_tweets(username=username, max_tweets=max_human_tweets)
    ai_tweets = generate_ai_persona(username=username, num_posts=num_ai_posts)

    human_rows: list[dict[str, Any]] = [
        {"username": tweet["username"], "text": tweet["text"], "timestamp": tweet["timestamp"], "label": 0}
        for tweet in human_tweets
    ]
    ai_rows: list[dict[str, Any]] = [
        {"username": tweet["username"], "text": tweet["text"], "timestamp": tweet["timestamp"], "label": 1}
        for tweet in ai_tweets
    ]

    if not human_rows:
        logger.warning("No human tweets were collected for %s.", username)
    if not ai_rows:
        logger.warning("No AI tweets were generated for %s.", username)

    dataset = pd.DataFrame(human_rows + ai_rows, columns=["username", "text", "timestamp", "label"])

    if dataset.empty:
        raise ValueError("Unable to build dataset because both human and AI rows are empty.")

    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    dataset.to_csv(DATASET_PATH, index=False)

    # Build Phase 2-ready preprocessing output from the same merged raw rows.
    preprocessing_input = dataset[["username", "text", "timestamp"]].rename(columns={"text": "post_text"})
    preprocessed = preprocess_pipeline(preprocessing_input)

    # Re-attach labels after preprocessing by matching canonical row keys.
    if not preprocessed.empty:
        label_map_source = dataset.rename(columns={"text": "original_text"})[
            ["username", "original_text", "timestamp", "label"]
        ].copy()
        label_map_source["timestamp"] = pd.to_datetime(label_map_source["timestamp"], utc=True, errors="coerce")

        preprocessed = preprocessed.merge(
            label_map_source,
            how="left",
            on=["username", "original_text", "timestamp"],
        )

    preprocessed.to_csv(PREPROCESSED_DATASET_PATH, index=False)

    logger.info("Saved dataset with %d rows to %s", len(dataset), DATASET_PATH)
    logger.info("Saved preprocessed dataset with %d rows to %s", len(preprocessed), PREPROCESSED_DATASET_PATH)
    return dataset
