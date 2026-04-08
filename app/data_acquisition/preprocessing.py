"""Phase 2 preprocessing pipeline for raw snscrape-style outputs.

This module accepts raw social posts as either pandas DataFrames or Python
lists/dicts, applies cleaning and normalization, and returns a DataFrame that
is ready for downstream feature extraction and ML training.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd


# Pre-compiled regex patterns keep cleaning fast on large datasets.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

EXPECTED_COLUMNS = ["username", "post_text", "timestamp"]
FINAL_COLUMNS = [
    "username",
    "clean_text",
    "timestamp",
    "hour_of_post",
    "day_of_week",
    "original_text",
]


def _canonicalize_username(username: Any) -> str:
    """Normalize cross-platform aliases to one canonical identity string."""

    if pd.isna(username):
        return ""

    user = str(username).strip().lower()

    # Remove common platform wrappers and prefixes.
    user = re.sub(r"^(https?://)?(www\.)?(twitter|x|instagram|reddit|tiktok)\.com/", "", user)
    user = re.sub(r"^(twitter|x|instagram|reddit|tiktok)\s*[:/]+", "", user)
    user = user.lstrip("@")

    # Keep stable identity characters only.
    user = re.sub(r"[^a-z0-9_.]", "", user)
    return user


def text_cleaning(text: Any) -> str:
    """Clean post text for NLP.

    Steps:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove special characters and emojis (non-alphanumeric)
    4. Normalize whitespace
    """

    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALNUM_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/noisy rows after cleaning.

    Removes rows where text becomes empty or token count is too small for
    meaningful downstream feature extraction.
    """

    working = df.copy()
    working["clean_text"] = working["clean_text"].fillna("")

    token_counts = working["clean_text"].str.count(r"\S+")
    has_text = working["clean_text"].str.len() > 0
    has_min_tokens = token_counts >= 2

    # Keep semantically useful rows and de-duplicate exact repeats.
    filtered = working[has_text & has_min_tokens]
    filtered = filtered.drop_duplicates(subset=["username", "clean_text", "timestamp"], keep="first")
    return filtered.reset_index(drop=True)


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to UTC and derive hour/day temporal features."""

    working = df.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"])

    working["hour_of_post"] = working["timestamp"].dt.hour.astype("int16")
    working["day_of_week"] = working["timestamp"].dt.day_name()
    return working.reset_index(drop=True)


def tokenize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Tokenize cleaned text into word-level tokens.

    Tokenization is intentionally lightweight and language-agnostic to keep the
    pipeline fast for large social datasets.
    """

    working = df.copy()
    working["_tokens"] = working["clean_text"].str.findall(TOKEN_PATTERN)
    return working


def identity_resolution(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve duplicate identities and cross-platform username variants."""

    working = df.copy()
    working["username"] = working["username"].map(_canonicalize_username)
    working = working[working["username"].str.len() > 0]
    return working.reset_index(drop=True)


def _normalize_single_source(source: Any) -> pd.DataFrame:
    """Normalize one source object into a DataFrame with expected columns."""

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    elif isinstance(source, dict):
        frame = pd.DataFrame([source])
    elif isinstance(source, list) and source and isinstance(source[0], dict):
        frame = pd.DataFrame(source)
    else:
        raise TypeError(
            "Unsupported raw_data source type. Use a DataFrame, dict, "
            "or list of dictionaries."
        )

    rename_map = {}
    if "text" in frame.columns and "post_text" not in frame.columns:
        rename_map["text"] = "post_text"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    missing = [col for col in EXPECTED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return frame[EXPECTED_COLUMNS].copy()


def _coerce_raw_to_dataframe(raw_data: Any) -> pd.DataFrame:
    """Merge one or many scrape outputs into a single canonical DataFrame."""

    if isinstance(raw_data, pd.DataFrame):
        return _normalize_single_source(raw_data)

    # Support automatic merge when multiple scrape batches are provided.
    if isinstance(raw_data, Iterable) and not isinstance(raw_data, (str, bytes, dict)):
        frames: list[pd.DataFrame] = []
        for item in raw_data:
            if isinstance(item, (pd.DataFrame, dict)):
                frames.append(_normalize_single_source(item))
            elif isinstance(item, list) and (not item or isinstance(item[0], dict)):
                frames.append(_normalize_single_source(item))
            else:
                raise TypeError(
                    "Unsupported item in raw_data iterable. "
                    "Expected DataFrame, dict, or list of dicts per item."
                )

        if not frames:
            return pd.DataFrame(columns=EXPECTED_COLUMNS)

        return pd.concat(frames, ignore_index=True, copy=False)

    return _normalize_single_source(raw_data)


def preprocess_pipeline(raw_data: Any) -> pd.DataFrame:
    """Run the complete Phase 2 preprocessing flow.

    Args:
        raw_data: One of the following:
            - pandas DataFrame with username, post_text/text, timestamp
            - list of dictionaries with username, post_text/text, timestamp
            - iterable of scrape batches (each batch can be a DataFrame,
              list of dicts, or a dict)

    Returns:
        Preprocessed DataFrame with columns:
        username, clean_text, timestamp, hour_of_post, day_of_week, original_text
    """

    df = _coerce_raw_to_dataframe(raw_data)

    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS)

    # Keep untouched text for auditability and model-debug traceability.
    df = df.rename(columns={"post_text": "original_text"})
    df["original_text"] = df["original_text"].fillna("").astype(str)

    # Vectorized cleaning keeps the pipeline efficient on large datasets.
    df["clean_text"] = df["original_text"].map(text_cleaning)

    df = identity_resolution(df)
    df = remove_noise(df)
    df = normalize_timestamps(df)
    df = tokenize_text(df)

    # Enforce final schema expected by Phase 3 feature extraction.
    df = df[FINAL_COLUMNS].copy()
    return df.reset_index(drop=True)
