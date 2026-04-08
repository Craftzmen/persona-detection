"""Synthetic AI persona generation utilities."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from app.utils.logging_utils import setup_logging

logger = setup_logging(__name__)

TONE_PROFILES = {
    "casual": [
        "just thinking about {topic} again.",
        "small update on {topic}: still testing the idea.",
        "not sure yet, but {topic} keeps coming back to mind.",
    ],
    "tech": [
        "experimenting with {topic} and comparing a few patterns.",
        "the signal around {topic} is interesting, but I want another pass.",
        "testing a slightly different angle on {topic} today.",
    ],
    "student": [
        "taking notes on {topic} and simplifying the core idea.",
        "reviewing {topic} one more time before I wrap up.",
        "still learning how {topic} connects to the bigger picture.",
    ],
    "professional": [
        "sharing a concise update on {topic} and the current direction.",
        "the first pass on {topic} looks promising, but more validation is needed.",
        "documenting a brief progress note on {topic} for today.",
    ],
}

TOPICS = [
    "automation",
    "workflow design",
    "model evaluation",
    "research notes",
    "data quality",
    "product strategy",
    "prompt iteration",
    "analysis pipelines",
    "feature extraction",
    "experiment tracking",
]

REPETITION_PHRASES = [
    "same thought, different angle.",
    "still circling back to this.",
    "worth revisiting later.",
    "keeping this on repeat for now.",
]

USERNAME_PREFIXES = ["neo", "byte", "nova", "echo", "orbit", "atlas", "signal", "pixel"]
USERNAME_SUFFIXES = ["think", "notes", "labs", "logic", "pulse", "craft", "wave", "mind"]


def _generate_username() -> str:
    """Generate a realistic fallback username."""

    prefix = random.choice(USERNAME_PREFIXES)
    suffix = random.choice(USERNAME_SUFFIXES)
    return f"{prefix}_{suffix}_{random.randint(10, 999)}"


def _build_timestamps(count: int) -> list[str]:
    """Create realistic timestamps spaced at irregular intervals."""

    current_time = datetime.now() - timedelta(days=random.randint(1, 14))
    timestamps: list[str] = []

    for _ in range(count):
        current_time += timedelta(minutes=random.randint(15, 240), seconds=random.randint(0, 59))
        timestamps.append(current_time.isoformat())

    return timestamps


def _build_tweet_text(tone: str, topic: str, repeated_text: str | None = None) -> str:
    """Create a tweet-like sentence with light repetition."""

    if repeated_text and random.random() < 0.35:
        return repeated_text

    template = random.choice(TONE_PROFILES[tone])
    text = template.format(topic=topic)

    if random.random() < 0.25:
        text = f"{text} {random.choice(REPETITION_PHRASES)}"

    return text


def generate_ai_persona(username: str = "", num_posts: int = 50) -> list[dict[str, Any]]:
    """Generate a synthetic AI persona tweet dataset.

    Args:
        username: Persona username. A fallback username is generated when blank.
        num_posts: Number of synthetic posts to create.

    Returns:
        A list of dictionaries with username, text, and timestamp fields.
    """

    if num_posts <= 0:
        logger.warning("num_posts must be positive; received %s.", num_posts)
        return []

    persona_username = username.strip() if username and username.strip() else _generate_username()
    timestamps = _build_timestamps(num_posts)
    posts: list[dict[str, Any]] = []
    repeated_text: str | None = None

    for index in range(num_posts):
        tone = random.choice(list(TONE_PROFILES.keys()))
        topic = random.choice(TOPICS)
        text = _build_tweet_text(tone, topic, repeated_text=repeated_text)

        if index % 6 == 0:
            repeated_text = text

        posts.append(
            {
                "username": persona_username,
                "text": text,
                "timestamp": timestamps[index],
            }
        )

    logger.info("Generated %d AI persona posts for %s", len(posts), persona_username)
    return posts
