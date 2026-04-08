"""Twitter/X scraping utilities powered by the X API v2."""

from __future__ import annotations

from urllib.parse import quote
from typing import Any

import requests

from app.config import DEFAULT_TWEET_COUNT, X_API_BASE_URL, X_BEARER_TOKEN
from app.utils.logging_utils import setup_logging

logger = setup_logging(__name__)


def _build_headers() -> dict[str, str]:
    """Build API request headers for X API."""

    return {
        "Authorization": f"Bearer {X_BEARER_TOKEN}",
        "Content-Type": "application/json",
    }


def _get_user_id(username: str) -> str | None:
    """Resolve a username to an X user id."""

    endpoint = f"{X_API_BASE_URL}/users/by/username/{quote(username)}"
    params = {"user.fields": "id,username"}

    try:
        response = requests.get(endpoint, headers=_build_headers(), params=params, timeout=20)
        if response.status_code == 404:
            logger.warning("Username not found on X API: %s", username)
            return None
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.exception("Failed to resolve username %s: %s", username, exc)
        return None

    user_data = payload.get("data", {})
    user_id = user_data.get("id")
    if not user_id:
        logger.warning("X API did not return a user id for %s", username)
        return None

    return user_id


def fetch_tweets(username: str, max_tweets: int = DEFAULT_TWEET_COUNT) -> list[dict[str, Any]]:
    """Fetch recent tweets for a given username.

    Args:
        username: The Twitter/X username to scrape.
        max_tweets: Maximum number of tweets to collect.

    Returns:
        A list of dictionaries with username, text, and timestamp fields.
    """

    if not username or not username.strip():
        logger.error("Cannot fetch tweets without a valid username.")
        return []

    if max_tweets <= 0:
        logger.warning("max_tweets must be positive; received %s.", max_tweets)
        return []

    if not X_BEARER_TOKEN:
        logger.error("Missing X_BEARER_TOKEN. Add it to your .env file before scraping.")
        return []

    cleaned_username = username.strip().lstrip("@")
    user_id = _get_user_id(cleaned_username)
    if not user_id:
        return []

    tweets: list[dict[str, Any]] = []
    next_token: str | None = None

    while len(tweets) < max_tweets:
        remaining = max_tweets - len(tweets)
        page_size = min(remaining, 100)
        params: dict[str, Any] = {
            "max_results": page_size,
            "tweet.fields": "created_at",
            "exclude": "retweets,replies",
        }
        if next_token:
            params["pagination_token"] = next_token

        endpoint = f"{X_API_BASE_URL}/users/{user_id}/tweets"

        try:
            response = requests.get(endpoint, headers=_build_headers(), params=params, timeout=20)
            if response.status_code in {401, 403}:
                logger.error(
                    "X API authorization failed (%s). Verify token permissions.",
                    response.status_code,
                )
                return []
            if response.status_code == 429:
                logger.error("X API rate limit reached while fetching tweets.")
                return []
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logger.exception("Failed to fetch tweets for %s: %s", cleaned_username, exc)
            return []

        data = payload.get("data", [])
        for item in data:
            tweets.append(
                {
                    "username": cleaned_username,
                    "text": item.get("text", ""),
                    "timestamp": item.get("created_at"),
                }
            )

        meta = payload.get("meta", {})
        next_token = meta.get("next_token")
        if not next_token or not data:
            break

    if not tweets:
        logger.warning("No tweets found for username: %s", cleaned_username)
        return []

    logger.info("Fetched %d tweets for %s via X API", len(tweets), cleaned_username)
    return tweets
