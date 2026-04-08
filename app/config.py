"""Global configuration for the persona detection project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
EXPORTS_DIR = OUTPUT_DIR / "exports"
SNAPSHOTS_DIR = OUTPUT_DIR / "snapshots"
LOGS_DIR = OUTPUT_DIR / "logs"
DATASET_PATH = PROCESSED_DATA_DIR / "dataset.csv"
PREPROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "dataset_preprocessed.csv"

DEFAULT_TWEET_COUNT = 50
DEFAULT_AI_POST_COUNT = 50
DEFAULT_RANDOM_SEED = 42
LOG_LEVEL = "INFO"
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "app.log")
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", str(5 * 1024 * 1024)))
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

X_API_BASE_URL = os.getenv("X_API_BASE_URL", "https://api.x.com/2")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")

API_AUTH_USERNAME = os.getenv("API_AUTH_USERNAME", "")
API_AUTH_PASSWORD = os.getenv("API_AUTH_PASSWORD", "")
DASHBOARD_AUTH_USERNAME = os.getenv("DASHBOARD_AUTH_USERNAME", "")
DASHBOARD_AUTH_PASSWORD = os.getenv("DASHBOARD_AUTH_PASSWORD", "")


def ensure_directories() -> None:
    """Create project data directories if they do not already exist."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
