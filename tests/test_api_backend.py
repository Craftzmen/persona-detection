"""API tests for health, analyze, history, and auth behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

import app.api.backend as api_backend
from app.config import DATASET_PATH


client = TestClient(api_backend.app)


def _test_username() -> str:
    if DATASET_PATH.exists():
        frame = pd.read_csv(DATASET_PATH)
        if "username" in frame.columns and not frame.empty:
            return str(frame.iloc[0]["username"])
    return "NASA"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_without_auth() -> None:
    original_user = api_backend.API_AUTH_USERNAME
    original_pass = api_backend.API_AUTH_PASSWORD
    api_backend.API_AUTH_USERNAME = ""
    api_backend.API_AUTH_PASSWORD = ""

    try:
        response = client.get("/analyze", params={"username": _test_username()})
        assert response.status_code == 200
        body = response.json()
        assert body["username"]
        assert body["prediction"] in {"AI", "Human"}
    finally:
        api_backend.API_AUTH_USERNAME = original_user
        api_backend.API_AUTH_PASSWORD = original_pass


def test_history_endpoint_without_auth() -> None:
    original_user = api_backend.API_AUTH_USERNAME
    original_pass = api_backend.API_AUTH_PASSWORD
    api_backend.API_AUTH_USERNAME = ""
    api_backend.API_AUTH_PASSWORD = ""

    try:
        response = client.get("/history", params={"limit": 5})
        assert response.status_code == 200
        payload = response.json()
        assert "entries" in payload
        assert "count" in payload
    finally:
        api_backend.API_AUTH_USERNAME = original_user
        api_backend.API_AUTH_PASSWORD = original_pass


def test_auth_required_when_configured() -> None:
    original_user = api_backend.API_AUTH_USERNAME
    original_pass = api_backend.API_AUTH_PASSWORD
    api_backend.API_AUTH_USERNAME = "tester"
    api_backend.API_AUTH_PASSWORD = "secret"

    try:
        unauth = client.get("/analyze", params={"username": _test_username()})
        assert unauth.status_code == 401

        bad_auth = client.get("/analyze", params={"username": _test_username()}, auth=("tester", "wrong"))
        assert bad_auth.status_code == 401

        ok_auth = client.get("/analyze", params={"username": _test_username()}, auth=("tester", "secret"))
        assert ok_auth.status_code == 200
    finally:
        api_backend.API_AUTH_USERNAME = original_user
        api_backend.API_AUTH_PASSWORD = original_pass
