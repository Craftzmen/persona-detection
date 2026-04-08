"""API tests for health, analyze, history, and auth behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.api.backend as api_backend


client = TestClient(api_backend.app)


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
        response = client.get("/analyze", params={"username": "NASA"})
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "nasa"
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
        unauth = client.get("/analyze", params={"username": "NASA"})
        assert unauth.status_code == 401

        bad_auth = client.get("/analyze", params={"username": "NASA"}, auth=("tester", "wrong"))
        assert bad_auth.status_code == 401

        ok_auth = client.get("/analyze", params={"username": "NASA"}, auth=("tester", "secret"))
        assert ok_auth.status_code == 200
    finally:
        api_backend.API_AUTH_USERNAME = original_user
        api_backend.API_AUTH_PASSWORD = original_pass
