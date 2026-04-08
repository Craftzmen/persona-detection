"""FastAPI backend for synthetic persona analysis."""

from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import API_AUTH_PASSWORD, API_AUTH_USERNAME
from app.integration_service import analyze_user, format_api_response, read_analysis_history

app = FastAPI(
    title="Synthetic Persona Detection API",
    version="1.1.0",
    description="OSINT pipeline API for detection and attribution of AI-generated personas.",
)

security = HTTPBasic(auto_error=False)


def _require_auth(credentials: HTTPBasicCredentials | None = Depends(security)) -> str:
    """Validate HTTP Basic credentials when auth is configured."""

    auth_enabled = bool(API_AUTH_USERNAME and API_AUTH_PASSWORD)
    if not auth_enabled:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    username_valid = secrets.compare_digest(credentials.username, API_AUTH_USERNAME)
    password_valid = secrets.compare_digest(credentials.password, API_AUTH_PASSWORD)
    if not (username_valid and password_valid):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health endpoint for readiness checks."""

    return {"status": "ok"}


@app.get("/analyze")
def analyze(
    username: str = Query(..., min_length=1, description="Target username"),
    _authenticated_user: str = Depends(_require_auth),
) -> dict:
    """Run the integrated analysis pipeline for one username."""

    try:
        analysis_data = analyze_user(username)
        return format_api_response(analysis_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - API top-level guard
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.get("/history")
def history(
    limit: int = Query(50, ge=1, le=1000, description="Maximum rows to return"),
    username: str | None = Query(None, description="Optional username filter"),
    _authenticated_user: str = Depends(_require_auth),
) -> dict[str, object]:
    """Return persisted analysis history entries."""

    try:
        entries = read_analysis_history(limit=limit, username=username)
        return {"count": len(entries), "entries": entries}
    except Exception as exc:  # pragma: no cover - API top-level guard
        raise HTTPException(status_code=500, detail=f"History lookup failed: {exc}") from exc
