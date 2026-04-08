"""ASGI entrypoint for FastAPI server.

Run with:
uvicorn api:app --reload
"""

from app.api.backend import app
