"""Compatibility wrapper for dashboard imports.

The dashboard implementation now lives in app.ui.dashboard.
"""

from app.ui.dashboard import render_dashboard

__all__ = ["render_dashboard"]
