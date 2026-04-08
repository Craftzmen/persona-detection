"""Visual regression checks for dashboard charts using deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.io as pio

from app.ui.visuals import (
    build_daily_frequency_figure,
    build_hourly_activity_figure,
    build_network_figure,
    build_wordcount_distribution_figure,
)

BASELINE_PATH = Path(__file__).parent / "baselines" / "visual_fingerprints.json"


def _figure_fingerprint(fig) -> str:
    encoded = pio.to_json(fig, pretty=False, remove_uids=True, engine="json")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _current_fingerprints() -> dict[str, str]:
    hour_df = pd.DataFrame({"hour": list(range(24)), "posts": [i % 5 for i in range(24)]})
    day_df = pd.DataFrame(
        {
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "posts": [2, 4, 6, 5, 7, 3, 1],
        }
    )
    wc_df = pd.DataFrame({"word_count": [4, 5, 9, 12, 6, 7, 3, 8, 10, 11, 12, 5, 4]})
    network_data = {
        "nodes": [{"id": "alpha"}, {"id": "beta"}, {"id": "gamma"}],
        "links": [
            {"source": "alpha", "target": "beta", "weight": 0.84},
            {"source": "beta", "target": "gamma", "weight": 0.72},
        ],
    }

    return {
        "hourly": _figure_fingerprint(build_hourly_activity_figure(hour_df)),
        "daily": _figure_fingerprint(build_daily_frequency_figure(day_df)),
        "wordcount": _figure_fingerprint(build_wordcount_distribution_figure(wc_df)),
        "network": _figure_fingerprint(build_network_figure(network_data, title="Network: baseline")),
    }


def test_visual_fingerprints_match_baseline() -> None:
    assert BASELINE_PATH.exists(), "Missing visual baseline file."

    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = _current_fingerprints()

    assert current == expected
