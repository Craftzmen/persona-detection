"""Integration-level tests for service orchestration and reporting."""

from __future__ import annotations

from pathlib import Path

from app.integration_service import (
    analyze_user,
    format_api_response,
    generate_report,
    generate_risk_score,
    read_analysis_history,
)


def test_generate_risk_score_boundaries() -> None:
    assert generate_risk_score(0.39) == "Low"
    assert generate_risk_score(0.4) == "Medium"
    assert generate_risk_score(0.7) == "Medium"
    assert generate_risk_score(0.71) == "High"


def test_analyze_user_contract_for_local_dataset_user() -> None:
    result = analyze_user("NASA")

    assert result["username"] == "nasa"
    assert result["prediction"] in {"AI", "Human"}
    assert 0.0 <= float(result["synthetic_score"]) <= 1.0
    assert result["risk_level"] in {"Low", "Medium", "High"}

    payload = format_api_response(result)
    assert set(payload.keys()) == {
        "username",
        "prediction",
        "synthetic_score",
        "behavioral_features",
        "stylometric_features",
        "cluster_id",
        "linked_personas",
        "risk_level",
    }


def test_generate_report_produces_pdf_bytes(tmp_path: Path) -> None:
    result = analyze_user("NASA")
    out = tmp_path / "report.pdf"

    data = generate_report(result, output_path=out)

    assert data.startswith(b"%PDF")
    assert out.exists()
    assert out.stat().st_size > 0


def test_read_analysis_history_returns_recent_rows() -> None:
    analyze_user("NASA")
    rows = read_analysis_history(limit=10)

    assert isinstance(rows, list)
    assert rows
    first = rows[0]
    assert "username" in first
    assert "synthetic_score" in first
