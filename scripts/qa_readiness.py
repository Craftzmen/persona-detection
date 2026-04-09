"""One-command QA readiness runner.

Usage:
    /Users/apple/Projects/fyps/persona-detection/.venv/bin/python scripts/qa_readiness.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATASET_PATH, LOGS_DIR, OUTPUT_DIR, REPORTS_DIR, SNAPSHOTS_DIR, ensure_directories
from app.integration_service import analyze_user, format_api_response


def _test_username() -> str:
    if DATASET_PATH.exists():
        frame = pd.read_csv(DATASET_PATH)
        if "username" in frame.columns and not frame.empty:
            return str(frame.iloc[0]["username"])
    return "NASA"


def _run_command(command: list[str], cwd: Path) -> tuple[int, str]:
    started = time.time()
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    elapsed = time.time() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    summary = f"$ {' '.join(command)}\n(exit={proc.returncode}, {elapsed:.2f}s)\n{output}"
    return proc.returncode, summary


def main() -> int:
    ensure_directories()

    print("QA Readiness Check")
    print(f"- output dir: {OUTPUT_DIR}")
    print(f"- logs dir:   {LOGS_DIR}")
    print(f"- reports dir:{REPORTS_DIR}")
    print(f"- snapshots:  {SNAPSHOTS_DIR}")

    checks: list[tuple[str, bool, str]] = []

    code, output = _run_command([sys.executable, "-m", "pytest", "-q"], PROJECT_ROOT)
    checks.append(("pytest", code == 0, output))

    try:
        result = analyze_user(_test_username())
        payload = format_api_response(result)
        response_time = float(result.get("response_time_seconds", 0.0))
        is_valid = (
            set(payload.keys())
            == {
                "username",
                "prediction",
                "synthetic_score",
                "behavioral_features",
                "stylometric_features",
                "cluster_id",
                "linked_personas",
                "risk_level",
            }
            and response_time < 10.0
        )
        checks.append(
            (
                "pipeline-smoke",
                is_valid,
                json.dumps(
                    {
                        "username": payload["username"],
                        "prediction": payload["prediction"],
                        "risk_level": payload["risk_level"],
                        "response_time_seconds": response_time,
                    },
                    indent=2,
                ),
            )
        )
    except Exception as exc:  # pragma: no cover - top-level QA guard
        checks.append(("pipeline-smoke", False, str(exc)))

    success = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {name}")
        print(detail.strip())
        success = success and passed

    print("\nOverall:", "READY" if success else "NOT READY")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
