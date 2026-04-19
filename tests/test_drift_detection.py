import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_drifted_ticket_profile_triggers_ks_fail(tmp_path):
    drifted = tmp_path / "tickets_drifted.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/simulate_ticket_drift.py"),
            "--output",
            str(drifted),
        ],
        cwd=ROOT,
        check=True,
    )
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/check_drift.py"),
            "--tickets",
            str(drifted),
            "--ks-threshold",
            "0.12",
            "--fail-on-drift",
        ],
        cwd=ROOT,
    )
    assert r.returncode == 1
