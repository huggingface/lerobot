import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "examples/cig_vla/check_cig_vla_training_readiness.py"


def test_offline_readiness_distinguishes_mock_and_real():
    result = subprocess.run([sys.executable, str(SCRIPT), "--offline"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "READY_FOR_MOCK_SMOKE_TRAINING: YES" in result.stdout
    assert "READY_FOR_REAL_SMOKE_TRAINING: NO" in result.stdout
    assert "real Qwen" in result.stdout
    assert "Action stats: PASS" in result.stdout
