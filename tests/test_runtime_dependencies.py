"""Exercise the frozen pipeline with removed dependencies unavailable."""

from pathlib import Path
import subprocess
import sys


def test_pipeline_without_scikit_learn_phyddle_or_joblib():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
sys.modules.update({name: None for name in ('sklearn', 'phyddle', 'joblib')})
import pytest
raise SystemExit(pytest.main(['-q', 'tests/test_pipeline_regression.py']))
"""],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
