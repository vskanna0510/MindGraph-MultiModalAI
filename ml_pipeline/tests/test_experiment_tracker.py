"""Tests for experiment tracking."""

import json
from pathlib import Path

import pytest

from ml_pipeline.experiments.experiment_tracker import ExperimentTracker, next_experiment_id


def test_next_experiment_id_empty(tmp_path: Path) -> None:
    assert next_experiment_id(tmp_path) == "EXP001"


def test_next_experiment_id_increment(tmp_path: Path) -> None:
    (tmp_path / "EXP001").mkdir()
    (tmp_path / "EXP002").mkdir()
    assert next_experiment_id(tmp_path) == "EXP003"


def test_create_experiment(tmp_path: Path) -> None:
    config = Path("configs/ml/baseline.yaml")
    if not config.exists():
        pytest.skip("baseline config missing")
    tracker = ExperimentTracker(tmp_path)
    exp_dir = tracker.create_experiment(config, experiment_id="EXP999")
    assert exp_dir.exists()
    assert (exp_dir / "config.yaml").exists()
    assert (exp_dir / "metrics.json").exists()
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["experiment_id"] == "EXP999"
