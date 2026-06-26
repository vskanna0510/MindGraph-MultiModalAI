"""Experiment tracking and immutable experiment directories."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml_pipeline.utils.reproducibility import SeedBundle, load_ml_config, seeds_from_config


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class EnvironmentInfo:
    """Hardware and software context for reproducibility."""

    hardware: str
    operating_system: str
    python_version: str
    cuda_version: str | None = None
    pytorch_version: str | None = None


@dataclass
class ExperimentManifest:
    """Complete provenance record for a research experiment."""

    experiment_id: str
    timestamp: str
    git_commit: str
    dataset_version: str
    model_version: str
    hyperparameters: dict[str, Any]
    seeds: dict[str, int]
    environment: EnvironmentInfo
    execution_time_seconds: float | None = None
    memory_usage_mb: float | None = None
    gpu_usage_percent: float | None = None
    cpu_usage_percent: float | None = None
    training_duration_seconds: float | None = None
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    inference_time_ms: float | None = None
    export_status: dict[str, str] = field(default_factory=dict)
    config_path: str = ""


def get_git_commit(root: Path | None = None) -> str:
    """Return current git commit hash or 'unknown'."""
    root = root or project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_environment() -> EnvironmentInfo:
    """Collect runtime environment metadata."""
    cuda_version = None
    pytorch_version = None
    try:
        import torch

        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
    except ImportError:
        pass

    return EnvironmentInfo(
        hardware=platform.processor() or platform.machine(),
        operating_system=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        cuda_version=cuda_version,
        pytorch_version=pytorch_version,
    )


def next_experiment_id(experiments_dir: Path) -> str:
    """Allocate the next sequential experiment ID (EXP001, EXP002, ...)."""
    existing = sorted(
        path.name
        for path in experiments_dir.glob("EXP*")
        if path.is_dir() and path.name[3:].isdigit()
    )
    if not existing:
        return "EXP001"
    last = int(existing[-1].replace("EXP", ""))
    return f"EXP{last + 1:03d}"


class ExperimentTracker:
    """Create and manage immutable experiment directories."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or project_root() / "experiments"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        config_path: str | Path,
        experiment_id: str | None = None,
        model_version: str = "MODEL_001_BASELINE",
    ) -> Path:
        """Create a new experiment folder with required artifacts."""
        config_path = Path(config_path)
        config = load_ml_config(str(config_path))
        exp_id = experiment_id or next_experiment_id(self.base_dir)
        exp_dir = self.base_dir / exp_id
        if exp_dir.exists():
            raise FileExistsError(f"Experiment directory already exists: {exp_dir}")

        exp_dir.mkdir(parents=True)
        seeds = seeds_from_config(config)

        try:
            rel_config = str(config_path.resolve().relative_to(project_root()))
        except ValueError:
            rel_config = str(config_path)

        manifest = ExperimentManifest(
            experiment_id=exp_id,
            timestamp=datetime.now(UTC).isoformat(),
            git_commit=get_git_commit(),
            dataset_version=str(config.get("dataset", {}).get("version", "unknown")),
            model_version=model_version,
            hyperparameters=config,
            seeds=seeds.as_dict(),
            environment=collect_environment(),
            config_path=rel_config,
        )

        self._write_templates(exp_dir, manifest, config, seeds)
        return exp_dir

    def finalize(
        self,
        exp_dir: Path,
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        start_time: float,
        checkpoint_name: str = "checkpoint.pt",
    ) -> ExperimentManifest:
        """Update experiment metrics after training completes."""
        metrics_path = exp_dir / "metrics.json"
        manifest_path = exp_dir / "manifest.json"
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        elapsed = time.perf_counter() - start_time
        data["execution_time_seconds"] = elapsed
        data["training_duration_seconds"] = elapsed
        data["validation_metrics"] = validation_metrics
        data["test_metrics"] = test_metrics
        data["checkpoint_path"] = checkpoint_name
        metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return ExperimentManifest(**{k: v for k, v in data.items() if k in ExperimentManifest.__dataclass_fields__})

    def _write_templates(
        self,
        exp_dir: Path,
        manifest: ExperimentManifest,
        config: dict[str, Any],
        seeds: SeedBundle,
    ) -> None:
        (exp_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (exp_dir / "hyperparameters.yaml").write_text(
            yaml.safe_dump(
                {
                    "model": config.get("model", {}),
                    "training": config.get("training", {}),
                    "dataset": config.get("dataset", {}),
                    "optimization": config.get("optimization", {}),
                    "evaluation": config.get("evaluation", {}),
                    "deployment": config.get("deployment", {}),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        metrics = asdict(manifest)
        metrics["environment"] = asdict(manifest.environment)
        (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (exp_dir / "manifest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (exp_dir / "training.log").write_text(
            f"{manifest.timestamp} INFO experiment_created id={manifest.experiment_id}\n",
            encoding="utf-8",
        )
        (exp_dir / "validation.csv").write_text("epoch,loss,accuracy,f1,roc_auc\n", encoding="utf-8")
        (exp_dir / "test.csv").write_text("metric,value\n", encoding="utf-8")
        (exp_dir / "README.md").write_text(self._readme_template(manifest), encoding="utf-8")
        (exp_dir / "seeds.json").write_text(json.dumps(seeds.as_dict(), indent=2), encoding="utf-8")

    def _readme_template(self, manifest: ExperimentManifest) -> str:
        return f"""# Experiment {manifest.experiment_id}

## Objective
Document why this experiment was performed.

## Configuration
- Config: `{manifest.config_path}`
- Dataset version: {manifest.dataset_version}
- Model version: {manifest.model_version}
- Git commit: `{manifest.git_commit}`

## What Changed
Describe changes from the previous experiment.

## Results
Populate after training from `metrics.json`.

## Observations
Record findings after analysis.

## Limitations
Document known limitations.

## Future Improvements
Define the next experiment step.

## Reproduce
```bash
python scripts/research/run_experiment.py --config {manifest.config_path} --experiment-id {manifest.experiment_id}
```
"""
