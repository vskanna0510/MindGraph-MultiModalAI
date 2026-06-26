"""Optuna hyperparameter search (optional)."""

from __future__ import annotations

from typing import Any, Callable


def run_optuna_study(objective: Callable, config: dict[str, Any]) -> dict[str, Any]:
    hpo = config.get("hpo", {})
    if not hpo.get("enabled", False):
        return {"status": "disabled"}
    try:
        import optuna
    except ImportError:
        return {"status": "optuna_not_installed"}

    trials = int(hpo.get("trials", 20))
    direction = "maximize" if "f1" in str(hpo.get("metric", "val_f1")) else "minimize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=trials)
    return {"status": "completed", "best_params": study.best_params, "best_value": study.best_value}
