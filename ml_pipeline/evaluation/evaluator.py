"""Core model evaluator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml_pipeline.evaluation.ablation.variants import ABLATION_VARIANTS, apply_ablation_mask
from ml_pipeline.evaluation.benchmark.compare import benchmark_inference
from ml_pipeline.evaluation.calibration.ece import calibrate_logits, expected_calibration_error
from ml_pipeline.evaluation.confusion.matrix import confusion_matrix_data, plot_confusion_matrix
from ml_pipeline.evaluation.config import evaluation_config, evaluation_paths
from ml_pipeline.evaluation.error_analysis.analyzer import analyze_errors
from ml_pipeline.evaluation.explainability.embedding_viz import plot_embedding_scatter, reduce_embeddings
from ml_pipeline.evaluation.explainability.xai import attention_summary, local_explanation, modality_contribution
from ml_pipeline.evaluation.metrics.classification import classification_metrics, per_class_report
from ml_pipeline.evaluation.metrics.regression import regression_metrics
from ml_pipeline.evaluation.pr.curves import plot_pr, pr_curve_data
from ml_pipeline.evaluation.roc.curves import plot_roc, roc_curve_data
from ml_pipeline.evaluation.statistics import confidence_interval, descriptive_stats
from ml_pipeline.evaluation.uncertainty.estimation import monte_carlo_dropout_uncertainty, predictive_entropy
from ml_pipeline.models.types import MultimodalBatch


class ModelEvaluator:
    """Publication-grade evaluation pipeline."""

    def __init__(self, model, config: dict[str, Any] | None = None) -> None:
        self.model = model
        self.cfg = config or evaluation_config()
        self.paths = evaluation_paths()
        self.paths["root"].mkdir(parents=True, exist_ok=True)
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def collect_predictions(self, dataloader) -> dict[str, Any]:
        self.model.eval()
        all_preds, all_labels, all_probs, all_fused = [], [], [], []
        pids, langs, sids = [], [], []
        fusion_weights_list = []
        for batch_dict in dataloader:
            batch = MultimodalBatch.from_collated(batch_dict).to(next(self.model.parameters()).device)
            if batch.labels is None:
                continue
            out = self.model(batch)
            probs = out.get("probs", torch.softmax(out["logits"], dim=-1))
            all_preds.append(out.get("preds", out["logits"].argmax(-1)).cpu())
            all_labels.append(batch.labels.cpu())
            all_probs.append(probs.cpu())
            if "fused" in out:
                all_fused.append(out["fused"].cpu())
            pids.extend(batch.participant_ids or [])
            langs.extend(batch.languages or [])
            sids.extend(batch.session_ids or [])
            if out.get("modality_weights"):
                fusion_weights_list.append(out["modality_weights"])
        if not all_labels:
            return {}
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        fused = torch.cat(all_fused).numpy() if all_fused else None
        return {
            "preds": preds,
            "labels": labels,
            "probs": probs,
            "fused": fused,
            "participant_ids": pids,
            "languages": langs,
            "session_ids": sids,
            "fusion_weights": fusion_weights_list,
        }

    def evaluate(self, dataloader, split: str = "test") -> dict[str, Any]:
        start = time.perf_counter()
        data = self.collect_predictions(dataloader)
        if not data:
            return {"split": split, "status": "empty"}

        preds, labels, probs = data["preds"], data["labels"], data["probs"]
        metrics = classification_metrics(preds, labels, probs)
        if "risk_score" in data:
            metrics.update(regression_metrics(data["risk_score"], labels.astype(float)))

        cal = expected_calibration_error(probs, labels)
        metrics["ece"] = cal.ece
        metrics["mce"] = cal.mce

        cm = confusion_matrix_data(preds, labels)
        roc = roc_curve_data(labels, probs[:, 1] if probs.ndim == 2 else probs)
        pr = pr_curve_data(labels, probs[:, 1] if probs.ndim == 2 else probs)
        errors = analyze_errors(preds, labels, probs, data.get("participant_ids"), data.get("languages"))
        per_class = per_class_report(preds, labels)

        dpi = int(self.cfg.get("publication", {}).get("dpi", 300))
        fig_dir = self.paths["figures"] / split
        plot_confusion_matrix(cm, fig_dir / "confusion_matrix.png", dpi=dpi)
        plot_roc([roc], [split], fig_dir / "roc.png", dpi=dpi)
        plot_pr([pr], [split], fig_dir / "pr.png", dpi=dpi)

        if data.get("fused") is not None and len(data["fused"]) >= 2:
            coords = reduce_embeddings(data["fused"], method="pca")
            plot_embedding_scatter(coords, labels, fig_dir / "embedding_pca.png", dpi=dpi)

        result = {
            "split": split,
            "metrics": metrics,
            "calibration": {"ece": cal.ece, "mce": cal.mce, "bins": cal.bins},
            "confusion_matrix": cm,
            "roc": roc,
            "pr": pr,
            "per_class": per_class,
            "error_analysis": errors,
            "eval_time_s": time.perf_counter() - start,
        }
        out_path = self.paths["exports"] / f"{split}_metrics.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    def explain_batch(self, batch: MultimodalBatch) -> dict[str, Any]:
        self.model.eval()
        out = self.model(batch)
        contrib = modality_contribution(out.get("modality_weights"))
        fusion_out = out.get("fusion_output")
        attn = attention_summary(fusion_out.attention_maps if fusion_out else None)
        return local_explanation(out, contrib, {"attention": attn})

    def run_ablation(self, dataloader, variants: list[str] | None = None) -> dict[str, Any]:
        variants = variants or list(ABLATION_VARIANTS.keys())
        results = {}
        for variant in variants:
            preds, labels, probs = [], [], []
            for batch_dict in dataloader:
                batch = apply_ablation_mask(MultimodalBatch.from_collated(batch_dict), variant)
                batch = batch.to(next(self.model.parameters()).device)
                if batch.labels is None:
                    continue
                out = self.model(batch)
                p = out.get("probs", torch.softmax(out["logits"], dim=-1))
                preds.append(out.get("preds", out["logits"].argmax(-1)).cpu())
                labels.append(batch.labels.cpu())
                probs.append(p.cpu())
            if not labels:
                continue
            preds_np = torch.cat(preds).numpy()
            labels_np = torch.cat(labels).numpy()
            probs_np = torch.cat(probs).numpy()
            results[variant] = classification_metrics(preds_np, labels_np, probs_np)
        table_path = self.paths["tables"] / "ablation_results.json"
        table_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    def publication_checklist(self, results: dict[str, Any]) -> dict[str, bool]:
        m = results.get("metrics", {})
        return {
            "metrics_computed": bool(m),
            "calibration_report": "ece" in m,
            "confusion_matrix": "confusion_matrix" in results,
            "roc_generated": "roc" in results,
            "error_analysis": "error_analysis" in results,
            "f1_above_baseline": m.get("f1", 0) > 0.5,
            "figures_exported": (self.paths["figures"]).exists(),
        }
