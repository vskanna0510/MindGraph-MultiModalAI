"""MindGraph++ trainer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ml_pipeline.models.device import resolve_device, resolve_dtype
from ml_pipeline.models.types import MultimodalBatch
from ml_pipeline.training.checkpoint import CheckpointManager
from ml_pipeline.training.early_stopping import EarlyStopping
from ml_pipeline.training.logging import TrainingLogger
from ml_pipeline.training.losses.multitask import MultiTaskLoss
from ml_pipeline.training.metrics import metrics_from_outputs
from ml_pipeline.training.optimizers import build_optimizer
from ml_pipeline.training.schedulers import build_scheduler
from ml_pipeline.training.stages import apply_training_stage, stage_for_epoch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        train_loader: Any = None,
        val_loader: Any = None,
    ) -> None:
        self.cfg = config
        tcfg = config.get("training", {})
        ocfg = config.get("optimization", {})
        ccfg = config.get("checkpoint", {})
        lcfg = config.get("logging", {})

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device(tcfg.get("device", "auto"))
        self.model.to(self.device)

        self.loss_fn = MultiTaskLoss(config)
        self.optimizer = build_optimizer(str(ocfg.get("optimizer", "adamw")), self.model.parameters(), ocfg)
        self.max_epochs = int(tcfg.get("max_epochs", tcfg.get("epochs", 100)))
        self.grad_clip = float(tcfg.get("gradient_clip_val", tcfg.get("gradient_clip_norm", 1.0)))
        self.accum_steps = int(tcfg.get("gradient_accumulation_steps", 1))
        self.use_amp = bool(tcfg.get("mixed_precision", False)) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.dtype = resolve_dtype("float16" if self.use_amp else "float32", self.device)

        steps = len(train_loader) if train_loader else 1
        self.scheduler = build_scheduler(str(ocfg.get("scheduler", "cosine")), self.optimizer, ocfg, steps, self.max_epochs)

        ckpt_dir = Path(ccfg.get("directory", "ml_pipeline/weights"))
        monitors = {}
        if ccfg.get("best_f1", True):
            monitors["val_f1"] = "max"
        if ccfg.get("best_loss", True):
            monitors["val_loss"] = "min"
        if ccfg.get("best_auc", True):
            monitors["val_roc_auc"] = "max"
        self.ckpt = CheckpointManager(ckpt_dir, int(ccfg.get("save_top_k", 3)), monitors, int(ccfg.get("every_n_epochs", 0)))
        self.early = EarlyStopping(
            patience=int(tcfg.get("early_stopping_patience", 10)),
            monitor=str(ccfg.get("monitor", "val_f1")),
            mode=str(ccfg.get("mode", "max")),
        )
        self.logger = TrainingLogger(Path(lcfg.get("log_dir", "ml_pipeline/experiments/logs")), lcfg.get("backend", "csv"))
        self.stages_cfg = config.get("stages", {})
        self.fixed_stage = tcfg.get("stage")

    def _to_batch(self, batch_dict: dict[str, Any]) -> MultimodalBatch:
        return MultimodalBatch.from_collated(batch_dict)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        stage = self.fixed_stage or stage_for_epoch(epoch, self.stages_cfg)
        apply_training_stage(self.model, stage)

        total_loss = 0.0
        n_batches = 0
        start = time.perf_counter()

        if self.train_loader is None:
            return {"train_loss": 0.0, "train_time": 0.0, "stage": stage}

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch_dict in enumerate(self.train_loader):
            batch = self._to_batch(batch_dict).to(self.device)
            if batch.labels is None:
                continue
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch.labels, self.model)
                loss = losses["loss"] / self.accum_steps

            self.scaler.scale(loss).backward()
            if (step + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(losses["loss"].detach())
            n_batches += 1

        if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_time": time.perf_counter() - start,
            "lr": self.optimizer.param_groups[0]["lr"],
            "stage": stage,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        if self.val_loader is None:
            return {"val_loss": 0.0, "val_f1": 0.0}

        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        start = time.perf_counter()

        for batch_dict in self.val_loader:
            batch = self._to_batch(batch_dict).to(self.device)
            if batch.labels is None:
                continue
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch.labels, self.model)
            total_loss += float(losses["loss"])
            all_preds.append(outputs.get("preds", outputs["logits"].argmax(-1)).cpu())
            all_labels.append(batch.labels.cpu())
            all_probs.append(outputs.get("probs", torch.softmax(outputs["logits"], dim=-1)).cpu())

        if not all_labels:
            return {"val_loss": 0.0, "val_f1": 0.0}

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        from ml_pipeline.training.metrics import compute_metrics

        metrics = compute_metrics(preds, labels, probs)
        metrics["val_loss"] = total_loss / len(all_labels)
        metrics["val_time"] = time.perf_counter() - start
        metrics = {f"val_{k}" if not k.startswith("val_") else k: v for k, v in metrics.items()}
        return metrics

    def fit(self, resume_path: Path | None = None) -> dict[str, Any]:
        start_epoch = 0
        if resume_path and resume_path.exists():
            payload = self.ckpt.load_resume(self.model, resume_path)
            start_epoch = int(payload.get("epoch", 0)) + 1

        history: list[dict[str, float]] = []
        for epoch in range(start_epoch, self.max_epochs):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)
            metrics = {**train_m, **val_m}
            history.append(metrics)

            self.logger.log(epoch, "train", train_m)
            self.logger.log(epoch, "val", val_m)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get("val_f1", 0.0))

            self.ckpt.save(self.model, epoch, metrics, tag="latest")
            self.ckpt.maybe_save_best(self.model, epoch, metrics)
            if self.ckpt.every_n_epochs and epoch % self.ckpt.every_n_epochs == 0:
                self.ckpt.save(self.model, epoch, metrics, tag=f"epoch")

            if self.early.step(metrics):
                break

        self.logger.flush()
        return {"history": history, "best": self.ckpt._best, "stopped_early": self.early.should_stop}
