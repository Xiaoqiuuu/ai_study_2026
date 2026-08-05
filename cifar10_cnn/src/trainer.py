"""Training loop with AMP, grad clipping, and EMA (optional)."""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import setup_logger


class Trainer:
    """Trainer with mixed precision, gradient clipping, and early stopping."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device, logger_name: str = "cifar10") -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epochs = config["epochs"]
        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = setup_logger(logger_name, log_file=f"{self.checkpoint_dir}/train.log")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        # Scheduler
        if config.get("scheduler", "step") == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["epochs"])
        else:
            self.scheduler = StepLR(self.optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_gamma"])

        # AMP
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None
        self.grad_clip = config.get("grad_clip", None)

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stop = False

        # TensorBoard
        self.use_tb = config.get("use_tensorboard", True)
        self.writer = SummaryWriter(log_dir=config["runs_dir"]) if self.use_tb else None

    def _run_epoch(self, dataloader: DataLoader, phase: str) -> Dict[str, float]:
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"{phase:5s}", leave=False)
        context = torch.enable_grad if is_train else torch.no_grad

        with context():
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                # Forward with AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                if is_train:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                total_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                if batch_idx % self.config["log_interval"] == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        self.logger.info("Training started.")
        self.logger.info(f"Config: {self.config}")

        for epoch in range(1, self.epochs + 1):
            if self.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            train_metrics = self._run_epoch(train_loader, "train")
            self.scheduler.step()

            log_str = (f"Epoch [{epoch:3d}/{self.epochs}] | "
                       f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")

            if val_loader is not None and epoch % self.config["val_interval"] == 0:
                val_metrics = self._run_epoch(val_loader, "val")
                log_str += (f" | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config["early_stopping_patience"]:
                        self.early_stop = True

                if self.writer:
                    self.writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
                    self.writer.add_scalars("Accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            else:
                if self.writer:
                    self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                    self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)

            self.logger.info(log_str)

        if self.writer:
            self.writer.close()
        self.logger.info("Training completed.")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        best_ckpt = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            self._load_checkpoint(best_ckpt)
            self.logger.info(f"Loaded best checkpoint: {best_ckpt}")
        metrics = self._run_epoch(test_loader, "test")
        self.logger.info(f"Test Loss: {metrics['loss']:.4f} | Test Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        path = os.path.join(self.checkpoint_dir, "best_model.pt" if is_best else f"ckpt_epoch_{epoch}.pt")
        torch.save(ckpt, path)
        if is_best:
            self.logger.info(f"New best model saved: val_acc={metrics['accuracy']:.4f}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
