"""Training, validation, and testing loops with checkpointing."""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils import setup_logger


class Trainer:
    """Orchestrates model training with validation, early stopping, and logging."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: torch.device,
        logger_name: str = "mnist_trainer",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epochs = config["epochs"]
        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = setup_logger(logger_name, log_file=f"{self.checkpoint_dir}/train.log")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config["scheduler_step"],
            gamma=config["scheduler_gamma"],
        )

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stop = False

        # TensorBoard
        self.use_tb = config.get("use_tensorboard", True)
        if self.use_tb:
            self.writer = SummaryWriter(log_dir=config["runs_dir"])
        else:
            self.writer = None

    def _run_epoch(self, dataloader: DataLoader, phase: str) -> Dict[str, float]:
        """Run one epoch for train or validation."""
        is_train = phase == "train"
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad if is_train else torch.no_grad
        pbar = tqdm(dataloader, desc=f"{phase:5s}", leave=False)

        with context():
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Metrics
                total_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                # Progress bar postfix
                if batch_idx % self.config["log_interval"] == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total
        accuracy = correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Full training loop with validation and early stopping."""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config}")

        for epoch in range(1, self.epochs + 1):
            if self.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            train_metrics = self._run_epoch(train_loader, "train")
            self.scheduler.step()

            log_str = f"Epoch [{epoch:3d}/{self.epochs}] | Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}"

            # Validation
            if val_loader is not None and epoch % self.config["val_interval"] == 0:
                val_metrics = self._run_epoch(val_loader, "val")
                log_str += f" | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"

                # Checkpointing (save best)
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config["early_stopping_patience"]:
                        self.early_stop = True

                # TensorBoard logging
                if self.writer:
                    self.writer.add_scalars("Loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
                    self.writer.add_scalars("Accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
            else:
                if self.writer:
                    self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                    self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)

            self.logger.info(log_str)

        if self.writer:
            self.writer.close()
        self.logger.info("Training completed.")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set using best checkpoint."""
        best_ckpt = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            self._load_checkpoint(best_ckpt)
            self.logger.info(f"Loaded best checkpoint: {best_ckpt}")

        metrics = self._run_epoch(test_loader, "test")
        self.logger.info(f"Test Loss: {metrics['loss']:.4f} | Test Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False) -> None:
        """Save model checkpoint."""
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
            self.logger.info(f"New best model saved with val_acc={metrics['accuracy']:.4f}")

    def _load_checkpoint(self, path: str) -> None:
        """Load checkpoint into model and optimizer."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
