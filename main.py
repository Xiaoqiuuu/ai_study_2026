"""Entry point for CIFAR-10 training."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_cifar10_loaders
from src.model import build_resnet18, get_trainable_layers
from src.trainer import Trainer
from src.utils import load_config, get_device, set_seed, count_parameters, count_all_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 with ResNet18")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.epochs: config["epochs"] = args.epochs
    if args.lr: config["lr"] = args.lr
    if args.batch_size: config["batch_size"] = args.batch_size

    device = get_device(config.get("device", "cuda"))
    set_seed(config["seed"], config.get("deterministic", True), config.get("benchmark", False))

    print(f"Device: {device}")
    print(f"Config: {config}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    model = build_resnet18(
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
        freeze_backbone=config["freeze_backbone"],
    )

    print(f"Total parameters: {count_all_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Trainable layers: {get_trainable_layers(model)}")

    trainer = Trainer(model, config, device)

    if not args.eval_only:
        trainer.train(train_loader, val_loader)

    trainer.evaluate(test_loader)
    print("\nDone! Run: tensorboard --logdir=./runs")


if __name__ == "__main__":
    main()
