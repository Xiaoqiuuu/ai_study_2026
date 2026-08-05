"""Entry point for MNIST training pipeline."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import get_mnist_loaders
from src.model import FeedForwardNet
from src.trainer import Trainer
from src.utils import load_config, get_device, set_seed, count_parameters


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train MNIST Classifier")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Setup
    device = get_device(config.get("device", "cuda"))
    set_seed(config["seed"], config.get("deterministic", True), config.get("benchmark", False))

    print(f"Device: {device}")
    print(f"Config: {config}")

    # Data
    train_loader, val_loader, test_loader = get_mnist_loaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # Model
    model = FeedForwardNet(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Trainer
    trainer = Trainer(model, config, device)

    if not args.eval_only:
        trainer.train(train_loader, val_loader)

    # Final evaluation
    trainer.evaluate(test_loader)
    print("\nDone! Check ./runs for TensorBoard logs and ./checkpoints for models.")
    print("Run: tensorboard --logdir=./runs")


if __name__ == "__main__":
    main()
