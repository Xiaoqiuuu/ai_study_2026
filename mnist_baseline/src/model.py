"""Neural network architectures for MNIST."""
from typing import List

import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """
    A configurable fully-connected network for MNIST classification.

    Architecture: Input -> [Linear -> BN -> ReLU -> Dropout] * N -> Linear -> Output
    """
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_size).
        Returns:
            Logits of shape (batch_size, num_classes).
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
