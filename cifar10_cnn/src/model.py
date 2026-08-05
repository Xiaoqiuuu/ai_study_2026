"""ResNet18 fine-tuning for CIFAR-10."""
from typing import List

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 10, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """
    Build a ResNet18 for CIFAR-10.

    CIFAR-10 is 32x32, but ResNet18 expects 224x224. However, torchvision ResNet18
    can still handle 32x32 because the first conv (7x7 stride 2) + maxpool will
    downsample aggressively. For better small-image performance, some people replace
    the first conv with 3x3 stride 1, but standard fine-tuning on CIFAR-10 with
    original architecture still works (~93% acc).

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all layers except layer4 and fc.

    Returns:
        ResNet18 model with modified fc layer.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final fc layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone and pretrained:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

    return model


def get_trainable_layers(model: nn.Module) -> List[str]:
    """Return names of layers that have requires_grad=True."""
    return [name for name, param in model.named_parameters() if param.requires_grad]
