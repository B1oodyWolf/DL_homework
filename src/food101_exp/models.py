from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    densenet121,
    resnet18,
)


def build_model(architecture: str, mode: str, num_classes: int) -> nn.Module:
    if architecture == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if mode == "finetune" else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if mode == "finetune" else None
        model = densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def build_optimizer(model: nn.Module, architecture: str, mode: str, lr: float, backbone_lr: float, weight_decay: float):
    if mode == "scratch":
        return __import__("torch").optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if architecture == "resnet18":
        head_params = list(model.fc.parameters())
        head_param_ids = {id(param) for param in head_params}
    elif architecture == "densenet121":
        head_params = list(model.classifier.parameters())
        head_param_ids = {id(param) for param in head_params}
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    backbone_params = [param for param in model.parameters() if id(param) not in head_param_ids]
    return __import__("torch").optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )

