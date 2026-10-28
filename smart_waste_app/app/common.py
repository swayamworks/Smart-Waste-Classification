"""Shared constants and model builder (used by train.py and the API).

Everything here mirrors the Phase 2 / Phase 3 reports:
  * six TrashNet classes (alphabetical = torchvision ImageFolder order)
  * MobileNetV3-Large, 224 x 224 RGB, ImageNet normalisation
  * confidence threshold 0.90 for automatic sorting, otherwise manual review
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CONF_THRESHOLD = 0.90          # validation-selected threshold from Phase 3
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

STREAMS = {
    "cardboard": "Dry: paper and cardboard",
    "paper": "Dry: paper and cardboard",
    "glass": "Dry: glass",
    "metal": "Dry: metal",
    "plastic": "Dry: plastic",
    "trash": "Residual (non-recyclable)",
}

# deterministic transform used for validation, test and inference
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model(pretrained: bool = True, dropout: float = 0.30) -> nn.Module:
    """MobileNetV3-Large with a new six-class head (dropout 0.30)."""
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_large(weights=weights)
    in_f = m.classifier[3].in_features
    m.classifier[2] = nn.Dropout(p=dropout)
    m.classifier[3] = nn.Linear(in_f, len(CLASSES))
    return m
