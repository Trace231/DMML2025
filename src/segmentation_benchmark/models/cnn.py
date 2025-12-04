"""Wrappers for torchvision semantic segmentation backbones."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import segmentation as tv_seg

from ..evaluation.registry import register_segmenter
from ..utils.pretrained import build_torchvision_segmentation_model
from .base import BaseSegmenter


def _default_weights(model_name: str):
    enum_name = f"{model_name.upper()}_Weights"
    return getattr(tv_seg, enum_name).DEFAULT if hasattr(tv_seg, enum_name) else None


def _adjust_head(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Conv2d) and last_layer.out_channels != num_classes:
            in_channels = last_layer.in_channels
            model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    if hasattr(model, "aux_classifier") and isinstance(model.aux_classifier, nn.Sequential):
        last_layer = model.aux_classifier[-1]
        if isinstance(last_layer, nn.Conv2d) and last_layer.out_channels != num_classes:
            in_channels = last_layer.in_channels
            model.aux_classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)


@register_segmenter("torchvision")
class TorchvisionSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_name: str = "fcn_resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        finetune_epochs: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 2,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(num_classes=num_classes, device=device, name=f"Torchvision-{model_name}")
        if not hasattr(tv_seg, model_name):
            raise ValueError(f"Unknown torchvision segmentation model '{model_name}'")
        # Use our central pretrained cache instead of letting torchvision download
        self.model = build_torchvision_segmentation_model(model_name, pretrained=pretrained)
        _adjust_head(self.model, num_classes)
        self.model.to(self.device)
        self.finetune_epochs = finetune_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_name = model_name

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        if self.finetune_epochs <= 0 or train_dataset is None:
            return
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        # Use ignore_index=255 so VOC "void" pixels are not penalised during training.
        # The dataset keeps 0-20 as valid class indices and 255 as void.
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].squeeze(1).long().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                logits = outputs["out"]
                loss = criterion(logits, masks)
                if "aux" in outputs and outputs["aux"] is not None:
                    loss = loss + 0.3 * criterion(outputs["aux"], masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"[TRAIN] {self.name} epoch {epoch+1}/{self.finetune_epochs} - avg loss: {avg_loss:.4f}")
        self.model.eval()

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            outputs = self.model(images)["out"]
            return outputs.detach().cpu().numpy()

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        logits = self.predict_logits(batch)
        preds = np.argmax(logits, axis=1).astype(np.int64)
        return preds


@register_segmenter("fcn_resnet50")
def build_fcn_resnet50(**kwargs: Any) -> TorchvisionSegmenter:
    kwargs.setdefault("model_name", "fcn_resnet50")
    return TorchvisionSegmenter(**kwargs)


@register_segmenter("deeplabv3_resnet50")
def build_deeplabv3_resnet50(**kwargs: Any) -> TorchvisionSegmenter:
    kwargs.setdefault("model_name", "deeplabv3_resnet50")
    return TorchvisionSegmenter(**kwargs)
