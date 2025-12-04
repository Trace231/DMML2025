"""End-to-end CNN + CRF segmenter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..evaluation.registry import register_segmenter
from .base import BaseSegmenter
from .cnn import TorchvisionSegmenter
from .crf_postprocess import CrfParams, CrfPostProcessor


@dataclass
class CnnCrfConfig:
    base_model: str = "fcn_resnet50"
    pretrained: bool = True
    finetune_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    num_workers: int = 0
    device: Optional[str] = None
    crf_params: Optional[Dict[str, Any]] = None


@register_segmenter("cnn_crf")
class CnnCrfSegmenter(BaseSegmenter):
    def __init__(self, num_classes: int = 2, config: Optional[CnnCrfConfig | Dict[str, Any]] = None) -> None:
        if isinstance(config, dict):
            config = CnnCrfConfig(**config)
        self.config = config or CnnCrfConfig()
        super().__init__(num_classes=num_classes, device=self.config.device, name=f"{self.config.base_model.upper()}-CRF")
        base_kwargs = {
            "model_name": self.config.base_model,
            "num_classes": num_classes,
            "pretrained": self.config.pretrained,
            "finetune_epochs": self.config.finetune_epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "device": str(self.device),
        }
        self.base_segmenter = TorchvisionSegmenter(**base_kwargs)
        crf_params = self.config.crf_params or {}
        params = CrfParams(**crf_params) if isinstance(crf_params, dict) else crf_params
        self.crf = CrfPostProcessor(num_classes=num_classes, params=params)

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        self.base_segmenter.prepare(train_dataset=train_dataset, val_dataset=val_dataset)

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        logits = self.base_segmenter.predict_logits(batch)
        return logits

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        logits = self.predict_logits(batch)
        if logits is None:
            preds = self.base_segmenter.predict_batch(batch)
            logits = self._one_hot_from_preds(preds)
        image = batch["image"]
        refined = []
        if logits.ndim == 4:
            for i in range(logits.shape[0]):
                refined_mask = self.crf.refine(image[i], logits[i])
                refined.append(refined_mask)
        else:
            refined.append(self.crf.refine(image, logits))
        return np.stack(refined, axis=0)

    def _one_hot_from_preds(self, preds: np.ndarray) -> np.ndarray:
        preds = preds.astype(np.int64)
        batch, height, width = preds.shape
        one_hot = np.zeros((batch, self.num_classes, height, width), dtype=np.float32)
        eye = np.eye(self.num_classes, dtype=np.float32)
        for b in range(batch):
            one_hot[b] = eye[preds[b]].transpose(2, 0, 1)
        return one_hot
