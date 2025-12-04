"""Common utilities and abstract base class for segmenters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


class BaseSegmenter(ABC):
    """Base class that all segmenters inherit from."""

    def __init__(self, num_classes: int = 2, device: Optional[str] = None, name: Optional[str] = None) -> None:
        self.num_classes = num_classes
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.name = name or self.__class__.__name__

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        """Optional hook to fit/train the model before evaluation."""
        # Default implementation does nothing.
        _ = train_dataset, val_dataset

    @abstractmethod
    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        """Predict segmentation masks for a batch.

        Returns a NumPy array of shape (N, H, W) containing integer labels.
        """

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        """Optionally return dense logits of shape (N, C, H, W).

        Subclasses that can expose probability maps should override this method. The default
        implementation returns ``None``.
        """

        return None

    @staticmethod
    def _to_numpy_image(batch: Dict[str, Any]) -> np.ndarray:
        image = batch["image"]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        image = np.transpose(image, (1, 2, 0)) if image.ndim == 3 else image
        return np.clip(image, 0.0, 1.0)

    @staticmethod
    def _ensure_numpy_mask(mask: Any) -> np.ndarray:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 3:
            mask = mask[0]
        return mask.astype(np.int64)
