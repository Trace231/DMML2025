"""Diffusion-inspired segmenter using the random walker algorithm."""

from __future__ import annotations

import numpy as np
from skimage import color
from skimage.segmentation import random_walker

from ..evaluation.registry import register_segmenter
from .base import BaseSegmenter


@register_segmenter("random_walker")
class RandomWalkerSegmenter(BaseSegmenter):
    """Random walker diffusion segmentation."""

    def __init__(self, num_classes: int = 2, beta: float = 130.0, tolerance: float = 1e-3, mode: str = "bf") -> None:
        super().__init__(num_classes=num_classes, name="RandomWalker")
        self.beta = beta
        self.tolerance = tolerance
        self.mode = mode

    def prepare(self, train_dataset=None, val_dataset=None) -> None:  # type: ignore[override]
        return

    def _create_markers(self, image: np.ndarray) -> np.ndarray:
        gray = color.rgb2gray(image)
        markers = np.zeros(gray.shape, dtype=np.int32)
        if self.num_classes == 2:
            low = np.percentile(gray, 15)
            high = np.percentile(gray, 85)
            markers[gray <= low] = 1
            markers[gray >= high] = 2
            if not np.any(markers == 1):
                markers[gray == gray.min()] = 1
            if not np.any(markers == 2):
                markers[gray == gray.max()] = 2
        else:
            percentiles = np.linspace(0, 100, self.num_classes + 1)
            for class_index in range(self.num_classes):
                lo = np.percentile(gray, percentiles[class_index])
                hi = np.percentile(gray, percentiles[class_index + 1])
                mask = (gray >= lo) & (gray <= hi)
                markers[mask] = class_index + 1
        return markers

    def predict_batch(self, batch):  # type: ignore[override]
        image = self._to_numpy_image(batch)
        markers = self._create_markers(image)
        labels = random_walker(
            data=image,
            labels=markers,
            beta=self.beta,
            mode=self.mode,
            tol=self.tolerance,
            channel_axis=-1,
        )
        labels = np.clip(labels - 1, 0, self.num_classes - 1)
        return labels[None, ...]
