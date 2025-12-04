"""Classical feature-based CRF segmenter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from skimage.color import rgb2gray
from skimage.filters import sobel

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:  # pragma: no cover - dependency missing at runtime
    dcrf = None
    unary_from_softmax = None

from ..evaluation.registry import register_segmenter
from .base import BaseSegmenter


@dataclass
class ClassicalCrfConfig:
    sample_pixels: int = 8000
    random_state: int = 13
    n_estimators: int = 80
    max_depth: Optional[int] = None
    crf_iterations: int = 5
    gaussian_sxy: int = 3
    bilateral_sxy: int = 50
    bilateral_srgb: int = 13
    add_position_features: bool = True
    add_gradient_feature: bool = True


@register_segmenter("classical_crf")
class ClassicalCRFSegmenter(BaseSegmenter):
    """CRF segmenter driven by hand-crafted features and a random forest classifier."""

    def __init__(self, num_classes: int = 2, config: Optional[ClassicalCrfConfig | Dict[str, Any]] = None) -> None:
        super().__init__(num_classes=num_classes)
        if isinstance(config, dict):
            config = ClassicalCrfConfig(**config)
        self.config = config or ClassicalCrfConfig()
        self._rng = check_random_state(self.config.random_state)
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        if dcrf is None or unary_from_softmax is None:
            raise ImportError(
                "pydensecrf is required for ClassicalCRFSegmenter. Install it via 'pip install pydensecrf'."
            )
        self._fitted = False

    # def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
    #     if train_dataset is None:
    #         raise ValueError("ClassicalCRFSegmenter requires a training dataset to fit the random forest")
    #     features_list = []
    #     labels_list = []
    #     for sample in train_dataset:
    #         image = sample["image"].detach().cpu().numpy().transpose(1, 2, 0)
    #         mask = sample["mask"].detach().cpu().numpy().squeeze()
    #         feats = self._extract_features(image)
    #         labels = mask.reshape(-1)
    #         if self.config.sample_pixels and len(labels) > self.config.sample_pixels:
    #             idx = self._rng.choice(len(labels), size=self.config.sample_pixels, replace=False)
    #             feats = feats[idx]
    #             labels = labels[idx]
    #         features_list.append(feats)
    #         labels_list.append(labels)
    #     X = np.concatenate(features_list, axis=0)
    #     y = np.concatenate(labels_list, axis=0)
    #     self.classifier.fit(X, y)
    #     self._fitted = True
    # ... existing code ...

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        if train_dataset is None:
            raise ValueError("ClassicalCRFSegmenter requires a training dataset to fit the random forest")

        # 将 sample_pixels 理解为"整个训练集总采样像素数"上限
        num_images = len(train_dataset)
        # 每张图最多采这么多像素，保证总数不超过 sample_pixels
        pixels_per_image = max(1, self.config.sample_pixels // max(1, num_images))

        features_list = []
        labels_list = []

        for sample in train_dataset:
            image = sample["image"].detach().cpu().numpy().transpose(1, 2, 0)
            mask = sample["mask"].detach().cpu().numpy().squeeze()
            feats = self._extract_features(image)
            labels = mask.reshape(-1)

            # 过滤掉无效标签（例如 VOC 数据集中的 255 忽略标签）
            # 只保留有效的类别标签（0 到 num_classes-1）
            valid_mask = labels < self.num_classes
            feats = feats[valid_mask]
            labels = labels[valid_mask]

            # 如果过滤后没有有效像素，跳过这张图
            if len(labels) == 0:
                continue

            # 这里不再用"每张图 sample_pixels 个像素"，
            # 而是每张图最多 pixels_per_image，整个训练集合计 <= sample_pixels
            if len(labels) > pixels_per_image:
                idx = self._rng.choice(len(labels), size=pixels_per_image, replace=False)
                feats = feats[idx]
                labels = labels[idx]

            features_list.append(feats)
            labels_list.append(labels)

        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        self.classifier.fit(X, y)
        self._fitted = True

# ... existing code ...
    
    # 或者：如果内存允许，最后一次性训练
    # 但这样还是会有内存问题
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        features = [image.reshape(-1, 3)]
        if self.config.add_position_features:
            yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
            features.append(np.stack([xx, yy], axis=-1).reshape(-1, 2))
        if self.config.add_gradient_feature:
            gray = rgb2gray(image)
            grad = sobel(gray)
            features.append(grad.reshape(-1, 1))
        return np.concatenate(features, axis=1)

    def _predict_proba(self, image: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ClassicalCRFSegmenter must be fitted before calling predict")
        features = self._extract_features(image)
        proba = self.classifier.predict_proba(features)
        full = np.zeros((features.shape[0], self.num_classes), dtype=np.float32)
        for class_index, label in enumerate(self.classifier.classes_):
            full[:, int(label)] = proba[:, class_index]
        full = full.reshape(image.shape[0], image.shape[1], self.num_classes)
        return np.moveaxis(full, -1, 0)

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        image = self._to_numpy_image(batch)
        probabilities = self._predict_proba(image)
        logits = np.clip(probabilities, 1e-5, 1.0)
        refined = self._apply_dense_crf(image, logits)
        return refined[None, ...]

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        image = self._to_numpy_image(batch)
        probabilities = self._predict_proba(image)
        return probabilities[None, ...]

    def _apply_dense_crf(self, image: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        dense = dcrf.DenseCRF2D(w, h, self.num_classes)
        probabilities = np.ascontiguousarray(probabilities)
        unary = unary_from_softmax(probabilities)
        dense.setUnaryEnergy(np.ascontiguousarray(unary))
        dense.addPairwiseGaussian(sxy=self.config.gaussian_sxy, compat=3)
        rgb_image = np.ascontiguousarray((image * 255).astype(np.uint8))
        dense.addPairwiseBilateral(
            sxy=self.config.bilateral_sxy,
            srgb=self.config.bilateral_srgb,
            rgbim=rgb_image,
            compat=10,
        )
        q = dense.inference(self.config.crf_iterations)
        prediction = np.array(q).reshape(self.num_classes, h, w)
        return np.argmax(prediction, axis=0)