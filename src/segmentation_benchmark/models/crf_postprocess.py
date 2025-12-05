"""CRF-based post-processing for segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:  # pragma: no cover - optional dependency
    dcrf = None
    unary_from_softmax = None

from ..evaluation.registry import build_segmenter, register_segmenter
from .base import BaseSegmenter


@dataclass
class CrfParams:
    """CRF post-processing parameters.
    
    Default values are conservative to avoid over-optimization that can degrade
    performance of well-trained deep models. For 256x256 images, smaller values
    are recommended.
    """
    iterations: int = 5  # Reduced from 20 to avoid over-optimization
    gaussian_sxy: int = 3
    bilateral_sxy: int = 30  # Reduced from 50-60 for 256x256 images
    bilateral_srgb: int = 13
    compat_gaussian: int = 3
    compat_bilateral: int = 10
    
    def __post_init__(self):
        """Validate parameters and warn about potentially problematic values."""
        if self.iterations > 10:
            import warnings
            warnings.warn(
                f"CRF iterations={self.iterations} may be too high and could "
                f"degrade performance. Consider using 5-10 iterations.",
                UserWarning
            )
        if self.bilateral_sxy > 50:
            import warnings
            warnings.warn(
                f"CRF bilateral_sxy={self.bilateral_sxy} may be too large for "
                f"256x256 images. Consider using 20-40.",
                UserWarning
            )


class CrfPostProcessor:
    def __init__(self, num_classes: int = 2, params: CrfParams | None = None) -> None:
        if dcrf is None or unary_from_softmax is None:
            raise ImportError("pydensecrf is required for CRF post processing")
        self.num_classes = num_classes
        self.params = params or CrfParams()

    def refine(self, image: Any, logits: Any) -> np.ndarray:
        image_np = self._ensure_image(image)
        probabilities = self._ensure_probabilities(logits)
        h, w = image_np.shape[:2]
        dense = dcrf.DenseCRF2D(w, h, self.num_classes)
        unary = unary_from_softmax(probabilities)
        dense.setUnaryEnergy(unary)
        dense.addPairwiseGaussian(sxy=self.params.gaussian_sxy, compat=self.params.compat_gaussian)
        dense.addPairwiseBilateral(
            sxy=self.params.bilateral_sxy,
            srgb=self.params.bilateral_srgb,
            rgbim=image_np,
            compat=self.params.compat_bilateral,
        )
        q = dense.inference(self.params.iterations)
        refined = np.argmax(np.array(q).reshape(self.num_classes, h, w), axis=0)
        return refined.astype(np.int64)

    @staticmethod
    def _ensure_image(image: Any) -> np.ndarray:
        import torch

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            image = np.transpose(image, (1, 2, 0))
        image = np.asarray(image)
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        # Ensure C-contiguous array for pydensecrf
        return np.ascontiguousarray(image)

    def _ensure_probabilities(self, logits: Any) -> np.ndarray:
        import torch

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        logits = np.asarray(logits)
        if logits.ndim == 4:
            logits = logits[0]
        if logits.shape[0] != self.num_classes:
            raise ValueError("Logits must have shape (C, H, W)")
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=0, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / (exp.sum(axis=0, keepdims=True) + 1e-10)
        # Ensure C-contiguous array for pydensecrf
        return np.ascontiguousarray(probabilities.astype(np.float32))


@register_segmenter("crf_wrapper")
class CrfWrappedSegmenter(BaseSegmenter):
    """Wrap an existing segmenter and apply DenseCRF post-processing."""

    def __init__(
        self,
        base_builder: str,
        base_params: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        crf_params: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        use_trained_base: bool = True,
        use_soft_onehot: bool = True,
        enable_crf: bool = True,
    ) -> None:
        suffix = "CRFPost" if enable_crf else "NoCRF"
        super().__init__(num_classes=num_classes, name=name or f"{base_builder}-{suffix}")
        base_params = dict(base_params or {})
        base_params.setdefault("num_classes", num_classes)
        base_params.setdefault("device", str(self.device))
        
        # If use_trained_base is True and finetune_epochs is 0, try to find a trained checkpoint
        # by ignoring finetune_epochs in the matching
        if use_trained_base and base_params.get("finetune_epochs", 0) == 0:
            from ..utils.checkpoint import find_matching_checkpoint
            
            # Try to find checkpoint ignoring finetune_epochs (so we can use any trained version)
            checkpoint_path = find_matching_checkpoint(
                base_builder, 
                base_params, 
                ignore_keys=["finetune_epochs"]
            )
            
            if checkpoint_path is not None:
                # Found a checkpoint, load its config to get the correct finetune_epochs
                from ..utils.checkpoint import get_config_from_checkpoint
                checkpoint_config = get_config_from_checkpoint(checkpoint_path)
                if checkpoint_config:
                    # Use the checkpoint's finetune_epochs so the model will load the checkpoint
                    base_params["finetune_epochs"] = checkpoint_config.get("finetune_epochs", 50)
                    print(f"[INFO] {self.name}: Found trained checkpoint for base model (epochs={base_params['finetune_epochs']}), will load it")
            else:
                print(f"[INFO] {self.name}: No trained checkpoint found for base model, using pretrained weights")
        
        self.base = build_segmenter(base_builder, **base_params)
        self.enable_crf = enable_crf
        self.use_soft_onehot = use_soft_onehot
        
        if enable_crf:
            params = CrfParams(**crf_params) if isinstance(crf_params, dict) else crf_params
            self.post = CrfPostProcessor(num_classes=num_classes, params=params)
        else:
            self.post = None

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        self.base.prepare(train_dataset=train_dataset, val_dataset=val_dataset)

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get logits from base model, with fallback to one-hot encoding.
        
        Prefers real logits (continuous scores) over hard predictions.
        If only hard predictions are available, converts to one-hot with warning.
        """
        logits = self.base.predict_logits(batch)
        if logits is None:
            import warnings
            warnings.warn(
                f"{self.name}: Base model '{self.base.name}' does not provide logits. "
                f"Using hard predictions converted to one-hot encoding. "
                f"This may degrade CRF performance as uncertainty information is lost. "
                f"Consider implementing predict_logits() in the base model.",
                UserWarning
            )
            preds = self.base.predict_batch(batch)
            logits = self._one_hot(preds, use_soft_onehot=self.use_soft_onehot)
        else:
            # Verify logits are not already hard (all 0 or 1)
            logits_np = np.asarray(logits)
            if logits_np.ndim == 4:
                logits_np = logits_np[0]
            # Check if logits are already probabilities (in [0, 1] range)
            if logits_np.min() >= 0 and logits_np.max() <= 1:
                # Check if they're hard (mostly 0 or 1)
                unique_vals = np.unique(logits_np)
                if len(unique_vals) <= 2 and all(v in [0.0, 1.0] for v in unique_vals):
                    import warnings
                    warnings.warn(
                        f"{self.name}: Logits appear to be hard one-hot (only 0 and 1). "
                        f"CRF may not work well. Ensure base model returns continuous logits.",
                        UserWarning
                    )
        return logits

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        """Predict with optional CRF post-processing."""
        if not self.enable_crf:
            # If CRF is disabled, just return base model predictions
            return self.base.predict_batch(batch)
        
        logits = self.predict_logits(batch)
        image = batch["image"]
        if logits.ndim == 4:
            refined = [self.post.refine(image[i], logits[i]) for i in range(logits.shape[0])]
        else:
            refined = [self.post.refine(image, logits)]
        return np.stack(refined, axis=0)

    def _one_hot(self, preds: np.ndarray, use_soft_onehot: bool = False) -> np.ndarray:
        """Convert hard predictions to logits format.
        
        Args:
            preds: Hard predictions with shape (batch, height, width)
            use_soft_onehot: If True, use soft one-hot (0.9 for predicted class, 0.1 for others)
                            instead of hard one-hot (1.0 for predicted class, 0.0 for others).
                            This preserves some uncertainty information.
        
        Returns:
            Logits with shape (batch, num_classes, height, width)
        """
        preds = preds.astype(np.int64)
        batch, height, width = preds.shape
        
        if use_soft_onehot:
            # Soft one-hot: 0.9 for predicted class, 0.1/(num_classes-1) for others
            # This preserves some uncertainty information for CRF
            logits = np.full(
                (batch, self.num_classes, height, width),
                0.1 / (self.num_classes - 1) if self.num_classes > 1 else 0.0,
                dtype=np.float32
            )
            for b in range(batch):
                for c in range(self.num_classes):
                    mask = (preds[b] == c)
                    logits[b, c, mask] = 0.9
        else:
            # Hard one-hot: 1.0 for predicted class, 0.0 for others
            eye = np.eye(self.num_classes, dtype=np.float32)
            logits = np.zeros((batch, self.num_classes, height, width), dtype=np.float32)
            for b in range(batch):
                logits[b] = eye[preds[b]].transpose(2, 0, 1)
        
        return logits
