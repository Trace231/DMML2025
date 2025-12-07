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


@dataclass
class AdaptiveCrfConfig:
    """Configuration for adaptive CRF parameter adjustment."""
    # Base parameters (used as starting point)
    base_iterations: int = 5
    base_gaussian_sxy: int = 3
    base_bilateral_sxy: int = 30
    base_bilateral_srgb: int = 13
    base_compat_gaussian: int = 3
    base_compat_bilateral: int = 10
    
    # Adaptive scaling factors
    scale_by_image_size: bool = True  # Adjust spatial params based on image size
    scale_by_entropy: bool = True  # Adjust iterations based on prediction uncertainty
    scale_by_contrast: bool = False  # Adjust bilateral params based on image contrast
    
    # Size scaling: reference size is 256x256
    reference_size: int = 256
    
    # Entropy-based scaling
    entropy_threshold_low: float = 0.3  # Low entropy -> fewer iterations
    entropy_threshold_high: float = 0.7  # High entropy -> more iterations
    min_iterations: int = 3
    max_iterations: int = 10
    
    # Contrast-based scaling (if enabled)
    contrast_threshold_low: float = 0.1
    contrast_threshold_high: float = 0.3


class CrfPostProcessor:
    def __init__(
        self, 
        num_classes: int = 2, 
        params: CrfParams | None = None,
        use_boundary_aware: bool = False,
        boundary_threshold: float = 0.3,
    ) -> None:
        if dcrf is None or unary_from_softmax is None:
            raise ImportError("pydensecrf is required for CRF post processing")
        self.num_classes = num_classes
        self.params = params or CrfParams()
        self.use_boundary_aware = use_boundary_aware
        self.boundary_threshold = boundary_threshold

    def refine(self, image: Any, logits: Any) -> np.ndarray:
        image_np = CrfPostProcessor._ensure_image(image)
        probabilities = self._ensure_probabilities(logits)
        h, w = image_np.shape[:2]
        
        # Boundary-aware CRF: apply stronger CRF only near boundaries
        if self.use_boundary_aware:
            return self._refine_boundary_aware(image_np, probabilities, h, w)
        
        # Standard CRF
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
    
    def _refine_boundary_aware(self, image_np: np.ndarray, probabilities: np.ndarray, h: int, w: int) -> np.ndarray:
        """Apply CRF with boundary-aware processing."""
        # Detect boundaries from predictions
        pred = np.argmax(probabilities, axis=0)
        
        # Compute gradient magnitude to detect boundaries
        from scipy import ndimage
        sobel_x = ndimage.sobel(pred.astype(float), axis=1)
        sobel_y = ndimage.sobel(pred.astype(float), axis=0)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        boundary_mask = gradient_magnitude > (gradient_magnitude.max() * self.boundary_threshold)
        
        # Apply standard CRF
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
        
        # Apply stronger CRF only on boundary regions
        if boundary_mask.any():
            # Create a mask for boundary regions (dilated)
            from scipy import ndimage
            boundary_dilated = ndimage.binary_dilation(boundary_mask, iterations=3)
            
            # Apply stronger CRF on boundary regions
            boundary_params = CrfParams(
                iterations=self.params.iterations + 2,  # More iterations
                gaussian_sxy=self.params.gaussian_sxy,
                bilateral_sxy=int(self.params.bilateral_sxy * 1.2),  # Larger spatial
                bilateral_srgb=self.params.bilateral_srgb,
                compat_gaussian=self.params.compat_gaussian,
                compat_bilateral=int(self.params.compat_bilateral * 1.2),  # Stronger compatibility
            )
            
            # Extract boundary region
            boundary_coords = np.where(boundary_dilated)
            if len(boundary_coords[0]) > 0:
                # For simplicity, apply CRF to entire image with stronger params
                # In a more sophisticated implementation, we could crop and process only boundary regions
                dense_boundary = dcrf.DenseCRF2D(w, h, self.num_classes)
                dense_boundary.setUnaryEnergy(unary)
                dense_boundary.addPairwiseGaussian(
                    sxy=boundary_params.gaussian_sxy, 
                    compat=boundary_params.compat_gaussian
                )
                dense_boundary.addPairwiseBilateral(
                    sxy=boundary_params.bilateral_sxy,
                    srgb=boundary_params.bilateral_srgb,
                    rgbim=image_np,
                    compat=boundary_params.compat_bilateral,
                )
                q_boundary = dense_boundary.inference(boundary_params.iterations)
                refined_boundary = np.argmax(np.array(q_boundary).reshape(self.num_classes, h, w), axis=0)
                
                # Blend: use boundary-refined result in boundary regions, standard result elsewhere
                refined = np.where(boundary_dilated, refined_boundary, refined)
        
        return refined.astype(np.int64)

    @staticmethod
    def _ensure_image(image: Any) -> np.ndarray:
        """Convert image to uint8 RGB format for CRF.
        
        Handles:
        - ImageNet-normalized tensors (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        - Normalized tensors in [0, 1] range
        - Already denormalized images in [0, 255] range
        """
        import torch

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 4:
                image = image[0]  # Take first image from batch
            # Convert from (C, H, W) to (H, W, C)
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        image = np.asarray(image, dtype=np.float32)
        
        # Check if image is ImageNet-normalized
        # ImageNet normalization: (x/255 - mean) / std
        # Normalized values are typically in [-2.5, 2.5] range
        # If we see negative values or values > 1.5, likely normalized
        img_min, img_max = image.min(), image.max()
        is_imagenet_normalized = img_min < -0.5 or (img_min < 0 and img_max > 1.5)
        
        if is_imagenet_normalized:
            # Denormalize: reverse ImageNet normalization
            # Original: normalized = (x/255 - mean) / std
            # Reverse: x/255 = normalized * std + mean, so x = (normalized * std + mean) * 255
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # Ensure image is (H, W, C) format
            if image.ndim == 3:
                if image.shape[0] == 3 and image.shape[2] != 3:
                    # Channels first, transpose
                    image = np.transpose(image, (1, 2, 0))
                
                # Apply denormalization per channel
                if image.shape[-1] == 3:
                    image = image * std + mean
                else:
                    # Single channel or unexpected format
                    image = image * std[0] + mean[0]
            
            # Clip to [0, 1] and convert to [0, 255]
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        elif img_max <= 1.0:
            # Already in [0, 1] range, scale to [0, 255]
            image = (image * 255.0).astype(np.uint8)
        else:
            # Already in [0, 255] range or higher, clip and convert
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Ensure C-contiguous array for pydensecrf
        return np.ascontiguousarray(image)

    def _ensure_probabilities(self, logits: Any) -> np.ndarray:
        """Convert logits to probability distribution."""
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


class AdaptiveCrfPostProcessor:
    """Adaptive CRF post-processor that adjusts parameters based on image characteristics."""
    
    def __init__(
        self, 
        num_classes: int = 2, 
        config: AdaptiveCrfConfig | Dict[str, Any] | None = None
    ) -> None:
        if dcrf is None or unary_from_softmax is None:
            raise ImportError("pydensecrf is required for CRF post processing")
        self.num_classes = num_classes
        if isinstance(config, dict):
            config = AdaptiveCrfConfig(**config)
        self.config = config or AdaptiveCrfConfig()
    
    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """Compute mean entropy of probability distribution."""
        # probabilities shape: (C, H, W)
        # Avoid log(0)
        probs_clipped = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=0)
        return float(np.mean(entropy))
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        """Compute image contrast (standard deviation of grayscale)."""
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        return float(np.std(gray) / 255.0)  # Normalize to [0, 1]
    
    def _adapt_params(
        self, 
        image: np.ndarray, 
        probabilities: np.ndarray
    ) -> CrfParams:
        """Adaptively compute CRF parameters based on image and prediction characteristics."""
        h, w = image.shape[:2]
        
        # Start with base parameters
        params = CrfParams(
            iterations=self.config.base_iterations,
            gaussian_sxy=self.config.base_gaussian_sxy,
            bilateral_sxy=self.config.base_bilateral_sxy,
            bilateral_srgb=self.config.base_bilateral_srgb,
            compat_gaussian=self.config.base_compat_gaussian,
            compat_bilateral=self.config.base_compat_bilateral,
        )
        
        # 1. Scale by image size
        if self.config.scale_by_image_size:
            size_ratio = np.sqrt((h * w) / (self.config.reference_size ** 2))
            # Scale spatial parameters proportionally
            params.bilateral_sxy = int(np.clip(
                self.config.base_bilateral_sxy * size_ratio,
                10, 80  # Reasonable bounds
            ))
            params.gaussian_sxy = max(1, int(self.config.base_gaussian_sxy * np.sqrt(size_ratio)))
        
        # 2. Scale iterations by entropy (uncertainty)
        if self.config.scale_by_entropy:
            entropy = self._compute_entropy(probabilities)
            # Normalize entropy to [0, 1] range (max entropy for 2 classes is log(2) ≈ 0.693)
            max_entropy = np.log(self.num_classes)
            normalized_entropy = entropy / max_entropy
            
            # Map entropy to iteration count
            if normalized_entropy < self.config.entropy_threshold_low:
                # Low uncertainty: fewer iterations
                iteration_factor = 0.6
            elif normalized_entropy > self.config.entropy_threshold_high:
                # High uncertainty: more iterations
                iteration_factor = 1.4
            else:
                # Medium uncertainty: linear interpolation
                t = (normalized_entropy - self.config.entropy_threshold_low) / (
                    self.config.entropy_threshold_high - self.config.entropy_threshold_low
                )
                iteration_factor = 0.6 + 0.8 * t
            
            params.iterations = int(np.clip(
                self.config.base_iterations * iteration_factor,
                self.config.min_iterations,
                self.config.max_iterations
            ))
        
        # 3. Scale bilateral parameters by image contrast (optional)
        if self.config.scale_by_contrast:
            contrast = self._compute_contrast(image)
            if contrast < self.config.contrast_threshold_low:
                # Low contrast: stronger bilateral smoothing
                contrast_factor = 1.2
            elif contrast > self.config.contrast_threshold_high:
                # High contrast: weaker bilateral smoothing
                contrast_factor = 0.8
            else:
                contrast_factor = 1.0
            
            params.compat_bilateral = int(np.clip(
                self.config.base_compat_bilateral * contrast_factor,
                5, 20
            ))
        
        return params
    
    def refine(self, image: Any, logits: Any) -> np.ndarray:
        """Refine predictions using adaptively computed CRF parameters."""
        image_np = AdaptiveCrfPostProcessor._ensure_image(image)
        probabilities = self._ensure_probabilities(logits)
        h, w = image_np.shape[:2]
        
        # Compute adaptive parameters
        params = self._adapt_params(image_np, probabilities)
        
        # Apply CRF with adaptive parameters
        dense = dcrf.DenseCRF2D(w, h, self.num_classes)
        unary = unary_from_softmax(probabilities)
        dense.setUnaryEnergy(unary)
        dense.addPairwiseGaussian(sxy=params.gaussian_sxy, compat=params.compat_gaussian)
        dense.addPairwiseBilateral(
            sxy=params.bilateral_sxy,
            srgb=params.bilateral_srgb,
            rgbim=image_np,
            compat=params.compat_bilateral,
        )
        q = dense.inference(params.iterations)
        refined = np.argmax(np.array(q).reshape(self.num_classes, h, w), axis=0)
        return refined.astype(np.int64)
    
    @staticmethod
    def _ensure_image(image: Any) -> np.ndarray:
        """Convert image to uint8 RGB format for CRF."""
        import torch

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
        
        image = np.asarray(image, dtype=np.float32)
        img_min, img_max = image.min(), image.max()
        is_imagenet_normalized = img_min < -0.5 or (img_min < 0 and img_max > 1.5)
        
        if is_imagenet_normalized:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            if image.ndim == 3:
                if image.shape[0] == 3 and image.shape[2] != 3:
                    image = np.transpose(image, (1, 2, 0))
                if image.shape[-1] == 3:
                    image = image * std + mean
                else:
                    image = image * std[0] + mean[0]
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        elif img_max <= 1.0:
            image = (image * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return np.ascontiguousarray(image)
    
    def _ensure_probabilities(self, logits: Any) -> np.ndarray:
        """Convert logits to probability distribution."""
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
        use_adaptive_crf: bool = True,
        adaptive_crf_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if enable_crf and use_adaptive_crf:
            suffix = "AdaptiveCRF"
        elif enable_crf:
            suffix = "CRFPost"
        else:
            suffix = "NoCRF"
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
            if use_adaptive_crf:
                self.post = AdaptiveCrfPostProcessor(
                    num_classes=num_classes, 
                    config=adaptive_crf_config
                )
            else:
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
