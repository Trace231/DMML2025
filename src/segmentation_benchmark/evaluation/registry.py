"""Simple registry used to access the available segmentation backbones."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, MutableMapping

from ..models.base import BaseSegmenter

logger = logging.getLogger(__name__)

_SEGMENTER_REGISTRY: Dict[str, Callable[..., BaseSegmenter]] = {}
_MODULES_IMPORTED = False

# Pre-import all model modules to ensure registration happens
_MODEL_MODULES = [
    "segmentation_benchmark.models.classical_crf",
    "segmentation_benchmark.models.cnn",
    "segmentation_benchmark.models.transformer",
    "segmentation_benchmark.models.diffusion",
    "segmentation_benchmark.models.hybrid",
    "segmentation_benchmark.models.cnn_crf",
    "segmentation_benchmark.models.crf_postprocess",
]

def _ensure_modules_imported() -> None:
    """Import all model modules to trigger registration."""
    global _MODULES_IMPORTED
    if _MODULES_IMPORTED:
        return
    for module_name in _MODEL_MODULES:
        try:
            __import__(module_name)
        except Exception as e:
            logger.warning(f"Failed to import {module_name}: {e}")
    _MODULES_IMPORTED = True


def register_segmenter(name: str) -> Callable[[Callable[..., BaseSegmenter]], Callable[..., BaseSegmenter]]:
    """Decorator used to register a segmenter factory or class."""

    def decorator(builder: Callable[..., BaseSegmenter]) -> Callable[..., BaseSegmenter]:
        if name in _SEGMENTER_REGISTRY:
            raise KeyError(f"Segmenter '{name}' already registered")
        _SEGMENTER_REGISTRY[name] = builder
        return builder

    return decorator


def build_segmenter(name: str, **kwargs: Any) -> BaseSegmenter:
    # Ensure all modules are imported
    _ensure_modules_imported()
    
    if name not in _SEGMENTER_REGISTRY:
        # Try to import model modules that might register this segmenter
        # Map of segmenter names to their module paths
        _MODULE_MAP = {
            "classical_crf": "segmentation_benchmark.models.classical_crf",
            "fcn_resnet50": "segmentation_benchmark.models.cnn",
            "deeplabv3_resnet50": "segmentation_benchmark.models.cnn",
            "torchvision": "segmentation_benchmark.models.cnn",
            "segformer": "segmentation_benchmark.models.transformer",
            "segformer_b0": "segmentation_benchmark.models.transformer",
            "random_walker": "segmentation_benchmark.models.diffusion",
            "hybrid_unet_transformer": "segmentation_benchmark.models.hybrid",
            "cnn_crf": "segmentation_benchmark.models.cnn_crf",
            "crf_wrapper": "segmentation_benchmark.models.crf_postprocess",
        }
        
        if name in _MODULE_MAP:
            try:
                __import__(_MODULE_MAP[name])
            except Exception as e:
                logger.warning(f"Failed to import {_MODULE_MAP[name]}: {e}")
        
        # Check again after potential import
        if name not in _SEGMENTER_REGISTRY:
            available = ", ".join(sorted(_SEGMENTER_REGISTRY)) or "<empty>"
            raise KeyError(f"Unknown segmenter '{name}'. Available: {available}")
    
    builder = _SEGMENTER_REGISTRY[name]
    return builder(**kwargs)


def list_segmenters() -> Iterable[str]:
    return sorted(_SEGMENTER_REGISTRY)


def registry() -> MutableMapping[str, Callable[..., BaseSegmenter]]:
    """Expose the registry for advanced use-cases (read-only please)."""
    return _SEGMENTER_REGISTRY
