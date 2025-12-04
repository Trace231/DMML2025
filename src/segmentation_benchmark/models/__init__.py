"""Model package exports with lazy loading to avoid circular imports."""

from .base import BaseSegmenter

__all__ = [
    "BaseSegmenter",
    "ClassicalCRFSegmenter",
    "TorchvisionSegmenter",
    "SegformerSegmenter",
    "DDPSegmenter",
    "HybridUNetTransformerSegmenter",
    "CnnCrfSegmenter",
    "CrfPostProcessor",
    "CrfWrappedSegmenter",
]

_LAZY_MAP = {
    "ClassicalCRFSegmenter": ".classical_crf",
    "TorchvisionSegmenter": ".cnn",
    "SegformerSegmenter": ".transformer",
    "DDPSegmenter": ".diffusion",
    "HybridUNetTransformerSegmenter": ".hybrid",
    "CnnCrfSegmenter": ".cnn_crf",
    "CrfPostProcessor": ".crf_postprocess",
    "CrfWrappedSegmenter": ".crf_postprocess",
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        module = __import__(f"{__name__}{_LAZY_MAP[name]}", fromlist=[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)