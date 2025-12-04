"""Segmentation benchmarking toolkit."""

from .evaluation.registry import build_segmenter, list_segmenters

__all__ = ["build_segmenter", "list_segmenters"]
