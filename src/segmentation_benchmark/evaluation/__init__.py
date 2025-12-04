"""Evaluation pipelines and registries."""

from .registry import build_segmenter, list_segmenters, register_segmenter
from .evaluator import BenchmarkEvaluator, EvaluationReport

__all__ = [
    "build_segmenter",
    "list_segmenters",
    "register_segmenter",
    "BenchmarkEvaluator",
    "EvaluationReport",
]
