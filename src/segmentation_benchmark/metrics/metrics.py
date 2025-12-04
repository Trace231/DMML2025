"""Metric computation helpers for segmentation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

_EPS = 1e-7


@dataclass
class SegmentationMetrics:
    pixel_accuracy: float
    mean_iou: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_dice: float
    per_class_iou: np.ndarray
    per_class_precision: np.ndarray
    per_class_recall: np.ndarray
    per_class_f1: np.ndarray
    per_class_dice: np.ndarray


def _confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    pred = pred.astype(np.int64).ravel()
    target = target.astype(np.int64).ravel()
    mask = (target >= 0) & (target < num_classes)
    combined = num_classes * target[mask] + pred[mask]
    counts = np.bincount(combined, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes).astype(np.float64)


def _metrics_from_confusion(confusion: np.ndarray) -> SegmentationMetrics:
    with np.errstate(divide="ignore", invalid="ignore"):
        true_positive = np.diag(confusion)
        support = confusion.sum(axis=1)
        predicted = confusion.sum(axis=0)

        precision = true_positive / (predicted + _EPS)
        recall = true_positive / (support + _EPS)
        f1 = 2 * precision * recall / (precision + recall + _EPS)
        dice = 2 * true_positive / (predicted + support + _EPS)
        union = support + predicted - true_positive
        iou = true_positive / (union + _EPS)

        pixel_accuracy = true_positive.sum() / (confusion.sum() + _EPS)

    mean_precision = float(np.nanmean(precision))
    mean_recall = float(np.nanmean(recall))
    mean_f1 = float(np.nanmean(f1))
    mean_dice = float(np.nanmean(dice))
    mean_iou = float(np.nanmean(iou))

    return SegmentationMetrics(
        pixel_accuracy=float(pixel_accuracy),
        mean_iou=mean_iou,
        mean_precision=mean_precision,
        mean_recall=mean_recall,
        mean_f1=mean_f1,
        mean_dice=mean_dice,
        per_class_iou=iou,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        per_class_dice=dice,
    )


def compute_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> SegmentationMetrics:
    """Compute segmentation metrics for a single prediction/target pair."""
    confusion = _confusion_matrix(pred, target, num_classes)
    return _metrics_from_confusion(confusion)


class MetricsAggregator:
    """Aggregate metrics across a dataset."""

    def __init__(self, num_classes: int, store_per_sample: bool = False) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
        self.store_per_sample = store_per_sample
        self.per_sample: List[SegmentationMetrics] = []

    def update(self, pred: np.ndarray, target: np.ndarray) -> SegmentationMetrics:
        confusion = _confusion_matrix(pred, target, self.num_classes)
        self.confusion += confusion
        metrics = _metrics_from_confusion(confusion)
        if self.store_per_sample:
            self.per_sample.append(metrics)
        return metrics

    def summary(self) -> SegmentationMetrics:
        return _metrics_from_confusion(self.confusion)

    def as_dict(self) -> Dict[str, float]:
        summary = self.summary()
        return {
            "pixel_accuracy": summary.pixel_accuracy,
            "mean_iou": summary.mean_iou,
            "mean_precision": summary.mean_precision,
            "mean_recall": summary.mean_recall,
            "mean_f1": summary.mean_f1,
            "mean_dice": summary.mean_dice,
        }
