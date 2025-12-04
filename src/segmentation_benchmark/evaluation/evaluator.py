"""Benchmark orchestration utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from ..metrics import MetricsAggregator, SegmentationMetrics
from ..models.base import BaseSegmenter


@dataclass
class EvaluationReport:
    model_name: str
    summary: Dict[str, Any]
    per_image: List[Dict[str, Any]]
    confusion_matrix: List[List[float]]
    output_dir: Path

    def save_json(self, path: Path) -> None:
        data = {
            "model_name": self.model_name,
            "summary": self.summary,
            "per_image": self.per_image,
            "confusion_matrix": self.confusion_matrix,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _metrics_to_dict(metrics: SegmentationMetrics) -> Dict[str, Any]:
    return {
        "pixel_accuracy": metrics.pixel_accuracy,
        "mean_iou": metrics.mean_iou,
        "mean_precision": metrics.mean_precision,
        "mean_recall": metrics.mean_recall,
        "mean_f1": metrics.mean_f1,
        "mean_dice": metrics.mean_dice,
        "per_class_iou": metrics.per_class_iou.tolist(),
        "per_class_precision": metrics.per_class_precision.tolist(),
        "per_class_recall": metrics.per_class_recall.tolist(),
        "per_class_f1": metrics.per_class_f1.tolist(),
        "per_class_dice": metrics.per_class_dice.tolist(),
    }


class BenchmarkEvaluator:
    """Run a suite of segmenters over a dataset and compute metrics."""

    def __init__(
        self,
        num_classes: int,
        output_dir: Path | str,
        save_predictions: bool = False,
        prediction_subdir: str = "predictions",
    ) -> None:
        self.num_classes = num_classes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_predictions = save_predictions
        self.prediction_dir = self.output_dir / prediction_subdir
        if save_predictions:
            self.prediction_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        segmenter: BaseSegmenter,
        test_loader: Iterable[Dict[str, Any]],
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        model_name: Optional[str] = None,
    ) -> EvaluationReport:
        model_name = model_name or segmenter.name
        aggregator = MetricsAggregator(self.num_classes, store_per_sample=True)
        segmenter.prepare(train_dataset=train_dataset, val_dataset=val_dataset)

        per_image_records: List[Dict[str, Any]] = []

        iterator = tqdm(test_loader, desc=f"Evaluating {model_name}")
        for batch_index, batch in enumerate(iterator):
            logits = segmenter.predict_logits(batch)
            preds = segmenter.predict_batch(batch)
            if preds.ndim == 2:
                preds = preds[None, ...]
            mask = batch["mask"]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = mask.astype(np.int64)
            # `mask` can come in different shapes depending on the dataset:
            # - (B, H, W)           e.g. our PascalVOCDataset
            # - (B, 1, H, W)        e.g. some torch datasets with channel dim
            # We only squeeze a singleton channel dimension, and leave (B, H, W) untouched.
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask[:, 0, ...]
            batch_size = preds.shape[0]
            for i in range(batch_size):
                metrics = aggregator.update(preds[i], mask[i])
                record = _metrics_to_dict(metrics)
                
                # Extract index - handle both tensor/array and scalar cases
                if "index" in batch:
                    index_val = batch["index"]
                    if hasattr(index_val, "item"):  # PyTorch tensor
                        if index_val.ndim == 0:
                            sample_index = int(index_val.item())
                        else:
                            sample_index = int(index_val[i].item())
                    elif hasattr(index_val, "__len__") and not isinstance(index_val, str):
                        # It's an array/list
                        sample_index = int(index_val[i])
                    else:
                        # It's a scalar
                        sample_index = int(index_val)
                else:
                    sample_index = batch_index * batch_size + i
                record["sample_index"] = sample_index
                
                # Extract image_id - handle both list/array and scalar cases
                if "image_id" in batch:
                    image_id_val = batch["image_id"]
                    if isinstance(image_id_val, (list, tuple)):
                        image_id = str(image_id_val[i])
                    elif hasattr(image_id_val, "__len__") and not isinstance(image_id_val, str):
                        # It's an array/tensor
                        image_id = str(image_id_val[i])
                    else:
                        # It's a scalar string
                        image_id = str(image_id_val)
                else:
                    image_id = str(record["sample_index"])
                record["image_id"] = image_id
                
                per_image_records.append(record)

                if self.save_predictions:
                    self._save_prediction(model_name, record["image_id"], preds[i])

        summary = aggregator.summary()
        summary_dict = _metrics_to_dict(summary)
        confusion = aggregator.confusion.tolist()

        report = EvaluationReport(
            model_name=model_name,
            summary=summary_dict,
            per_image=per_image_records,
            confusion_matrix=confusion,
            output_dir=self.output_dir,
        )

        report_path = self.output_dir / f"{model_name}_metrics.json"
        report.save_json(report_path)

        return report

    def _save_prediction(self, model_name: str, image_id: str, prediction: np.ndarray) -> None:
        path = self.prediction_dir / model_name
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / f"{image_id}.npy", prediction.astype(np.uint8))
