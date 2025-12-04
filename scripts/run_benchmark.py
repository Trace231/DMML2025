"""Run the segmentation benchmark configured via YAML."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from segmentation_benchmark.data import create_dataloaders
from segmentation_benchmark.evaluation import BenchmarkEvaluator, build_segmenter
from segmentation_benchmark.utils import load_config
from segmentation_benchmark.utils.paths import ProjectPaths


LOGGER = logging.getLogger("segmentation_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation benchmarking runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "crackforest_benchmark.yaml",
        help="Path to the benchmark configuration file",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional override for output directory")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda)")
    parser.add_argument("--skip-train", action="store_true", help="Disable using training split for model preparation")
    parser.add_argument("--save-predictions", action="store_true", help="Persist predictions as numpy arrays")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging (may show verbose output from libraries)")
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    # Set root logger level
    root_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=root_level,
        format="[%(levelname)s] %(message)s",
    )
    
    # Suppress verbose output from third-party libraries unless verbose mode is enabled
    if not verbose:
        # Suppress pydensecrf and other third-party debug output
        logging.getLogger("pydensecrf").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("Pillow").setLevel(logging.WARNING)


def run_benchmark(config: Dict[str, Any], output_dir: Path, device_override: str | None = None, skip_train: bool = False, save_predictions: bool = False) -> List[Dict[str, Any]]:
    paths = ProjectPaths.from_root()
    paths.ensure()

    dataset_cfg = config.get("dataset", {})
    LOGGER.info("Creating dataloaders (shared across all models)...")
    dataloaders = create_dataloaders(dataset_cfg)
    LOGGER.info("Dataloaders created. Dataset will be reused for all models.")

    evaluation_cfg = config.get("evaluation", {})
    num_classes = int(dataset_cfg.get("num_classes", 2))
    evaluator = BenchmarkEvaluator(
        num_classes=num_classes,
        output_dir=output_dir,
        save_predictions=save_predictions,
    )

    models_cfg = config.get("models", [])
    summaries: List[Dict[str, Any]] = []
    
    LOGGER.info("Starting evaluation of %d model(s)...", len(models_cfg))

    for model_cfg in models_cfg:
        name = model_cfg.get("name")
        if not name:
            raise ValueError("Each model configuration requires a 'name'")
        builder = model_cfg.get("builder", name)
        params = dict(model_cfg.get("params", {}))
        params.setdefault("num_classes", num_classes)
        if device_override is not None:
            params.setdefault("device", device_override)

        LOGGER.info("Preparing model %s using builder '%s'", name, builder)
        segmenter = build_segmenter(builder, **params)

        train_dataset = None if skip_train else dataloaders.get("train_dataset")
        val_dataset = None if skip_train else dataloaders.get("val_dataset")

        report = evaluator.evaluate(
            segmenter=segmenter,
            test_loader=dataloaders["test_loader"],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_name=name,
        )

        summary_row = {"model": name, **report.summary}
        summaries.append(summary_row)
        LOGGER.info("Finished %s — mIoU: %.4f | Dice: %.4f", name, report.summary["mean_iou"], report.summary["mean_dice"])

    table = pd.DataFrame(summaries)
    table_path = output_dir / "benchmark_summary.csv"
    table.to_csv(table_path, index=False)
    LOGGER.info("Saved summary table to %s", table_path)

    overall_path = output_dir / "benchmark_summary.json"
    overall_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    return summaries


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    config = load_config(args.config)
    output_dir = args.output or config.get("evaluation", {}).get("output_dir")
    if output_dir is None:
        run_name = config.get("evaluation", {}).get("run_name", "benchmark")
        output_dir = ProjectPaths.from_root().reports_dir / run_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_benchmark(config, output_dir, device_override=args.device, skip_train=args.skip_train, save_predictions=args.save_predictions)


if __name__ == "__main__":
    main()
