"""Evaluate all epoch checkpoints for models in a benchmark configuration.

This script finds all epoch checkpoints (saved every 10 epochs) and evaluates
each one to track model performance during training.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from segmentation_benchmark.data import create_dataloaders
from segmentation_benchmark.evaluation import BenchmarkEvaluator, build_segmenter
from segmentation_benchmark.utils import load_config
from segmentation_benchmark.utils.checkpoint import list_epoch_checkpoints, load_epoch_checkpoint
from segmentation_benchmark.utils.paths import ProjectPaths


LOGGER = logging.getLogger("segmentation_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate epoch checkpoints from training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "voc_benchmark.yaml",
        help="Path to the benchmark configuration file",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional override for output directory")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda)")
    parser.add_argument("--save-predictions", action="store_true", help="Persist predictions as numpy arrays")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Reduce noise from libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("Pillow").setLevel(logging.WARNING)


def get_model_builder_name(model_cfg: Dict[str, Any]) -> str:
    """Get the builder name for a model configuration."""
    builder = model_cfg.get("builder")
    if not builder:
        name = model_cfg.get("name")
        # Map common model names to builders
        if name and "fcn" in name.lower():
            return "fcn_resnet50"
        elif name and "deeplab" in name.lower():
            return "deeplabv3_resnet50"
        elif name and "segformer" in name.lower():
            return "segformer_b0"
        elif name and "hybrid" in name.lower():
            return "hybrid_unet_transformer"
        return name
    return builder


def evaluate_epoch_checkpoints(
    config: Dict[str, Any],
    output_dir: Path,
    device_override: str | None = None,
    save_predictions: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate all epoch checkpoints for models in the configuration."""
    paths = ProjectPaths.from_root()
    paths.ensure()

    dataset_cfg = config.get("dataset", {})
    LOGGER.info("Creating dataloaders...")
    dataloaders = create_dataloaders(dataset_cfg)
    LOGGER.info("Dataloaders created.")

    evaluation_cfg = config.get("evaluation", {})
    num_classes = int(dataset_cfg.get("num_classes", 2))
    evaluator = BenchmarkEvaluator(
        num_classes=num_classes,
        output_dir=output_dir,
        save_predictions=save_predictions,
    )

    models_cfg = config.get("models", [])
    all_summaries: List[Dict[str, Any]] = []
    
    LOGGER.info("Starting evaluation of epoch checkpoints for %d model(s)...", len(models_cfg))

    for model_cfg in models_cfg:
        name = model_cfg.get("name")
        if not name:
            continue
        
        builder = get_model_builder_name(model_cfg)
        params = dict(model_cfg.get("params", {}))
        params.setdefault("num_classes", num_classes)
        if device_override is not None:
            params.setdefault("device", device_override)
        
        # Skip models that don't support epoch checkpoints (e.g., crf_wrapper, classical_crf)
        if builder in ["crf_wrapper", "classical_crf", "ddp", "random_walker"]:
            LOGGER.info("Skipping %s (does not support epoch checkpoints)", name)
            continue
        
        # Get the config that would be used for checkpoint matching
        # We need to build a temporary segmenter to get its config
        # But we need to set finetune_epochs to match what was used during training
        # to find the correct checkpoints
        try:
            # Temporarily set finetune_epochs to a value that would match training checkpoints
            # The actual value doesn't matter for epoch checkpoint matching, but we need it
            # to build the segmenter
            original_finetune_epochs = params.get("finetune_epochs", 0)
            # Use a reasonable default if not specified (e.g., 100 for VOC)
            if original_finetune_epochs == 0:
                params["finetune_epochs"] = 100  # Default training epochs
            
            temp_segmenter = build_segmenter(builder, **params)
            if hasattr(temp_segmenter, "_get_config"):
                model_config = temp_segmenter._get_config()
            else:
                LOGGER.warning("Model %s does not have _get_config method, skipping", name)
                continue
        except Exception as e:
            LOGGER.warning("Could not build model %s to get config: %s", name, e)
            continue
        
        # Find all epoch checkpoints for this model
        epoch_checkpoints = list_epoch_checkpoints(builder, model_config)
        
        if not epoch_checkpoints:
            LOGGER.info("No epoch checkpoints found for %s (builder: %s)", name, builder)
            continue
        
        LOGGER.info("Found %d epoch checkpoints for %s", len(epoch_checkpoints), name)
        
        # Evaluate each epoch checkpoint
        for epoch, checkpoint_path in epoch_checkpoints:
            LOGGER.info("Evaluating %s at epoch %d...", name, epoch)
            
            # Build the segmenter
            segmenter = build_segmenter(builder, **params)
            
            # Load the epoch checkpoint
            device = torch.device(device_override or str(segmenter.device))
            
            # Get the model object to load weights into
            model_to_load = None
            if hasattr(segmenter, "model"):
                model_to_load = segmenter.model
            elif hasattr(segmenter, "base_segmenter") and hasattr(segmenter.base_segmenter, "model"):
                model_to_load = segmenter.base_segmenter.model
            
            checkpoint = load_epoch_checkpoint(
                builder,
                model_config,
                epoch,
                model=model_to_load,
                device=device,
            )
            
            if checkpoint is None:
                LOGGER.warning("Could not load checkpoint for %s at epoch %d", name, epoch)
                continue
            
            # Mark as checkpoint loaded to skip training
            if hasattr(segmenter, "_checkpoint_loaded"):
                segmenter._checkpoint_loaded = True
            if hasattr(segmenter, "base_segmenter") and hasattr(segmenter.base_segmenter, "_checkpoint_loaded"):
                segmenter.base_segmenter._checkpoint_loaded = True
            
            # Evaluate
            model_name = f"{name}_epoch_{epoch}"
            try:
                report = evaluator.evaluate(
                    segmenter=segmenter,
                    test_loader=dataloaders["test_loader"],
                    train_dataset=None,  # Skip training, we're using checkpoints
                    val_dataset=None,
                    model_name=model_name,
                )
                
                summary_row = {
                    "model": model_name,
                    "base_model": name,
                    "epoch": epoch,
                    **report.summary
                }
                all_summaries.append(summary_row)
                LOGGER.info(
                    "Finished %s (epoch %d) — mIoU: %.4f | Dice: %.4f",
                    name, epoch, report.summary["mean_iou"], report.summary["mean_dice"]
                )
            except Exception as e:
                LOGGER.error("Error evaluating %s at epoch %d: %s", name, epoch, e, exc_info=True)
                continue
    
    # Save results
    if all_summaries:
        table = pd.DataFrame(all_summaries)
        table_path = output_dir / "epoch_checkpoints_summary.csv"
        table.to_csv(table_path, index=False)
        LOGGER.info("Saved epoch checkpoints summary to %s", table_path)
        
        # Also save as JSON
        json_path = output_dir / "epoch_checkpoints_summary.json"
        with open(json_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        LOGGER.info("Saved epoch checkpoints summary (JSON) to %s", json_path)
    else:
        LOGGER.warning("No epoch checkpoints were evaluated")
    
    return all_summaries


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    config = load_config(args.config)
    output_dir = args.output or config.get("evaluation", {}).get("output_dir")
    if output_dir is None:
        run_name = config.get("evaluation", {}).get("run_name", "benchmark")
        output_dir = ProjectPaths.from_root().reports_dir / run_name / "epoch_checkpoints"
    else:
        output_dir = Path(output_dir) / "epoch_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_epoch_checkpoints(
        config,
        output_dir,
        device_override=args.device,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()

