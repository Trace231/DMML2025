"""Evaluate all epoch checkpoints for models in a benchmark configuration.

This script finds all epoch checkpoints (saved every 10 epochs) and evaluates
each one to track model performance during training. Optionally compares with
CRF post-processing and generates visualization plots.
"""
#debug
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
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
    parser.add_argument("--compare-crf", action="store_true", help="Also evaluate with CRF post-processing for comparison")
    parser.add_argument("--crf-params", type=json.loads, default=None, help="JSON string for CRF parameters (e.g., '{\"iterations\": 5}')")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating visualization plots")
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


def create_visualizations(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Create visualization plots for epoch checkpoint evaluation results."""
    if not summaries:
        return
    
    df = pd.DataFrame(summaries)
    
    # Group by base model
    base_models = df["base_model"].unique()
    
    for base_model in base_models:
        model_df = df[df["base_model"] == base_model].copy()
        
        # Separate original and CRF results
        original_df = model_df[~model_df["model"].str.contains("_crf", case=False)].copy()
        crf_df = model_df[model_df["model"].str.contains("_crf", case=False)].copy()
        
        if original_df.empty:
            continue
        
        # Create figure with subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Training Progress: {base_model}", fontsize=16, fontweight="bold")
        
        metrics_to_plot = [
            ("mean_iou", "Mean IoU", axes[0, 0]),
            ("mean_dice", "Mean Dice", axes[0, 1]),
            ("overall_accuracy", "Overall Accuracy", axes[1, 0]),
            ("mean_precision", "Mean Precision", axes[1, 1]),
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            if metric_key not in original_df.columns:
                ax.text(0.5, 0.5, f"{metric_name}\n(not available)", 
                       ha="center", va="center", transform=ax.transAxes)
                ax.set_title(metric_name)
                continue
            
            # Plot original model
            original_sorted = original_df.sort_values("epoch")
            ax.plot(original_sorted["epoch"], original_sorted[metric_key], 
                   marker="o", linestyle="-", linewidth=2, markersize=6,
                   label="Original", color="#1f77b4")
            
            # Plot CRF version if available
            if not crf_df.empty:
                crf_sorted = crf_df.copy()
                # Ensure epoch column exists
                if "epoch" not in crf_sorted.columns:
                    crf_sorted["epoch"] = crf_sorted["model"].str.extract(r"epoch_(\d+)")[0].astype(int)
                crf_sorted = crf_sorted.sort_values("epoch")
                if metric_key in crf_sorted.columns:
                    ax.plot(crf_sorted["epoch"], crf_sorted[metric_key],
                           marker="s", linestyle="--", linewidth=2, markersize=6,
                           label="With CRF", color="#ff7f0e")
            
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        plot_path = output_dir / f"{base_model}_training_progress.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved training progress plot to %s", plot_path)
        
        # Create a comparison plot if CRF results are available
        if not crf_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Merge original and CRF results by epoch
            original_sorted = original_df.sort_values("epoch")
            crf_sorted = crf_df.copy()
            # Ensure epoch column exists
            if "epoch" not in crf_sorted.columns:
                crf_sorted["epoch"] = crf_sorted["model"].str.extract(r"epoch_(\d+)")[0].astype(int)
            crf_sorted = crf_sorted.sort_values("epoch")
            
            # Plot main metrics comparison
            ax.plot(original_sorted["epoch"], original_sorted["mean_iou"],
                   marker="o", linestyle="-", linewidth=2.5, markersize=8,
                   label="Original (mIoU)", color="#1f77b4")
            ax.plot(crf_sorted["epoch"], crf_sorted["mean_iou"],
                   marker="s", linestyle="--", linewidth=2.5, markersize=8,
                   label="With CRF (mIoU)", color="#ff7f0e")
            
            ax2 = ax.twinx()
            ax2.plot(original_sorted["epoch"], original_sorted["mean_dice"],
                    marker="^", linestyle="-", linewidth=2.5, markersize=8,
                    label="Original (Dice)", color="#2ca02c")
            ax2.plot(crf_sorted["epoch"], crf_sorted["mean_dice"],
                    marker="v", linestyle="--", linewidth=2.5, markersize=8,
                    label="With CRF (Dice)", color="#d62728")
            
            ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean IoU", fontsize=12, fontweight="bold", color="#1f77b4")
            ax2.set_ylabel("Mean Dice", fontsize=12, fontweight="bold", color="#2ca02c")
            ax.tick_params(axis="y", labelcolor="#1f77b4")
            ax2.tick_params(axis="y", labelcolor="#2ca02c")
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)
            
            ax.set_title(f"CRF Post-processing Comparison: {base_model}", 
                        fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            comparison_path = output_dir / f"{base_model}_crf_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
            plt.close()
            LOGGER.info("Saved CRF comparison plot to %s", comparison_path)


def evaluate_epoch_checkpoints(
    config: Dict[str, Any],
    output_dir: Path,
    device_override: str | None = None,
    save_predictions: bool = False,
    compare_crf: bool = False,
    crf_params: Optional[Dict[str, Any]] = None,
    generate_plots: bool = True,
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
            
            # Evaluate original model
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
                    "has_crf": False,
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
            
            # Evaluate with CRF post-processing if requested
            if compare_crf:
                try:
                    LOGGER.info("Evaluating %s at epoch %d with CRF post-processing...", name, epoch)
                    
                    # Build CRF-wrapped segmenter
                    crf_wrapper_params = {
                        "base_builder": builder,
                        "base_params": params.copy(),
                        "num_classes": num_classes,
                        "crf_params": crf_params or {},
                        "use_trained_base": False,  # We'll load checkpoint manually
                        "enable_crf": True,
                    }
                    if device_override is not None:
                        crf_wrapper_params["base_params"]["device"] = device_override
                    
                    crf_segmenter = build_segmenter("crf_wrapper", **crf_wrapper_params)
                    
                    # Load the same checkpoint into the base model of CRF wrapper
                    crf_model_to_load = None
                    if hasattr(crf_segmenter, "base") and hasattr(crf_segmenter.base, "model"):
                        crf_model_to_load = crf_segmenter.base.model
                    
                    crf_checkpoint = load_epoch_checkpoint(
                        builder,
                        model_config,
                        epoch,
                        model=crf_model_to_load,
                        device=device,
                    )
                    
                    if crf_checkpoint is None:
                        LOGGER.warning("Could not load checkpoint for CRF version of %s at epoch %d", name, epoch)
                    else:
                        # Mark as checkpoint loaded
                        if hasattr(crf_segmenter.base, "_checkpoint_loaded"):
                            crf_segmenter.base._checkpoint_loaded = True
                        
                        # Evaluate CRF version
                        crf_model_name = f"{name}_epoch_{epoch}_crf"
                        crf_report = evaluator.evaluate(
                            segmenter=crf_segmenter,
                            test_loader=dataloaders["test_loader"],
                            train_dataset=None,
                            val_dataset=None,
                            model_name=crf_model_name,
                        )
                        
                        crf_summary_row = {
                            "model": crf_model_name,
                            "base_model": name,
                            "epoch": epoch,
                            "has_crf": True,
                            **crf_report.summary
                        }
                        all_summaries.append(crf_summary_row)
                        LOGGER.info(
                            "Finished %s with CRF (epoch %d) — mIoU: %.4f | Dice: %.4f",
                            name, epoch, crf_report.summary["mean_iou"], crf_report.summary["mean_dice"]
                        )
                except Exception as e:
                    LOGGER.error("Error evaluating CRF version of %s at epoch %d: %s", name, epoch, e, exc_info=True)
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
        
        # Generate visualization plots
        if generate_plots:
            try:
                create_visualizations(all_summaries, output_dir)
            except Exception as e:
                LOGGER.warning("Failed to generate plots: %s", e, exc_info=True)
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
        compare_crf=args.compare_crf,
        crf_params=args.crf_params,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()

