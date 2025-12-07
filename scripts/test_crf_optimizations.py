"""Test CRF optimizations using existing checkpoints.

This script allows testing new CRF optimizations (adaptive CRF, boundary-aware CRF, etc.)
without retraining models. It loads existing checkpoints and applies different CRF configurations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from segmentation_benchmark.data import create_dataloaders
from segmentation_benchmark.evaluation import BenchmarkEvaluator, build_segmenter
from segmentation_benchmark.utils import load_config
from segmentation_benchmark.utils.checkpoint import list_epoch_checkpoints, load_epoch_checkpoint
from segmentation_benchmark.utils.paths import ProjectPaths


LOGGER = logging.getLogger("segmentation_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CRF optimizations with existing checkpoints")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "voc_benchmark.yaml",
        help="Path to the benchmark configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for results (default: reports/{run_name}/crf_optimizations)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to test (default: latest)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("Pillow").setLevel(logging.WARNING)


def test_crf_configurations(
    config: Dict[str, Any],
    output_dir: Path,
    device_override: str | None = None,
    target_epoch: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Test different CRF configurations using existing checkpoints."""
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
        save_predictions=False,
    )

    models_cfg = config.get("models", [])
    all_results: List[Dict[str, Any]] = []
    
    # Define CRF configurations to test
    crf_configs = [
        {
            "name": "baseline",
            "description": "Standard CRF (baseline)",
            "params": {
                "use_adaptive_crf": False,
                "crf_params": {
                    "iterations": 5,
                    "gaussian_sxy": 3,
                    "bilateral_sxy": 80,
                    "bilateral_srgb": 13,
                    "compat_gaussian": 3,
                    "compat_bilateral": 10,
                },
            },
        },
        {
            "name": "adaptive_crf",
            "description": "Adaptive CRF (auto-adjusts parameters)",
            "params": {
                "use_adaptive_crf": True,
                "adaptive_crf_config": {
                    "scale_by_image_size": True,
                    "scale_by_entropy": True,
                    "scale_by_contrast": False,
                },
            },
        },
        {
            "name": "adaptive_crf_full",
            "description": "Adaptive CRF (all features enabled)",
            "params": {
                "use_adaptive_crf": True,
                "adaptive_crf_config": {
                    "scale_by_image_size": True,
                    "scale_by_entropy": True,
                    "scale_by_contrast": True,
                },
            },
        },
        {
            "name": "crf_more_iterations",
            "description": "Standard CRF with more iterations",
            "params": {
                "use_adaptive_crf": False,
                "crf_params": {
                    "iterations": 10,
                    "gaussian_sxy": 3,
                    "bilateral_sxy": 80,
                    "bilateral_srgb": 13,
                    "compat_gaussian": 3,
                    "compat_bilateral": 10,
                },
            },
        },
    ]
    
    LOGGER.info("Testing %d CRF configurations on %d model(s)...", len(crf_configs), len(models_cfg))

    for model_cfg in models_cfg:
        name = model_cfg.get("name", "unknown")
        builder = model_cfg.get("builder")
        params = model_cfg.get("params", {})
        
        # Only test models that have a base model (for loading checkpoints)
        # Skip CRF models themselves
        if builder in ["crf_wrapper", "cnn_crf"]:
            LOGGER.info("Skipping %s (already a CRF model)", name)
            continue
        
        # Skip models that don't support cnn_crf (cnn_crf only supports TorchvisionSegmenter models)
        # SegFormer and other transformer models are not supported by cnn_crf
        if builder not in ["fcn_resnet50", "deeplabv3_resnet50"]:
            LOGGER.info("Skipping %s (cnn_crf only supports fcn_resnet50 and deeplabv3_resnet50)", name)
            continue
        
        # Find available checkpoints
        # Build checkpoint config matching the format saved in checkpoints
        # Checkpoints use "model_name" instead of "builder", and include weight_decay
        # Different models have different config formats
        if builder == "segformer_b0":
            # SegFormer uses HuggingFace model name and doesn't have pretrained param
            checkpoint_config = {
                "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
                "num_classes": num_classes,
                "finetune_epochs": params.get("finetune_epochs", 80),  # Will be ignored in matching
                "learning_rate": params.get("learning_rate", 5e-5),
                "weight_decay": params.get("weight_decay", 1e-4),
                "batch_size": params.get("batch_size", 12),
                "num_workers": params.get("num_workers", 14),
            }
        else:
            # CNN models (fcn_resnet50, deeplabv3_resnet50) use builder name as model_name
            checkpoint_config = {
                "model_name": builder,  # Checkpoints use "model_name" not "builder"
                "num_classes": num_classes,
                "pretrained": params.get("pretrained", True),
                "finetune_epochs": params.get("finetune_epochs", 80),  # Will be ignored in matching
                "learning_rate": params.get("learning_rate", 0.001),
                "weight_decay": params.get("weight_decay", 1e-4),  # Default weight_decay
                "batch_size": params.get("batch_size", 12),
                "num_workers": params.get("num_workers", 14),
            }
        
        # Remove device from config for checkpoint matching (if present)
        checkpoint_config.pop("device", None)
        
        # Use ignore_keys to ignore finetune_epochs when matching (since we're loading specific epoch)
        available_checkpoints = list_epoch_checkpoints(
            builder, 
            checkpoint_config,
            ignore_keys=["finetune_epochs", "resume_epoch"]
        )
        if not available_checkpoints:
            LOGGER.warning("No checkpoints found for %s (builder: %s)", name, builder)
            continue
        
        # Select epoch
        if target_epoch is not None:
            selected_checkpoints = [(e, p) for e, p in available_checkpoints if e == target_epoch]
            if not selected_checkpoints:
                LOGGER.warning("Epoch %d not found for %s, using latest", target_epoch, name)
                selected_checkpoints = [available_checkpoints[-1]]
        else:
            # Use latest checkpoint
            selected_checkpoints = [available_checkpoints[-1]]
        
        epoch, checkpoint_path = selected_checkpoints[0]
        LOGGER.info("Testing %s at epoch %d with %d CRF configurations...", name, epoch, len(crf_configs))
        
        # Test each CRF configuration
        for crf_cfg in crf_configs:
            try:
                LOGGER.info("  Testing %s: %s", crf_cfg["name"], crf_cfg["description"])
                
                # Build CNN-CRF segmenter with this configuration
                # Map builder names to base_model names for cnn_crf
                base_model_name = builder
                if builder == "fcn_resnet50":
                    base_model_name = "fcn_resnet50"
                elif builder == "deeplabv3_resnet50":
                    base_model_name = "deeplabv3_resnet50"
                elif builder == "segformer_b0":
                    base_model_name = "segformer_b0"
                
                cnn_crf_config = {
                    "base_model": base_model_name,
                    "pretrained": params.get("pretrained", True),
                    "finetune_epochs": 0,  # Don't train, use checkpoint
                    "device": device_override or params.get("device"),
                    **crf_cfg["params"],
                }
                
                segmenter = build_segmenter("cnn_crf", num_classes=num_classes, config=cnn_crf_config)
                
                # Load checkpoint from base model (not cnn_crf)
                # The checkpoint was saved with the base model's builder and config
                device = torch.device(device_override or str(segmenter.device))
                model_to_load = segmenter.base_segmenter.model
                
                checkpoint = load_epoch_checkpoint(
                    builder,  # Use base model builder (e.g., "fcn_resnet50")
                    checkpoint_config,  # Use base model config
                    epoch,
                    model=model_to_load,
                    device=device,
                    ignore_keys=["finetune_epochs", "resume_epoch"],  # Ignore these for matching
                )
                
                if checkpoint is None:
                    LOGGER.warning("Could not load checkpoint for %s at epoch %d", name, epoch)
                    continue
                
                # Mark as checkpoint loaded
                segmenter._checkpoint_loaded = True
                segmenter.base_segmenter._checkpoint_loaded = True
                
                # Evaluate
                model_name = f"{name}_epoch_{epoch}_{crf_cfg['name']}"
                eval_loader = dataloaders.get("val_loader") or dataloaders.get("test_loader")
                if eval_loader is None:
                    LOGGER.warning("No validation or test loader available for %s", name)
                    continue
                
                report = evaluator.evaluate(
                    segmenter=segmenter,
                    test_loader=eval_loader,
                    train_dataset=None,
                    val_dataset=None,
                    model_name=model_name,
                )
                
                result = {
                    "base_model": name,
                    "epoch": epoch,
                    "crf_config": crf_cfg["name"],
                    "crf_description": crf_cfg["description"],
                    **report.summary
                }
                all_results.append(result)
                
                LOGGER.info(
                    "    %s — mIoU: %.4f | Dice: %.4f",
                    crf_cfg["name"], report.summary["mean_iou"], report.summary["mean_dice"]
                )
            except Exception as e:
                LOGGER.error("Error testing %s with %s: %s", name, crf_cfg["name"], e, exc_info=True)
                continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = output_dir / "crf_optimizations_results.csv"
        df.to_csv(csv_path, index=False)
        LOGGER.info("Saved results to %s", csv_path)
        
        json_path = output_dir / "crf_optimizations_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        LOGGER.info("Saved results (JSON) to %s", json_path)
        
        # Create comparison table
        if len(df) > 0:
            comparison = df.groupby("base_model").agg({
                "mean_iou": ["max", "mean", "std"],
                "mean_dice": ["max", "mean", "std"],
            }).round(4)
            comparison_path = output_dir / "crf_optimizations_comparison.csv"
            comparison.to_csv(comparison_path)
            LOGGER.info("Saved comparison to %s", comparison_path)
            
            # Print summary
            print("\n" + "="*80)
            print("CRF Optimization Test Results Summary")
            print("="*80)
            for base_model in df["base_model"].unique():
                model_df = df[df["base_model"] == base_model]
                print(f"\n{base_model}:")
                print(f"  Best mIoU: {model_df['mean_iou'].max():.4f} ({model_df.loc[model_df['mean_iou'].idxmax(), 'crf_config']})")
                print(f"  Best Dice: {model_df['mean_dice'].max():.4f} ({model_df.loc[model_df['mean_dice'].idxmax(), 'crf_config']})")
                print(f"  Configurations tested: {len(model_df)}")
    else:
        LOGGER.warning("No results to save")
    
    return all_results


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    config = load_config(args.config)
    
    if args.output is None:
        run_name = config.get("evaluation", {}).get("run_name", "benchmark")
        output_dir = ProjectPaths.from_root().reports_dir / run_name / "crf_optimizations"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_crf_configurations(
        config,
        output_dir,
        device_override=args.device,
        target_epoch=args.epoch,
    )


if __name__ == "__main__":
    main()

