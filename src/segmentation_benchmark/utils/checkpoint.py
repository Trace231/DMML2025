"""Checkpoint management for trained models.

This module provides functionality to save and load trained model checkpoints
with automatic configuration matching. Checkpoints are saved based on a hash
of the model configuration, allowing automatic loading when the same configuration
is used again.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .paths import ProjectPaths


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a hash of the configuration dictionary.
    
    The hash is computed from a sorted JSON representation to ensure
    consistent hashing regardless of key order.
    """
    # Sort keys and convert to JSON string for consistent hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_checkpoint_dir() -> Path:
    """Get the directory for storing checkpoints."""
    paths = ProjectPaths.from_root()
    checkpoint_dir = paths.artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(
    builder: str,
    config: Dict[str, Any],
    create_dir: bool = True,
    epoch: Optional[int] = None,
) -> Path:
    """Get the checkpoint path for a given builder and configuration.
    
    Args:
        builder: The model builder name (e.g., 'fcn_resnet50')
        config: The model configuration dictionary
        create_dir: Whether to create the checkpoint directory if it doesn't exist
        epoch: Optional epoch number to include in filename (e.g., epoch_10, epoch_20)
        
    Returns:
        Path to the checkpoint file
    """
    checkpoint_dir = get_checkpoint_dir()
    config_hash = _compute_config_hash(config)
    if epoch is not None:
        checkpoint_path = checkpoint_dir / f"{builder}_{config_hash}_epoch_{epoch}.pth"
    else:
        checkpoint_path = checkpoint_dir / f"{builder}_{config_hash}.pth"
    
    if create_dir:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_path


def save_checkpoint(
    model: torch.nn.Module,
    builder: str,
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a model checkpoint with configuration metadata.
    
    Args:
        model: The PyTorch model to save
        builder: The model builder name
        config: The model configuration dictionary
        metadata: Optional additional metadata to save (e.g., training metrics)
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_path = get_checkpoint_path(builder, config, create_dir=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "builder": builder,
        "config": config,
        "metadata": metadata or {},
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    builder: str,
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint matching the given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        model: Optional model to load weights into. If None, only returns the checkpoint dict.
        device: Device to load the checkpoint to (default: CPU)
        
    Returns:
        The checkpoint dictionary if found, None otherwise
    """
    checkpoint_path = get_checkpoint_path(builder, config, create_dir=False)
    
    if not checkpoint_path.exists():
        return None
    
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    return checkpoint


def find_matching_checkpoint(
    builder: str,
    config: Dict[str, Any],
    ignore_keys: Optional[list[str]] = None,
) -> Optional[Path]:
    """Find a checkpoint matching the given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        ignore_keys: Optional list of keys to ignore when matching (e.g., ['finetune_epochs'])
        
    Returns:
        Path to the matching checkpoint if found, None otherwise
    """
    # First try exact match
    checkpoint_path = get_checkpoint_path(builder, config, create_dir=False)
    if checkpoint_path.exists():
        return checkpoint_path
    
    # If ignore_keys provided, try to find checkpoint ignoring those keys
    if ignore_keys:
        filtered_config = {k: v for k, v in config.items() if k not in ignore_keys}
        # List all checkpoints for this builder and check if any match
        checkpoint_dir = get_checkpoint_dir()
        if checkpoint_dir.exists():
            for checkpoint_path in checkpoint_dir.glob(f"{builder}_*.pth"):
                try:
                    checkpoint_config = get_config_from_checkpoint(checkpoint_path)
                    if checkpoint_config:
                        filtered_checkpoint_config = {k: v for k, v in checkpoint_config.items() if k not in ignore_keys}
                        if filtered_config == filtered_checkpoint_config:
                            return checkpoint_path
                except Exception:
                    continue
    
    return None


def list_checkpoints(builder: Optional[str] = None) -> list[Path]:
    """List all available checkpoints.
    
    Args:
        builder: Optional builder name to filter by
        
    Returns:
        List of checkpoint paths
    """
    checkpoint_dir = get_checkpoint_dir()
    
    if not checkpoint_dir.exists():
        return []
    
    pattern = f"{builder}_*.pth" if builder else "*.pth"
    return sorted(checkpoint_dir.glob(pattern))


def get_config_from_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Extract configuration from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Configuration dictionary if checkpoint is valid, None otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint.get("config")
    except Exception:
        return None

