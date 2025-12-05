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
    epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Path:
    """Save a model checkpoint with configuration metadata and training state.
    
    Args:
        model: The PyTorch model to save
        builder: The model builder name
        config: The model configuration dictionary
        metadata: Optional additional metadata to save (e.g., training metrics)
        epoch: Optional epoch number to include in filename and metadata
        optimizer: Optional optimizer state to save (for resume training)
        scheduler: Optional scheduler state to save (for resume training)
        scaler: Optional gradient scaler state to save (for resume training with AMP)
        
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_path = get_checkpoint_path(builder, config, create_dir=True, epoch=epoch)
    
    metadata = metadata or {}
    if epoch is not None:
        metadata["epoch"] = epoch
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "builder": builder,
        "config": config,
        "metadata": metadata,
    }
    
    # Save training state for resume
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_epoch_checkpoint(
    model: torch.nn.Module,
    builder: str,
    config: Dict[str, Any],
    epoch: int,
    metadata: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Path:
    """Save a model checkpoint for a specific epoch.
    
    This is a convenience wrapper around save_checkpoint that requires epoch
    as a positional parameter for clarity when saving epoch-specific checkpoints.
    
    Args:
        model: The PyTorch model to save
        builder: The model builder name
        config: The model configuration dictionary
        epoch: The epoch number (required)
        metadata: Optional additional metadata to save (e.g., training metrics)
        optimizer: Optional optimizer state to save (for resume training)
        scheduler: Optional scheduler state to save (for resume training)
        scaler: Optional gradient scaler state to save (for resume training with AMP)
        
    Returns:
        Path to the saved checkpoint file
    """
    return save_checkpoint(
        model=model,
        builder=builder,
        config=config,
        metadata=metadata,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )


def load_checkpoint(
    builder: str,
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint matching the given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        model: Optional model to load weights into. If None, only returns the checkpoint dict.
        device: Device to load the checkpoint to (default: CPU)
        optimizer: Optional optimizer to load state into (for resume training)
        scheduler: Optional scheduler to load state into (for resume training)
        scaler: Optional gradient scaler to load state into (for resume training with AMP)
        
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
    
    # Load training state for resume
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
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


def list_epoch_checkpoints(
    builder: str,
    config: Dict[str, Any],
) -> list[tuple[int, Path]]:
    """List all epoch checkpoints for a given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        
    Returns:
        List of (epoch, path) tuples, sorted by epoch
    """
    checkpoint_dir = get_checkpoint_dir()
    if not checkpoint_dir.exists():
        return []
    
    config_hash = _compute_config_hash(config)
    pattern = f"{builder}_{config_hash}_epoch_*.pth"
    
    epoch_checkpoints = []
    for checkpoint_path in checkpoint_dir.glob(pattern):
        # Extract epoch number from filename: {builder}_{hash}_epoch_{epoch}.pth
        try:
            epoch_str = checkpoint_path.stem.split("_epoch_")[-1]
            epoch = int(epoch_str)
            epoch_checkpoints.append((epoch, checkpoint_path))
        except (ValueError, IndexError):
            continue
    
    return sorted(epoch_checkpoints, key=lambda x: x[0])


def load_epoch_checkpoint(
    builder: str,
    config: Dict[str, Any],
    epoch: int,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint for a specific epoch.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        epoch: The epoch number to load
        model: Optional model to load weights into
        device: Device to load the checkpoint to (default: CPU)
        optimizer: Optional optimizer to load state into (for resume training)
        scheduler: Optional scheduler to load state into (for resume training)
        scaler: Optional gradient scaler to load state into (for resume training with AMP)
        
    Returns:
        The checkpoint dictionary if found, None otherwise
    """
    checkpoint_path = get_checkpoint_path(builder, config, create_dir=False, epoch=epoch)
    
    if not checkpoint_path.exists():
        return None
    
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Load training state for resume
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint


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


def find_latest_epoch_checkpoint(
    builder: str,
    config: Dict[str, Any],
) -> Optional[tuple[int, Path]]:
    """Find the latest epoch checkpoint for a given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        
    Returns:
        Tuple of (epoch, path) for the latest checkpoint if found, None otherwise
    """
    epoch_checkpoints = list_epoch_checkpoints(builder, config)
    if epoch_checkpoints:
        return epoch_checkpoints[-1]  # Return the latest (highest epoch)
    return None


def load_latest_epoch_checkpoint(
    builder: str,
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Optional[tuple[int, Dict[str, Any]]]:
    """Load the latest epoch checkpoint for a given builder and configuration.
    
    Args:
        builder: The model builder name
        config: The model configuration dictionary
        model: Optional model to load weights into
        device: Device to load the checkpoint to (default: CPU)
        optimizer: Optional optimizer to load state into (for resume training)
        scheduler: Optional scheduler to load state into (for resume training)
        scaler: Optional gradient scaler to load state into (for resume training with AMP)
        
    Returns:
        Tuple of (epoch, checkpoint_dict) if found, None otherwise
    """
    latest = find_latest_epoch_checkpoint(builder, config)
    if latest is None:
        return None
    
    epoch, checkpoint_path = latest
    
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # Load training state for resume
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return (epoch, checkpoint)

