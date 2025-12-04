"""Helpers for managing pretrained model weights in a central folder.

This module provides:
- A fixed project-level directory for pretrained weights.
- Utilities to load torchvision segmentation models using that directory,
  preferring local files and only downloading when missing.
- Helpers to download all known pretrained weights ahead of time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torchvision.models import segmentation as tv_seg
from torchvision.models._api import WeightsEnum

from .paths import ProjectPaths


LOGGER = logging.getLogger(__name__)


def get_pretrained_root() -> Path:
    """Return the root directory for all pretrained weights and ensure it exists.

    The layout is:
        artifacts/pretrained/
            torchvision/
            huggingface/
    under the detected project root.
    """
    paths = ProjectPaths.from_root()
    root = paths.artifacts_dir / "pretrained"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Torchvision segmentation helpers
# ---------------------------------------------------------------------------

def _get_segmentation_weights_enum(model_name: str) -> Optional[type[WeightsEnum]]:
    """Return the WeightsEnum class for a torchvision segmentation model, if any."""
    enum_name = f"{model_name.upper()}_Weights"
    return getattr(tv_seg, enum_name, None)


def get_torchvision_weights_file(weights: WeightsEnum) -> Path:
    """Return the expected local cache file path for a torchvision Weights entry."""
    root = get_pretrained_root() / "torchvision"
    root.mkdir(parents=True, exist_ok=True)
    # Use the file name from the official URL so that manual copies also work.
    filename = os.path.basename(weights.url)
    return root / filename


def ensure_torchvision_weights_downloaded(weights: WeightsEnum, *, force: bool = False) -> Path:
    """Ensure the given torchvision weights are downloaded into our cache folder.

    This does NOT change PyTorch's global cache; it only manages a project-local copy.
    """
    import urllib.request

    dst = get_torchvision_weights_file(weights)
    if dst.exists() and not force:
        LOGGER.info("Using cached torchvision weights at %s", dst)
        return dst

    url = weights.url
    LOGGER.info("Downloading torchvision weights from %s to %s", url, dst)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst)
    except Exception as e:  # pragma: no cover - network dependent
        LOGGER.error("Failed to download torchvision weights from %s: %s", url, e)
        raise
    return dst


def load_torchvision_weights_state_dict(model_name: str, *, pretrained: bool) -> Optional[Dict[str, torch.Tensor]]:
    """Load a state dict for a torchvision segmentation model from the local cache.

    If ``pretrained`` is False or no official weights exist, returns None.
    If the file is missing it will be downloaded automatically.
    """
    if not pretrained:
        return None

    enum_cls = _get_segmentation_weights_enum(model_name)
    if enum_cls is None:
        LOGGER.warning("No official weights enum found for '%s'; using random init", model_name)
        return None

    weights = enum_cls.DEFAULT
    path = get_torchvision_weights_file(weights)
    if not path.exists():
        path = ensure_torchvision_weights_downloaded(weights)

    LOGGER.info("Loading pretrained weights for '%s' from %s", model_name, path)
    state_dict = torch.load(path, map_location="cpu")
    return state_dict


def build_torchvision_segmentation_model(
    model_name: str,
    *,
    pretrained: bool = True,
) -> torch.nn.Module:
    """Construct a torchvision segmentation model using the managed pretrained cache.

    - If ``pretrained=False``: returns the model with random initialization.
    - If ``pretrained=True`` and weights are available:
        * checks the project cache folder;
        * downloads the weights if necessary;
        * loads the state dict manually into a model constructed with ``weights=None``.

    Important: We also pass ``weights_backbone=None`` so that torchvision does NOT
    try to download backbone weights from the internet (which would fail on
    offline clusters).
    """
    if not hasattr(tv_seg, model_name):
        raise ValueError(f"Unknown torchvision segmentation model '{model_name}'")

    state_dict = load_torchvision_weights_state_dict(model_name, pretrained=pretrained)
    if state_dict is None:
        # Fallback: random initialization
        LOGGER.info("Building '%s' with random initialisation (no pretrained weights)", model_name)
        # Disable torchvision's own backbone-pretrained download path.
        try:
            return getattr(tv_seg, model_name)(weights=None, weights_backbone=None)
        except TypeError:
            # Older torchvision versions may not have weights_backbone; fall back gracefully.
            return getattr(tv_seg, model_name)(weights=None)

    LOGGER.info("Building '%s' with pretrained weights from local cache", model_name)
    model_fn = getattr(tv_seg, model_name)
    try:
        # Newer torchvision: disable backbone pretraining as well.
        model = model_fn(weights=None, weights_backbone=None)
    except TypeError:
        # Older torchvision: only weights argument exists.
        model = model_fn(weights=None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning("While loading '%s' weights: missing=%s, unexpected=%s", model_name, missing, unexpected)
    return model


def iter_all_torchvision_segmentation_weights() -> Iterable[WeightsEnum]:
    """Iterate over DEFAULT weights for all known torchvision segmentation models.

    This is used by the download script to fetch everything in one go.
    """
    for attr in dir(tv_seg):
        obj = getattr(tv_seg, attr)
        if isinstance(obj, type) and issubclass(obj, WeightsEnum):
            try:
                default = obj.DEFAULT
            except Exception:
                continue
            # Heuristic: only segmentation backbones live in tv_seg.*_Weights
            yield default


def download_all_torchvision_segmentation_weights(*, force: bool = False) -> List[Path]:
    """Download all known torchvision segmentation weights into the local cache."""
    paths: List[Path] = []
    for weights in iter_all_torchvision_segmentation_weights():
        try:
            path = ensure_torchvision_weights_downloaded(weights, force=force)
            paths.append(path)
        except Exception:
            # Keep going even if one model fails, but log a warning.
            LOGGER.warning("Skipping weights %s due to download error", weights)
    return paths


# ---------------------------------------------------------------------------
# HuggingFace / SegFormer helpers
# ---------------------------------------------------------------------------

def get_huggingface_cache_dir() -> Path:
    """Return the cache dir to use for HuggingFace models."""
    root = get_pretrained_root() / "huggingface"
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_segformer_model(model_name: str, *, force: bool = False) -> None:
    """Download a SegFormer backbone and its processor into our cache.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. 'nvidia/segformer-b0-finetuned-ade-512-512'.
    force:
        If True, forces re-download even if files already exist.
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    cache_dir = get_huggingface_cache_dir()
    LOGGER.info("Ensuring SegFormer weights for '%s' in %s", model_name, cache_dir)

    # The transformers library manages individual files inside cache_dir.
    # We rely on its own caching; 'force' is implemented by passing force_download.
    SegformerImageProcessor.from_pretrained(model_name, cache_dir=cache_dir, force_download=force)
    SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir, force_download=force)



