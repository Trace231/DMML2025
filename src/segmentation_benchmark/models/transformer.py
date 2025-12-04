"""Transformer-based segmenters (SegFormer)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from ..evaluation.registry import register_segmenter
from ..utils.pretrained import get_huggingface_cache_dir
from .base import BaseSegmenter


@register_segmenter("segformer")
class SegformerSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes: int = 2,
        finetune_epochs: int = 0,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 2,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        display_name = model_name.split("/")[-1]
        super().__init__(num_classes=num_classes, device=device, name=f"SegFormer-{display_name}")
        cache_dir = get_huggingface_cache_dir()
        # Use a shared project-local cache dir for all HuggingFace models, and
        # force offline usage at runtime (weights & configs must be pre-downloaded).
        self.processor = SegformerImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if self.model.config.num_labels != num_classes:
            in_channels = self.model.decode_head.classifier.in_channels
            self.model.decode_head.classifier = torch.nn.Conv2d(in_channels, num_classes, kernel_size=1)
            self.model.config.num_labels = num_classes
            self.model.config.id2label = {i: f"class_{i}" for i in range(num_classes)}
            self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}
        self.model.to(self.device)
        self.finetune_epochs = finetune_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        if self.finetune_epochs <= 0 or train_dataset is None:
            return
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model.train()
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in loader:
                images = self._prepare_images(batch["image"])
                masks = batch["mask"].squeeze(1).long()  # (N, H, W)
                
                # Process images - this may resize them
                encoded = self.processor(images=images, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Resize labels to match processed image size
                # The processor may resize images, so we need to resize labels accordingly
                processed_size = encoded["pixel_values"].shape[-2:]  # (H, W)
                original_size = masks.shape[-2:]  # (H, W)
                
                if processed_size != original_size:
                    # Resize labels using nearest neighbor to preserve class indices
                    masks_resized = F.interpolate(
                        masks.unsqueeze(1).float(),  # Add channel dim: (N, 1, H, W)
                        size=processed_size,
                        mode="nearest"
                    ).squeeze(1).long()  # Remove channel dim: (N, H, W)
                else:
                    masks_resized = masks
                
                labels = masks_resized.to(self.device)
                outputs = self.model(**encoded, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"[TRAIN] {self.name} epoch {epoch+1}/{self.finetune_epochs} - avg loss: {avg_loss:.4f}")
        self.model.eval()

    def _prepare_images(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """Convert normalized tensor back to RGB images in [0, 255] range.
        
        The input tensor is expected to be ImageNet-normalized (mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]). We denormalize it first, then convert to uint8.
        """
        tensor = tensor.detach().cpu()
        tensor = tensor.permute(0, 2, 3, 1)  # N H W C
        
        # Denormalize: reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_np = tensor.numpy()
        tensor_np = tensor_np * std + mean  # Denormalize
        tensor_np = np.clip(tensor_np, 0.0, 1.0)  # Clip to [0, 1]
        images = (tensor_np * 255.0).astype(np.uint8)
        return [img for img in images]

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            images = self._prepare_images(batch["image"])
            encoded = self.processor(images=images, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            logits = outputs.logits
            original_size = batch["image"].shape[-2:]
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
            return logits.detach().cpu().numpy()

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        logits = self.predict_logits(batch)
        preds = np.argmax(logits, axis=1).astype(np.int64)
        return preds


@register_segmenter("segformer_b0")
def build_segformer_b0(**kwargs: Any) -> SegformerSegmenter:
    kwargs.setdefault("model_name", "nvidia/segformer-b0-finetuned-ade-512-512")
    return SegformerSegmenter(**kwargs)
