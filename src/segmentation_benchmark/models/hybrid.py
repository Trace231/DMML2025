"""Hybrid CNN-Transformer segmenter."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from ..evaluation.registry import register_segmenter
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .base import BaseSegmenter


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).permute(0, 2, 1)  # B, N, C
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        x = tokens.permute(0, 2, 1).view(b, c, h, w)
        return x


class HybridUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.attn = AttentionBlock(base_channels * 8, num_heads=8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))
        b = self.attn(b)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return logits


@register_segmenter("hybrid_unet_transformer")
class HybridUNetTransformerSegmenter(BaseSegmenter):
    def __init__(
        self,
        num_classes: int = 2,
        base_channels: int = 32,
        finetune_epochs: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        num_workers: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(num_classes=num_classes, device=device, name="HybridUNetTransformer")
        # Ensure numeric parameters are correct types (YAML may parse as strings)
        base_channels = int(base_channels)
        self.model = HybridUNet(in_channels=3, num_classes=num_classes, base_channels=base_channels).to(self.device)
        self.finetune_epochs = int(finetune_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.base_channels = base_channels
        
        # Try to load checkpoint if available (only if finetune_epochs > 0, meaning we expect trained weights)
        if finetune_epochs > 0:
            config = self._get_config()
            checkpoint = load_checkpoint("hybrid_unet_transformer", config, model=self.model, device=self.device)
            if checkpoint is not None:
                print(f"[INFO] {self.name}: Loaded checkpoint from previous training")
                self._checkpoint_loaded = True
            else:
                self._checkpoint_loaded = False
        else:
            self._checkpoint_loaded = False

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for checkpoint matching."""
        return {
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "finetune_epochs": self.finetune_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def _compute_class_weights(self, loader: DataLoader) -> Optional[torch.Tensor]:
        """Compute simple inverse-frequency class weights from one pass of the loader.

        This is mainly useful when foreground pixels are very sparse (e.g. CrackForest).
        For VOC-style multi-class segmentation the imbalance is smaller, but this still
        gives a small boost by down-weighting dominant background.
        """
        # Only attempt if num_classes is reasonably small
        if self.num_classes <= 1 or self.num_classes > 256:
            return None

        counts = torch.zeros(self.num_classes, dtype=torch.long)
        with torch.no_grad():
            for batch in loader:
                masks = batch["mask"].squeeze(1).long()
                # Ignore void label 255 if present
                valid = (masks >= 0) & (masks < self.num_classes)
                if not valid.any():
                    continue
                flat = masks[valid].view(-1)
                counts.scatter_add_(0, flat.cpu(), torch.ones_like(flat.cpu(), dtype=torch.long))

        if counts.sum() == 0:
            return None

        freqs = counts.float() / counts.sum().float()
        # Inverse frequency; add epsilon to avoid division by zero
        inv_freq = 1.0 / (freqs + 1e-6)
        weights = inv_freq / inv_freq.sum()
        return weights.to(self.device)

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss for multi-class segmentation.

        - logits: (N, C, H, W)
        - targets: (N, H, W) with values in [0, C-1] or 255 for void.
        """
        num_classes = logits.shape[1]
        # Create one-hot targets, ignore void=255
        with torch.no_grad():
            valid_mask = targets != 255
            valid_targets = targets.clone()
            valid_targets[~valid_mask] = 0  # temporary value, will be masked out
            one_hot = F.one_hot(valid_targets.clamp(min=0, max=num_classes - 1), num_classes=num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)
            one_hot = one_hot * valid_mask.unsqueeze(1)  # zero-out void positions

        probs = torch.softmax(logits, dim=1)
        probs = probs * valid_mask.unsqueeze(1)  # ignore void

        dims = (0, 2, 3)
        numerator = 2.0 * (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims) + 1e-6
        dice_per_class = 1.0 - numerator / denominator

        # Average over classes
        return dice_per_class.mean()

    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        # Skip training if checkpoint was already loaded
        if self._checkpoint_loaded:
            print(f"[INFO] {self.name}: Skipping training (using loaded checkpoint)")
            self.model.eval()
            return
        if self.finetune_epochs <= 0 or train_dataset is None:
            return

        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # Compute simple class-balanced weights to mitigate foreground/background imbalance.
        class_weights = self._compute_class_weights(loader)
        if class_weights is not None:
            print(f"[INFO] {self.name}: Using class-balanced CE weights: {class_weights.detach().cpu().numpy()}")

        # VOC-style masks use 0–20 as valid class indices and 255 as "void/ignore".
        # Use ignore_index=255 to avoid invalid-label errors when training on VOC.
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Cosine LR schedule from initial LR to 10% of initial LR over finetune_epochs.
        scheduler = CosineAnnealingLR(optimizer, T_max=self.finetune_epochs, eta_min=self.learning_rate * 0.1)

        # Enable mixed precision training when possible (mirrors CNN segmenter behaviour).
        scaler = GradScaler('cuda')
        use_amp = torch.cuda.is_available() and hasattr(torch.amp, 'autocast')
        if use_amp:
            print(f"[INFO] {self.name}: Mixed precision training (AMP) enabled")

        self.model.train()
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].squeeze(1).long().to(self.device)
                optimizer.zero_grad()

                if use_amp:
                    with autocast('cuda'):
                        logits = self.model(images)
                        ce_loss = ce_criterion(logits, masks)
                        dice_loss = self._dice_loss(logits, masks)
                        loss = ce_loss + dice_loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(images)
                    ce_loss = ce_criterion(logits, masks)
                    dice_loss = self._dice_loss(logits, masks)
                    loss = ce_loss + dice_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            scheduler.step()

            avg_loss = epoch_loss / max(batch_count, 1)
            print(f"[TRAIN] {self.name} epoch {epoch+1}/{self.finetune_epochs} - avg loss: {avg_loss:.4f}")

        self.model.eval()
        
        # Save checkpoint after training
        config = self._get_config()
        checkpoint_path = save_checkpoint(
            self.model,
            "hybrid_unet_transformer",
            config,
            metadata={"final_loss": avg_loss, "epochs": self.finetune_epochs}
        )
        print(f"[INFO] {self.name}: Saved checkpoint to {checkpoint_path}")

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            logits = self.model(images)
            return logits.detach().cpu().numpy()

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        logits = self.predict_logits(batch)
        preds = np.argmax(logits, axis=1).astype(np.int64)
        return preds
