"""Hybrid CNN-Transformer segmenter."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..evaluation.registry import register_segmenter
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
        self.model = HybridUNet(in_channels=3, num_classes=num_classes, base_channels=base_channels).to(self.device)
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
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.finetune_epochs):
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].squeeze(1).long().to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
        self.model.eval()

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
