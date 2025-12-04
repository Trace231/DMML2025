"""Diffusion-based segmenter using DDP (Diffusion Dense Prediction)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..evaluation.registry import register_segmenter
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .base import BaseSegmenter


def _get_num_groups(num_channels: int, preferred_groups: int = 8) -> int:
    """Get number of groups for GroupNorm that divides num_channels.
    
    Args:
        num_channels: Number of input channels
        preferred_groups: Preferred number of groups (default: 8)
    
    Returns:
        Number of groups that divides num_channels
    """
    # Try preferred number of groups first
    if num_channels % preferred_groups == 0:
        return preferred_groups
    
    # Try smaller divisors
    for groups in [4, 2, 1]:
        if num_channels % groups == 0:
            return groups
    
    # Fallback to 1 (LayerNorm-like behavior)
    return 1


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class UNetBlock(nn.Module):
    """Basic UNet block with time embedding."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(_get_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(_get_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class DiffusionUNet(nn.Module):
    """UNet for diffusion-based segmentation."""
    
    def __init__(self, image_channels: int = 3, mask_channels: int = 1, num_classes: int = 2, base_channels: int = 64) -> None:
        super().__init__()
        time_dim = base_channels * 4
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Image encoder (for conditioning)
        self.image_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down1 = UNetBlock(mask_channels + base_channels, base_channels, time_dim)
        self.down2 = UNetBlock(base_channels, base_channels * 2, time_dim)
        self.down3 = UNetBlock(base_channels * 2, base_channels * 4, time_dim)
        
        self.downsample = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_channels * 4, base_channels * 8, time_dim)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = UNetBlock(base_channels * 8 + base_channels * 4, base_channels * 4, time_dim)
        self.up2 = UNetBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up1 = UNetBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        
        # Output head - predict noise
        self.out = nn.Sequential(
            nn.GroupNorm(_get_num_groups(base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, mask_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy mask [B, 1, H, W]
            timestep: Time step [B]
            image: Conditioning image [B, 3, H, W]
        Returns:
            Predicted noise [B, 1, H, W]
        """
        t = self.time_mlp(timestep)
        
        # Encode image for conditioning
        img_emb = self.image_conv(image)
        
        # Concatenate noisy mask with image embedding
        x = torch.cat([x, img_emb], dim=1)
        
        # Downsampling
        d1 = self.down1(x, t)
        d2 = self.down2(self.downsample(d1), t)
        d3 = self.down3(self.downsample(d2), t)
        
        # Bottleneck
        b = self.bottleneck(self.downsample(d3), t)
        
        # Upsampling with skip connections
        u3 = self.up3(torch.cat([self.upsample(b), d3], dim=1), t)
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1), t)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), t)
        
        # Output
        return self.out(u1)


@register_segmenter("ddp")
class DDPSegmenter(BaseSegmenter):
    """DDP (Diffusion Dense Prediction) segmenter for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int = 2,
        finetune_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 4,
        num_workers: int = 0,
        device: Optional[str] = None,
        num_timesteps: int = 1000,
        inference_steps: int = 50,
        base_channels: int = 64,
    ) -> None:
        super().__init__(num_classes=num_classes, device=device, name="DDP")
        # Ensure numeric parameters are correct types (YAML may parse as strings)
        self.finetune_epochs = int(finetune_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.num_timesteps = int(num_timesteps)
        self.inference_steps = int(inference_steps)
        self.base_channels = int(base_channels)
        
        # Build model
        self.model = DiffusionUNet(
            image_channels=3,
            mask_channels=1,
            num_classes=num_classes,
            base_channels=base_channels
        )
        self.model.to(self.device)
        
        # Setup noise schedule (linear schedule)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Try to load checkpoint
        if finetune_epochs > 0:
            config = self._get_config()
            checkpoint = load_checkpoint("ddp", config, model=self.model, device=self.device)
            if checkpoint is not None:
                print(f"[INFO] {self.name}: Loaded checkpoint from previous training")
                self._checkpoint_loaded = True
            else:
                self._checkpoint_loaded = False
        else:
            self._checkpoint_loaded = False
        
        # Enable model compilation
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"[INFO] {self.name}: Model compilation enabled")
            except Exception as e:
                print(f"[WARNING] {self.name}: Model compilation failed ({e}), continuing without it")
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for checkpoint matching."""
        return {
            "num_classes": self.num_classes,
            "finetune_epochs": self.finetune_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_timesteps": self.num_timesteps,
            "inference_steps": self.inference_steps,
            "base_channels": self.base_channels,
        }
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from tensor ``a`` at indices ``t``, keeping everything on the same device."""
        # Ensure indices are integer type and on the same device as the source tensor
        if t.dtype != torch.long:
            t = t.long()
        if a.device != t.device:
            a = a.to(t.device)
        
        batch_size = t.shape[0]
        # ``a`` is a 1D tensor of length ``num_timesteps``; we gather along dim 0
        out = a.gather(0, t)
        # Reshape to broadcast correctly with ``x_shape``
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: add noise to masks."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _p_sample(self, x: torch.Tensor, t: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion: denoise one step."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Predict noise
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t, image) / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def _p_sample_loop(self, shape: tuple, image: torch.Tensor) -> torch.Tensor:
        """Full reverse diffusion process."""
        b = shape[0]
        # Start from random noise
        img = torch.randn(shape, device=self.device)
        
        # Create time steps for inference
        steps = torch.linspace(self.num_timesteps - 1, 0, self.inference_steps, device=self.device).long()
        
        for i in steps:
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self._p_sample(img, t, image)
        
        return img
    
    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None) -> None:
        """Train the diffusion model."""
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
        
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.finetune_epochs, eta_min=self.learning_rate * 0.1)
        
        # Mixed precision training
        scaler = GradScaler('cuda')
        use_amp = torch.cuda.is_available() and hasattr(torch.amp, 'autocast')
        if use_amp:
            print(f"[INFO] {self.name}: Mixed precision training (AMP) enabled")
        
        self.model.train()
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in loader:
                images = batch["image"].to(self.device)  # [B, 3, H, W]
                masks = batch["mask"].squeeze(1).long().to(self.device)  # [B, H, W]
                
                # Convert masks to one-hot for diffusion (we'll use class indices as continuous values)
                # For diffusion, we normalize masks to [-1, 1] range
                # Convert class indices to normalized values
                mask_normalized = (masks.float() / (self.num_classes - 1) * 2.0 - 1.0).unsqueeze(1)  # [B, 1, H, W]
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (images.shape[0],), device=self.device).long()
                
                # Sample noise
                noise = torch.randn_like(mask_normalized)
                
                # Add noise to masks
                noisy_masks = self._q_sample(mask_normalized, t, noise)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with autocast('cuda'):
                        # Predict noise
                        noise_pred = self.model(noisy_masks, t, images)
                        loss = F.mse_loss(noise_pred, noise)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    noise_pred = self.model(noisy_masks, t, images)
                    loss = F.mse_loss(noise_pred, noise)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            scheduler.step()
            avg_loss = epoch_loss / max(batch_count, 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[TRAIN] {self.name} epoch {epoch+1}/{self.finetune_epochs} - avg loss: {avg_loss:.4f} (LR: {current_lr:.6f})")
        
        self.model.eval()
        
        # Save checkpoint
        config = self._get_config()
        checkpoint_path = save_checkpoint(
            self.model,
            "ddp",
            config,
            metadata={"final_loss": avg_loss, "epochs": self.finetune_epochs}
        )
        print(f"[INFO] {self.name}: Saved checkpoint to {checkpoint_path}")
    
    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        """Predict segmentation masks using diffusion."""
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)  # [B, 3, H, W]
            b, _, h, w = images.shape
            
            # Generate masks using diffusion
            generated_masks = self._p_sample_loop((b, 1, h, w), images)  # [B, 1, H, W]
            
            # Denormalize: convert from [-1, 1] back to [0, num_classes-1]
            generated_masks = (generated_masks + 1.0) / 2.0 * (self.num_classes - 1)
            generated_masks = torch.clamp(generated_masks, 0, self.num_classes - 1)
            
            # Convert to class indices
            preds = generated_masks.round().long().squeeze(1).cpu().numpy()
            return preds.astype(np.int64)
    
    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        """Predict logits by running diffusion and converting to probabilities."""
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            b, _, h, w = images.shape
            
            # Generate masks using diffusion
            generated_masks = self._p_sample_loop((b, 1, h, w), images)
            
            # Denormalize
            generated_masks = (generated_masks + 1.0) / 2.0 * (self.num_classes - 1)
            generated_masks = torch.clamp(generated_masks, 0, self.num_classes - 1)
            
            # Convert to logits by creating one-hot and smoothing
            # This is a simple approximation - in practice, you might want to run multiple samples
            masks_float = generated_masks.squeeze(1)  # [B, H, W]
            
            # Create logits by converting to one-hot and adding small noise
            logits = torch.zeros(b, self.num_classes, h, w, device=self.device)
            for c in range(self.num_classes):
                mask = (masks_float - c).abs() < 0.5
                logits[:, c] = mask.float() * 10.0 - 5.0  # High confidence for predicted class
            
            return logits.cpu().numpy()
