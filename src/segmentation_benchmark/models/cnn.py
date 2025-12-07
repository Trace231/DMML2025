"""Wrappers for torchvision semantic segmentation backbones."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import segmentation as tv_seg

from ..evaluation.registry import register_segmenter
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.pretrained import build_torchvision_segmentation_model
from .base import BaseSegmenter


def _default_weights(model_name: str):
    enum_name = f"{model_name.upper()}_Weights"
    return getattr(tv_seg, enum_name).DEFAULT if hasattr(tv_seg, enum_name) else None


def _adjust_head(model: nn.Module, num_classes: int) -> None:
    """Adjust model head to match num_classes and initialize new layers properly."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_layer = model.classifier[-1]
        if isinstance(last_layer, nn.Conv2d) and last_layer.out_channels != num_classes:
            in_channels = last_layer.in_channels
            new_layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            # Initialize new classification layer with Kaiming initialization
            nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
            if new_layer.bias is not None:
                nn.init.constant_(new_layer.bias, 0)
            model.classifier[-1] = new_layer
    if hasattr(model, "aux_classifier") and isinstance(model.aux_classifier, nn.Sequential):
        last_layer = model.aux_classifier[-1]
        if isinstance(last_layer, nn.Conv2d) and last_layer.out_channels != num_classes:
            in_channels = last_layer.in_channels
            new_layer = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            # Initialize auxiliary classifier with Kaiming initialization
            nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')
            if new_layer.bias is not None:
                nn.init.constant_(new_layer.bias, 0)
            model.aux_classifier[-1] = new_layer


@register_segmenter("torchvision")
class TorchvisionSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_name: str = "fcn_resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        finetune_epochs: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 2,
        num_workers: int = 0,
        device: Optional[str] = None,
        resume_epoch: Optional[int] = None,
    ) -> None:
        super().__init__(num_classes=num_classes, device=device, name=f"Torchvision-{model_name}")
        if not hasattr(tv_seg, model_name):
            raise ValueError(f"Unknown torchvision segmentation model '{model_name}'")
        
        # Ensure numeric parameters are correct types (YAML may parse as strings)
        self.finetune_epochs = int(finetune_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.model_name = model_name
        self.pretrained = pretrained
        self.resume_epoch = int(resume_epoch) if resume_epoch is not None else None
        
        # Use our central pretrained cache instead of letting torchvision download
        self.model = build_torchvision_segmentation_model(model_name, pretrained=pretrained)
        _adjust_head(self.model, num_classes)
        self.model.to(self.device)
        
        # Try to load checkpoint if available (only if finetune_epochs > 0, meaning we expect trained weights)
        # Load checkpoint BEFORE compilation, as compiled models may have issues loading state_dict
        # If resume_epoch is specified, we'll load it during training
        if finetune_epochs > 0 and self.resume_epoch is None:
            config = self._get_config()
            checkpoint = load_checkpoint(self.model_name, config, model=self.model, device=self.device)
            if checkpoint is not None:
                print(f"[INFO] {self.name}: Loaded checkpoint from previous training")
                # If checkpoint found, we can skip training
                self._checkpoint_loaded = True
            else:
                self._checkpoint_loaded = False
        else:
            self._checkpoint_loaded = False
        
        # Enable model compilation for faster training (PyTorch 2.0+)
        # This provides 10-30% speedup on compatible GPUs
        # Do this AFTER checkpoint loading
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print(f"[INFO] {self.name}: Model compilation enabled for faster training")
            except Exception as e:
                print(f"[WARNING] {self.name}: Model compilation failed ({e}), continuing without it")

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for checkpoint matching."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "finetune_epochs": self.finetune_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
    
    def prepare(self, train_dataset: Optional[Any] = None, val_dataset: Optional[Any] = None, checkpoint_builder_name: Optional[str] = None) -> None:
        # Skip training if checkpoint was already loaded
        if self._checkpoint_loaded:
            print(f"[INFO] {self.name}: Skipping training (using loaded checkpoint)")
            self.model.eval()
            return
        
        if self.finetune_epochs <= 0 or train_dataset is None:
            return
        
        # Use provided checkpoint builder name, or fall back to model_name
        builder_name = checkpoint_builder_name if checkpoint_builder_name is not None else self.model_name
        loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        # Use ignore_index=255 so VOC "void" pixels are not penalised during training.
        # The dataset keeps 0-20 as valid class indices and 255 as void.
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # Use different learning rates for backbone and classification head
        # Backbone (pretrained) uses smaller LR, head (random init) uses larger LR
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name or 'aux_classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # Head gets 10x larger learning rate since it's randomly initialized
        optimizer = Adam(
            [
                {'params': backbone_params, 'lr': self.learning_rate * 0.1},
                {'params': head_params, 'lr': self.learning_rate}
            ],
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler: cosine annealing from initial LR to 1/10 of initial LR
        # This allows high LR early (fast learning) and moderate LR late (fine-tuning)
        # Final LR will be 10% of initial, which is a good fine-tuning rate
        scheduler = CosineAnnealingLR(optimizer, T_max=self.finetune_epochs, eta_min=self.learning_rate * 0.1)
        
        # Enable mixed precision training for 1.5-2x speedup on V100
        # V100 has Tensor Cores that accelerate FP16 operations significantly
        scaler = GradScaler('cuda')
        use_amp = torch.cuda.is_available() and hasattr(torch.amp, 'autocast')
        if use_amp:
            print(f"[INFO] {self.name}: Mixed precision training (AMP) enabled")
        
        # Get config for checkpoint operations
        config = self._get_config()
        
        # Check if we should resume from a checkpoint
        start_epoch = 0
        from ..utils.checkpoint import load_epoch_checkpoint, load_latest_epoch_checkpoint
        
        if self.resume_epoch is not None:
            # Manual resume: load specific epoch
            checkpoint = load_epoch_checkpoint(
                builder_name, config, self.resume_epoch, 
                model=self.model, device=self.device,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler
            )
            if checkpoint is not None:
                start_epoch = self.resume_epoch
                print(f"[INFO] {self.name}: Resuming training from epoch {start_epoch} (manual)")
            else:
                print(f"[WARNING] {self.name}: Resume epoch {self.resume_epoch} checkpoint not found, starting from epoch 0")
        else:
            # Auto resume: find and load latest checkpoint
            result = load_latest_epoch_checkpoint(
                builder_name, config,
                model=self.model, device=self.device,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler
            )
            if result is not None:
                latest_epoch, checkpoint = result
                start_epoch = latest_epoch
                print(f"[INFO] {self.name}: Auto-resuming training from latest checkpoint (epoch {start_epoch})")
        
        self.model.train()
        checkpoint_interval = 10  # Save checkpoint every 10 epochs
        
        for epoch in range(start_epoch, self.finetune_epochs):
            epoch_loss = 0.0
            batch_count = 0
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].squeeze(1).long().to(self.device)
                optimizer.zero_grad()
                
                # Use mixed precision training if available
                if use_amp:
                    with autocast('cuda'):
                        outputs = self.model(images)
                        logits = outputs["out"]
                        loss = criterion(logits, masks)
                        if "aux" in outputs and outputs["aux"] is not None:
                            loss = loss + 0.3 * criterion(outputs["aux"], masks)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Fallback to FP32 training
                    outputs = self.model(images)
                    logits = outputs["out"]
                    loss = criterion(logits, masks)
                    if "aux" in outputs and outputs["aux"] is not None:
                        loss = loss + 0.3 * criterion(outputs["aux"], masks)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Update learning rate at end of each epoch
            scheduler.step()
            
            # Get current learning rates for logging
            current_lr_backbone = optimizer.param_groups[0]['lr']
            current_lr_head = optimizer.param_groups[1]['lr']
            
            avg_loss = epoch_loss / max(batch_count, 1)
            current_epoch = epoch + 1
            print(f"[TRAIN] {self.name} epoch {current_epoch}/{self.finetune_epochs} - avg loss: {avg_loss:.4f} "
                  f"(LR: backbone={current_lr_backbone:.6f}, head={current_lr_head:.6f})")
            
            # Save checkpoint every checkpoint_interval epochs
            if current_epoch % checkpoint_interval == 0:
                checkpoint_path = save_checkpoint(
                    self.model,
                    builder_name,
                    config,
                    metadata={"loss": avg_loss, "epoch": current_epoch, "total_epochs": self.finetune_epochs},
                    epoch=current_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_amp else None,
                )
                print(f"[INFO] {self.name}: Saved checkpoint at epoch {current_epoch} to {checkpoint_path}")
        
        self.model.eval()
        
        # Save final checkpoint after training
        checkpoint_path = save_checkpoint(
            self.model,
            builder_name,
            config,
            metadata={"final_loss": avg_loss, "epochs": self.finetune_epochs},
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
        )
        print(f"[INFO] {self.name}: Saved final checkpoint to {checkpoint_path}")

    def predict_logits(self, batch: Dict[str, Any]) -> Optional[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            images = batch["image"].to(self.device)
            outputs = self.model(images)["out"]
            return outputs.detach().cpu().numpy()

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        logits = self.predict_logits(batch)
        preds = np.argmax(logits, axis=1).astype(np.int64)
        return preds


@register_segmenter("fcn_resnet50")
def build_fcn_resnet50(**kwargs: Any) -> TorchvisionSegmenter:
    kwargs.setdefault("model_name", "fcn_resnet50")
    return TorchvisionSegmenter(**kwargs)


@register_segmenter("deeplabv3_resnet50")
def build_deeplabv3_resnet50(**kwargs: Any) -> TorchvisionSegmenter:
    kwargs.setdefault("model_name", "deeplabv3_resnet50")
    return TorchvisionSegmenter(**kwargs)
