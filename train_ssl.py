"""SSL Pre-training with Ordinal Position Prediction (Pure Jigsaw - NO Masking).

This script implements Stage 1 pre-training using ordinal position prediction.
The model receives 9 semantic patches (P, QRS, T from 3 consecutive heartbeats)
in a random shuffled order, and must predict the original temporal position
(0-8) of each patch.

NO MASKING is used - all patches are visible (intact) but shuffled.

The model learns:
- Morphological features of each wave (P, QRS, T)
- Temporal dependencies across heartbeats (P → QRS → T sequence)
- Physiological validity (recognizing correct cardiac timing)

Key Differences from Original Implementation:
- NO masked reconstruction loss
- NO reconstruction head in model
- Uses R-peak detection (Pan-Tompkins) to extract semantic patches
- Ordinal position prediction (9 classes) instead of full permutation (9! classes)
- Per-patch position classification for better learning signal
- Dataset returns shuffled patches from multiple heartbeats
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ptbxl_dataset import PTBXLDataset
from models import PatchEncoder, PuzzleClassifier
from utils import set_seed, save_checkpoint, AverageMeter


class SSLEncoder(nn.Module):
    """Encoder for SSL pre-training with Ordinal Position Prediction.
    
    Architecture:
        - PatchEncoder: CNN to encode each patch independently
        - Position Classifier: Per-patch MLP to predict original position (0-8)
    
    Task: Given 9 shuffled patches from 3 heartbeats, predict each patch's
    original temporal position. This is more learnable than full permutation
    prediction (9 classes vs 362,880 classes).
    """
    
    def __init__(
        self,
        patch_size: int = 64,
        patch_encoder_hidden: int = 64,
        embedding_dim: int = 128,
        num_positions: int = 9,
        dropout: float = 0.1,
        deep_encoder: bool = False,
        input_channels: int = 12,
        encoder_depth: str = "resnet18",
    ):
        super().__init__()
        
        self.patch_encoder = PatchEncoder(
            hidden_dim=patch_encoder_hidden,
            embedding_dim=embedding_dim,
            deep=deep_encoder,
            input_channels=input_channels,
            encoder_depth=encoder_depth,
        )
        
        # Per-patch position classifier
        self.position_head = PuzzleClassifier(
            embedding_dim=embedding_dim,
            num_classes=num_positions,
            dropout=dropout,
        )
        
    def forward(self, patches: torch.Tensor, return_embeddings: bool = False):
        """Forward pass.
        
        Args:
            patches: (batch, 9, patch_size) or (batch, 9, channels, patch_size)
                     shuffled patches from 3 heartbeats
            return_embeddings: if True, also return patch embeddings
            
        Returns:
            position_logits: (batch, 9, 9) position classification logits per patch
            encoded (optional): (batch, 9, embedding_dim)
        """
        if patches.dim() == 3:
            batch_size, n_patches, _ = patches.shape
        elif patches.dim() == 4:
            batch_size, n_patches, _, _ = patches.shape
        else:
            raise ValueError(f"Unexpected patches shape: {patches.shape}")
        
        # Encode each patch
        encoded = self.patch_encoder(patches)  # (batch, 9, embedding_dim)
        
        # Predict position for each patch independently
        encoded_flat = encoded.view(-1, encoded.size(-1))  # (batch*9, embedding_dim)
        position_logits_flat = self.position_head(encoded_flat)  # (batch*9, 9)
        position_logits = position_logits_flat.view(batch_size, n_patches, -1)
        
        if return_embeddings:
            return position_logits, encoded
        return position_logits


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    
    # Metrics
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        patches = batch['patches'].to(device)  # (B, 9, patch_size) - shuffled patches
        position_targets = batch['position_labels'].to(device)  # (B, 9) - original positions [0-8]
        
        # Forward pass (optionally return embeddings for contrastive)
        position_logits, encoded = model(patches, return_embeddings=True)  # (B, 9, 9)
        
        # Compute loss (cross-entropy per patch, then average)
        batch_size, n_patches, n_classes = position_logits.shape
        
        # Reshape for cross-entropy: (B*9, 9) and (B*9,)
        position_logits_flat = position_logits.view(-1, n_classes)
        position_targets_flat = position_targets.view(-1)
        
        ce_loss = nn.CrossEntropyLoss()(position_logits_flat, position_targets_flat)
        
        # Contrastive auxiliary (SimCLR-style on sample embeddings)
        contrastive_weight = config['training'].get('lambda_contrastive', 0.0)
        if contrastive_weight > 0:
            # Create two noisy views of patches and get mean embeddings
            noise_std = config['training'].get('contrastive_noise_std', 0.05)
            patches_v1 = patches + noise_std * torch.randn_like(patches)
            patches_v2 = patches + noise_std * torch.randn_like(patches)
            _, enc_v1 = model(patches_v1, return_embeddings=True)
            _, enc_v2 = model(patches_v2, return_embeddings=True)
            z1 = enc_v1.mean(dim=1)
            z2 = enc_v2.mean(dim=1)
            z1 = torch.nn.functional.normalize(z1, dim=1)
            z2 = torch.nn.functional.normalize(z2, dim=1)
            logits_sim = torch.matmul(z1, z2.t()) / config['training'].get('contrastive_tau', 0.5)
            labels_sim = torch.arange(z1.size(0), device=device)
            contrastive = nn.CrossEntropyLoss()(logits_sim, labels_sim)
            loss = ce_loss + contrastive_weight * contrastive
        else:
            loss = ce_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Compute accuracy (per-patch prediction)
        position_pred = position_logits.argmax(dim=-1)  # (B, 9)
        acc = (position_pred == position_targets).float().mean()
        
        # Update meters
        batch_size = patches.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        
        # Log
        if batch_idx % config['logging']['log_interval'] == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc: {acc_meter.avg:.4f} ({acc_meter.avg*100:.2f}%)")
    
    return {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for batch in dataloader:
        patches = batch['patches'].to(device)  # (B, 9, patch_size)
        position_targets = batch['position_labels'].to(device)  # (B, 9)
        
        # Forward pass
        position_logits = model(patches)  # (B, 9, 9)
        
        # Compute loss
        batch_size, n_patches, n_classes = position_logits.shape
        position_logits_flat = position_logits.view(-1, n_classes)
        position_targets_flat = position_targets.view(-1)
        
        loss = nn.CrossEntropyLoss()(position_logits_flat, position_targets_flat)
        
        # Compute accuracy (per-patch)
        position_pred = position_logits.argmax(dim=-1)  # (B, 9)
        acc = (position_pred == position_targets).float().mean()
        
        # Update meters
        batch_size = patches.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
    
    return {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
    }


def main():
    parser = argparse.ArgumentParser(description='SSL Pre-training with ECGWavePuzzle (Pure Jigsaw)')
    parser.add_argument('--config', type=str, default='configs/stage1_ssl.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, 
                        default=None,
                        help='Path to PTB-XL data (defaults to config.dataset.path)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--exclude_patients_file', type=str, default=None,
                        help='Optional file with patient_ids to exclude (one per line)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("SSL Pre-training: ECGWavePuzzle (Pure Jigsaw - NO Masking)")
    print("=" * 80)
    data_path = args.data_path or config['dataset'].get('path') \
        or '/path/to/ptbxl/processed'
    print(f"Config: {args.config}")
    print(f"Data: {data_path}")
    print()
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create datasets (NO masking - removed mask_ratio parameter)
    print("Loading datasets...")
    exclude_patients = None
    if args.exclude_patients_file:
        with open(args.exclude_patients_file) as f:
            exclude_patients = [int(line.strip()) for line in f if line.strip()]
        print(f"Excluding {len(exclude_patients)} patient_ids from train/val")
    patch_size = config['dataset'].get('patch_size', 64)

    train_dataset = PTBXLDataset(
        data_path=data_path,
        split='train',
        mode='ssl',
        patch_size=patch_size,
        seed=config['seed'],
        exclude_patients=exclude_patients,
    )
    
    val_dataset = PTBXLDataset(
        data_path=data_path,
        split='val',
        mode='ssl',
        patch_size=patch_size,
        seed=config['seed'],
        exclude_patients=exclude_patients,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Random guess accuracy baseline: {100/9:.2f}% (1 out of 9 positions)")
    print()
    
    # Create model
    print("Creating model...")
    
    deep_encoder = config['model'].get('deep_encoder', False)
    encoder_depth = config['model'].get('encoder_depth', 'resnet18')
    input_channels = config['model'].get('input_channels', 12)
    if deep_encoder:
        print(f"  ✓ Using Deep Encoder (ResNet - {encoder_depth})")
        
    model = SSLEncoder(
        patch_size=patch_size,
        patch_encoder_hidden=config['model']['patch_encoder_hidden'],
        embedding_dim=config['model']['embedding_dim'],
        num_positions=9,  # Changed from num_permutations=6
        dropout=config['training'].get('dropout', 0.1),
        deep_encoder=deep_encoder,
        input_channels=input_channels,
        encoder_depth=encoder_depth,
    )
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Warm-up + Cosine
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
        print()
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = validate(model, val_loader, device, config)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"\nEpoch [{epoch+1}] Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['acc']:.4f} ({train_metrics['acc']*100:.2f}%)")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}  Acc: {val_metrics['acc']:.4f} ({val_metrics['acc']*100:.2f}%)")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_loss': best_val_loss,
                'config': config,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename=f'ssl_epoch_{epoch+1}.pt',
        )
        
        print(f"  Checkpoint saved: {checkpoint_dir}")
        if is_best:
            print(f"  ✨ New best validation loss: {best_val_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
