"""Stage 2b: Standard Fine-tuning for Arrhythmia Classification.

This script implements standard supervised fine-tuning following SOTA protocol:
- Load pre-trained encoder from Stage 1 (SSL)
- Add single linear classification head
- Train with low learning rate
- Evaluate with macro AUROC (SOTA metric)
- Compare with baseline (random init + supervised)

This serves as a fair comparison baseline for the MoE approach.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ptbxl_dataset import PTBXLDataset
from models import PatchEncoder
from utils import set_seed, save_checkpoint, AverageMeter


class SimpleClassifier(nn.Module):
    """Simple classifier with pre-trained encoder + linear head.
    
    This follows the standard fine-tuning protocol used in SOTA papers.
    """
    
    def __init__(
        self,
        patch_size: int = 64,
        embedding_dim: int = 256,
        num_classes: int = 5,
        patch_encoder_hidden: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        
        # Patch encoder (from Stage 1 SSL)
        self.patch_encoder = PatchEncoder(
            hidden_dim=patch_encoder_hidden,
            embedding_dim=embedding_dim,
        )
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.patch_encoder.parameters():
                param.requires_grad = False
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        
        self.freeze_encoder = freeze_encoder
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch, 9, 64) semantic patches
            
        Returns:
            logits: (batch, num_classes) classification logits
        """
        # Encode patches
        encoded = self.patch_encoder(patches)  # (batch, 9, embedding_dim)
        
        # Aggregate via mean pooling
        aggregated = encoded.mean(dim=1)  # (batch, embedding_dim)
        
        # Classify
        logits = self.classifier(aggregated)  # (batch, num_classes)
        
        return logits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute SOTA evaluation metrics for multi-label classification.
    
    Args:
        y_true: (N, num_classes) ground truth binary labels
        y_pred: (N, num_classes) predicted binary labels
        y_score: (N, num_classes) predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy (exact match and per-sample)
    exact_match = (y_true == y_pred).all(axis=1).mean()
    per_sample_acc = (y_true == y_pred).mean()
    
    metrics['exact_match'] = exact_match
    metrics['accuracy'] = per_sample_acc
    
    # Macro AUROC (SOTA primary metric)
    try:
        macro_auroc = roc_auc_score(y_true, y_score, average='macro')
        metrics['macro_auroc'] = macro_auroc
    except ValueError:
        metrics['macro_auroc'] = 0.0
    
    # Per-class AUROC
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    for i, name in enumerate(class_names):
        try:
            auroc = roc_auc_score(y_true[:, i], y_score[:, i])
            metrics[f'auroc_{name}'] = auroc
        except ValueError:
            metrics[f'auroc_{name}'] = 0.0
    
    # Macro AUPRC
    try:
        macro_auprc = average_precision_score(y_true, y_score, average='macro')
        metrics['macro_auprc'] = macro_auprc
    except ValueError:
        metrics['macro_auprc'] = 0.0
    
    return metrics


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(dataloader):
        patches = batch['patches'].to(device)  # (B, 9, 64)
        labels = batch['labels'].to(device)  # (B, 5)
        
        # Forward pass
        logits = model(patches)
        
        # Compute loss (multi-label BCE)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # Compute accuracy
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == labels).float().mean()
        
        batch_size = patches.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        
        if batch_idx % config['logging']['log_interval'] == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc: {acc_meter.avg:.4f} ({acc_meter.avg*100:.2f}%)")
    
    return {
        'loss': loss_meter.avg,
        'acc': acc_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, device, config, compute_full_metrics=False):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    
    all_labels = []
    all_preds = []
    all_scores = []
    
    for batch in dataloader:
        patches = batch['patches'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(patches)
        
        # Compute loss
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss_meter.update(loss.item(), patches.size(0))
        
        # Store predictions for metrics
        scores = torch.sigmoid(logits)
        preds = (scores > 0.5).float()
        
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
    
    # Concatenate all batches
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    
    # Compute metrics
    if compute_full_metrics:
        metrics = compute_metrics(all_labels, all_preds, all_scores)
        metrics['loss'] = loss_meter.avg
        # Add 'acc' alias for consistency
        metrics['acc'] = metrics['accuracy']
    else:
        # Quick metrics for training loop
        acc = (all_labels == all_preds).mean()
        metrics = {
            'loss': loss_meter.avg,
            'acc': acc,
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Standard Fine-tuning for Arrhythmia Classification')
    parser.add_argument('--config', type=str, default='configs/stage2_finetune.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str, 
                        default='/path/to/ptbxl/processed',
                        help='Path to PTB-XL data')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pre-trained Stage 1 checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights during fine-tuning')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("Standard Fine-tuning for Arrhythmia Classification")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")
    print(f"Pre-trained: {args.pretrained}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print()
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create data augmentation
    from augmentation import ECGAugmentation
    
    train_transform = ECGAugmentation(
        amplitude_scale_range=(0.8, 1.2),
        time_warp_range=(0.9, 1.1),
        baseline_wander_amplitude=0.1,
        baseline_wander_freq_range=(0.2, 0.5),
        gaussian_noise_std=0.05,
        prob=0.5,  # Apply each augmentation with 50% probability
    )
    
    print("Data Augmentation enabled for training:")
    print("  • Amplitude scaling (0.8-1.2x)")
    print("  • Time warping (0.9-1.1x)")
    print("  • Baseline wander (0.2-0.5 Hz)")
    print("  • Gaussian noise (std=0.05)")
    print("  • Probability: 0.5 per augmentation")
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = PTBXLDataset(
        data_path=args.data_path,
        split='train',
        mode='finetune',
        patch_size=64,
        seed=config['seed'],
        transform=train_transform,  # Apply augmentation to training
    )
    
    val_dataset = PTBXLDataset(
        data_path=args.data_path,
        split='val',
        mode='finetune',
        patch_size=64,
        seed=config['seed'],
        transform=None,  # No augmentation for validation
    )
    
    test_dataset = PTBXLDataset(
        data_path=args.data_path,
        split='test',
        mode='finetune',
        patch_size=64,
        seed=config['seed'],
        transform=None,  # No augmentation for test
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print()
    
    # Create model
    print("Creating model...")
    model = SimpleClassifier(
        patch_size=64,
        embedding_dim=config['model']['embedding_dim'],
        num_classes=5,
        patch_encoder_hidden=config['model']['patch_encoder_hidden'],
        dropout=config['training'].get('dropout', 0.1),
        freeze_encoder=args.freeze_encoder,
    )
    model = model.to(device)
    
    # Load pre-trained weights
    print(f"Loading pre-trained Stage 1 checkpoint: {args.pretrained}")
    checkpoint = torch.load(args.pretrained, map_location=device)
    
    # Extract patch encoder weights
    state_dict = checkpoint['model_state_dict']
    patch_encoder_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('patch_encoder.'):
            new_key = key.replace('patch_encoder.', '')
            patch_encoder_dict[new_key] = value
    
    model.patch_encoder.load_state_dict(patch_encoder_dict)
    print("✓ Stage 1 patch encoder weights loaded successfully")
    
    # Count parameters
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params_total:,}")
    print(f"Trainable parameters: {n_params_trainable:,}")
    print()
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Learning rate scheduler: Warm-up + Cosine Annealing
    warmup_epochs = config['training'].get('warmup_epochs', 3)
    
    # Warm-up scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - warmup_epochs,
        eta_min=1e-7,
    )
    
    # Combine schedulers
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    print(f"LR Schedule: {warmup_epochs} epochs warm-up + Cosine Annealing")
    print(f"  Initial LR: {config['training']['lr'] * 0.1:.2e} → {config['training']['lr']:.2e}")
    print(f"  Min LR: 1e-7")
    
    # Training loop
    print("Starting training...")
    print("=" * 80)
    print()
    
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_auroc = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"Epoch [{epoch}/{config['training']['epochs']}]")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = validate(model, val_loader, device, config, compute_full_metrics=True)
        
        # Update learning rate
        scheduler.step()
        
        # Print summary
        print(f"\nEpoch [{epoch}] Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['acc']:.4f} ({train_metrics['acc']*100:.2f}%)")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}  Acc: {val_metrics['acc']:.4f} ({val_metrics['acc']*100:.2f}%)")
        print(f"  Val Macro AUROC: {val_metrics['macro_auroc']:.4f}")
        print(f"  Val Macro AUPRC: {val_metrics['macro_auprc']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['macro_auroc'] > best_val_auroc
        if is_best:
            best_val_auroc = val_metrics['macro_auroc']
        
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_val_auroc': best_val_auroc,
                'config': config,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename=f'finetune_epoch_{epoch}.pt',
        )
        
        if is_best:
            print(f"  ✨ New best validation AUROC: {best_val_auroc:.4f}")
        print()
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)
    
    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, device, config, compute_full_metrics=True)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['acc']:.4f} ({test_metrics['acc']*100:.2f}%)")
    print(f"  Exact Match: {test_metrics['exact_match']:.4f} ({test_metrics['exact_match']*100:.2f}%)")
    print(f"  Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"\nPer-Class AUROC:")
    for name in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        print(f"  {name}: {test_metrics[f'auroc_{name}']:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    print(f"Test AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
