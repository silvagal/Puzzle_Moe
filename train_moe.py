"""Stage 2: MoE Pre-training with Symbolic Gating.

This script implements Stage 2 pre-training using Mixture of Experts with
symbolic routing priors. The model receives pre-trained encoders from Stage 1
and learns specialized experts for different cardiac patterns.

Architecture:
- Input: 9 semantic patches from Stage 1 (P, QRS, T from 3 heartbeats)
- Experts: 3 specialized experts (Rhythm, Morphology, Quality)
- Gating: Hybrid neuro-symbolic routing with medical priors
- Task: Arrhythmia classification (5 superclasses: NORM, MI, STTC, CD, HYP)
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ptbxl_dataset import PTBXLDataset
from models import PatchEncoder, SwiGLUExpert
from utils import set_seed, save_checkpoint, AverageMeter


def load_balancing_loss(gating_weights: torch.Tensor) -> torch.Tensor:
    """Compute load balancing loss to prevent expert collapse.
    
    Penalizes uneven expert utilization by computing the coefficient of variation
    squared. Lower values indicate more balanced routing.
    
    Args:
        gating_weights: (batch, num_experts) routing weights
        
    Returns:
        loss: scalar load balancing loss
    """
    # Compute mean expert usage across batch
    expert_usage = gating_weights.mean(dim=0)  # (num_experts,)
    
    # Coefficient of variation: std / mean
    mean_usage = expert_usage.mean()
    std_usage = expert_usage.std()
    cv = std_usage / (mean_usage + 1e-8)
    
    # Return squared CV as penalty
    return cv ** 2


def map_labels_to_gate_target(labels: torch.Tensor) -> torch.Tensor:
    """Map PTB-XL multi-label targets (B, 5) to a single expert index (B,).

    Mapping with fixed priority (morphology > conduction > normal):
        Expert 0: NORM
        Expert 1: MI/STTC/HYP (morphology)
        Expert 2: CD (rhythm/conduction)
    """
    # labels order: [NORM, MI, STTC, CD, HYP]
    norm = labels[:, 0] > 0.5
    morph = (labels[:, 1] > 0.5) | (labels[:, 2] > 0.5) | (labels[:, 4] > 0.5)
    conduction = labels[:, 3] > 0.5
    
    target = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)
    target[morph] = 1
    target[~morph & conduction] = 2
    target[~morph & ~conduction & norm] = 0
    return target


class MoEExpert(nn.Module):
    """Individual expert module for specialized classification."""
    
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float = 0.1, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or embedding_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SparseMoEExpert(nn.Module):
    """Switch-style sparse FFN with top-k gating over sub-experts."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
        num_subexperts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()
        self.top_k = top_k
        hidden = hidden_dim or embedding_dim // 2
        self.subexperts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            for _ in range(num_subexperts)
        ])
        self.gating = nn.Linear(embedding_dim, num_subexperts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gating(x)  # (B, num_subexperts)
        top_vals, top_idx = torch.topk(gate_logits, k=self.top_k, dim=-1)
        gate = torch.zeros_like(gate_logits).scatter(-1, top_idx, torch.softmax(top_vals, dim=-1))

        outputs = []
        for i, expert in enumerate(self.subexperts):
            out = expert(x)
            outputs.append(out)
        stacked = torch.stack(outputs, dim=1)  # (B, num_subexperts, num_classes)
        mixed = (gate.unsqueeze(-1) * stacked).sum(dim=1)
        return mixed


class SymbolicGating(nn.Module):
    """Gating network with symbolic priors for expert routing."""
    
    def __init__(self, embedding_dim: int, num_experts: int, symbolic_dim: int = 6):
        super().__init__()
        
        # Neural component: learns from embeddings
        self.neural_gating = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts),
        )
        
        # Symbolic component: learns from medical features
        self.symbolic_gating = nn.Sequential(
            nn.Linear(symbolic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )
    
    def forward(self, embeddings: torch.Tensor, symbolic_features: torch.Tensor = None):
        """
        Args:
            embeddings: (batch, embedding_dim) aggregated embeddings
            symbolic_features: (batch, symbolic_dim) medical biomarkers
            
        Returns:
            gating_weights: (batch, num_experts) normalized routing weights
        """
        neural_logits = self.neural_gating(embeddings)  # (batch, num_experts)
        
        if symbolic_features is not None:
            symbolic_logits = self.symbolic_gating(symbolic_features)  # (batch, num_experts)
            combined_logits = 0.5 * neural_logits + 0.5 * symbolic_logits
        else:
            combined_logits = neural_logits
        
        gating_weights = torch.softmax(combined_logits, dim=1)
        return gating_weights


class MoEClassifier(nn.Module):
    """Mixture of Experts classifier with semantic patch encoding."""
    
    def __init__(
        self,
        patch_size: int = 64,
        embedding_dim: int = 256,
        num_experts: int = 3,
        num_classes: int = 5,
        patch_encoder_hidden: int = 128,
        dropout: float = 0.1,
        use_symbolic_gating: bool = True,
        deep_encoder: bool = False,
        use_attention: bool = False,
        attention_heads: int = 4,
        input_channels: int = 12,
        encoder_depth: str = "resnet18",
        expert_hidden_dim: Optional[int] = None,
        expert_type: str = "mlp",
        num_subexperts: int = 4,
        top_k: int = 1,
    ):
        super().__init__()
        
        # Patch encoder (can be frozen from Stage 1 or fine-tuned)
        self.patch_encoder = PatchEncoder(
            hidden_dim=patch_encoder_hidden,
            embedding_dim=embedding_dim,
            deep=deep_encoder,  # Tier A/B: Deep encoder (ResNet if deep=True)
            input_channels=input_channels,
            encoder_depth=encoder_depth,
        )
        
        # Attention aggregation (Tier A improvement)
        self.use_attention = use_attention
        if use_attention:
            from models import AttentionAggregation
            self.attention_agg = AttentionAggregation(
                embedding_dim=embedding_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
        
        # Experts: choose architecture
        experts: list[nn.Module] = []
        for _ in range(num_experts):
            if expert_type == "swiglu":
                experts.append(SwiGLUExpert(embedding_dim, num_classes, hidden_multiplier=expert_hidden_dim / embedding_dim if expert_hidden_dim else 2.0))
            elif expert_type == "sparse":
                experts.append(SparseMoEExpert(
                    embedding_dim=embedding_dim,
                    num_classes=num_classes,
                    dropout=dropout,
                    hidden_dim=expert_hidden_dim,
                    num_subexperts=num_subexperts,
                    top_k=top_k,
                ))
            else:
                experts.append(MoEExpert(embedding_dim, num_classes, dropout, hidden_dim=expert_hidden_dim))
        self.experts = nn.ModuleList(experts)
        
        # Gating network
        self.gating = SymbolicGating(embedding_dim, num_experts)
        self.use_symbolic_gating = use_symbolic_gating
        
        self.num_experts = num_experts
        self.num_classes = num_classes
    
    def forward(
        self,
        patches: torch.Tensor,
        symbolic_features: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            patches: (batch, 9, 12, 64) semantic patches
            symbolic_features: (batch, 5) optional medical biomarkers
            
        Returns:
            logits: (batch, num_classes) classification logits
            gating_weights: (batch, num_experts) expert routing weights
            expert_logits: (batch, num_experts, num_classes) per-expert predictions
            attn_weights: (batch, num_heads, seq_len, seq_len) attention weights if use_attention=True, else None
        """
        # Encode patches
        encoded = self.patch_encoder(patches)  # (batch, 9, embedding_dim)
        
        # Aggregate for gating decision
        attn_weights = None
        if self.use_attention:
            # Tier A: Use attention aggregation with weight extraction
            result = self.attention_agg(encoded, return_weights=True)
            if isinstance(result, tuple):
                aggregated, attn_weights = result  # (batch, embedding_dim), (batch, heads, seq, seq)
            else:
                aggregated = result
        else:
            # Original: mean pooling
            aggregated = encoded.mean(dim=1)  # (batch, embedding_dim)
        
        # Compute gating weights
        if self.use_symbolic_gating and symbolic_features is not None:
            gating_weights = self.gating(aggregated, symbolic_features)
        else:
            gating_weights = self.gating(aggregated)
        
        # Get predictions from each expert
        expert_logits = torch.stack([
            expert(aggregated) for expert in self.experts
        ], dim=1)  # (batch, num_experts, num_classes)
        
        # Combine expert predictions via gating
        # Weighted sum: (batch, num_experts, 1) × (batch, num_experts, num_classes)
        logits = (gating_weights.unsqueeze(-1) * expert_logits).sum(dim=1)
        
        return logits, gating_weights, expert_logits, attn_weights


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    gate_meter = AverageMeter()
    
    gating_strategy = config['training'].get('gating_strategy', 'symbolic')
    lambda_gate_supervised = config['training'].get('lambda_gate_supervised', 1.0)
    lambda_lb = 0.0 if gating_strategy == 'supervised_tuler' else config['training'].get('lambda_load_balance', 0.01)
    
    for batch_idx, batch in enumerate(dataloader):
        patches = batch['patches'].to(device)  # (B, 9, 64)
        labels = batch['labels'].to(device)  # (B, 5)
        symbolic = batch.get('symbolic_features')
        if symbolic is not None:
            symbolic = symbolic.to(device)
        
        # Forward pass with symbolic features
        logits, gating_weights, expert_logits, _ = model(patches, symbolic_features=symbolic)
        
        # Compute classification loss (multi-label BCE or focal)
        if config['training'].get('use_focal_loss', False):
            gamma = config['training'].get('focal_gamma', 1.5)
            prob = torch.sigmoid(logits)
            bce = -(labels * torch.log(prob + 1e-8) + (1 - labels) * torch.log(1 - prob + 1e-8))
            loss_ce = ((1 - prob) ** gamma * bce).mean()
        else:
            loss_ce = nn.BCEWithLogitsLoss()(logits, labels)
        
        # Compute load balancing loss
        loss_lb = load_balancing_loss(gating_weights) if lambda_lb > 0 else torch.tensor(0.0, device=device)

        # Supervised gating (Tuler proxy)
        if gating_strategy == 'supervised_tuler':
            gate_target = map_labels_to_gate_target(labels)
            gate_loss = nn.NLLLoss()(torch.log(gating_weights + 1e-8), gate_target)
        else:
            gate_loss = torch.tensor(0.0, device=device)
        
        # Combine losses
        loss = loss_ce + lambda_lb * loss_lb + lambda_gate_supervised * gate_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        # Compute accuracy (multi-label: match if predicted and actual agree)
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == labels).float().mean()
        
        batch_size = patches.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        gate_meter.update(gate_loss.item(), batch_size)
        
        if batch_idx % config['logging']['log_interval'] == 0:
            print(f"  Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc: {acc_meter.avg:.4f} ({acc_meter.avg*100:.2f}%)")
    
    return {'loss': loss_meter.avg, 'acc': acc_meter.avg, 'gate_loss': gate_meter.avg}


@torch.no_grad()
def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    gate_meter = AverageMeter()
    gating_strategy = config['training'].get('gating_strategy', 'symbolic')
    lambda_gate_supervised = config['training'].get('lambda_gate_supervised', 1.0)
    lambda_lb = 0.0 if gating_strategy == 'supervised_tuler' else config['training'].get('lambda_load_balance', 0.01)
    
    for batch in dataloader:
        patches = batch['patches'].to(device)
        labels = batch['labels'].to(device)
        symbolic = batch.get('symbolic_features')
        if symbolic is not None:
            symbolic = symbolic.to(device)
        
        logits, gating_weights, expert_logits, _ = model(patches, symbolic_features=symbolic)
        
        if config['training'].get('use_focal_loss', False):
            gamma = config['training'].get('focal_gamma', 1.5)
            prob = torch.sigmoid(logits)
            bce = -(labels * torch.log(prob + 1e-8) + (1 - labels) * torch.log(1 - prob + 1e-8))
            loss = ((1 - prob) ** gamma * bce).mean()
        else:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        
        # Gating losses
        loss_lb = load_balancing_loss(gating_weights) if lambda_lb > 0 else torch.tensor(0.0, device=device)
        if gating_strategy == 'supervised_tuler':
            gate_target = map_labels_to_gate_target(labels)
            gate_loss = nn.NLLLoss()(torch.log(gating_weights + 1e-8), gate_target)
        else:
            gate_loss = torch.tensor(0.0, device=device)

        loss = loss + lambda_lb * loss_lb + lambda_gate_supervised * gate_loss
        
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = (pred == labels).float().mean()
        
        batch_size = patches.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)
        gate_meter.update(gate_loss.item(), batch_size)
    
    return {'loss': loss_meter.avg, 'acc': acc_meter.avg, 'gate_loss': gate_meter.avg}


def main():
    parser = argparse.ArgumentParser(description='MoE Pre-training with Symbolic Gating')
    parser.add_argument('--config', type=str, default='configs/stage2_moe.yaml',
                        help='Path to config file')
    parser.add_argument('--data_path', type=str,
                        default=None,
                        help='Path to PTB-XL data (defaults to config.dataset.path)')
    parser.add_argument('--pretrained', type=str, default='',
                        help='Path to pre-trained Stage 1 checkpoint (optional)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--exclude_patients_file', type=str, default=None,
                        help='Optional file with patient_ids to exclude from train/val')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("MoE Pre-training: Mixture of Experts with Symbolic Gating")
    print("=" * 80)
    data_path = args.data_path or config['dataset'].get('path') \
        or '/path/to/ptbxl/processed'
    print(f"Config: {args.config}")
    print(f"Data: {data_path}")
    if args.pretrained:
        print(f"Pre-trained: {args.pretrained}")
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
        prob=0.7,  # Stronger augmentation to combat overfitting
    )
    
    print("Data Augmentation enabled for training:")
    print("  • Amplitude scaling (0.8-1.2x)")
    print("  • Time warping (0.9-1.1x)")
    print("  • Baseline wander (0.2-0.5 Hz)")
    print("  • Gaussian noise (std=0.05)")
    print("  • Probability: 0.7 per augmentation")
    print()
    
    # Create datasets
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
        mode='finetune',  # Not SSL mode - use real labels
        patch_size=patch_size,
        seed=config['seed'],
        transform=train_transform,  # Apply augmentation to training
        exclude_patients=exclude_patients,
    )
    
    val_dataset = PTBXLDataset(
        data_path=data_path,
        split='val',
        mode='finetune',
        patch_size=patch_size,
        seed=config['seed'],
        transform=None,  # No augmentation for validation
        exclude_patients=exclude_patients,
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
    
    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print()
    
    # Create model
    print("Creating MoE model...")
    
    # Tier A improvements
    deep_encoder = config['model'].get('deep_encoder', False)
    use_attention = config['model'].get('use_attention', False)
    encoder_depth = config['model'].get('encoder_depth', 'resnet18')
    input_channels = config['model'].get('input_channels', 12)
    attention_heads = config['model'].get('attention_heads', 4)
    
    if deep_encoder:
        print(f"  ✓ Using Deep Encoder (ResNet - {encoder_depth})")
    if use_attention:
        print(f"  ✓ Using Attention Aggregation ({attention_heads} heads)")
    
    model = MoEClassifier(
        patch_size=patch_size,
        embedding_dim=config['model']['embedding_dim'],
        num_experts=config['model']['num_experts'],
        num_classes=5,  # 5 superclasses
        patch_encoder_hidden=config['model']['patch_encoder_hidden'],
        dropout=config['training'].get('dropout', 0.1),
        use_symbolic_gating=config['model'].get('use_symbolic_gating', True),
        deep_encoder=deep_encoder,
        use_attention=use_attention,
        attention_heads=attention_heads,
        input_channels=input_channels,  # Tier B: 12-lead input
        encoder_depth=encoder_depth,
        expert_hidden_dim=config['model'].get('expert_hidden_dim'),
        expert_type=config['model'].get('expert_type', 'mlp'),
        num_subexperts=config['model'].get('num_subexperts', 4),
        top_k=config['model'].get('top_k', 1),
    )
    model = model.to(device)
    
    # Load pre-trained weights if provided
    if args.pretrained:
        print(f"Loading pre-trained Stage 1 checkpoint: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        
        # Extract only the patch encoder weights from the full SSL model
        state_dict = checkpoint['model_state_dict']
        patch_encoder_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('patch_encoder.'):
                # Remove 'patch_encoder.' prefix
                new_key = key.replace('patch_encoder.', '')
                patch_encoder_dict[new_key] = value
        
        if patch_encoder_dict:
            # Try to load with strict=False to handle architecture differences
            try:
                missing_keys, unexpected_keys = model.patch_encoder.load_state_dict(patch_encoder_dict, strict=False)
                print("✓ Stage 1 patch encoder weights loaded with adaptations:")
                if missing_keys:
                    print(f"  Missing keys (new layers): {len(missing_keys)}")
                if unexpected_keys:
                    print(f"  Unexpected keys (old layers): {len(unexpected_keys)}")
                print("  Continuing with partial weight transfer")
            except RuntimeError as e:
                print(f"⚠ Architecture mismatch with SSL checkpoint: {e}")
                print("  Skipping SSL weight loading - using random initialization")
                print("  (Tier A has different architecture: deep encoder + attention)")
        else:
            # Fallback: try loading the entire state dict as-is
            try:
                missing_keys, unexpected_keys = model.patch_encoder.load_state_dict(state_dict, strict=False)
                print("✓ Stage 1 weights loaded into patch encoder with adaptations")
                if missing_keys or unexpected_keys:
                    print(f"  Adapted {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
            except RuntimeError as e:
                print(f"⚠ Could not load pre-trained weights: {e}")
                print("  Continuing with random initialization")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create learning rate scheduler: Warm-up + Cosine Annealing with Restarts
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    cosine_t0 = config['training'].get('cosine_t0', 10)
    cosine_tmult = config['training'].get('cosine_tmult', 2)
    cosine_eta_min = config['training'].get('eta_min', 1e-7)
    
    # Warm-up scheduler: linearly increase LR from 10% to 100%
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Cosine annealing with warm restarts
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cosine_t0,
        T_mult=cosine_tmult,
        eta_min=cosine_eta_min
    )
    
    # Combine schedulers: warmup first, then cosine
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    print(f"LR Schedule: {warmup_epochs} epochs warm-up + Cosine Annealing with Restarts")
    print(f"  Initial LR: {config['training']['lr'] * 0.1:.2e} → {config['training']['lr']:.2e}")
    print(f"  Min LR: 1e-7")
    print()
    
    # Checkpoint directory
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
        print("-" * 80)
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        val_metrics = validate(model, val_loader, device, config)
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch+1}] Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}  Acc: {train_metrics['acc']:.4f} ({train_metrics['acc']*100:.2f}%)")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}  Acc: {val_metrics['acc']:.4f} ({val_metrics['acc']*100:.2f}%)")
        print(f"  Learning Rate: {current_lr:.2e}")
        
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
            filename=f'moe_epoch_{epoch+1}.pt',
        )
        
        if is_best:
            print(f"  ✨ New best validation loss: {best_val_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
