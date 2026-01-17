#!/usr/bin/env python3
"""
Comprehensive Model Interpretability Analysis for Puzzle-MoE.

This script generates all interpretability artifacts for the paper:
1. Attention Heatmaps: Visualize which patches the model focuses on
2. Expert Specialization Matrix: Show which experts handle which pathologies
3. Gating Distribution Analysis: Understand routing behavior
4. Symbolic Correlation: Verify symbolic features influence routing
5. Noise Robustness Probe: Test expert behavior under noise

Usage:
    python analyze_model_interpretability.py \
        --checkpoint checkpoints/stage2_moe/best_model.pt \
        --config configs/stage2_moe.yaml \
        --data_path ecg_ssl_project/data/processed/ptbxl/fs500/superclasses \
        --output_dir plots/interpretability
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repository src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_moe import MoEClassifier
from ptbxl_dataset import PTBXLDataset


class InterpretabilityAnalyzer:
    """Comprehensive interpretability analysis for Puzzle-MoE."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        class_names: List[str],
        output_dir: Path,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for collected data
        self.all_gating = []
        self.all_labels = []
        self.all_symbolic = []
        self.all_attention = []
        self.all_predictions = []
        
    def collect_activations(self, num_batches: int = None):
        """Collect model activations for analysis."""
        print("📊 Collecting model activations...")
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                if num_batches is not None and num_batches > 0 and batch_idx >= num_batches:
                    break
                    
                patches = batch['patches'].to(self.device)
                labels = batch['labels'].to(self.device)
                symbolic = batch.get('symbolic_features')
                if symbolic is not None:
                    symbolic = symbolic.to(self.device)
                
                # Forward pass (assuming model returns logits, gating, expert_logits, attn_weights)
                outputs = self.model(patches, symbolic_features=symbolic)
                
                # Unpack based on what the model returns
                if len(outputs) == 4:
                    logits, gating, expert_logits, attn_weights = outputs
                else:
                    logits, gating = outputs[:2]
                    attn_weights = None
                    
                # Store
                self.all_gating.append(gating.cpu())
                self.all_labels.append(labels.cpu())
                if symbolic is not None:
                    self.all_symbolic.append(symbolic.cpu())
                if attn_weights is not None:
                    self.all_attention.append(attn_weights.cpu())
                self.all_predictions.append(torch.sigmoid(logits).cpu())
        
        # Concatenate
        self.all_gating = torch.cat(self.all_gating, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        self.all_predictions = torch.cat(self.all_predictions, dim=0)
        if self.all_symbolic:
            self.all_symbolic = torch.cat(self.all_symbolic, dim=0)
        
        print(f"✅ Collected {len(self.all_gating)} samples")
        
    def plot_expert_specialization(self):
        """Generate expert specialization heatmap."""
        print("🎨 Generating expert specialization heatmap...")
        
        num_experts = self.all_gating.shape[1]
        expert_usage = np.zeros((len(self.class_names), num_experts))
        expert_counts = np.zeros((len(self.class_names), num_experts))
        
        # For each class, compute mean gating weight per expert
        for cls_idx, cls_name in enumerate(self.class_names):
            mask = self.all_labels[:, cls_idx] == 1
            if mask.sum() > 0:
                # Average gating weights for samples of this class
                expert_usage[cls_idx] = self.all_gating[mask].mean(dim=0).numpy()
                
                # Also count which expert was most used
                top_experts = self.all_gating[mask].argmax(dim=1)
                for exp_idx in range(num_experts):
                    expert_counts[cls_idx, exp_idx] = (top_experts == exp_idx).sum().item()
        
        # Plot 1: Mean gating weights
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        im1 = ax1.imshow(expert_usage, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(num_experts))
        ax1.set_yticks(range(len(self.class_names)))
        ax1.set_xticklabels([f'Expert {i+1}' for i in range(num_experts)])
        ax1.set_yticklabels(self.class_names)
        ax1.set_title('Mean Gating Weight per Class')
        ax1.set_xlabel('Expert')
        ax1.set_ylabel('Pathology')
        
        # Add values
        for i in range(len(self.class_names)):
            for j in range(num_experts):
                text = ax1.text(j, i, f'{expert_usage[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot 2: Count of samples routed to each expert
        expert_counts_norm = expert_counts / expert_counts.sum(axis=1, keepdims=True)
        im2 = ax2.imshow(expert_counts_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(num_experts))
        ax2.set_yticks(range(len(self.class_names)))
        ax2.set_xticklabels([f'Expert {i+1}' for i in range(num_experts)])
        ax2.set_yticklabels(self.class_names)
        ax2.set_title('Primary Expert Selection (Fraction)')
        ax2.set_xlabel('Expert')
        
        for i in range(len(self.class_names)):
            for j in range(num_experts):
                text = ax2.text(j, i, f'{expert_counts_norm[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_specialization.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'expert_specialization.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved to {self.output_dir / 'expert_specialization.png'}")
        
    def plot_gating_distributions(self):
        """Plot gating weight distributions."""
        print("📈 Plotting gating distributions...")
        
        num_experts = self.all_gating.shape[1]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Overall distribution per expert
        ax = axes[0, 0]
        for exp_idx in range(num_experts):
            ax.hist(self.all_gating[:, exp_idx].numpy(), bins=50, alpha=0.6, 
                   label=f'Expert {exp_idx+1}', density=True)
        ax.set_xlabel('Gating Weight')
        ax.set_ylabel('Density')
        ax.set_title('Overall Gating Weight Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Expert entropy over samples
        ax = axes[0, 1]
        eps = 1e-8
        probs = self.all_gating.clamp(min=eps)
        entropy = -(probs * torch.log(probs)).sum(dim=1).numpy()
        ax.hist(entropy, bins=50, color='teal', alpha=0.7, edgecolor='black')
        ax.axvline(entropy.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {entropy.mean():.2f}')
        ax.set_xlabel('Expert Entropy (per sample)')
        ax.set_ylabel('Count')
        ax.set_title('Routing Diversity Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Per-class gating boxplot
        ax = axes[1, 0]
        gating_by_class = []
        for cls_idx in range(len(self.class_names)):
            mask = self.all_labels[:, cls_idx] == 1
            if mask.sum() > 0:
                # Get primary expert for each sample
                primary_expert = self.all_gating[mask].argmax(dim=1).numpy()
                gating_by_class.append(primary_expert)
            else:
                gating_by_class.append(np.array([]))
        
        ax.boxplot([g for g in gating_by_class if len(g) > 0], 
                   tick_labels=[self.class_names[i] for i, g in enumerate(gating_by_class) if len(g) > 0])
        ax.set_ylabel('Primary Expert Index')
        ax.set_title('Primary Expert Selection by Class')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Correlation matrix between experts
        ax = axes[1, 1]
        gating_np = self.all_gating.numpy()
        corr_matrix = np.corrcoef(gating_np.T)
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(num_experts))
        ax.set_yticks(range(num_experts))
        ax.set_xticklabels([f'E{i+1}' for i in range(num_experts)])
        ax.set_yticklabels([f'E{i+1}' for i in range(num_experts)])
        ax.set_title('Expert Weight Correlation')
        
        for i in range(num_experts):
            for j in range(num_experts):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gating_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved to {self.output_dir / 'gating_distributions.png'}")
        
    def analyze_symbolic_correlation(self):
        """Analyze correlation between symbolic features and expert routing."""
        if len(self.all_symbolic) == 0:
            print("⚠️  No symbolic features available, skipping correlation analysis")
            return
            
        print("🔬 Analyzing symbolic feature correlation with routing...")
        
        symbolic_np = self.all_symbolic.numpy()
        gating_np = self.all_gating.numpy()
        
        num_experts = gating_np.shape[1]
        num_features = symbolic_np.shape[1]
        
        # Feature names (update based on your actual features)
        # Dynamically generate names if we don't know them
        default_feature_names = ['RR Mean', 'RR Std', 'QRS Width', 'QT Interval', 'P Amplitude', 'T Amplitude']
        feature_names = default_feature_names[:num_features] if num_features <= len(default_feature_names) else [f'Feature {i+1}' for i in range(num_features)]
        
        # Compute correlations
        correlations = np.zeros((num_features, num_experts))
        p_values = np.zeros((num_features, num_experts))
        
        for feat_idx in range(num_features):
            for exp_idx in range(num_experts):
                corr, pval = pearsonr(symbolic_np[:, feat_idx], gating_np[:, exp_idx])
                correlations[feat_idx, exp_idx] = corr
                p_values[feat_idx, exp_idx] = pval
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Correlation values
        im1 = ax1.imshow(correlations, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        ax1.set_xticks(range(num_experts))
        ax1.set_yticks(range(num_features))
        ax1.set_xticklabels([f'Expert {i+1}' for i in range(num_experts)])
        ax1.set_yticklabels(feature_names)
        ax1.set_title('Symbolic Feature → Expert Correlation (Pearson)')
        ax1.set_xlabel('Expert')
        ax1.set_ylabel('Symbolic Feature')
        
        # Add values and significance markers
        for i in range(num_features):
            for j in range(num_experts):
                sig_marker = '***' if p_values[i, j] < 0.001 else ('**' if p_values[i, j] < 0.01 else ('*' if p_values[i, j] < 0.05 else ''))
                text = ax1.text(j, i, f'{correlations[i, j]:.2f}{sig_marker}',
                              ha="center", va="center", color="black", fontsize=8)
        
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Significance p-values (log scale)
        im2 = ax2.imshow(-np.log10(p_values + 1e-10), cmap='Greens', aspect='auto')
        ax2.set_xticks(range(num_experts))
        ax2.set_yticks(range(num_features))
        ax2.set_xticklabels([f'Expert {i+1}' for i in range(num_experts)])
        ax2.set_yticklabels(feature_names)
        ax2.set_title('Statistical Significance (-log10 p-value)')
        ax2.set_xlabel('Expert')
        
        fig.colorbar(im2, ax=ax2, label='-log10(p)', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'symbolic_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save to text file
        with open(self.output_dir / 'symbolic_correlation.txt', 'w') as f:
            f.write("Symbolic Feature → Expert Routing Correlation\n")
            f.write("=" * 60 + "\n\n")
            for feat_idx in range(num_features):
                f.write(f"{feature_names[feat_idx]}:\n")
                for exp_idx in range(num_experts):
                    sig = "***" if p_values[feat_idx, exp_idx] < 0.001 else ("**" if p_values[feat_idx, exp_idx] < 0.01 else ("*" if p_values[feat_idx, exp_idx] < 0.05 else "ns"))
                    f.write(f"  Expert {exp_idx+1}: r={correlations[feat_idx, exp_idx]:.3f}, p={p_values[feat_idx, exp_idx]:.4f} {sig}\n")
                f.write("\n")
        
        print(f"✅ Saved to {self.output_dir / 'symbolic_correlation.png'}")
        
    def plot_attention_examples(self, num_samples: int = 5):
        """Plot attention heatmaps for sample ECGs."""
        if len(self.all_attention) == 0:
            print("⚠️  No attention weights available, skipping attention visualization")
            return
            
        print(f"🎨 Plotting attention examples for {num_samples} samples...")
        
        # Get first batch for visualization
        sample_batch = next(iter(self.dataloader))
        patches = sample_batch['patches'][:num_samples].to(self.device)
        labels = sample_batch['labels'][:num_samples]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(patches, symbolic_features=None)
            if len(outputs) >= 4:
                _, _, _, attn_weights = outputs
            else:
                print("⚠️  Model does not return attention weights")
                return
        
        if attn_weights is None:
            print("⚠️  Attention weights are None")
            return
            
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        # seq_len = num_patches + 1 (CLS token)
        
        for sample_idx in range(num_samples):
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(f'Sample {sample_idx} - Attention Analysis\nLabels: {[self.class_names[i] for i, v in enumerate(labels[sample_idx]) if v == 1]}', 
                        fontsize=14)
            
            # Get attention from CLS token to patches
            # attn_weights[sample_idx]: (num_heads, seq_len, seq_len)
            attn = attn_weights[sample_idx].cpu().numpy()
            
            # Average over heads
            attn_avg = attn.mean(axis=0)  # (seq_len, seq_len)
            cls_to_patches = attn_avg[0, 1:]  # Attention from CLS to 9 patches
            
            # Normalize
            cls_to_patches = (cls_to_patches - cls_to_patches.min()) / (cls_to_patches.max() - cls_to_patches.min() + 1e-8)
            
            patch_names = ['Beat 1 P', 'Beat 1 QRS', 'Beat 1 T',
                          'Beat 2 P', 'Beat 2 QRS', 'Beat 2 T',
                          'Beat 3 P', 'Beat 3 QRS', 'Beat 3 T']
            
            # Plot each patch (Lead II for visualization)
            lead_idx = 1
            patches_np = patches[sample_idx].cpu().numpy()  # (9, 12, 64)
            
            for patch_idx, ax in enumerate(axes.flat):
                if patch_idx < 9:
                    signal = patches_np[patch_idx, lead_idx, :]
                    attn_val = cls_to_patches[patch_idx]
                    
                    # Color background by attention
                    ax.set_facecolor(plt.cm.Reds(attn_val * 0.4))
                    
                    ax.plot(signal, color='darkblue', linewidth=1.5)
                    ax.set_title(f'{patch_names[patch_idx]}\nAttn: {attn_val:.3f}', fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'attention_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Saved attention plots to {self.output_dir}")
        
    def noise_robustness_probe(self, noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5]):
        """Test expert routing behavior under different noise levels."""
        print("🔊 Running noise robustness probe...")
        
        # Get a clean batch
        sample_batch = next(iter(self.dataloader))
        patches_clean = sample_batch['patches'][:32].to(self.device)
        
        num_experts = self.all_gating.shape[1]
        noise_results = {noise: [] for noise in noise_levels}
        
        self.model.eval()
        with torch.no_grad():
            for noise_std in noise_levels:
                if noise_std == 0.0:
                    patches = patches_clean
                else:
                    patches = patches_clean + noise_std * torch.randn_like(patches_clean)
                
                outputs = self.model(patches, symbolic_features=None)
                gating = outputs[1] if len(outputs) >= 2 else None
                
                if gating is not None:
                    noise_results[noise_std] = gating.cpu().mean(dim=0).numpy()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Expert usage vs noise
        expert_usage = np.array([noise_results[n] for n in noise_levels])
        for exp_idx in range(num_experts):
            ax1.plot(noise_levels, expert_usage[:, exp_idx], marker='o', label=f'Expert {exp_idx+1}', linewidth=2)
        
        ax1.set_xlabel('Noise STD')
        ax1.set_ylabel('Mean Gating Weight')
        ax1.set_title('Expert Routing vs. Noise Level')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Change from baseline
        baseline = expert_usage[0]
        delta = expert_usage - baseline
        
        for exp_idx in range(num_experts):
            ax2.plot(noise_levels[1:], delta[1:, exp_idx], marker='s', label=f'Expert {exp_idx+1}', linewidth=2)
        
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Noise STD')
        ax2.set_ylabel('Δ Gating Weight (from clean)')
        ax2.set_title('Routing Shift Under Noise')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved to {self.output_dir / 'noise_robustness.png'}")
        
    def generate_report(self):
        """Generate a comprehensive text report."""
        print("📝 Generating analysis report...")
        
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PUZZLE-MOE INTERPRETABILITY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write(f"Dataset: {len(self.all_labels)} samples\n")
            f.write(f"Classes: {self.class_names}\n")
            f.write(f"Number of Experts: {self.all_gating.shape[1]}\n\n")
            
            # Class distribution
            f.write("Class Distribution:\n")
            for cls_idx, cls_name in enumerate(self.class_names):
                count = self.all_labels[:, cls_idx].sum().item()
                pct = 100 * count / len(self.all_labels)
                f.write(f"  {cls_name}: {count} ({pct:.1f}%)\n")
            f.write("\n")
            
            # Expert utilization
            f.write("Expert Utilization (mean gating weight):\n")
            mean_gating = self.all_gating.mean(dim=0).numpy()
            for exp_idx, weight in enumerate(mean_gating):
                f.write(f"  Expert {exp_idx+1}: {weight:.4f}\n")
            f.write("\n")
            
            # Routing diversity
            eps = 1e-8
            probs = self.all_gating.clamp(min=eps)
            entropy = -(probs * torch.log(probs)).sum(dim=1)
            f.write(f"Routing Entropy: {entropy.mean():.4f} ± {entropy.std():.4f}\n")
            f.write(f"  (Higher entropy = more diverse routing, max = {np.log(self.all_gating.shape[1]):.2f})\n\n")
            
            # Expert collapse check
            primary_experts = self.all_gating.argmax(dim=1)
            expert_usage_counts = [(primary_experts == i).sum().item() for i in range(self.all_gating.shape[1])]
            f.write("Primary Expert Selection Counts:\n")
            for exp_idx, count in enumerate(expert_usage_counts):
                pct = 100 * count / len(self.all_labels)
                f.write(f"  Expert {exp_idx+1}: {count} samples ({pct:.1f}%)\n")
            f.write("\n")
            
            # Check for collapse
            min_usage_pct = min(expert_usage_counts) / len(self.all_labels) * 100
            if min_usage_pct < 5:
                f.write("⚠️  WARNING: Expert collapse detected! ")
                f.write(f"Expert {expert_usage_counts.index(min(expert_usage_counts))+1} is underutilized ({min_usage_pct:.1f}%)\n")
            else:
                f.write(f"✅ No expert collapse detected (min usage: {min_usage_pct:.1f}%)\n")
            f.write("\n")
            
            # Per-class expert preference
            f.write("Dominant Expert per Class:\n")
            for cls_idx, cls_name in enumerate(self.class_names):
                mask = self.all_labels[:, cls_idx] == 1
                if mask.sum() > 0:
                    class_gating = self.all_gating[mask].mean(dim=0)
                    dominant_expert = class_gating.argmax().item() + 1
                    confidence = class_gating.max().item()
                    f.write(f"  {cls_name}: Expert {dominant_expert} (weight: {confidence:.3f})\n")
            
        print(f"✅ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive interpretability analysis for Puzzle-MoE")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_path', type=str, required=True, help='Path to PTB-XL data')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='plots/interpretability')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_batches', type=int, default=None, help='Limit number of batches for faster analysis')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = MoEClassifier(
        patch_size=64,
        embedding_dim=config['model']['embedding_dim'],
        num_experts=config['model']['num_experts'],
        num_classes=5,
        patch_encoder_hidden=config['model']['patch_encoder_hidden'],
        dropout=config['training'].get('dropout', 0.1),
        use_symbolic_gating=config['training'].get('use_symbolic_gating', True),
        deep_encoder=config['model'].get('deep_encoder', False),
        use_attention=config['model'].get('use_attention', False),
        attention_heads=config['model'].get('attention_heads', 4),
        input_channels=config['model'].get('input_channels', 12),
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    
    # Load data
    print(f"Loading {args.split} data...")
    dataset = PTBXLDataset(
        data_path=args.data_path,
        split=args.split,
        mode='finetune',
        patch_size=64,
        seed=config.get('seed', 42),
        transform=None  # No augmentation for analysis
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Create analyzer
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    output_dir = Path(args.output_dir)
    
    analyzer = InterpretabilityAnalyzer(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        output_dir=output_dir,
    )
    
    # Run all analyses
    print("\n" + "="*80)
    print("STARTING INTERPRETABILITY ANALYSIS")
    print("="*80 + "\n")
    
    analyzer.collect_activations(num_batches=args.num_batches)
    analyzer.plot_expert_specialization()
    analyzer.plot_gating_distributions()
    analyzer.analyze_symbolic_correlation()
    analyzer.plot_attention_examples(num_samples=5)
    analyzer.noise_robustness_probe()
    analyzer.generate_report()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print("="*80 + "\n")
    
    print("Generated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
