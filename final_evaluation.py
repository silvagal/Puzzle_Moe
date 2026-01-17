#!/usr/bin/env python3
"""
Final evaluation of all experimental variants.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

project_dir = "/path/to/puzzle-moe"
sys.path.append(project_dir)

from src.ptbxl_dataset import PTBXLDataset
from train_moe import MoEClassifier

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(checkpoint_path, config):
    """Load model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MoEClassifier(
        patch_encoder_hidden=config['model']['patch_encoder_hidden'],
        embedding_dim=config['model']['embedding_dim'],
        num_experts=config['model']['num_experts'],
        num_classes=config['model']['num_classes'],
        deep_encoder=config['model'].get('deep_encoder', False),
        use_attention=config['model'].get('use_attention', False),
        attention_heads=config['model'].get('attention_heads', 4),
        input_channels=config['model'].get('input_channels', 12),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if 'patches' in batch:
                ecg = batch['patches'].to(device)
            elif 'ecg' in batch:
                ecg = batch['ecg'].to(device)
            else:
                raise KeyError(f"Unexpected batch keys: {batch.keys()}")
            
            if 'labels' in batch:
                labels = batch['labels'].to(device)
            elif 'label' in batch:
                labels = batch['label'].to(device)
            else:
                raise KeyError(f"No label key found in batch: {batch.keys()}")
            
            if labels.dim() > 1 and labels.shape[1] > 1:
                labels = labels.argmax(dim=1)
            
            symbolic = batch.get('symbolic_features', None)
            if symbolic is not None:
                symbolic = symbolic.to(device)
            
            output = model(ecg, symbolic_features=symbolic)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    num_classes = all_probs.shape[1]
    aurocs = []
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    for i in range(num_classes):
        y_true = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        auroc = roc_auc_score(y_true, y_score)
        aurocs.append(auroc)
    
    macro_auroc = np.mean(aurocs)
    
    return macro_auroc, aurocs, class_names

def main():
    print("=" * 80)
    print("FINAL EVALUATION - ALL EXPERIMENTAL VARIANTS")
    print("=" * 80)
    
    base_dir = Path('/path/to/puzzle-moe')
    
    # All models to evaluate
    models_to_eval = [
        {
            'name': 'Baseline: Original (4-experts, λ_LB=0.01, 50 epochs)',
            'checkpoint': str(base_dir / 'checkpoints/stage2_moe/best_model.pt'),
            'config': str(base_dir / 'configs/stage2_moe_original.yaml'),
        },
        {
            'name': 'Step 1: Fine-tune (3-experts, λ_LB=0.005)',
            'checkpoint': str(base_dir / 'checkpoints/stage2_moe_lb005/best_model.pt'),
            'config': str(base_dir / 'configs/stage2_moe_lb005.yaml'),
        },
        {
            'name': 'Step 2: 4-Experts Architecture',
            'checkpoint': str(base_dir / 'checkpoints/stage2_moe_4experts/best_model.pt'),
            'config': str(base_dir / 'configs/stage2_moe_4experts.yaml'),
        },
        {
            'name': 'Step 4: No SSL Pre-training (Ablation)',
            'checkpoint': str(base_dir / 'checkpoints/stage2_moe_no_ssl/best_model.pt'),
            'config': str(base_dir / 'configs/stage2_moe_no_ssl.yaml'),
        },
    ]
    
    print("\nLoading test dataset...")
    test_dataset = PTBXLDataset(
        data_path='/path/to/ptbxl/processed',
        split='test',
        mode='moe',
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_dataset)} samples\n")
    
    results = []
    
    for model_info in models_to_eval:
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_info['name']}")
        print(f"{'=' * 80}")
        
        if not Path(model_info['checkpoint']).exists():
            print(f"❌ Checkpoint not found: {model_info['checkpoint']}")
            continue
        
        if not Path(model_info['config']).exists():
            print(f"❌ Config not found: {model_info['config']}")
            continue
        
        config = load_config(model_info['config'])
        model, device = load_model(model_info['checkpoint'], config)
        
        print(f"✓ Model loaded ({config['model']['num_experts']} experts)\n")
        
        macro_auroc, aurocs, class_names = evaluate_model(model, test_loader, device)
        
        results.append({
            'name': model_info['name'],
            'macro_auroc': macro_auroc,
            'aurocs': dict(zip(class_names, aurocs))
        })
        
        print(f"Macro AUROC: {macro_auroc:.4f}")
        print(f"\nPer-class AUROC:")
        for name, auroc in zip(class_names, aurocs):
            print(f"  {name:6s}: {auroc:.4f}")
    
    # Summary table
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY - ALL EXPERIMENTS")
    print(f"{'=' * 90}\n")
    
    print(f"{'Model':<55} {'Macro':>8} {'NORM':>8} {'MI':>8} {'STTC':>8} {'CD':>8} {'HYP':>8}")
    print(f"{'-' * 90}")
    
    for result in results:
        print(f"{result['name']:<55} {result['macro_auroc']:>8.4f}", end='')
        for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
            print(f" {result['aurocs'][cls]:>8.4f}", end='')
        print()
    
    print(f"{'=' * 90}\n")
    
    # Findings
    print("KEY FINDINGS:")
    print("-" * 90)
    
    # Best overall
    best_result = max(results, key=lambda x: x['macro_auroc'])
    print(f"🏆 Best Overall: {best_result['name']}")
    print(f"   Macro AUROC: {best_result['macro_auroc']:.4f}")
    
    # Best per class
    print(f"\n🎯 Best per class:")
    for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
        best_for_class = max(results, key=lambda x: x['aurocs'][cls])
        print(f"   {cls:6s}: {best_for_class['aurocs'][cls]:.4f} ({best_for_class['name'][:40]}...)")
    
    # SSL ablation
    print(f"\n🔬 SSL Pre-training Impact:")
    baseline = next((r for r in results if 'Baseline' in r['name']), None)
    no_ssl = next((r for r in results if 'No SSL' in r['name']), None)
    if baseline and no_ssl:
        diff = (baseline['macro_auroc'] - no_ssl['macro_auroc']) * 100
        print(f"   With SSL:    {baseline['macro_auroc']:.4f}")
        print(f"   Without SSL: {no_ssl['macro_auroc']:.4f}")
        print(f"   Improvement: +{diff:.2f}% (SSL helps!)" if diff > 0 else f"   Degradation: {diff:.2f}%")
    
    print("-" * 90)
    
    # Save results
    output_file = str(base_dir / 'outputs/final_evaluation_summary.txt')
    with open(output_file, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("FINAL EVALUATION - ALL EXPERIMENTAL VARIANTS\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"{'Model':<55} {'Macro':>8} {'NORM':>8} {'MI':>8} {'STTC':>8} {'CD':>8} {'HYP':>8}\n")
        f.write("-" * 90 + "\n")
        for result in results:
            f.write(f"{result['name']:<55} {result['macro_auroc']:>8.4f}")
            for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
                f.write(f" {result['aurocs'][cls]:>8.4f}")
            f.write("\n")
        
        f.write("\n" + "=" * 90 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Best Overall: {best_result['name']}\n")
        f.write(f"Macro AUROC: {best_result['macro_auroc']:.4f}\n\n")
        
        f.write("Best per class:\n")
        for cls in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
            best_for_class = max(results, key=lambda x: x['aurocs'][cls])
            f.write(f"  {cls:6s}: {best_for_class['aurocs'][cls]:.4f}\n")
    
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == '__main__':
    main()
