"""Evaluate baseline models on PTB-XL test set."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ptbxl_dataset import PTBXLDataset
from baselines import Ribeiro2020, Strodthoff2020


def evaluate(config_path, checkpoint_path):
    """Evaluate model on test set."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    data_path = Path(config['dataset']['path'])
    test_dataset = PTBXLDataset(
        data_path=data_path,
        split='test',
        mode='finetune',
        patch_size=64,
        seed=config['seed'],
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset'].get('num_workers', 0),
        pin_memory=True,
    )
    
    print(f"Test: {len(test_dataset)} samples")
    
    # Load model
    model_name = config['model']['name']
    input_channels = config['model'].get('input_channels', 12)
    
    if model_name == "ribeiro":
        model = Ribeiro2020(input_channels=input_channels, num_classes=5)
    elif model_name == "strodthoff":
        model = Strodthoff2020(input_channels=input_channels, num_classes=5)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x = batch['signal'].to(device)
            y = batch['labels'].cpu().numpy()
            
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_preds.append(probs)
            all_labels.append(y)
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Compute metrics
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    # Per-class AUROC
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    aucs = []
    for i, name in enumerate(class_names):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
            print(f"{name:6s} AUROC: {auc:.4f}")
        else:
            print(f"{name:6s} AUROC: N/A (single class)")
    
    # Macro AUROC
    macro_auc = np.mean(aucs)
    print(f"\nMacro AUROC: {macro_auc:.4f}")
    
    # Accuracy (using threshold 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    print(f"Accuracy: {acc:.4f}")
    
    # F1 Score (macro)
    f1 = f1_score(all_labels, preds_binary, average='macro')
    print(f"F1 Score (macro): {f1:.4f}")
    
    print("="*60)
    
    return {
        'macro_auroc': macro_auc,
        'per_class_auroc': dict(zip(class_names, aucs)),
        'accuracy': acc,
        'f1_macro': f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()
    
    evaluate(args.config, args.checkpoint)
