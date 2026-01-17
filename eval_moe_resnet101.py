#!/usr/bin/env python3
"""
Evaluate Stage 2 MoE (ResNet-101 backbone) on PTB-XL test split using the
same MoEClassifier architecture that was used for training in train_moe.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import create_logger, set_seed
from train_moe import MoEClassifier
from ptbxl_dataset import PTBXLDataset


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> MoEClassifier:
    """Load MoEClassifier with the same settings used in train_moe.py."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = MoEClassifier(
        patch_size=cfg["dataset"].get("patch_size", 64),
        embedding_dim=cfg["model"]["embedding_dim"],
        num_experts=cfg["model"]["num_experts"],
        num_classes=cfg["model"].get("num_classes", 5),
        patch_encoder_hidden=cfg["model"]["patch_encoder_hidden"],
        dropout=cfg["training"].get("dropout", 0.1),
        use_symbolic_gating=True,
        deep_encoder=cfg["model"].get("deep_encoder", False),
        use_attention=cfg["model"].get("use_attention", False),
        attention_heads=cfg["model"].get("attention_heads", 4),
        input_channels=cfg["model"].get("input_channels", 12),
        encoder_depth=cfg["model"].get("encoder_depth", "resnet18"),
        expert_hidden_dim=cfg["model"].get("expert_hidden_dim"),
        expert_type=cfg["model"].get("expert_type", "mlp"),
        num_subexperts=cfg["model"].get("num_subexperts", 4),
        top_k=cfg["model"].get("top_k", 1),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing={len(missing)} unexpected={len(unexpected)} when loading checkpoint")

    model.to(device)
    model.eval()
    return model


def evaluate(model: MoEClassifier, test_loader: DataLoader, device: torch.device, logger) -> dict:
    """Evaluate model on test set with multi-label AUROC."""
    all_probs = []
    all_labels = []
    all_routing = []
    all_acc = []

    logger.info("Evaluating on test set...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            patches = batch["patches"].to(device)
            labels = batch.get("labels", batch.get("label")).to(device)
            symbolic = batch.get("symbolic_features")
            if symbolic is not None:
                symbolic = symbolic.to(device)

            logits, routing, _, _ = model(patches, symbolic_features=symbolic)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            routing_np = routing.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels_np)
            all_routing.append(routing_np)
            preds = (probs > 0.5).astype(np.float32)
            all_acc.append((preds == labels_np).mean())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_routing = np.concatenate(all_routing, axis=0)
    mean_acc = float(np.mean(all_acc)) if all_acc else 0.0

    class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    auroc_per_class = []
    logger.info("\nPer-class AUROC:")
    for i, class_name in enumerate(class_names):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            logger.info(f"  {class_name}: {auroc:.4f}")
        except ValueError:
            auroc = 0.0
            logger.warning(f"  {class_name}: N/A (only one class present)")
        auroc_per_class.append(auroc)

    macro_auroc = float(np.mean([x for x in auroc_per_class if x > 0]))
    logger.info(f"\nMacro AUROC: {macro_auroc:.4f}")
    logger.info(f"Mean sample accuracy (@0.5): {mean_acc:.4f}")

    expert_usage = all_routing.mean(axis=0)
    expert_entropy = -np.sum(expert_usage * np.log(expert_usage + 1e-8))
    logger.info(f"\nExpert Routing Analysis:")
    logger.info(f"  Expert usage: {expert_usage}")
    logger.info(f"  Routing entropy: {expert_entropy:.4f}")

    return {
        "macro_auroc": macro_auroc,
        "per_class_auroc": dict(zip(class_names, auroc_per_class)),
        "mean_acc": mean_acc,
        "expert_usage": expert_usage,
        "routing_entropy": expert_entropy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MoE ResNet-101 on PTB-XL test set")
    parser.add_argument("--config", type=str, default="configs/stage2_moe_resnet101.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage2_moe_resnet101/best_model.pt")
    args = parser.parse_args()

    logger = create_logger("EvalMoE")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model(args.config, args.checkpoint, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    data_path = cfg["dataset"]["path"]
    batch_size = cfg["dataset"].get("batch_size", 32)
    num_workers = cfg["dataset"].get("num_workers", 4)

    test_dataset = PTBXLDataset(
        data_path=data_path,
        split="test",
        mode="finetune",
        patch_size=cfg["dataset"].get("patch_size", 64),
        seed=cfg["seed"],
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    logger.info(f"Test set size: {len(test_dataset)}")

    # Evaluate
    results = evaluate(model, test_loader, device, logger)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY - MoE ResNet-101")
    logger.info("=" * 60)
    logger.info(f"Macro AUROC: {results['macro_auroc']:.4f}")
    logger.info(f"Per-class AUROC: {results['per_class_auroc']}")
    logger.info(f"Routing Entropy: {results['routing_entropy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
