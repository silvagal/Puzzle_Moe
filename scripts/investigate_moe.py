#!/usr/bin/env python3
"""
Investigation utilities for Puzzle-MoE.

Generates interpretability assets:
- Attention heatmaps (if attention aggregation is enabled)
- Gating analysis: per-class expert usage and histograms
- Noise probe: compare gating on clean vs noisy inputs

Outputs are saved in docs/figs/ for direct inclusion in the paper.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_moe import MoEClassifier  # type: ignore
from ptbxl_dataset import PTBXLDataset  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate MoE interpretability artifacts.")
    parser.add_argument("--checkpoint", required=True, help="Path to MoE checkpoint (.pt)")
    parser.add_argument("--data_path", required=True, help="PTB-XL processed data path")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for analysis")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--output_dir", default="docs/figs", help="Directory to save figures")
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config")
    if config is None:
        raise ValueError("Checkpoint missing config; cannot reconstruct model.")

    model = MoEClassifier(
        patch_size=64,
        embedding_dim=config["model"]["embedding_dim"],
        num_experts=config["model"]["num_experts"],
        num_classes=config["model"]["num_classes"],
        patch_encoder_hidden=config["model"]["patch_encoder_hidden"],
        dropout=config["training"].get("dropout", 0.1),
        use_symbolic_gating=True,
        deep_encoder=config["model"].get("deep_encoder", False),
        use_attention=config["model"].get("use_attention", False),
        attention_heads=config["model"].get("attention_heads", 4),
        input_channels=config["model"].get("input_channels", 12),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, config


def plot_attention(attn_weights: torch.Tensor, save_path: Path):
    """attn_weights: (batch, heads, tokens, tokens)"""
    batch_attn = attn_weights.mean(dim=1).cpu().numpy()  # (batch, tokens, tokens)
    fig, axes = plt.subplots(1, min(4, len(batch_attn)), figsize=(12, 3))
    if len(batch_attn) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(batch_attn[i], cmap="inferno", vmin=0, vmax=batch_attn[i].max())
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("Key tokens")
        ax.set_ylabel("Query tokens")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_gating_distribution(gating: torch.Tensor, labels: torch.Tensor, class_names: List[str], save_dir: Path):
    gating_np = gating.cpu().numpy()  # (batch, num_experts)
    labels_np = labels.cpu().numpy()  # (batch, num_classes)
    num_experts = gating_np.shape[1]

    # Histogram of gating weights
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(num_experts):
        ax.hist(gating_np[:, k], bins=30, alpha=0.6, label=f"Expert {k}")
    ax.set_title("Gating weight distribution (val batch)")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "gating_hist.png", dpi=200)
    plt.close(fig)

    # Class vs expert average weights
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(num_experts):
        cls_means = []
        for c_idx in range(labels_np.shape[1]):
            mask = labels_np[:, c_idx] == 1
            if mask.sum() == 0:
                cls_means.append(0.0)
            else:
                cls_means.append(gating_np[mask, k].mean())
        ax.plot(class_names, cls_means, marker="o", label=f"Expert {k}")
    ax.set_title("Mean gating weight per class (multi-label)")
    ax.set_ylabel("Mean weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "gating_class_means.png", dpi=200)
    plt.close(fig)


def noise_probe(model, batch, device, save_dir: Path):
    patches = batch["patches"].to(device)
    labels = batch["labels"].to(device)
    noise_std = 0.2

    def forward(p):
        logits, gating, _ = model(p, symbolic_features=None)
        return torch.sigmoid(logits), gating

    probs_clean, gating_clean = forward(patches)
    noisy = patches + noise_std * torch.randn_like(patches)
    probs_noisy, gating_noisy = forward(noisy)

    # Plot gating change per expert
    delta = (gating_noisy - gating_clean).mean(dim=0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(len(delta)), delta)
    ax.set_title("Gating shift under noise (mean across batch)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Δ weight (noisy - clean)")
    fig.tight_layout()
    fig.savefig(save_dir / "gating_noise_probe.png", dpi=200)
    plt.close(fig)

    # Optional: AUROC on the batch (macro)
    labels_np = labels.cpu().numpy()
    auroc = roc_auc_score(labels_np, probs_clean.detach().cpu().numpy(), average="macro")
    return {"batch_macro_auroc": auroc}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(Path(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Dataloader (no augmentation)
    ds = PTBXLDataset(
        data_path=args.data_path,
        split=args.split,
        mode="finetune",
        patch_size=64,
        transform=None,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Grab one batch for analysis
    batch = next(iter(loader))
    patches = batch["patches"].to(device)
    labels = batch["labels"]

    # Attention rollout (if available)
    if config["model"].get("use_attention", False):
        attn_weights = {}

        def hook(module, inp, out):
            # out is (attn_output, attn_weights)
            attn_output, attn_w = out
            attn_weights["weights"] = attn_w.detach()

        handle = model.attention_agg.attention.register_forward_hook(hook)
    else:
        attn_weights = None
        handle = None

    with torch.no_grad():
        logits, gating, _ = model(patches, symbolic_features=None)

    if handle:
        handle.remove()
        if "weights" in attn_weights:
            plot_attention(attn_weights["weights"], out_dir / "attention_heatmaps.png")

    # Gating analysis
    plot_gating_distribution(gating, labels, class_names=["NORM", "MI", "STTC", "CD", "HYP"], save_dir=out_dir)

    # Noise probe
    probe_metrics = noise_probe(model, batch, device, out_dir)
    with open(out_dir / "probe_metrics.txt", "w") as f:
        for k, v in probe_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"Figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
