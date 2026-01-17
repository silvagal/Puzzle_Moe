"""Training utilities for Puzzle-MoE."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import ECGDataset, load_ecg_dataset
from losses import puzzle_loss, total_loss
from models import PuzzleMoE
from utils import create_logger, expert_entropy, save_checkpoint, set_seed


@dataclass
class TrainingConfig:
    """Configuration for training stages."""

    experiment: str
    seed: int
    dataset: Dict[str, object]
    model: Dict[str, object]
    training: Dict[str, object]
    logging: Dict[str, object]


class Trainer:
    """Trainer handling SSL pre-training and MoE fine-tuning."""

    def __init__(self, config: TrainingConfig) -> None:
        set_seed(int(config.seed))
        self.config = config
        self.logger = create_logger("Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Using device %s", self.device)
        self.dataset_cfg = config.dataset
        self.model_cfg = config.model
        self.train_cfg = config.training
        self.log_cfg = config.logging

        # Determine backbone and input channels from config
        deep = self.model_cfg.get("deep_encoder", False)
        backbone = "resnet" if deep else "patch_encoder"
        # Allow explicit backbone override if needed
        if "backbone" in self.model_cfg:
            backbone = self.model_cfg["backbone"]

        input_channels = int(self.model_cfg.get("input_channels", 12))

        self.model = PuzzleMoE(
            embedding_dim=int(self.model_cfg.get("embedding_dim", 128)),
            hidden_dim=int(self.model_cfg.get("patch_encoder_hidden", 64)),
            num_experts=int(self.model_cfg.get("num_experts", 3)),
            num_classes=int(self.train_cfg.get("num_classes", 5)),
            puzzle_classes=int(self.train_cfg.get("puzzle_classes", 6)),
            backbone=backbone,
            input_channels=input_channels,
            physio_cfg=self.model_cfg.get("physio_cfg", None),
        ).to(self.device)

    def _dataloader(self, mode: str) -> DataLoader:
        path = self.dataset_cfg.get("path", "data/mock")
        kwargs = {k: v for k, v in self.dataset_cfg.items() if k != "path"}
        dataset = load_ecg_dataset(path, mode=mode, **kwargs)
        return DataLoader(dataset, batch_size=int(self.dataset_cfg.get("batch_size", 32)), num_workers=int(self.dataset_cfg.get("num_workers", 0)))

    def _optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters(), lr=float(self.train_cfg.get("lr", 1e-3)), weight_decay=float(self.train_cfg.get("weight_decay", 0.0)))

    def train_ssl(self) -> None:
        dataloader = self._dataloader(mode="ssl")
        optimizer = self._optimizer()
        epochs = int(self.train_cfg.get("epochs", 5))
        self.logger.info("Starting SSL pre-training for %d epochs", epochs)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(dataloader, desc=f"SSL Epoch {epoch+1}/{epochs}"):
                patches = batch["patches"].to(self.device)
                permutation = batch["permutation"].to(self.device)
                optimizer.zero_grad()
                logits = self.model.forward_ssl(patches)
                loss = puzzle_loss(logits, permutation)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.logger.info("SSL Epoch %d - Loss %.4f", epoch + 1, epoch_loss / len(dataloader))
            self._save(epoch, optimizer, stage="ssl")

    def train_moe(self) -> None:
        dataloader = self._dataloader(mode="moe")
        optimizer = self._optimizer()
        epochs = int(self.train_cfg.get("epochs", 5))
        lambda_sym = float(self.train_cfg.get("lambda_sym", 0.1))
        lambda_ssl = float(self.train_cfg.get("ssl_weight", 0.0))
        warmup_epochs = max(1, epochs // 3)
        self.logger.info("Starting MoE fine-tuning for %d epochs", epochs)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            entropy_vals = []
            current_lambda_sym = lambda_sym * min(1.0, (epoch + 1) / warmup_epochs)
            for batch in tqdm(dataloader, desc=f"MoE Epoch {epoch+1}/{epochs}"):
                patches = batch["patches"].to(self.device)
                labels = batch["label"].to(self.device)
                symbolic = batch["symbolic_features"].to(self.device)
                optimizer.zero_grad()
                class_logits, gating_probs = self.model.forward_moe(patches, symbolic)
                ssl_logits = None
                ssl_targets = None
                if lambda_ssl > 0.0:
                    ssl_logits = self.model.forward_ssl(patches)
                    ssl_targets = torch.zeros_like(labels)
                loss = total_loss(
                    class_logits=class_logits,
                    class_targets=labels,
                    gating_probs=gating_probs,
                    symbolic_features=symbolic,
                    lambda_sym=current_lambda_sym,
                    ssl_logits=ssl_logits,
                    ssl_targets=ssl_targets,
                    lambda_ssl=lambda_ssl,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                entropy_vals.append(expert_entropy(gating_probs).item())
            self.logger.info(
                "MoE Epoch %d - Loss %.4f - Expert Entropy %.4f - lambda_sym %.4f",
                epoch + 1,
                epoch_loss / len(dataloader),
                sum(entropy_vals) / len(entropy_vals),
                current_lambda_sym,
            )
            self._save(epoch, optimizer, stage="moe")

    def _save(self, epoch: int, optimizer: optim.Optimizer, stage: str) -> None:
        checkpoint_dir = self.log_cfg.get("checkpoint_dir", f"checkpoints/{stage}")
        filename = f"epoch_{epoch+1}.pt"
        save_checkpoint(
            state={"epoch": epoch + 1, "model_state": self.model.state_dict(), "optim_state": optimizer.state_dict()},
            checkpoint_dir=checkpoint_dir,
            filename=filename
        )


def load_config(path: str) -> TrainingConfig:
    """Load YAML configuration into a TrainingConfig dataclass."""

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TrainingConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Puzzle-MoE Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--stage", type=str, choices=["ssl", "moe"], required=True, help="Training stage")
    parser.add_argument("--subset_ratio", type=float, default=None, help="Optional subset ratio for data efficiency studies")
    parser.add_argument("--lambda_sym", type=float, default=None, help="Override symbolic consistency weight")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.subset_ratio is not None:
        cfg.dataset["subset_ratio"] = args.subset_ratio
    if args.lambda_sym is not None:
        cfg.training["lambda_sym"] = args.lambda_sym
    trainer = Trainer(cfg)
    if args.stage == "ssl":
        trainer.train_ssl()
    else:
        trainer.train_moe()


if __name__ == "__main__":
    main()
