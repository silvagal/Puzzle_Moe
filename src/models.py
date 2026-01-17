"""Model architectures for Puzzle-MoE."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    """Learnable spectral tokenizer implemented with Sinc filters.

    This layer parameterizes lower cutoff and bandwidth instead of raw weights,
    encouraging band-pass filters that align with ECG physiology. The filters are
    generated on the fly via ``torch.sinc`` and applied with ``conv1d``.

    Args:
        in_channels: Number of input channels (typically 1 for ECG).
        out_channels: Number of learnable band-pass filters.
        kernel_size: Length of each filter; must be odd for symmetry.
        min_low_hz: Minimum low cutoff to keep filters above 0 Hz.
        min_band_hz: Minimum bandwidth to avoid degenerate filters.
        sample_rate: Sampling frequency of the signal for proper scaling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        min_low_hz: float = 0.5,
        min_band_hz: float = 60.0,
        sample_rate: float = 250.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric Sinc filters")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.sample_rate = sample_rate

        low_hz_init = torch.linspace(self.min_low_hz, self.min_band_hz, out_channels)
        band_hz_init = torch.full((out_channels,), self.min_band_hz / out_channels)

        self.low_hz_ = nn.Parameter(low_hz_init)
        self.band_hz_ = nn.Parameter(band_hz_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable band-pass filters.

        Args:
            x: Input tensor of shape ``(batch, channels, time)``.

        Returns:
            Spectral token embeddings of shape ``(batch, out_channels, time)``.
        """

        device = x.device
        dtype = x.dtype

        f_low = self.min_low_hz + torch.abs(self.low_hz_)
        f_high = torch.clamp(
            f_low + self.min_band_hz + torch.abs(self.band_hz_),
            max=self.sample_rate / 2 - 1e-3,
        )

        n_lin = torch.linspace(
            -(self.kernel_size // 2), self.kernel_size // 2, steps=self.kernel_size, device=device, dtype=dtype
        )
        t = n_lin / self.sample_rate
        t = torch.where(t == 0, torch.tensor(1e-6, device=device, dtype=dtype), t)

        window = torch.hamming_window(self.kernel_size, periodic=False, device=device, dtype=dtype)

        band_pass_list = []
        for low, high in zip(f_low, f_high):
            high_pass = 2 * high * torch.sinc(2 * torch.pi * high * t)
            low_pass = 2 * low * torch.sinc(2 * torch.pi * low * t)
            band = (high_pass - low_pass) * window
            band_pass_list.append(band)

        filters = torch.stack(band_pass_list).view(self.out_channels, 1, self.kernel_size)
        if self.in_channels > 1:
            filters = filters.repeat(1, self.in_channels, 1)

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, groups=1)


class CPE(nn.Module):
    """Conditional Positional Encoding via depthwise convolution.

    Adds elasticity to token positions by mixing local neighborhoods, which helps
    handle variable heart rates and temporal stretching.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conditional positional encoding.

        Args:
            x: Input tensor of shape ``(batch, dim, seq_len)``.

        Returns:
            Tensor with positional conditioning preserved.
        """

        return x + self.dwconv(x)


class PhysioTransformer(nn.Module):
    """Transformer backbone with Sinc tokenization and CPE positional bias.

    Acts as a physiologically-aware alternative to agnostic backbones. Signals are
    first tokenized by a learnable spectral layer, enhanced with depthwise
    convolutional positional encoding, and then modeled by a Transformer encoder.
    """

    def __init__(
        self,
        sinc_conv_cfg: dict,
        transformer_depth: int = 4,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.tokenizer = SincConv1d(**sinc_conv_cfg)
        embedding_dim = sinc_conv_cfg["out_channels"]
        self.cpe = CPE(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physiologically-aware embeddings.

        Args:
            x: Input tensor shaped ``(batch, channels, time)`` or
                ``(batch, num_patches, patch_len)``.

        Returns:
            Aggregated token embeddings of shape ``(batch, embedding_dim)``.
        """

        if x.dim() == 3 and x.size(1) != self.tokenizer.in_channels:
            x = x.view(x.size(0), self.tokenizer.in_channels, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        tokens = self.tokenizer(x)
        tokens = self.cpe(tokens)
        seq_first = tokens.permute(2, 0, 1)
        encoded = self.transformer(seq_first)
        pooled = encoded.mean(dim=0)
        return pooled


class PatchEncoder(nn.Module):
    """Lightweight 1D CNN encoder for semantic ECG patches."""

    def __init__(self, hidden_dim: int = 64, embedding_dim: int = 128, deep: bool = False, input_channels: int = 12) -> None:
        super().__init__()
        
        self.deep = deep
        self.input_channels = input_channels
        
        if deep:
            # Tier B: ResNet-18 Backbone for high capacity
            # Uses [2, 2, 2, 2] blocks structure
            self.backbone = ResNet1d(BasicBlock1d, [2, 2, 2, 2], input_channels=input_channels, num_classes=embedding_dim)
        else:
            # Original lightweight encoder (updated for multi-lead)
            self.conv = nn.Sequential(
                nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Expanded to 9 patches (3 beats × P/QRS/T).
        self.positional = nn.Parameter(torch.randn(9, embedding_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches.

        Args:
            patches: Tensor of shape ``(batch, num_patches, channels, patch_len)`` 
                     OR ``(batch, num_patches, patch_len)`` (legacy single channel).

        Returns:
            Patch embeddings of shape ``(batch, num_patches, embedding_dim)``.
        """

        bsz, num_patches = patches.shape[:2]
        
        # Handle input shape
        if patches.dim() == 3:
            # Legacy single channel: (batch, num_patches, patch_len)
            # Reshape to (batch*num_patches, 1, patch_len)
            x = patches.view(bsz * num_patches, 1, -1)
            # If model expects 12 channels, we must repeat or fail.
            # Assuming legacy model expects 1 channel.
            if self.input_channels > 1:
                 # Replicate to match input channels (fallback)
                 x = x.repeat(1, self.input_channels, 1)
        else:
            # Multi-channel: (batch, num_patches, channels, patch_len)
            # Reshape to (batch*num_patches, channels, patch_len)
            channels = patches.shape[2]
            patch_len = patches.shape[3]
            x = patches.view(bsz * num_patches, channels, patch_len)
        
        if self.deep:
            # ResNet backbone returns (bsz*num_patches, embedding_dim)
            features = self.backbone(x)
            projected = features.view(bsz, num_patches, -1)
        else:
            # Original encoder uses mean pooling
            features = self.conv(x).mean(dim=2)  # (bsz*num_patches, hidden_dim)
            projected = self.proj(features).view(bsz, num_patches, -1)
        
        positional = self.positional[:num_patches].unsqueeze(0)
        return projected + positional


class AttentionAggregation(nn.Module):
    """Multi-head attention aggregation for patch embeddings (Tier A improvement).
    
    Replaces simple mean pooling with learned attention mechanism.
    Uses a CLS token (like BERT) to aggregate information from all patches.
    """
    
    def __init__(self, embedding_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, patch_embeddings: torch.Tensor, return_weights: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate patch embeddings using attention.
        
        Args:
            patch_embeddings: (batch, num_patches, embedding_dim)
            return_weights: If True, returns attention weights
            
        Returns:
            aggregated: (batch, embedding_dim)
            weights: (batch, num_heads, num_patches+1, num_patches+1) if return_weights=True
        """
        batch_size = patch_embeddings.size(0)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (batch, num_patches+1, embedding_dim)
        
        # Self-attention
        # need_weights=True is default, but we need to ensure we get them
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=False)
        
        # Normalize and return CLS token embedding
        aggregated = self.norm(attn_out[:, 0, :])  # (batch, embedding_dim)
        
        if return_weights:
            return aggregated, attn_weights
            
        return aggregated


class SymbolicGating(nn.Module):
    """Symbolic gating network combining neural embeddings and features."""

    def __init__(self, embedding_dim: int, num_experts: int, symbolic_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + symbolic_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_experts),
        )

    def forward(self, embedding: torch.Tensor, symbolic_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([embedding, symbolic_features], dim=1)
        logits = self.net(fused)
        return F.softmax(logits, dim=1)


class Expert(nn.Module):
    """Simple expert head for classification."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(embedding)


class PuzzleClassifier(nn.Module):
    """Classifier head for permutation prediction in SSL."""
    
    def __init__(self, embedding_dim: int, num_classes: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            embedding: (batch, embedding_dim) aggregated embedding
            
        Returns:
            logits: (batch, num_classes) classification logits
        """
        return self.mlp(embedding)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    """ResNet-18 1D for ECG."""

    def __init__(self, block, layers, input_channels=12, num_classes=128):
        super(ResNet1d, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PuzzleMoE(nn.Module):
    """Main Puzzle-MoE model supporting SSL and MoE forward passes."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_experts: int = 3,
        num_classes: int = 5,
        puzzle_classes: int = 6,
        backbone: str = "patch_encoder",
        physio_cfg: dict | None = None,
        input_channels: int = 12,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone
        if backbone == "physio_transformer":
            if physio_cfg is None:
                raise ValueError("physio_cfg must be provided for physio_transformer backbone")
            self.encoder = PhysioTransformer(
                sinc_conv_cfg=physio_cfg.get(
                    "sinc_conv",
                    {
                        "in_channels": input_channels,
                        "out_channels": embedding_dim,
                        "kernel_size": 129,
                        "min_low_hz": 0.5,
                        "min_band_hz": 60.0,
                    },
                ),
                transformer_depth=physio_cfg.get("transformer_depth", 4),
                transformer_heads=physio_cfg.get("transformer_heads", 4),
            )
        else:
            # Check if we want deep encoder (ResNet)
            deep = (backbone == "resnet" or backbone == "deep")
            self.encoder = PatchEncoder(
                hidden_dim=hidden_dim, 
                embedding_dim=embedding_dim, 
                deep=deep,
                input_channels=input_channels
            )

        self.gating = SymbolicGating(embedding_dim=embedding_dim, num_experts=num_experts)
        self.experts = nn.ModuleList([Expert(embedding_dim, num_classes) for _ in range(num_experts)])
        self.puzzle_head = nn.Linear(embedding_dim * 3, puzzle_classes)

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches with the selected backbone."""

        if self.backbone_type == "physio_transformer":
            if patches.dim() == 4:
                # (bsz, num_patches, channels, patch_len)
                bsz, num_patches, channels, patch_len = patches.shape
                flat = patches.view(bsz * num_patches, channels, patch_len)
            else:
                # (bsz, num_patches, patch_len)
                bsz, num_patches, patch_len = patches.shape
                flat = patches.view(bsz * num_patches, 1, patch_len)
            
            encoded = self.encoder(flat).view(bsz, num_patches, -1)
            return encoded
        return self.encoder(patches)

    def _aggregate_embedding(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate patch embeddings by averaging over patches."""

        return patch_embeddings.mean(dim=1)

    def forward_ssl(self, patches: torch.Tensor) -> torch.Tensor:
        """Forward pass for the permutation prediction task."""

        patch_embeddings = self._encode_patches(patches)
        concatenated = patch_embeddings.reshape(patches.size(0), -1)
        return self.puzzle_head(concatenated)

    def forward_moe(self, patches: torch.Tensor, symbolic_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for classification with MoE routing."""

        patch_embeddings = self._encode_patches(patches)
        aggregated = self._aggregate_embedding(patch_embeddings)
        gating_probs = self.gating(aggregated, symbolic_features)
        expert_outputs: List[torch.Tensor] = [expert(aggregated) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)
        mixed = torch.sum(gating_probs.unsqueeze(-1) * stacked, dim=1)
        return mixed, gating_probs


__all__ = [
    "SincConv1d",
    "CPE",
    "PhysioTransformer",
    "PatchEncoder",
    "SymbolicGating",
    "Expert",
    "PuzzleClassifier",
    "PuzzleMoE",
    "BasicBlock1d",
    "ResNet1d",
]
