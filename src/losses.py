"""Loss functions for Puzzle-MoE."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rule_table_lookup(symbolic_features: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Generate a prior distribution from symbolic features.

    The heuristic maps HRV and QRS width proxies to expert preferences. This is
    a placeholder for domain rules and can be replaced by a clinical rule table.
    """

    hrv = symbolic_features[:, 0]
    qrs_width = symbolic_features[:, 1]
    prior = torch.zeros(symbolic_features.size(0), num_experts, device=symbolic_features.device)
    for i in range(num_experts):
        weight = torch.sigmoid(hrv * (i + 1)) * torch.exp(-qrs_width * abs(i - 1))
        prior[:, i] = weight
    prior = prior + 1e-6
    return prior / prior.sum(dim=1, keepdim=True)


def puzzle_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for permutation prediction."""

    return F.cross_entropy(logits, targets)


def symbolic_consistency_loss(gating: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """KL divergence between gating distribution and symbolic prior."""

    log_gating = torch.log(gating + 1e-8)
    return F.kl_div(log_gating, prior, reduction="batchmean")


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for reconstructing masked patches.
    
    Args:
        predictions: (batch, n_patches, patch_size) predicted patches
        targets: (batch, n_patches, patch_size) ground truth patches
        mask: (batch, n_patches) binary mask (1 = masked, 0 = not masked)
        
    Returns:
        Scalar loss averaged over masked positions only
    """
    # Compute MSE
    mse = F.mse_loss(predictions, targets, reduction='none')  # (B, n_patches, patch_size)
    
    # Average over patch dimension
    mse = mse.mean(dim=-1)  # (B, n_patches)
    
    # Apply mask and normalize
    mask = mask.float()
    masked_mse = (mse * mask).sum() / (mask.sum() + 1e-8)
    
    return masked_mse


def ssl_combined_loss(
    perm_logits: torch.Tensor,
    perm_targets: torch.Tensor,
    recon_predictions: torch.Tensor | None = None,
    recon_targets: torch.Tensor | None = None,
    recon_mask: torch.Tensor | None = None,
    lambda_recon: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined SSL loss: permutation + masked reconstruction.
    
    Args:
        perm_logits: (batch, n_permutations) logits for permutation classification
        perm_targets: (batch,) ground truth permutation indices
        recon_predictions: (batch, n_patches, patch_size) reconstructed patches
        recon_targets: (batch, n_patches, patch_size) original patches
        recon_mask: (batch, n_patches) binary mask
        lambda_recon: weight for reconstruction loss
        
    Returns:
        total_loss: scalar combined loss
        loss_dict: dictionary with individual loss components
    """
    # Permutation loss
    perm_loss = puzzle_loss(perm_logits, perm_targets)
    
    loss_dict = {
        'permutation': perm_loss,
    }
    
    total = perm_loss
    
    # Add reconstruction loss if available
    if recon_predictions is not None and recon_targets is not None and recon_mask is not None:
        recon_loss = masked_reconstruction_loss(recon_predictions, recon_targets, recon_mask)
        loss_dict['reconstruction'] = recon_loss
        total = total + lambda_recon * recon_loss
    
    loss_dict['total'] = total
    
    return total, loss_dict


def total_loss(
    class_logits: torch.Tensor,
    class_targets: torch.Tensor,
    gating_probs: torch.Tensor,
    symbolic_features: torch.Tensor,
    lambda_sym: float,
    ssl_logits: torch.Tensor | None = None,
    ssl_targets: torch.Tensor | None = None,
    lambda_ssl: float = 0.0,
) -> torch.Tensor:
    """Composite loss for MoE fine-tuning and optional SSL regularization."""

    ce = F.cross_entropy(class_logits, class_targets)
    prior = rule_table_lookup(symbolic_features, gating_probs.shape[1])
    sym = symbolic_consistency_loss(gating_probs, prior)
    loss = ce + lambda_sym * sym
    if ssl_logits is not None and ssl_targets is not None:
        loss = loss + lambda_ssl * puzzle_loss(ssl_logits, ssl_targets)
    return loss


__all__ = [
    "rule_table_lookup",
    "puzzle_loss",
    "symbolic_consistency_loss",
    "masked_reconstruction_loss",
    "ssl_combined_loss",
    "total_loss",
]
