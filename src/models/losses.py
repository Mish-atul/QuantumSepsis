"""
QuantumSepsis Shield — Asymmetric Focal Loss
==============================================

Custom loss function for sepsis detection with:
  - Focal loss for hard example mining (γ = 2.0)
  - Asymmetric class weighting: FN penalty = 10× FP penalty
  - α_pos = 0.9, α_neg = 0.1

This ensures the model prioritizes detecting sepsis (minimizing false negatives)
at the cost of more false positives — clinically appropriate because a missed
sepsis case is far more dangerous than an unnecessary workup.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for binary classification.
    
    L(p, y) = -α_t · (1 - p_t)^γ · log(p_t)
    
    Where:
        - For y=1 (sepsis):     α_t = alpha_pos, penalizes FN heavily
        - For y=0 (non-sepsis): α_t = alpha_neg
        - p_t = p if y=1, else (1-p)
        - γ (gamma) = focusing parameter for hard examples
    
    With alpha_pos=0.9 and alpha_neg=0.1:
        - FN loss contribution ≈ 9× FP loss contribution
        - Combined with class frequency, effective FN:FP penalty ratio ≈ 10:1
    
    Args:
        alpha_pos: Weight for positive class (sepsis). Default 0.9.
        alpha_neg: Weight for negative class (non-sepsis). Default 0.1.
        gamma: Focusing parameter. Default 2.0.
        reduction: 'mean', 'sum', or 'none'. Default 'mean'.
    """
    
    def __init__(
        self,
        alpha_pos: float = 0.9,
        alpha_neg: float = 0.1,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric focal loss.
        
        Args:
            logits: Raw model output (before sigmoid), shape (N,) or (N, 1)
            targets: Binary labels, shape (N,) or (N, 1)
        
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        # Ensure correct shape
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Compute BCE loss per sample (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute p_t (probability assigned to the true class)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t (class-dependent weight)
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        
        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma
        
        # Final loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
    
    def __repr__(self) -> str:
        return (
            f"AsymmetricFocalLoss("
            f"α_pos={self.alpha_pos}, α_neg={self.alpha_neg}, "
            f"γ={self.gamma}, reduction='{self.reduction}')"
        )


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross-Entropy Loss as a simpler alternative.
    
    Uses pos_weight to directly penalize false negatives.
    
    Args:
        pos_weight: Weight for positive class. Default 10.0 (FN = 10× FP).
    """
    
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        
        # Move pos_weight to correct device
        pos_weight = self.pos_weight.to(logits.device)
        
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )


def verify_asymmetric_loss():
    """Verify the asymmetric property of the loss function."""
    loss_fn = AsymmetricFocalLoss(alpha_pos=0.9, alpha_neg=0.1, gamma=2.0)
    
    # Case 1: False Negative (sepsis patient predicted as non-sepsis)
    logit_fn = torch.tensor([-2.0])   # Model says "not sepsis" (low probability)
    target_fn = torch.tensor([1.0])    # Truth: sepsis
    loss_fn_val = loss_fn(logit_fn, target_fn)
    
    # Case 2: False Positive (non-sepsis patient predicted as sepsis)
    logit_fp = torch.tensor([2.0])     # Model says "sepsis" (high probability)
    target_fp = torch.tensor([0.0])    # Truth: not sepsis
    loss_fp_val = loss_fn(logit_fp, target_fp)
    
    # Case 3: True Positive (correct sepsis prediction)
    logit_tp = torch.tensor([2.0])
    target_tp = torch.tensor([1.0])
    loss_tp_val = loss_fn(logit_tp, target_tp)
    
    # Case 4: True Negative (correct non-sepsis prediction)
    logit_tn = torch.tensor([-2.0])
    target_tn = torch.tensor([0.0])
    loss_tn_val = loss_fn(logit_tn, target_tn)
    
    print("Asymmetric Focal Loss Verification:")
    print(f"  False Negative loss: {loss_fn_val.item():.4f}")
    print(f"  False Positive loss: {loss_fp_val.item():.4f}")
    print(f"  True Positive loss:  {loss_tp_val.item():.4f}")
    print(f"  True Negative loss:  {loss_tn_val.item():.4f}")
    print(f"  FN/FP ratio:         {loss_fn_val.item() / loss_fp_val.item():.1f}×")
    
    assert loss_fn_val > loss_fp_val, "FN loss should be > FP loss!"
    assert loss_fn_val > loss_tp_val, "FN loss should be > TP loss!"
    print("\n[OK] Asymmetric property verified: FN >> FP")


if __name__ == "__main__":
    verify_asymmetric_loss()
