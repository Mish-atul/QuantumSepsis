"""
QuantumSepsis Shield — Hierarchical LSTM with Spatial Attention
================================================================

Enhanced architecture inspired by the FOU paper (Wang et al., 2023):
  - Spatial attention mechanism on time series features
  - Hierarchical classification approach
  - Multimodal fusion (static + temporal features)
  - Layer-wise relevance propagation for interpretability

Key improvements over baseline:
  1. Spatial attention learns which features (HR, BP, etc.) are most important
  2. Hierarchical task decomposition (sepsis → severe sepsis → septic shock)
  3. Better static feature integration via dedicated pathway
  4. Improved handling of class imbalance through hierarchy

Architecture:
    Static pathway:  (batch, static_dim) → FC → (batch, 64)
    Temporal pathway: (batch, 6, 12) → SpatialAttention → GRU-D → (batch, 256)
    Fusion: Concat → FC → Hierarchical heads
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import LSTMConfig, get_default_config

logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """Spatial attention over time series features (FOU paper Section II.C.4).
    
    Learns which features (HR, BP, lactate, etc.) are most informative
    at each time step, similar to the attention module in the FOU paper.
    
    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, n_features) — feature-weighted input
            + attention_weights (batch, seq_len, n_features)
    """
    
    def __init__(self, n_features: int, attention_dim: int = 32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, n_features),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial attention.
        
        Args:
            x: (batch, seq_len, n_features)
        
        Returns:
            x_weighted: (batch, seq_len, n_features)
            weights: (batch, seq_len, n_features) — attention weights
        """
        # Compute attention scores for each feature at each time step
        scores = self.attention(x)  # (batch, seq_len, n_features)
        
        # Softmax over features dimension
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len, n_features)
        
        # Apply attention weights
        x_weighted = x * weights
        
        return x_weighted, weights


class TemporalAttention(nn.Module):
    """Temporal attention over time steps (original implementation)."""
    
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
    
    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention(lstm_output).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(
            weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        
        return context, weights


class HierarchicalSepsisLSTM(nn.Module):
    """Hierarchical LSTM with spatial + temporal attention.
    
    Implements a 3-level hierarchy:
        Level 1 (Root): Sepsis vs No Sepsis
        Level 2: Severe Sepsis vs Non-Severe
        Level 3: Septic Shock vs Non-Shock
    
    This decomposition helps with:
        - Class imbalance (each level is more balanced)
        - Interpretability (clinicians think hierarchically)
        - Performance (specialized classifiers per level)
    
    Args:
        config: LSTMConfig with model hyperparameters
        static_dim: Dimension of static features (demographics, labs)
    """
    
    def __init__(
        self,
        config: Optional[LSTMConfig] = None,
        static_dim: int = 0,
    ):
        super().__init__()
        
        if config is None:
            config = get_default_config().lstm
        
        self.config = config
        self.static_dim = static_dim
        
        # === Temporal Pathway ===
        
        # Input normalization
        self.layer_norm = nn.LayerNorm([config.seq_len, config.input_size])
        
        # Spatial attention (FOU paper innovation)
        self.spatial_attention = SpatialAttention(
            n_features=config.input_size,
            attention_dim=32,
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            attention_dim=config.attention_dim,
        )
        
        # === Static Pathway ===
        
        if static_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            fusion_dim = lstm_output_dim + 64
        else:
            self.static_encoder = None
            fusion_dim = lstm_output_dim
        
        # === Fusion and Embedding ===
        
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, config.fc1_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc1_dim, config.embedding_dim),
            nn.Tanh(),  # Bound to [-1, 1] for quantum encoding
        )
        
        # === Hierarchical Classification Heads ===
        
        # Level 1: Sepsis vs No Sepsis (root classifier)
        self.head_level1 = nn.Linear(config.embedding_dim, 1)
        
        # Level 2: Severe Sepsis vs Non-Severe (conditional on sepsis=1)
        self.head_level2 = nn.Linear(config.embedding_dim, 1)
        
        # Level 3: Septic Shock vs Non-Shock (conditional on severe=1)
        self.head_level3 = nn.Linear(config.embedding_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        for module in [self.projection, self.head_level1, self.head_level2, self.head_level3]:
            for m in module.modules() if hasattr(module, 'modules') else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_temporal: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
        return_attention: bool = False,
        level: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with hierarchical outputs.
        
        Args:
            x_temporal: (batch, 6, 12) temporal features
            x_static: (batch, static_dim) static features (optional)
            return_embedding: Include 16-dim embedding
            return_attention: Include attention weights
            level: Which hierarchical level to compute (1, 2, or 3)
        
        Returns:
            Dictionary with:
                'logits_l1': Level 1 logits (sepsis vs no sepsis)
                'logits_l2': Level 2 logits (severe vs non-severe)
                'logits_l3': Level 3 logits (shock vs non-shock)
                'risk_score_l1': Level 1 probability
                'risk_score_l2': Level 2 probability
                'risk_score_l3': Level 3 probability
                'embedding': 16-dim latent (if requested)
                'spatial_attention': Feature attention weights (if requested)
                'temporal_attention': Time attention weights (if requested)
        """
        batch_size = x_temporal.size(0)
        
        # === Temporal Pathway ===
        
        # Layer normalization
        x = self.layer_norm(x_temporal)
        
        # Spatial attention (learn which features matter)
        x, spatial_attn = self.spatial_attention(x)
        # x: (batch, 6, 12), spatial_attn: (batch, 6, 12)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, 6, 256)
        
        # Temporal attention (learn which time steps matter)
        temporal_context, temporal_attn = self.temporal_attention(lstm_out)
        # temporal_context: (batch, 256)
        
        # === Static Pathway ===
        
        if self.static_encoder is not None and x_static is not None:
            static_context = self.static_encoder(x_static)
            # static_context: (batch, 64)
            
            # Fusion
            context = torch.cat([temporal_context, static_context], dim=-1)
        else:
            context = temporal_context
        
        # === Embedding ===
        
        embedding = self.projection(context)
        # embedding: (batch, 16)
        
        # === Hierarchical Classification ===
        
        # Level 1: Sepsis detection
        logits_l1 = self.head_level1(embedding)
        risk_l1 = torch.sigmoid(logits_l1).squeeze(-1)
        
        output = {
            'logits_l1': logits_l1,
            'risk_score_l1': risk_l1,
            'logits': logits_l1,  # Backward compatibility
            'risk_score': risk_l1,
        }
        
        # Level 2: Severe sepsis (only if level >= 2)
        if level >= 2:
            logits_l2 = self.head_level2(embedding)
            risk_l2 = torch.sigmoid(logits_l2).squeeze(-1)
            output['logits_l2'] = logits_l2
            output['risk_score_l2'] = risk_l2
        
        # Level 3: Septic shock (only if level >= 3)
        if level >= 3:
            logits_l3 = self.head_level3(embedding)
            risk_l3 = torch.sigmoid(logits_l3).squeeze(-1)
            output['logits_l3'] = logits_l3
            output['risk_score_l3'] = risk_l3
        
        if return_embedding:
            output['embedding'] = embedding
        
        if return_attention:
            # Average spatial attention over time for interpretability
            output['spatial_attention'] = spatial_attn.mean(dim=1)  # (batch, 12)
            output['temporal_attention'] = temporal_attn  # (batch, 6)
        
        return output
    
    def extract_embeddings(
        self,
        x_temporal: torch.Tensor,
        x_static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract 16-dim embeddings for quantum kernel."""
        output = self.forward(
            x_temporal,
            x_static,
            return_embedding=True,
            level=1,
        )
        return output['embedding']
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return model summary string."""
        config = self.config
        lines = [
            f"HierarchicalSepsisLSTM Summary:",
            f"  Input (temporal):  ({config.seq_len}, {config.input_size})",
            f"  Input (static):    {self.static_dim}",
            f"  LSTM layers:       {config.n_layers}",
            f"  Hidden dim:        {config.hidden_dim}",
            f"  Bidirectional:     {config.bidirectional}",
            f"  Spatial attention: 32-dim",
            f"  Temporal attention: {config.attention_dim}-dim",
            f"  Embedding dim:     {config.embedding_dim}",
            f"  Hierarchical levels: 3",
            f"  Dropout:           {config.dropout}",
            f"  Total params:      {self.count_parameters():,}",
        ]
        return "\n".join(lines)


def test_hierarchical_model():
    """Test the hierarchical model with synthetic data."""
    config = get_default_config().lstm
    model = HierarchicalSepsisLSTM(config, static_dim=20)
    
    print(model.summary())
    print()
    
    # Test forward pass
    batch_size = 32
    x_temporal = torch.randn(batch_size, config.seq_len, config.input_size)
    x_static = torch.randn(batch_size, 20)
    
    # Test all levels
    output = model(
        x_temporal,
        x_static,
        return_embedding=True,
        return_attention=True,
        level=3,
    )
    
    print("Forward pass (all levels):")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: {tuple(value.shape)}")
    
    # Verify shapes
    assert output['logits_l1'].shape == (batch_size, 1)
    assert output['logits_l2'].shape == (batch_size, 1)
    assert output['logits_l3'].shape == (batch_size, 1)
    assert output['embedding'].shape == (batch_size, config.embedding_dim)
    assert output['spatial_attention'].shape == (batch_size, config.input_size)
    assert output['temporal_attention'].shape == (batch_size, config.seq_len)
    
    # Verify bounds
    assert output['embedding'].min() >= -1.0
    assert output['embedding'].max() <= 1.0
    assert output['risk_score_l1'].min() >= 0.0
    assert output['risk_score_l1'].max() <= 1.0
    
    # Verify attention sums
    spatial_sum = output['spatial_attention'].sum(dim=1)
    temporal_sum = output['temporal_attention'].sum(dim=1)
    assert torch.allclose(spatial_sum, torch.ones(batch_size), atol=1e-5)
    assert torch.allclose(temporal_sum, torch.ones(batch_size), atol=1e-5)
    
    print("\n✓ All checks passed!")
    
    # Test embedding extraction
    embeddings = model.extract_embeddings(x_temporal, x_static)
    print(f"\nEmbedding extraction: {embeddings.shape}")
    
    # Test gradient flow
    loss = output['logits_l1'].sum() + output['logits_l2'].sum() + output['logits_l3'].sum()
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"\nGradient flow: {len(grad_norms)} parameters have gradients")
    print(f"  Max grad norm: {max(grad_norms.values()):.4f}")
    print(f"  Min grad norm: {min(grad_norms.values()):.6f}")
    
    print("\n✓ Model test complete!")


if __name__ == "__main__":
    test_hierarchical_model()
