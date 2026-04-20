"""
QuantumSepsis Shield — SepsisLSTM Model
========================================

Bidirectional LSTM with temporal attention for sepsis risk prediction.

Architecture:
    Input:  (batch, 6, 12)  — 6-hour window of 12 features
    →  LayerNorm
    →  Bidirectional LSTM (2 layers, hidden=128, dropout=0.3)
    →  Temporal Attention Pooling
    →  FC: 256 → 64 → 16 (Tanh)     ← Latent embedding for quantum kernel
    →  FC: 16 → 1 (Sigmoid)          ← Classical prediction head

The 16-dim embedding layer is the interface to the quantum kernel module.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import LSTMConfig, get_default_config

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Learnable attention mechanism over LSTM time steps.
    
    Instead of taking the last hidden state or mean-pooling,
    this module learns which time steps are most informative
    for sepsis prediction (e.g., weighting recent deterioration
    more than earlier stable periods).
    
    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim) — weighted sum over time steps
    """
    
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
        """Compute attention-weighted sum.
        
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            mask: Optional (batch, seq_len) boolean mask for padding
        
        Returns:
            context: (batch, hidden_dim) — weighted output
            weights: (batch, seq_len) — attention weights (sum to 1)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output             # (batch, seq_len, hidden_dim)
        ).squeeze(1)               # (batch, hidden_dim)
        
        return context, weights


class SepsisLSTM(nn.Module):
    """Bidirectional LSTM with temporal attention for sepsis prediction.
    
    Produces both:
        1. Classical risk prediction (sigmoid output)
        2. 16-dim latent embedding (for quantum kernel module)
    
    Args:
        config: LSTMConfig with model hyperparameters
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        
        if config is None:
            config = get_default_config().lstm
        
        self.config = config
        
        # Input normalization
        self.layer_norm = nn.LayerNorm([config.seq_len, config.input_size])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        
        # Compute output dimension after LSTM
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            attention_dim=config.attention_dim,
        )
        
        # Projection to latent embedding
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, config.fc1_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc1_dim, config.embedding_dim),
            nn.Tanh(),  # Bound to [-1, 1] for quantum encoding
        )
        
        # Classification head (for classical standalone + pre-training)
        self.classifier = nn.Linear(config.embedding_dim, config.n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers
        and orthogonal for LSTM."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with long-term memory)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        
        for module in [self.projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (batch, 6, 12) input tensor
            return_embedding: If True, include 16-dim embedding in output
            return_attention: If True, include attention weights in output
        
        Returns:
            Dictionary with keys:
                'logits': (batch, 1) raw classification logits
                'risk_score': (batch,) sigmoid probabilities
                'embedding': (batch, 16) latent embedding (if return_embedding)
                'attention_weights': (batch, 6) attention weights (if return_attention)
        """
        batch_size = x.size(0)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, 6, 256) for bidirectional
        
        # Temporal attention pooling
        context, attn_weights = self.attention(lstm_out)
        # context: (batch, 256)
        
        # Project to latent embedding
        embedding = self.projection(context)
        # embedding: (batch, 16)
        
        # Classification
        logits = self.classifier(embedding)
        # logits: (batch, 1)
        
        risk_score = torch.sigmoid(logits).squeeze(-1)
        
        output = {
            'logits': logits,
            'risk_score': risk_score,
        }
        
        if return_embedding:
            output['embedding'] = embedding
        
        if return_attention:
            output['attention_weights'] = attn_weights
        
        return output
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the 16-dim latent embedding (for quantum kernel).
        
        Args:
            x: (batch, 6, 12) input tensor
        
        Returns:
            (batch, 16) latent embedding bounded to [-1, 1]
        """
        output = self.forward(x, return_embedding=True)
        return output['embedding']
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Return model summary string."""
        config = self.config
        lines = [
            f"SepsisLSTM Summary:",
            f"  Input:          ({config.seq_len}, {config.input_size})",
            f"  LSTM layers:    {config.n_layers}",
            f"  Hidden dim:     {config.hidden_dim}",
            f"  Bidirectional:  {config.bidirectional}",
            f"  Attention dim:  {config.attention_dim}",
            f"  Embedding dim:  {config.embedding_dim}",
            f"  Dropout:        {config.dropout}",
            f"  Total params:   {self.count_parameters():,}",
        ]
        return "\n".join(lines)


def test_model():
    """Test the SepsisLSTM model with synthetic data."""
    config = get_default_config().lstm
    model = SepsisLSTM(config)
    
    print(model.summary())
    print()
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, config.seq_len, config.input_size)
    
    output = model(x, return_embedding=True, return_attention=True)
    
    print("Forward pass:")
    print(f"  logits:           {output['logits'].shape}")
    print(f"  risk_score:       {output['risk_score'].shape}")
    print(f"  embedding:        {output['embedding'].shape}")
    print(f"  attention_weights: {output['attention_weights'].shape}")
    
    # Verify shapes
    assert output['logits'].shape == (batch_size, 1)
    assert output['risk_score'].shape == (batch_size,)
    assert output['embedding'].shape == (batch_size, config.embedding_dim)
    assert output['attention_weights'].shape == (batch_size, config.seq_len)
    
    # Verify embedding bounds
    assert output['embedding'].min() >= -1.0
    assert output['embedding'].max() <= 1.0
    
    # Verify attention weights sum to 1
    attn_sum = output['attention_weights'].sum(dim=1)
    assert torch.allclose(attn_sum, torch.ones(batch_size), atol=1e-5)
    
    # Verify risk score bounds
    assert output['risk_score'].min() >= 0.0
    assert output['risk_score'].max() <= 1.0
    
    print("\n✓ All shape and bound checks passed!")
    
    # Test embedding extraction
    embeddings = model.extract_embeddings(x)
    print(f"\nEmbedding extraction: {embeddings.shape}")
    assert embeddings.shape == (batch_size, config.embedding_dim)
    
    # Test gradient flow
    loss = output['logits'].sum()
    loss.backward()
    
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"\nGradient flow check: {len(grad_norms)} parameters have gradients")
    print(f"  Max grad norm: {max(grad_norms.values()):.4f}")
    print(f"  Min grad norm: {min(grad_norms.values()):.6f}")
    
    print("\n✓ Model test complete!")


if __name__ == "__main__":
    test_model()
