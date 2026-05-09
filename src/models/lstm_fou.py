"""
FOU LSTM Model
==============
BiLSTM + Temporal Attention for multi-class FOU detection.

Architecture:
    Input: (batch, 24, 27)  # 24-hour window, 27 features
    → LayerNorm([24, 27])
    → BiLSTM(input=27, hidden=128, layers=2, dropout=0.3)
    → TemporalAttention(256, attn_dim=64)
    → FC(256 → 64, ReLU, Dropout)
    → FC(64 → 16, Tanh)  # Embedding for quantum kernel
    → FC(16 → 4, Softmax)  # 4-class output

Output:
    - logits: (batch, 4)
    - probabilities: (batch, 4)
    - embedding: (batch, 16)
    - attention_weights: (batch, 24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""

    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        attn_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores.squeeze(-1), dim=1)  # (batch, seq_len)

        # Compute weighted context
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)

        return context, attn_weights


class FouLSTM(nn.Module):
    """BiLSTM + Attention for FOU detection (multi-class)."""

    def __init__(
        self,
        input_size: int = 27,
        seq_len: int = 24,
        hidden_dim: int = 128,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        attention_dim: int = 64,
        fc1_dim: int = 64,
        embedding_dim: int = 16,
        n_classes: int = 4
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Layer normalization
        self.layer_norm = nn.LayerNorm([seq_len, input_size])

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0
        )

        # Attention
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(lstm_output_dim, attention_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, fc1_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc_embedding = nn.Linear(fc1_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            return_embedding: If True, return embedding for quantum kernel

        Returns:
            logits: (batch, n_classes)
            probs: (batch, n_classes)
            embedding: (batch, embedding_dim) if return_embedding else None
            attention_weights: (batch, seq_len)
        """
        # Layer normalization
        x = self.layer_norm(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)

        # Attention
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_dim * 2), (batch, seq_len)

        # FC layers
        h = F.relu(self.fc1(context))
        h = self.dropout1(h)

        # Embedding (for quantum kernel)
        embedding = torch.tanh(self.fc_embedding(h))  # (batch, embedding_dim)

        # Output layer
        logits = self.fc_out(embedding)  # (batch, n_classes)
        probs = F.softmax(logits, dim=1)  # (batch, n_classes)

        if return_embedding:
            return logits, probs, embedding, attn_weights
        else:
            return logits, probs, None, attn_weights

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for quantum kernel."""
        with torch.no_grad():
            _, _, embedding, _ = self.forward(x, return_embedding=True)
        return embedding


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test FOU LSTM
    print("=== FOU LSTM Test ===")

    model = FouLSTM(
        input_size=27,
        seq_len=24,
        hidden_dim=128,
        n_layers=2,
        bidirectional=True,
        dropout=0.3,
        attention_dim=64,
        fc1_dim=64,
        embedding_dim=16,
        n_classes=4
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 24, 27)

    logits, probs, embedding, attn_weights = model(x, return_embedding=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    print(f"\nProbs sum (should be ~1.0): {probs[0].sum().item():.4f}")
    print(f"Sample probs: {probs[0].tolist()}")

    print("\n✓ FOU LSTM test passed")
