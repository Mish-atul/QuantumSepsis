"""
QuantumSepsis Shield — Enhanced LSTM V2 Model
===============================================

Multi-Scale Temporal Fusion architecture with:
  - Multi-scale 1D convolutions (kernel sizes 2, 3, 5) to capture short/medium/long patterns
  - Channel Attention (SE-block style) to learn feature importance
  - Bidirectional LSTM for sequential modeling
  - Multi-head temporal attention (4 heads) for richer time-step weighting
  - Residual connections for gradient flow
  - 16-dim Tanh embedding for quantum kernel compatibility

Input:  (batch, 6, 33)  — 6-hour window × 33 enriched features
Output: risk_score ∈ [0, 1], embedding ∈ [-1, 1]^16

Expected AUROC improvement: +0.03–0.06 over vanilla BiLSTM.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import get_default_config

logger = logging.getLogger(__name__)


# ── Multi-Scale Temporal Convolution Block ─────────────────────────────────────

class MultiScaleConv(nn.Module):
    """Parallel 1D convolutions at multiple kernel sizes.

    Captures temporal patterns at different scales (2h, 3h, 5h) and
    concatenates the outputs.  Each branch produces `out_channels` features.
    """

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.out_channels = out_channels * 3  # 96 default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features) → (batch, seq_len, out_channels*3)."""
        # Conv1d expects (batch, channels, seq_len)
        x_t = x.transpose(1, 2)

        c2 = self.conv2(x_t)[:, :, :x.size(1)]  # trim to seq_len
        c3 = self.conv3(x_t)[:, :, :x.size(1)]
        c5 = self.conv5(x_t)[:, :, :x.size(1)]

        out = torch.cat([c2, c3, c5], dim=1)  # (batch, 96, seq_len)
        out = self.bn(out)
        out = F.gelu(out)
        return out.transpose(1, 2)  # (batch, seq_len, 96)


# ── Channel (Feature) Attention ────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.

    Learns which feature channels are most important for sepsis prediction.
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, channels) → same shape, channel-weighted."""
        # Global average pooling over time
        gap = x.mean(dim=1)  # (batch, channels)
        weights = self.fc(gap).unsqueeze(1)  # (batch, 1, channels)
        return x * weights


# ── Multi-Head Temporal Attention ──────────────────────────────────────────────

class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention over time steps (not self-attention).

    Each head learns a different aspect of temporal importance.
    Final output is the concatenation of per-head weighted sums, projected back.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, attn_dim: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, attn_dim),
                nn.Tanh(),
                nn.Linear(attn_dim, 1, bias=False),
            )
            for _ in range(n_heads)
        ])
        self.projection = nn.Linear(hidden_dim * n_heads, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, hidden_dim) → (batch, hidden_dim), (batch, n_heads, seq_len)."""
        contexts = []
        all_weights = []

        for head in self.heads:
            scores = head(x).squeeze(-1)  # (batch, seq_len)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
            contexts.append(context)
            all_weights.append(weights)

        # Concatenate heads and project
        multi = torch.cat(contexts, dim=-1)  # (batch, hidden_dim * n_heads)
        output = self.projection(multi)       # (batch, hidden_dim)
        output = self.layer_norm(output)

        weights_tensor = torch.stack(all_weights, dim=1)  # (batch, n_heads, seq_len)

        return output, weights_tensor


# ── Main V2 Model ─────────────────────────────────────────────────────────────

class SepsisLSTMv2(nn.Module):
    """Enhanced multi-scale temporal fusion model for sepsis prediction.

    Architecture:
        Input (batch, 6, 33)
        → MultiScaleConv(33 → 96)
        → ChannelAttention(96)
        → Residual: concat [conv_out, raw_projected] → 128
        → LayerNorm
        → BiLSTM(128, hidden=128, 2 layers, dropout=0.3) → 256
        → MultiHeadTemporalAttention(256, 4 heads) → 256
        → FC: 256 → 128 → ReLU → Dropout
        → FC: 128 → 16 → Tanh  (quantum embedding)
        → FC: 16 → 1 → Sigmoid (risk score)
    """

    def __init__(
        self,
        input_size: int = 33,
        seq_len: int = 6,
        conv_channels: int = 32,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # ── Multi-scale convolution ────────────────────────────────────────
        self.multi_conv = MultiScaleConv(input_size, conv_channels)
        conv_out_dim = self.multi_conv.out_channels  # 96

        # ── Channel attention ──────────────────────────────────────────────
        self.channel_attn = ChannelAttention(conv_out_dim, reduction=4)

        # ── Residual projection (raw features → same dim as conv output) ──
        self.raw_proj = nn.Linear(input_size, conv_out_dim)

        # ── Pre-LSTM layer norm ────────────────────────────────────────────
        self.pre_lstm_norm = nn.LayerNorm(conv_out_dim)

        # ── Bidirectional LSTM ─────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2  # 256

        # ── Multi-head temporal attention ──────────────────────────────────
        self.temporal_attn = MultiHeadTemporalAttention(
            lstm_out_dim, n_heads=n_heads, attn_dim=64
        )

        # ── Projection to embedding ───────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),  # Bounded [-1, 1] for quantum encoding
        )

        # ── Classification head ───────────────────────────────────────────
        self.classifier = nn.Linear(embedding_dim, 1)

        # ── Weight initialization ─────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        for module in [self.projection, self.classifier, self.raw_proj]:
            for m in module.modules() if isinstance(module, nn.Sequential) else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 6, 33) enriched features

        Returns:
            dict with logits, risk_score, embedding, attention_weights
        """
        # ── Multi-scale convolution ────────────────────────────────────────
        conv_out = self.multi_conv(x)            # (batch, 6, 96)
        conv_out = self.channel_attn(conv_out)   # (batch, 6, 96)

        # ── Residual connection with projected raw input ───────────────────
        raw_proj = self.raw_proj(x)              # (batch, 6, 96)
        fused = conv_out + raw_proj              # (batch, 6, 96)
        fused = self.pre_lstm_norm(fused)

        # ── BiLSTM ─────────────────────────────────────────────────────────
        lstm_out, _ = self.lstm(fused)           # (batch, 6, 256)

        # ── Multi-head temporal attention ──────────────────────────────────
        context, attn_weights = self.temporal_attn(lstm_out)  # (batch, 256)

        # ── Embedding + classifier ─────────────────────────────────────────
        embedding = self.projection(context)      # (batch, 16)
        logits = self.classifier(embedding)       # (batch, 1)
        risk_score = torch.sigmoid(logits).squeeze(-1)

        return {
            "logits": logits.squeeze(-1),
            "risk_score": risk_score,
            "embedding": embedding,
            "attention_weights": attn_weights,
        }

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only the 16-dim embedding for quantum kernel."""
        with torch.no_grad():
            return self.forward(x)["embedding"]

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"SepsisLSTMv2(\n"
            f"  input={self.input_size}, seq={self.seq_len}\n"
            f"  MultiScaleConv → ChannelAttn → BiLSTM({self.hidden_dim}×2)\n"
            f"  MultiHeadTemporalAttn(4 heads) → Embed({self.embedding_dim})\n"
            f"  Parameters: {total:,} total, {trainable:,} trainable\n"
            f")"
        )


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = SepsisLSTMv2(input_size=33, seq_len=6)
    print(model.summary())

    x = torch.randn(4, 6, 33)
    out = model(x)
    print(f"logits:     {out['logits'].shape}")
    print(f"risk_score: {out['risk_score'].shape}")
    print(f"embedding:  {out['embedding'].shape}")
    print(f"attn:       {out['attention_weights'].shape}")

    emb = model.extract_embeddings(x)
    print(f"embeddings: {emb.shape}, range=[{emb.min():.3f}, {emb.max():.3f}]")
    print("\n✓ SepsisLSTMv2 smoke test passed!")
