"""
Particle Transformer for Jet Classification
=============================================
Based on: Qu et al., "Particle Transformer for Jet Tagging" ICML 2022

Architecture:
  - Self-attention over particle constituents
  - Pairwise interaction features injected into attention weights
  - Class token aggregation for classification

Physics motivation:
  The Transformer's attention mechanism naturally captures pairwise
  particle correlations. In QCD, the radiation pattern is governed by
  the angular ordering of emissions — the attention weights learn to
  focus on physically meaningful pairs (e.g., particles from the same
  prong of a top quark decay).

  The pairwise features (ΔR, kT, z, m_ij) encode the physics of:
  - Angular separation (ΔR) → jet clustering metric
  - Transverse momentum scale (kT) → QCD splitting function
  - Momentum sharing (z) → collinear radiation pattern
  - Invariant mass (m_ij) → resonance structure (W mass, top mass)

Key innovation of Particle Transformer:
  Standard Transformers use dot-product attention: a_ij = q_i · k_j
  Particle Transformer adds interaction terms:
    a_ij = q_i · k_j + MLP(pair_features_ij)
  This allows the model to directly use physics-motivated pairwise
  information, rather than learning it from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PARTFORMER_CONFIG


class PairwiseInteraction(nn.Module):
    """
    MLP that maps pairwise features to attention bias.

    Takes physics-motivated pair features (ΔR, kT, z, m_ij) and produces
    a scalar bias for each attention head, which is added to the
    dot-product attention logits.
    """

    def __init__(self, pair_input_dim, num_heads):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_heads),
        )

    def forward(self, pair_features):
        """
        Args:
            pair_features: (batch, N, N, pair_dim)

        Returns:
            bias: (batch, num_heads, N, N)
        """
        bias = self.mlp(pair_features)  # (B, N, N, num_heads)
        return bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional pairwise interaction bias.

    Standard attention: Attention(Q, K, V) = softmax(QK^T/√d) V
    With pair bias:     Attention(Q, K, V) = softmax(QK^T/√d + B) V

    The bias B comes from the pairwise interaction features, allowing
    the attention to be informed by physics-motivated particle pair
    properties.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, pair_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.pair_interaction = None
        if pair_dim is not None:
            self.pair_interaction = PairwiseInteraction(pair_dim, num_heads)

    def forward(self, x, mask=None, pair_features=None):
        """
        Args:
            x: (batch, N, embed_dim)
            mask: (batch, N) boolean mask
            pair_features: (batch, N, N, pair_dim) optional

        Returns:
            out: (batch, N, embed_dim)
            attn_weights: (batch, num_heads, N, N)
        """
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # Project to Q, K, V
        Q = self.W_q(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        K = self.W_k(x).view(B, N, H, d).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, d).transpose(1, 2)

        # Scaled dot-product attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, N, N)

        # Add pairwise interaction bias (key innovation of Particle Transformer)
        if self.pair_interaction is not None and pair_features is not None:
            pair_bias = self.pair_interaction(pair_features)  # (B, H, N, N)
            attn_logits = attn_logits + pair_bias

        # Apply mask: set padded positions to -inf
        if mask is not None:
            # mask: (B, N) → (B, 1, 1, N) for broadcasting. The dataset
            # supplies the mask as float (1.0 = real particle, 0.0 = pad),
            # so use an equality test rather than bitwise-NOT (~), which
            # only works on bool/int tensors.
            attn_mask = (mask == 0).unsqueeze(1).unsqueeze(2)  # True where padded
            attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)

        return out, attn_weights


class TransformerBlock(nn.Module):
    """
    Transformer encoder block: Attention → Add&Norm → FFN → Add&Norm
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, pair_dim=None):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout, pair_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, pair_features=None):
        """
        Returns:
            x: (batch, N, embed_dim) updated features
            attn_weights: (batch, num_heads, N, N) attention map
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, mask, pair_features)
        x = self.norm1(x + attn_out)

        # Feedforward with residual
        x = self.norm2(x + self.ffn(x))

        # Mask padded positions
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return x, attn_weights


class ParticleTransformer(nn.Module):
    """
    Particle Transformer for jet classification.

    Architecture:
      Input embedding → [TransformerBlock × N_layers] → Class token → FC → logit

    The class [CLS] token is a learnable embedding prepended to the
    particle sequence. After the transformer layers, its representation
    encodes information about the entire jet and is used for classification.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or PARTFORMER_CONFIG
        embed_dim = cfg["embed_dim"]
        num_heads = cfg["num_heads"]
        num_layers = cfg["num_layers"]
        ff_dim = cfg["ff_dim"]
        dropout = cfg["dropout"]
        input_features = cfg["input_features"]
        pair_features = cfg["pair_features"]

        # Input embedding: project particle features to embed_dim
        self.input_embed = nn.Sequential(
            nn.Linear(input_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, ff_dim, dropout,
                pair_dim=pair_features if i == 0 else None  # Pair features only in first layer (efficiency)
            )
            for i in range(num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following the original transformer paper."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, mask, pair_features=None):
        """
        Args:
            features: (batch, N, input_dim) particle features
            mask: (batch, N) boolean mask
            pair_features: (batch, N, N, pair_dim) optional

        Returns:
            logits: (batch,) classification logits
            attention_maps: list of (batch, num_heads, N+1, N+1) attention weights
        """
        B, N, _ = features.shape

        # Embed particles
        x = self.input_embed(features)  # (B, N, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, embed_dim)

        # Extend mask for [CLS] token (always valid)
        cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
        full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, N+1)

        # Extend pair features for [CLS] (pad with zeros)
        full_pair = None
        if pair_features is not None:
            # Pad pair features: (B, N, N, D) → (B, N+1, N+1, D)
            D = pair_features.shape[-1]
            full_pair = torch.zeros(B, N + 1, N + 1, D, device=pair_features.device)
            full_pair[:, 1:, 1:, :] = pair_features

        # Apply transformer blocks
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x, full_mask, full_pair)
            attention_maps.append(attn)
            full_pair = None  # Only use pair features in first block

        # Extract [CLS] token representation
        cls_out = self.norm(x[:, 0])  # (B, embed_dim)

        # Classify
        logits = self.classifier(cls_out).squeeze(-1)  # (B,)

        return logits, attention_maps

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ParticleTransformer()
    print(f"ParticleTransformer: {model.count_parameters():,} trainable parameters")

    B, N = 4, 100
    features = torch.randn(B, N, 7)
    # Use a FLOAT mask here — this is what the real ParticleDataset feeds
    # the model. (A bool mask would mask the ~ operator bug that floats hit.)
    mask = torch.ones(B, N, dtype=torch.float32)
    mask[:, 60:] = 0.0
    pair = torch.randn(B, N, N, 4)

    logits, attn_maps = model(features, mask, pair)
    print(f"Input: features={features.shape}, pair={pair.shape}")
    print(f"Output: logits={logits.shape}")
    print(f"Attention maps: {len(attn_maps)} layers, each {attn_maps[0].shape}")
