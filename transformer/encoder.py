"""
Transformer Encoder Layer

Integrates all components:
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Position-wise Feedforward Network
4. Residual Connection + Layer Normalization

Architecture:
    Input
     ↓
    [Multi-Head Attention] → Add & Norm
     ↓
    [Feedforward Network]  → Add & Norm
     ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    A complete encoder layer containing:
    1. Multi-Head Self-Attention
    2. Add & Norm (Residual connection + Layer normalization)
    3. Position-wise FFN
    4. Add & Norm (Residual connection + Layer normalization)

    Complete flow:
        x → [Self-Attention] → Add(x, ·) → LayerNorm →
        → [FFN] → Add(·, ·) → LayerNorm → output

    Why this order?
    - First Attention: gather information
    - Then FFN: process information
    - Residual + Norm at each step: stabilize training

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout rate
        activation: FFN activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # Component 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # Component 2: Position-wise Feedforward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # Two Layer Normalization layers
        # Why two? Because we have two sublayers (Attention and FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Two Dropout layers (after residual connections)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer

        Args:
            x: Input sequence of shape (batch_size, seq_len, d_model)
            mask: Padding mask of shape (batch_size, 1, 1, seq_len) or None
                 Used to ignore <PAD> tokens

        Returns:
            output: Encoder layer output of shape (batch_size, seq_len, d_model)

        Flow details:
            1. Self-Attention: each token attends to entire sequence
            2. Add & Norm: add input + normalize
            3. FFN: process each token independently
            4. Add & Norm: add input again + normalize
        """
        # ===== Sublayer 1: Multi-Head Self-Attention =====

        # Step 1: Self-Attention
        # Q = K = V = x (self-attention)
        # Each token in the sequence can "see" the entire sequence
        attn_output = self.self_attention(x, x, x, mask)

        # Step 2: Residual Connection
        # Why add x?
        # - Allows gradients to flow directly back to earlier layers
        # - Lets model learn "modifications" instead of "reconstruction"
        # - Analogy: tell model "make adjustments to the original"
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # This is the residual connection!

        # Step 3: Layer Normalization
        # Why normalize?
        # - Stabilize training (prevent values from exploding or vanishing)
        # - Speed up convergence
        # - Keep output distribution stable across layers
        x = self.norm1(x)

        # ===== Sublayer 2: Position-wise Feedforward =====

        # Step 4: Feedforward Network
        # Process each position independently
        # Provides non-linear transformation
        ff_output = self.feed_forward(x)

        # Step 5: Residual Connection + Dropout
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Second residual connection

        # Step 6: Layer Normalization
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    Complete Transformer Encoder

    Stacks multiple EncoderLayers.

    Why stack multiple layers?
    - Each layer can learn different levels of features
    - Layer 1: might learn local patterns (e.g., phrases)
    - Layer 2: might learn mid-level patterns (e.g., clauses)
    - Layer N: might learn high-level semantics

    Analogy to deep CNN:
    - Shallow layers: learn edges, textures
    - Middle layers: learn shapes, parts
    - Deep layers: learn objects, scenes

    Args:
        num_layers: Number of encoder layers (original paper uses 6)
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: FFN hidden dimension
        dropout: Dropout rate
        activation: FFN activation function
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # Create num_layers encoder layers with identical structure
        # Note: each layer has independent weights (not shared)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # Final Layer Normalization
        # Why one more at the end?
        # - Ensures encoder output has stable distribution
        # - Easier for downstream processing (e.g., connecting to decoder)
        self.norm = nn.LayerNorm(d_model)

        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of encoder

        Args:
            x: Input sequence of shape (batch_size, seq_len, d_model)
               (typically word embeddings + positional encoding)
            mask: Padding mask of shape (batch_size, 1, 1, seq_len) or None

        Returns:
            output: Encoder output of shape (batch_size, seq_len, d_model)

        Flow:
            Input → Layer1 → Layer2 → ... → LayerN → Norm → Output
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization
        x = self.norm(x)

        return x


if __name__ == "__main__":
    # Test code
    print("=== Testing Encoder Layer ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # Create single encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = encoder_layer(x)
    print(f"Output shape: {output.shape}")

    # Test with mask
    # Assume first 7 tokens are real, last 3 are padding
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0  # Last 3 positions masked
    output_with_mask = encoder_layer(x, mask)
    print(f"Output with mask shape: {output_with_mask.shape}")

    print("\n=== Testing Full Encoder (6 layers) ===\n")

    num_layers = 6
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = encoder(x, mask)
    print(f"Full encoder output shape: {output_full.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nFull encoder ({num_layers} layers) total parameters: {total_params:,}")
    print(f"Approximately {total_params / 1e6:.1f}M parameters")
