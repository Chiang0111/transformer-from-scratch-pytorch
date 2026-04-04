"""
Attention Mechanisms

This module implements:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention

    This is the core of the Transformer architecture.

    Args:
        query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
        key:   Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
        value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
        mask:  Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
               or (batch_size, num_heads, seq_len_q, seq_len_k)

    Returns:
        output: Attention-weighted output of shape (batch_size, num_heads, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)

    Formula: Attention(Q,K,V) = softmax(QK^T / √d_k) * V
    """
    # Step 1: Get d_k for scaling
    # Why scale? To prevent dot products from growing too large,
    # which would push softmax into regions with extremely small gradients
    d_k = query.size(-1)

    # Step 2: Compute attention scores (Q·K^T)
    # torch.matmul handles batch matrix multiplication automatically
    # key.transpose(-2, -1) swaps last two dimensions: (seq_len_k, d_k) -> (d_k, seq_len_k)
    # Result shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 3: Scale by √d_k
    # This is what makes it "scaled" dot-product attention
    scores = scores / math.sqrt(d_k)

    # Step 4: Apply mask if provided
    # Mask is used to:
    # - Ignore padding tokens (padding mask)
    # - Prevent attending to future tokens (causal mask for decoder)
    if mask is not None:
        # Set masked positions to large negative value
        # so they become ~0 after softmax
        # Use -1e9 instead of -inf to avoid NaN issues
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 5: Apply softmax to get attention weights
    # dim=-1 means softmax over the last dimension (seq_len_k)
    # Result: attention weights sum to 1 for each query position
    attention_weights = F.softmax(scores, dim=-1)

    # Step 6: Multiply by values (weighted sum)
    # This is where we actually "attend" to the values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism

    Why multiple heads?
    - A single attention head might only focus on one type of relationship (e.g., syntactic)
    - Multiple heads can attend to different types of patterns simultaneously
      (syntax, semantics, position, etc.)
    - Analogy: Multiple experts analyzing the same problem from different perspectives

    Args:
        d_model: Model dimension (input/output dimension)
        num_heads: Number of attention heads
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # d_model must be divisible by num_heads
        # because we split d_model evenly across heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        # Dimension of each head
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        # Why project? To learn different "views" of the input
        # These weights are learnable parameters
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        # Combines information from all heads back to d_model dimension
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split input into multiple attention heads

        Input shape:  (batch_size, seq_len, d_model)
        Output shape: (batch_size, num_heads, seq_len, d_k)

        What this does:
        1. Split d_model into num_heads chunks of size d_k
        2. Rearrange dimensions so each head can be processed independently
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch_size, num_heads, seq_len, d_k)
        # Why transpose? So we can do batch operations over all heads
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple attention heads back together

        Input shape:  (batch_size, num_heads, seq_len, d_k)
        Output shape: (batch_size, seq_len, d_model)

        This is the inverse of split_heads
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose back: (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # Merge last two dimensions: (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass

        Args:
            query, key, value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

        Returns:
            output: shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Step 1: Linear projections for Q, K, V
        # These projections are learnable - the model learns
        # what "questions to ask", "keys to match", and "values to return"
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)

        # Step 2: Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # Step 3: Apply scaled dot-product attention
        # Each head computes attention independently
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: (batch_size, num_heads, seq_len, d_k)

        # Step 4: Combine heads
        output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)

        # Step 5: Final linear projection
        # This projection integrates information from all heads
        output = self.W_o(output)  # (batch_size, seq_len, d_model)

        return output
