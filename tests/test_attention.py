"""
Unit tests for attention mechanisms

Test goals:
1. Verify output shapes are correct
2. Verify attention weights sum to 1
3. Verify masking works properly
"""

import torch
import pytest
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Test Scaled Dot-Product Attention"""

    def test_output_shape(self):
        """Test that output shapes are correct"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64

        # Create random Q, K, V
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Apply attention
        output, attn_weights = scaled_dot_product_attention(Q, K, V)

        # Verify output shape
        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        # Verify attention weights shape
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 (probability distribution property)"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64

        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        # Check if each row (attention over all keys for each query) sums to 1
        sum_weights = attn_weights.sum(dim=-1)
        assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

    def test_mask_works(self):
        """Test that masking works correctly"""
        batch_size = 1
        num_heads = 1
        seq_len = 5
        d_k = 8

        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Create a mask: only attend to first 3 positions
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, :3] = 1  # First 3 positions allowed

        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Check that masked positions (last 2) have near-zero attention weights
        masked_weights = attn_weights[:, :, :, 3:]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)


class TestMultiHeadAttention:
    """Test Multi-Head Attention"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        # Create multi-head attention module
        mha = MultiHeadAttention(d_model, num_heads)

        # Create input
        x = torch.randn(batch_size, seq_len, d_model)

        # Self-attention (Q=K=V)
        output = mha(x, x, x)

        # Verify output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_sequence_lengths(self):
        """Test cross-attention with different sequence lengths for Q and K,V"""
        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        d_model = 256
        num_heads = 4

        mha = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)

        output = mha(query, key, value)

        # Output sequence length should match query
        assert output.shape == (batch_size, seq_len_q, d_model)

    def test_parameters_exist(self):
        """Test that module has correct learnable parameters"""
        d_model = 512
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)

        # Check that there are 4 linear layers (W_q, W_k, W_v, W_o)
        params = list(mha.parameters())
        # Each linear layer has weight and bias, so 8 parameters total
        assert len(params) == 8

    def test_d_model_divisible_by_num_heads(self):
        """Test that assertion fails when d_model is not divisible by num_heads"""
        with pytest.raises(AssertionError):
            # 512 is not divisible by 7, should raise error
            MultiHeadAttention(d_model=512, num_heads=7)


if __name__ == "__main__":
    # Can run this file directly for testing
    pytest.main([__file__, "-v"])
