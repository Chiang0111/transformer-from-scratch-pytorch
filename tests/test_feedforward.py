"""
Unit tests for feedforward networks

Test goals:
1. Verify output shapes are correct
2. Verify dimension expansion and compression
3. Verify different activation functions
4. Verify gated version
"""

import torch
import pytest
from transformer.feedforward import PositionwiseFeedForward, GatedFeedForward


class TestPositionwiseFeedForward:
    """Test Position-wise Feedforward Network"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ffn = PositionwiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_relu_activation(self):
        """Test ReLU activation function"""
        d_model = 256
        d_ff = 1024

        ffn = PositionwiseFeedForward(d_model, d_ff, activation='relu')
        x = torch.randn(1, 5, d_model)

        output = ffn(x)
        assert output.shape == (1, 5, d_model)

    def test_gelu_activation(self):
        """Test GELU activation function"""
        d_model = 256
        d_ff = 1024

        ffn = PositionwiseFeedForward(d_model, d_ff, activation='gelu')
        x = torch.randn(1, 5, d_model)

        output = ffn(x)
        assert output.shape == (1, 5, d_model)

    def test_invalid_activation(self):
        """Test that invalid activation raises error"""
        with pytest.raises(ValueError):
            PositionwiseFeedForward(d_model=512, d_ff=2048, activation='sigmoid')

    def test_parameters_count(self):
        """Test that parameter count is correct"""
        d_model = 512
        d_ff = 2048

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # Calculate expected parameter count
        # Linear1: d_model * d_ff + d_ff (weight + bias)
        # Linear2: d_ff * d_model + d_model (weight + bias)
        expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_params

    def test_dropout_in_training_mode(self):
        """Test that dropout works in training mode"""
        d_model = 256
        d_ff = 1024
        dropout = 0.5

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        ffn.train()  # Set to training mode

        x = torch.randn(1, 10, d_model)

        # Run multiple times, results should differ (due to dropout)
        output1 = ffn(x)
        output2 = ffn(x)

        # Due to dropout, two outputs should not be identical
        assert not torch.allclose(output1, output2)

    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is disabled in eval mode"""
        d_model = 256
        d_ff = 1024
        dropout = 0.5

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        ffn.eval()  # Set to eval mode

        x = torch.randn(1, 10, d_model)

        # Run multiple times, results should be identical (dropout off)
        output1 = ffn(x)
        output2 = ffn(x)

        # In eval mode, two outputs should be identical
        assert torch.allclose(output1, output2)

    def test_variable_sequence_lengths(self):
        """Test different sequence lengths"""
        d_model = 256
        d_ff = 1024
        batch_size = 2

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # Test with different sequence lengths
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            assert output.shape == (batch_size, seq_len, d_model)


class TestGatedFeedForward:
    """Test Gated Feedforward Network"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        gated_ffn = GatedFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = gated_ffn(x)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_parameters_count(self):
        """Test that gated version has approximately 2x parameters"""
        d_model = 512
        d_ff = 2048

        standard_ffn = PositionwiseFeedForward(d_model, d_ff)
        gated_ffn = GatedFeedForward(d_model, d_ff)

        standard_params = sum(p.numel() for p in standard_ffn.parameters())
        gated_params = sum(p.numel() for p in gated_ffn.parameters())

        # Gated version has two parallel linear1 layers, so more parameters
        assert gated_params > standard_params

        # Specifically, should be one extra d_model * d_ff + d_ff
        expected_diff = d_model * d_ff + d_ff
        assert gated_params - standard_params == expected_diff

    def test_gating_mechanism(self):
        """Test that gating mechanism works"""
        d_model = 256
        d_ff = 1024

        gated_ffn = GatedFeedForward(d_model, d_ff)
        x = torch.randn(1, 10, d_model)

        # Just ensure it runs without error
        output = gated_ffn(x)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
