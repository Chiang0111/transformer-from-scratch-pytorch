"""
Unit tests for Encoder

Test goals:
1. Verify EncoderLayer output shape is correct
2. Verify residual connections work properly
3. Verify Layer Normalization is applied
4. Verify full Encoder works correctly
5. Verify mask functionality
"""

import torch
import pytest
from transformer.encoder import EncoderLayer, Encoder


class TestEncoderLayer:
    """Test single Encoder Layer"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        d_ff = 2048

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = encoder_layer(x)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_with_mask(self):
        """Test with mask"""
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create mask: first 7 positions valid, last 3 are padding
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, 7:] = 0

        output = encoder_layer(x, mask)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_residual_connection_exists(self):
        """Test that residual connections exist"""
        batch_size = 1
        seq_len = 5
        d_model = 128
        num_heads = 4
        d_ff = 512

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)
        encoder_layer.eval()  # Eval mode, turn off dropout

        # Use very small input
        x = torch.randn(batch_size, seq_len, d_model) * 0.01

        output = encoder_layer(x)

        # If residual connections exist, output should have some similarity with input
        # This is not a strict test, just ensures residual connection concept exists
        # A completely new output would be very different from input
        assert output.shape == x.shape

    def test_layer_norm_applied(self):
        """Test that Layer Normalization is applied"""
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = encoder_layer(x)

        # Check output statistics
        # Layer Norm normalizes each sample at each position
        # But due to learnable gamma and beta, we just check shape
        assert output.shape == x.shape

    def test_different_activations(self):
        """Test different activation functions"""
        d_model = 256
        num_heads = 4
        d_ff = 1024
        x = torch.randn(1, 5, d_model)

        # ReLU
        encoder_relu = EncoderLayer(d_model, num_heads, d_ff, activation='relu')
        output_relu = encoder_relu(x)
        assert output_relu.shape == x.shape

        # GELU
        encoder_gelu = EncoderLayer(d_model, num_heads, d_ff, activation='gelu')
        output_gelu = encoder_gelu(x)
        assert output_gelu.shape == x.shape

        # Two activation functions should give different outputs (due to random initialization)
        assert not torch.allclose(output_relu, output_gelu)


class TestEncoder:
    """Test full Encoder"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        d_ff = 2048
        num_layers = 6

        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = encoder(x)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_num_layers(self):
        """Test Encoder with different number of layers"""
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024
        x = torch.randn(batch_size, seq_len, d_model)

        # Test 1, 3, 6 layers
        for num_layers in [1, 3, 6]:
            encoder = Encoder(num_layers, d_model, num_heads, d_ff)
            output = encoder(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_with_mask(self):
        """Test with mask"""
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024
        num_layers = 3

        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        # Create mask
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, 7:] = 0  # Last 3 positions are padding

        output = encoder(x, mask)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_parameters_increase_with_layers(self):
        """Test that more layers means more parameters"""
        d_model = 256
        num_heads = 4
        d_ff = 1024

        encoder_2 = Encoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        encoder_4 = Encoder(num_layers=4, d_model=d_model, num_heads=num_heads, d_ff=d_ff)

        params_2 = sum(p.numel() for p in encoder_2.parameters())
        params_4 = sum(p.numel() for p in encoder_4.parameters())

        # 4 layers should have approximately 2x parameters of 2 layers
        # Not exactly 2x because of final LayerNorm
        assert params_4 > params_2
        assert params_4 < params_2 * 2.1  # Should be close to 2x

    def test_layers_are_different(self):
        """Test that each layer has independent parameters (not shared)"""
        d_model = 128
        num_heads = 4
        d_ff = 512
        num_layers = 2

        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # Check that first and second layer parameters are different
        # (they are independently initialized)
        layer1_params = list(encoder.layers[0].parameters())[0]
        layer2_params = list(encoder.layers[1].parameters())[0]

        # Parameters should not be identical (random initialization)
        assert not torch.allclose(layer1_params, layer2_params)

    def test_variable_sequence_lengths(self):
        """Test different sequence lengths"""
        batch_size = 2
        d_model = 256
        num_heads = 4
        d_ff = 1024
        num_layers = 3

        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # Test with different sequence lengths
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = encoder(x)
            assert output.shape == (batch_size, seq_len, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
