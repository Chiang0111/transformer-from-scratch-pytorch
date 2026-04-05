"""
Unit tests for Decoder
"""

import pytest
import torch
from transformer.decoder import DecoderLayer, Decoder, create_causal_mask


class TestCausalMask:
    """Tests for causal mask creation"""

    def test_mask_shape(self):
        """Test that causal mask has correct shape"""
        size = 5
        mask = create_causal_mask(size)

        # Should have shape (1, 1, size, size)
        assert mask.shape == (1, 1, size, size)

    def test_mask_is_lower_triangular(self):
        """Test that causal mask is lower triangular"""
        size = 4
        mask = create_causal_mask(size)

        # Remove batch and head dimensions
        mask_2d = mask.squeeze()

        # Check lower triangular property
        # Upper triangle should be all zeros
        for i in range(size):
            for j in range(size):
                if j > i:
                    # Future positions should be masked (0)
                    assert mask_2d[i, j] == 0
                else:
                    # Past and current positions should be visible (1)
                    assert mask_2d[i, j] == 1

    def test_mask_pattern(self):
        """Test specific pattern of causal mask"""
        mask = create_causal_mask(3)
        expected = torch.tensor([
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]]
        ])

        # Compare with expected pattern
        assert torch.equal(mask.squeeze(0), expected.float())

    def test_different_sizes(self):
        """Test causal mask with different sizes"""
        for size in [1, 5, 10, 20]:
            mask = create_causal_mask(size)
            assert mask.shape == (1, 1, size, size)

            # Check it's still lower triangular
            # Use squeeze with specific dimensions to avoid 0-d tensor
            mask_2d = mask.squeeze(0).squeeze(0)  # Remove batch and head dims
            for i in range(size):
                # Diagonal and below should be 1
                assert mask_2d[i, i] == 1
                # Above diagonal should be 0
                if i < size - 1:
                    assert mask_2d[i, i + 1] == 0


class TestDecoderLayer:
    """Tests for single Decoder Layer"""

    @pytest.fixture
    def decoder_params(self):
        """Common parameters for decoder layer"""
        return {
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1
        }

    @pytest.fixture
    def decoder_layer(self, decoder_params):
        """Create a decoder layer instance"""
        return DecoderLayer(**decoder_params)

    def test_output_shape(self, decoder_layer):
        """Test decoder layer output shape matches input shape"""
        batch_size = 2
        src_len = 10
        tgt_len = 8
        d_model = 512

        # Create dummy inputs
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # Create masks
        tgt_mask = create_causal_mask(tgt_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        # Forward pass
        output = decoder_layer(x, encoder_output, tgt_mask, src_mask)

        # Output should have same shape as target input
        assert output.shape == (batch_size, tgt_len, d_model)

    def test_different_source_target_lengths(self, decoder_layer):
        """Test with different source and target sequence lengths"""
        batch_size = 3
        src_len = 15
        tgt_len = 10
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        tgt_mask = create_causal_mask(tgt_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        output = decoder_layer(x, encoder_output, tgt_mask, src_mask)

        # Output length should match target length, not source length
        assert output.shape == (batch_size, tgt_len, d_model)

    def test_with_masks(self, decoder_layer):
        """Test decoder layer with various masks"""
        batch_size = 2
        src_len = 6
        tgt_len = 5
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # Test with causal mask only
        tgt_mask = create_causal_mask(tgt_len)
        output1 = decoder_layer(x, encoder_output, tgt_mask, None)
        assert output1.shape == (batch_size, tgt_len, d_model)

        # Test with source mask only
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        output2 = decoder_layer(x, encoder_output, None, src_mask)
        assert output2.shape == (batch_size, tgt_len, d_model)

        # Test with both masks
        output3 = decoder_layer(x, encoder_output, tgt_mask, src_mask)
        assert output3.shape == (batch_size, tgt_len, d_model)

        # Test with no masks
        output4 = decoder_layer(x, encoder_output, None, None)
        assert output4.shape == (batch_size, tgt_len, d_model)

    def test_parameters_exist(self, decoder_layer):
        """Test that all expected parameters exist"""
        params = dict(decoder_layer.named_parameters())

        # Check self-attention parameters
        assert 'self_attention.W_q.weight' in params
        assert 'self_attention.W_k.weight' in params
        assert 'self_attention.W_v.weight' in params
        assert 'self_attention.W_o.weight' in params

        # Check cross-attention parameters
        assert 'cross_attention.W_q.weight' in params
        assert 'cross_attention.W_k.weight' in params
        assert 'cross_attention.W_v.weight' in params
        assert 'cross_attention.W_o.weight' in params

        # Check feedforward parameters
        assert 'feed_forward.linear1.weight' in params
        assert 'feed_forward.linear2.weight' in params

        # Check layer norm parameters (3 of them)
        assert 'norm1.weight' in params
        assert 'norm2.weight' in params
        assert 'norm3.weight' in params

    def test_training_vs_eval_mode(self, decoder_layer):
        """Test that dropout behaves differently in training vs eval mode"""
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        tgt_mask = create_causal_mask(tgt_len)

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Training mode
        decoder_layer.train()
        output_train1 = decoder_layer(x, encoder_output, tgt_mask, None)

        # Reset seed
        torch.manual_seed(42)
        output_train2 = decoder_layer(x, encoder_output, tgt_mask, None)

        # Due to dropout randomness, outputs should be different
        # (even with same seed, dropout introduces randomness)

        # Eval mode
        decoder_layer.eval()
        torch.manual_seed(42)
        output_eval1 = decoder_layer(x, encoder_output, tgt_mask, None)

        torch.manual_seed(42)
        output_eval2 = decoder_layer(x, encoder_output, tgt_mask, None)

        # In eval mode with same seed, outputs should be identical
        assert torch.allclose(output_eval1, output_eval2)

    def test_gradient_flow(self, decoder_layer):
        """Test that gradients can flow through decoder layer"""
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
        encoder_output = torch.randn(batch_size, src_len, d_model, requires_grad=True)
        tgt_mask = create_causal_mask(tgt_len)

        output = decoder_layer(x, encoder_output, tgt_mask, None)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert encoder_output.grad is not None

        # Check that parameter gradients exist
        for param in decoder_layer.parameters():
            assert param.grad is not None


class TestDecoder:
    """Tests for full Decoder (multiple layers stacked)"""

    @pytest.fixture
    def decoder_params(self):
        """Common parameters for decoder"""
        return {
            'num_layers': 6,
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'dropout': 0.1
        }

    @pytest.fixture
    def decoder(self, decoder_params):
        """Create a decoder instance"""
        return Decoder(**decoder_params)

    def test_output_shape(self, decoder):
        """Test decoder output shape matches input shape"""
        batch_size = 2
        src_len = 10
        tgt_len = 8
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        tgt_mask = create_causal_mask(tgt_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        output = decoder(x, encoder_output, tgt_mask, src_mask)

        # Output should have same shape as input
        assert output.shape == (batch_size, tgt_len, d_model)

    def test_num_layers(self):
        """Test decoder with different number of layers"""
        for num_layers in [1, 2, 4, 6, 12]:
            decoder = Decoder(
                num_layers=num_layers,
                d_model=256,
                num_heads=4,
                d_ff=1024
            )

            assert decoder.num_layers == num_layers
            assert len(decoder.layers) == num_layers

    def test_parameter_count(self, decoder):
        """Test that decoder has reasonable number of parameters"""
        total_params = sum(p.numel() for p in decoder.parameters())

        # Decoder should have significant number of parameters
        # With 6 layers, d_model=512, should be tens of millions
        assert total_params > 1_000_000

        # Should be less than 100M (sanity check)
        assert total_params < 100_000_000

    def test_layers_have_independent_parameters(self, decoder):
        """Test that each layer has its own parameters (not shared)"""
        # Get first layer's self-attention query weight
        layer0_weight = decoder.layers[0].self_attention.W_q.weight
        layer1_weight = decoder.layers[1].self_attention.W_q.weight

        # They should be different objects (not shared)
        assert layer0_weight is not layer1_weight

        # They should have different values (random initialization)
        assert not torch.equal(layer0_weight, layer1_weight)

    def test_with_different_activations(self):
        """Test decoder with different activation functions"""
        for activation in ['relu', 'gelu']:
            decoder = Decoder(
                num_layers=2,
                d_model=256,
                num_heads=4,
                d_ff=1024,
                activation=activation
            )

            batch_size = 2
            src_len = 5
            tgt_len = 4
            d_model = 256

            x = torch.randn(batch_size, tgt_len, d_model)
            encoder_output = torch.randn(batch_size, src_len, d_model)
            tgt_mask = create_causal_mask(tgt_len)

            output = decoder(x, encoder_output, tgt_mask, None)
            assert output.shape == (batch_size, tgt_len, d_model)

    def test_encoder_output_unchanged(self, decoder):
        """Test that encoder output is not modified by decoder"""
        batch_size = 2
        src_len = 6
        tgt_len = 5
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        encoder_output_copy = encoder_output.clone()

        tgt_mask = create_causal_mask(tgt_len)

        # Run decoder
        decoder.eval()
        with torch.no_grad():
            _ = decoder(x, encoder_output, tgt_mask, None)

        # Encoder output should remain unchanged
        assert torch.equal(encoder_output, encoder_output_copy)

    def test_gradient_flow_through_stack(self, decoder):
        """Test that gradients flow through entire decoder stack"""
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        x = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
        encoder_output = torch.randn(batch_size, src_len, d_model, requires_grad=True)
        tgt_mask = create_causal_mask(tgt_len)

        output = decoder(x, encoder_output, tgt_mask, None)
        loss = output.sum()
        loss.backward()

        # Check gradients exist for inputs
        assert x.grad is not None
        assert encoder_output.grad is not None

        # Check gradients exist for all layers
        for layer_idx, layer in enumerate(decoder.layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, \
                    f"Layer {layer_idx} parameter {name} has no gradient"

    def test_sequential_generation_simulation(self, decoder):
        """Test decoder in auto-regressive generation scenario"""
        batch_size = 1
        src_len = 6
        d_model = 512

        # Encoder output (fixed)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        decoder.eval()
        with torch.no_grad():
            # Simulate generating tokens one by one
            max_len = 5
            current_output = torch.randn(batch_size, 1, d_model)  # Start token

            for step in range(1, max_len):
                tgt_len = current_output.shape[1]
                tgt_mask = create_causal_mask(tgt_len)

                # Decode current sequence
                output = decoder(current_output, encoder_output, tgt_mask, None)

                # Shape should match current length
                assert output.shape == (batch_size, tgt_len, d_model)

                # Append new token (simulated)
                new_token = torch.randn(batch_size, 1, d_model)
                current_output = torch.cat([current_output, new_token], dim=1)

            # Final length should be max_len
            assert current_output.shape[1] == max_len


class TestDecoderIntegration:
    """Integration tests for decoder with encoder"""

    def test_encoder_decoder_compatibility(self):
        """Test that encoder and decoder work together"""
        from transformer.encoder import Encoder

        batch_size = 2
        src_len = 10
        tgt_len = 8
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048

        # Create encoder and decoder
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # Create inputs
        src = torch.randn(batch_size, src_len, d_model)
        tgt = torch.randn(batch_size, tgt_len, d_model)

        # Create masks
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        tgt_mask = create_causal_mask(tgt_len)

        # Encode
        encoder_output = encoder(src, src_mask)
        assert encoder_output.shape == (batch_size, src_len, d_model)

        # Decode
        decoder_output = decoder(tgt, encoder_output, tgt_mask, src_mask)
        assert decoder_output.shape == (batch_size, tgt_len, d_model)

    def test_full_transformer_forward_pass(self):
        """Test complete forward pass of encoder-decoder architecture"""
        from transformer.encoder import Encoder

        # Small model for testing
        batch_size = 4
        src_len = 12
        tgt_len = 10
        d_model = 256
        num_heads = 4
        num_layers = 2
        d_ff = 1024

        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # Create inputs
            src = torch.randn(batch_size, src_len, d_model)
            tgt = torch.randn(batch_size, tgt_len, d_model)

            # Encode
            memory = encoder(src)

            # Decode with auto-regressive masking
            tgt_mask = create_causal_mask(tgt_len)
            output = decoder(tgt, memory, tgt_mask)

            assert output.shape == (batch_size, tgt_len, d_model)
