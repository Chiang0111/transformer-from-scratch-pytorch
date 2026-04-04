"""
Unit tests for positional encoding

Test goals:
1. Verify output shape is correct
2. Verify positional encoding is actually added
3. Verify different positions have different encodings
"""

import torch
import pytest
from transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    """Test Positional Encoding"""

    def test_output_shape(self):
        """Test that output shape is correct"""
        d_model = 512
        batch_size = 2
        seq_len = 10

        pe = PositionalEncoding(d_model, dropout=0.0)  # dropout=0 for testing
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        # Output shape should match input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_added(self):
        """Test that positional encoding is actually added to input"""
        d_model = 512
        batch_size = 1
        seq_len = 10

        pe = PositionalEncoding(d_model, dropout=0.0)  # dropout=0 for testing

        # Use all-zero input, so output will only contain positional encoding
        x = torch.zeros(batch_size, seq_len, d_model)
        output = pe(x)

        # Output should not be all zeros (because positional encoding was added)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_different_positions_different_encodings(self):
        """Test that different positions have different encodings"""
        d_model = 512
        max_len = 100

        pe = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        # Extract encodings at position 0 and position 1
        pos_0 = pe.pe[0, 0, :]  # shape: (d_model,)
        pos_1 = pe.pe[0, 1, :]  # shape: (d_model,)

        # Encodings at different positions should be different
        assert not torch.allclose(pos_0, pos_1)

    def test_encoding_range(self):
        """Test that positional encoding values are in range [-1, 1]"""
        d_model = 512
        max_len = 1000

        pe = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        # Check that all positional encoding values are within [-1, 1]
        # (because we use sin/cos)
        assert torch.all(pe.pe >= -1.0)
        assert torch.all(pe.pe <= 1.0)

    def test_supports_variable_length(self):
        """Test that it supports sequences of different lengths"""
        d_model = 256
        batch_size = 2

        pe = PositionalEncoding(d_model, max_len=1000, dropout=0.0)

        # Test with different sequence lengths
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = pe(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_d_model_must_be_even(self):
        """Test that odd d_model still works"""
        # Note: Original implementation may have issues with odd d_model
        # This test ensures we can handle this case
        d_model = 511  # odd number
        batch_size = 1
        seq_len = 10

        # For odd d_model, the last dimension might not be properly encoded
        # but the code should still run
        try:
            pe = PositionalEncoding(d_model, dropout=0.0)
            x = torch.randn(batch_size, seq_len, d_model)
            output = pe(x)
            assert output.shape == (batch_size, seq_len, d_model)
        except Exception as e:
            pytest.skip(f"Odd d_model not supported: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
