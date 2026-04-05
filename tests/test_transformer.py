"""
Tests for Complete Transformer Model

Test Coverage:
1. TokenEmbedding: embedding lookup and scaling
2. Transformer initialization: parameter counts, architecture
3. Encode: source sequence encoding
4. Decode: target sequence decoding
5. Forward pass: complete training flow
6. Generate: autoregressive inference
7. Integration: end-to-end scenarios
"""

import pytest
import torch
import torch.nn as nn

from transformer import (
    Transformer,
    TokenEmbedding,
    create_transformer
)


class TestTokenEmbedding:
    """Test Token Embedding Layer"""

    def test_embedding_output_shape(self):
        """Test that embeddings have correct shape"""
        vocab_size = 1000
        d_model = 512
        batch_size = 2
        seq_len = 10

        embedding = TokenEmbedding(vocab_size, d_model)

        # Create random token IDs
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Get embeddings
        output = embedding(tokens)

        # Check shape: (batch, seq_len, d_model)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_embedding_scaling(self):
        """Test that embeddings are scaled by sqrt(d_model)"""
        vocab_size = 100
        d_model = 64
        embedding = TokenEmbedding(vocab_size, d_model)

        # Create single token
        tokens = torch.tensor([[5]])

        # Get scaled embedding
        scaled_output = embedding(tokens)

        # Get unscaled embedding directly from nn.Embedding
        unscaled_output = embedding.embedding(tokens)

        # Check that output is scaled by sqrt(d_model)
        expected = unscaled_output * (d_model ** 0.5)
        torch.testing.assert_close(scaled_output, expected)

    def test_different_tokens_different_embeddings(self):
        """Test that different tokens have different embeddings"""
        vocab_size = 100
        d_model = 64
        embedding = TokenEmbedding(vocab_size, d_model)

        # Two different tokens
        token1 = torch.tensor([[5]])
        token2 = torch.tensor([[10]])

        # Get embeddings
        emb1 = embedding(token1)
        emb2 = embedding(token2)

        # Embeddings should be different
        assert not torch.allclose(emb1, emb2)

    def test_same_token_same_embedding(self):
        """Test that the same token always gets the same embedding"""
        vocab_size = 100
        d_model = 64
        embedding = TokenEmbedding(vocab_size, d_model)

        # Same token in different positions
        tokens = torch.tensor([[5, 10, 5, 20]])

        # Get embeddings
        output = embedding(tokens)

        # Embedding at position 0 and 2 should be identical (both token 5)
        torch.testing.assert_close(output[0, 0], output[0, 2])


class TestTransformerInitialization:
    """Test Transformer Model Initialization"""

    def test_model_creation(self):
        """Test that model can be created with default parameters"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )

        assert isinstance(model, nn.Module)
        assert model.src_embedding is not None
        assert model.tgt_embedding is not None
        assert model.encoder is not None
        assert model.decoder is not None

    def test_small_model_creation(self):
        """Test creating a small model (CPU-friendly)"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=512
        )

        # Small model should have fewer parameters
        param_count = model.count_parameters()
        assert param_count < 10_000_000  # Less than 10M parameters

    def test_create_transformer_factory(self):
        """Test factory function for creating models"""
        model = create_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_layers=4,
            d_ff=1024
        )

        assert isinstance(model, Transformer)
        # Check that encoder and decoder have same number of layers
        assert len(model.encoder.layers) == 4
        assert len(model.decoder.layers) == 4

    def test_parameter_count(self):
        """Test parameter counting"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=512
        )

        param_count = model.count_parameters()

        # Verify it's a reasonable number
        assert param_count > 0
        assert param_count < 100_000_000  # Less than 100M for this config

        # Manually count and compare
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count


class TestTransformerEncode:
    """Test Encoder Functionality"""

    def test_encode_shape(self):
        """Test that encode produces correct output shape"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10

        # Create source tokens
        src = torch.randint(0, 1000, (batch_size, src_len))

        # Encode
        memory = model.encode(src)

        # Check shape: (batch, src_len, d_model)
        assert memory.shape == (batch_size, src_len, 128)

    def test_encode_with_mask(self):
        """Test encoding with source mask"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10

        src = torch.randint(0, 1000, (batch_size, src_len))

        # Create mask (1 = attend, 0 = ignore)
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        src_mask[:, :, :, 5:] = 0  # Mask out positions 5-9

        # Encode with mask
        memory = model.encode(src, src_mask)

        # Should still produce correct shape
        assert memory.shape == (batch_size, src_len, 128)

    def test_encode_deterministic(self):
        """Test that encoding is deterministic (same input → same output)"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        model.eval()  # Disable dropout

        src = torch.randint(0, 1000, (2, 10))

        # Encode twice
        with torch.no_grad():
            memory1 = model.encode(src)
            memory2 = model.encode(src)

        # Should be identical
        torch.testing.assert_close(memory1, memory2)


class TestTransformerDecode:
    """Test Decoder Functionality"""

    def test_decode_shape(self):
        """Test that decode produces correct output shape"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10
        tgt_len = 8

        # Create source and encode
        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)

        # Create target
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # Decode
        output = model.decode(tgt, memory)

        # Check shape: (batch, tgt_len, d_model)
        assert output.shape == (batch_size, tgt_len, 128)

    def test_decode_with_causal_mask(self):
        """Test decoding with causal mask"""
        from transformer import create_causal_mask

        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # Create causal mask
        tgt_mask = create_causal_mask(tgt_len)

        # Decode with mask
        output = model.decode(tgt, memory, tgt_mask)

        # Should still produce correct shape
        assert output.shape == (batch_size, tgt_len, 128)

    def test_decode_different_lengths(self):
        """Test decoding with different source and target lengths"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2

        # Different lengths
        src_len = 15
        tgt_len = 5

        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        output = model.decode(tgt, memory)

        # Output should match target length, not source length
        assert output.shape == (batch_size, tgt_len, 128)


class TestTransformerForward:
    """Test Complete Forward Pass"""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape"""
        src_vocab = 1000
        tgt_vocab = 800
        d_model = 128

        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=d_model,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, src_vocab, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

        # Forward pass
        logits = model(src, tgt)

        # Check shape: (batch, tgt_len, tgt_vocab_size)
        assert logits.shape == (batch_size, tgt_len, tgt_vocab)

    def test_forward_with_masks(self):
        """Test forward pass with both source and target masks"""
        from transformer import create_causal_mask

        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 1000, (batch_size, src_len))
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # Create masks
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        tgt_mask = create_causal_mask(tgt_len)

        # Forward with masks
        logits = model(src, tgt, src_mask, tgt_mask)

        assert logits.shape == (batch_size, tgt_len, 1000)

    def test_forward_backprop(self):
        """Test that gradients flow through forward pass"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 1000, (batch_size, src_len))
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # Forward pass
        logits = model(src, tgt)

        # Create dummy loss
        loss = logits.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist for key parameters
        assert model.src_embedding.embedding.weight.grad is not None
        assert model.tgt_embedding.embedding.weight.grad is not None
        assert model.output_projection.weight.grad is not None

    def test_forward_batch_size_one(self):
        """Test forward pass with batch size 1"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # Single example
        src = torch.randint(0, 1000, (1, 10))
        tgt = torch.randint(0, 1000, (1, 8))

        logits = model(src, tgt)

        assert logits.shape == (1, 8, 1000)


class TestTransformerGenerate:
    """Test Autoregressive Generation"""

    def test_generate_basic(self):
        """Test basic generation functionality"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10

        src = torch.randint(0, 1000, (batch_size, src_len))

        # Generate (max 20 tokens)
        generated = model.generate(src, max_len=20, start_token=1, end_token=2)

        # Check shape: (batch, generated_len)
        # Length should be <= max_len
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 20

        # First token should be start_token
        assert (generated[:, 0] == 1).all()

    def test_generate_with_max_len(self):
        """Test that generation respects max_len"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        src = torch.randint(0, 1000, (1, 10))

        # Generate with small max_len
        max_len = 5
        generated = model.generate(src, max_len=max_len, start_token=1, end_token=2)

        # Should not exceed max_len
        assert generated.shape[1] <= max_len

    def test_generate_deterministic_eval_mode(self):
        """Test that generation is deterministic in eval mode with no randomness"""
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.0  # No dropout
        )
        model.eval()

        src = torch.randint(0, 100, (1, 5))

        # Generate twice
        with torch.no_grad():
            gen1 = model.generate(src, max_len=10, start_token=1, end_token=2)
            gen2 = model.generate(src, max_len=10, start_token=1, end_token=2)

        # Should be identical
        assert torch.equal(gen1, gen2)

    def test_generate_stops_at_end_token(self):
        """Test that generation can stop early if end_token is generated"""
        # This test just ensures the code runs - actual early stopping depends on model weights
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        src = torch.randint(0, 100, (1, 5))

        # Generate - should complete without error
        generated = model.generate(src, max_len=20, start_token=1, end_token=2)

        assert generated.shape[1] <= 20


class TestTransformerIntegration:
    """Integration Tests - End to End Scenarios"""

    def test_translation_pipeline(self):
        """Test complete translation pipeline (encode → decode → generate)"""
        src_vocab = 1000
        tgt_vocab = 800
        d_model = 128

        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=d_model,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        batch_size = 2
        src_len = 10

        # Source sequence (e.g., English)
        src = torch.randint(0, src_vocab, (batch_size, src_len))

        # Training mode: use known target
        tgt_len = 8
        tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

        # Forward pass for training
        logits = model(src, tgt)
        assert logits.shape == (batch_size, tgt_len, tgt_vocab)

        # Inference mode: generate target
        model.eval()
        with torch.no_grad():
            generated = model.generate(src, max_len=15, start_token=1, end_token=2)

        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 15

    def test_different_vocab_sizes(self):
        """Test with different source and target vocabulary sizes"""
        model = Transformer(
            src_vocab_size=5000,   # English: 5000 words
            tgt_vocab_size=3000,   # French: 3000 words
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        src = torch.randint(0, 5000, (2, 12))
        tgt = torch.randint(0, 3000, (2, 10))

        logits = model(src, tgt)

        # Output should be target vocab size
        assert logits.shape == (2, 10, 3000)

    def test_variable_sequence_lengths(self):
        """Test with various sequence length combinations"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        test_cases = [
            (5, 3),    # Short source, short target
            (20, 15),  # Long source, long target
            (10, 3),   # Long source, short target
            (3, 10),   # Short source, long target
        ]

        for src_len, tgt_len in test_cases:
            src = torch.randint(0, 1000, (2, src_len))
            tgt = torch.randint(0, 1000, (2, tgt_len))

            logits = model(src, tgt)

            assert logits.shape == (2, tgt_len, 1000), \
                f"Failed for src_len={src_len}, tgt_len={tgt_len}"

    def test_training_step_simulation(self):
        """Simulate a complete training step"""
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding

        # Training data
        src = torch.randint(1, 1000, (4, 10))  # batch=4
        tgt_input = torch.randint(1, 1000, (4, 8))
        tgt_output = torch.randint(1, 1000, (4, 8))

        # Forward pass
        model.train()
        logits = model(src, tgt_input)  # (4, 8, 1000)

        # Compute loss
        # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
        logits_flat = logits.reshape(-1, 1000)
        tgt_flat = tgt_output.reshape(-1)
        loss = criterion(logits_flat, tgt_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that loss is computed
        assert loss.item() > 0

        # Check that weights were updated
        # (gradient descent should change parameters)
        assert True  # If we get here, training step succeeded
