"""
Positional Encoding

Why is this needed?
- The Transformer's attention mechanism is order-agnostic
- We need to explicitly inject position information

Implementation:
- Uses sinusoidal functions to generate position encodings
- Added to word embeddings
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module

    【Why Do We Need Positional Encoding?】
    Problem: Attention is completely order-agnostic!

    Concrete example:
        Sentence A: "I love eating apples"
        Sentence B: "apples eating love I"

    For Attention mechanism:
        - These two sentences have identical Q, K, V matrices!
        - Because Attention only looks at "which words are related"
        - It doesn't care about "order"
        - Like throwing words into a bag and losing all order information

    This is a huge problem because:
        - "I love eating apples" and "apples eating love I" mean totally different things
        - "The cat sat on the mat" vs "The mat sat on the cat" - completely different!
        - In language, ORDER MATTERS!

    Solution: Positional Encoding
        Add position information to each word's embedding
        → Let the model know: this word is at position 0, 1, 2, etc.

    【Why Use sin/cos Functions?】
    There are many ways to encode position. Why choose sin/cos?

    Option 1: Learned Positional Embeddings
        Drawback: Can only handle lengths seen during training
        → If trained on max 100 words, fails on 150-word sentences

    Option 2: Fixed sin/cos Encoding (This Implementation)
        Advantages:
        ✓ Can handle ANY sequence length (generalizes beyond training)
        ✓ No parameters to learn (reduces model size)
        ✓ Mathematical properties: relative positions can be expressed
          as linear combinations (model can learn "3 positions ahead")

    【The Clock Analogy】
    Think of a clock:
        - Second hand: rotates fast (high frequency)
        - Minute hand: rotates slower (medium frequency)
        - Hour hand: rotates slowest (low frequency)
        - Together, they uniquely identify any moment!

    Positional encoding is similar:
        - Low dimensions: high frequency waves (like second hand)
        - Middle dimensions: medium frequency waves (like minute hand)
        - High dimensions: low frequency waves (like hour hand)
        - Together, they uniquely identify each position!

    【Formula Explained】
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
        pos     = position index (0, 1, 2, 3, ...)
        i       = dimension index (0, 1, 2, 3, ...)
        d_model = model dimension (e.g., 512)
        2i      = even dimensions use sin
        2i+1    = odd dimensions use cos

    Concrete example (assume d_model=512):
        Position 0 encoding:
            dim 0 (even): sin(0 / 10000^(0/512))   = sin(0) = 0
            dim 1 (odd):  cos(0 / 10000^(0/512))   = cos(0) = 1
            dim 2 (even): sin(0 / 10000^(2/512))   = sin(0) = 0
            dim 3 (odd):  cos(0 / 10000^(2/512))   = cos(0) = 1
            ...

        Position 1 encoding:
            dim 0: sin(1 / 10000^(0/512))   = sin(1) ≈ 0.841
            dim 1: cos(1 / 10000^(0/512))   = cos(1) ≈ 0.540
            dim 2: sin(1 / 10000^(2/512))   ≈ 0.841
            dim 3: cos(1 / 10000^(2/512))   ≈ 0.540
            ...

    【Why 10000?】
        - This is an empirical value (used in original paper)
        - Purpose: create different frequencies for different dimensions
        - 10000^(2i/d_model) increases with i
        - → Frequency decreases (from fast oscillation to slow oscillation)

    Args:
        d_model: Model dimension (e.g., 512)
        max_len: Maximum sequence length (default 5000, can handle 5000-word sentences)
        dropout: Dropout rate (default 0.1, prevents overfitting)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ========== Step 1: Initialize Positional Encoding Matrix ==========
        # Create an all-zero matrix, will fill with sin/cos values later
        # shape: (max_len, d_model)
        # Example: (5000, 512) → can handle sentences up to 5000 words
        #
        # Why (max_len, d_model)?
        # - Each row: represents one position (word 0, word 1, word 2, ...)
        # - Each column: represents one dimension (512 dimensions)
        # - Each value: the encoding for that position at that dimension
        #
        # This matrix is FIXED (not learned!), so use register_buffer, not nn.Parameter
        pe = torch.zeros(max_len, d_model)

        # ========== Step 2: Create Position Indices ==========
        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Example: [0, 1, 2, ..., 4999]
        position = torch.arange(0, max_len, dtype=torch.float)  # shape: (max_len,)

        # unsqueeze(1) changes shape from (max_len,) to (max_len, 1)
        # Example: (5000,) → (5000, 1)
        #
        # Why unsqueeze?
        # Because we'll multiply with div_term (shape: (d_model/2,))
        # Broadcasting rule: (5000, 1) * (256,) → (5000, 256)
        position = position.unsqueeze(1)  # shape: (max_len, 1)

        # ========== Step 3: Compute Frequency Terms (Denominator) ==========
        # Goal: compute 10000^(2i/d_model)
        #
        # torch.arange(0, d_model, 2) generates even indices
        # Example for d_model=512: [0, 2, 4, 6, ..., 510]
        # → shape: (256,), which is d_model/2
        #
        # Why only even indices?
        # Because in the formula:
        # - Even dimensions (2i) use sin
        # - Odd dimensions (2i+1) use cos
        # But they share the SAME frequency term (denominator)!
        # So we only need to compute d_model/2 frequency values

        # Mathematical derivation:
        # Original denominator: 10000^(2i/d_model)
        # Convert to exp form: 10000^(2i/d_model) = exp(log(10000^(2i/d_model)))
        #                                          = exp((2i/d_model) * log(10000))
        # But we want 1 / 10000^(2i/d_model)
        # = exp(-(2i/d_model) * log(10000))
        #
        # Why this conversion?
        # - Direct computation 10000^x can overflow when x is large
        # - exp(-x * log(10000)) is numerically more stable
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term shape: (d_model/2,) example: (256,)
        #
        # Concrete example values (d_model=512):
        # i=0:   exp(-(0/512) * log(10000)) = exp(0) = 1.0
        # i=2:   exp(-(2/512) * log(10000)) ≈ 0.912
        # i=4:   exp(-(4/512) * log(10000)) ≈ 0.832
        # ...
        # i=510: exp(-(510/512) * log(10000)) ≈ 0.0001  (very low frequency)
        #
        # → Frequencies decrease from high to low (like clock hands)

        # ========== Step 4: Compute sin and cos Encodings ==========
        # Even dimensions use sin
        # position * div_term broadcasts:
        # (max_len, 1) * (d_model/2,) → (max_len, d_model/2)
        # Example: (5000, 1) * (256,) → (5000, 256)
        #
        # pe[:, 0::2] means what?
        # - [:, 0::2] = "all rows, columns starting from 0, step by 2"
        # - i.e., columns 0, 2, 4, 6, ... (even columns)
        # - shape: (max_len, d_model/2)
        #
        # Concrete example (position pos=1, dimension i=0):
        # sin(1 * 1.0) = sin(1) ≈ 0.841
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions use cos
        # pe[:, 1::2] means:
        # - "all rows, columns starting from 1, step by 2"
        # - i.e., columns 1, 3, 5, 7, ... (odd columns)
        #
        # Concrete example (position pos=1, dimension i=0):
        # cos(1 * 1.0) = cos(1) ≈ 0.540
        pe[:, 1::2] = torch.cos(position * div_term)

        # Now pe shape: (max_len, d_model)
        # Example: (5000, 512)
        # Each row is a complete encoding for one position (512-dim vector)

        # ========== Step 5: Add Batch Dimension ==========
        # Change shape from (max_len, d_model) to (1, max_len, d_model)
        # Example: (5000, 512) → (1, 5000, 512)
        #
        # Why add this 1?
        # Because actual input shape is (batch_size, seq_len, d_model)
        # Example: (32, 50, 512)  ← 32 samples, each 50 words, each word 512 dims
        #
        # When adding:
        # (32, 50, 512) + (1, 5000, 512)
        # PyTorch broadcasts:
        # (32, 50, 512) + (1, 50, 512) → (32, 50, 512)
        #             ↑
        #     Only take first 50 positions' encoding
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        # ========== Step 6: Register as Buffer ==========
        # Register positional encoding as a buffer (not a parameter)
        #
        # register_buffer vs nn.Parameter difference:
        #
        # nn.Parameter:
        # ✓ Updated by optimizer during training (weights change)
        # ✓ Used for learnable weights (like W_q, W_k, W_v)
        #
        # register_buffer:
        # ✓ NOT updated by optimizer (fixed constant)
        # ✓ Moves with model (model.to('cuda') automatically moves it to GPU)
        # ✓ Saved in state_dict (included in checkpoints)
        # ✓ Used for fixed constants (like positional encoding)
        #
        # Why use buffer for positional encoding?
        # - Positional encoding is computed from math formula, it's fixed
        # - No need to learn it during training
        # - But needs to move to GPU/CPU with model
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input word embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            output: Input with positional encoding added, shape (batch_size, seq_len, d_model)

        Complete flow example: "I love eating apples"
            Assume: batch_size=1, seq_len=5, d_model=512

            Step 1: Input (word embeddings)
                x.shape = (1, 5, 512)
                x[0, 0, :] = embedding of "I" (512-dim vector)
                x[0, 1, :] = embedding of "love" (512-dim vector)
                x[0, 2, :] = embedding of "eating" (512-dim vector)
                x[0, 3, :] = embedding of "apples" (512-dim vector)
                x[0, 4, :] = embedding of "<END>" (512-dim vector)

                Problem: These embeddings have NO position information!
                         "I love eating apples" and "apples eating love I"
                         have the same embeddings

            Step 2: Extract positional encodings
                self.pe.shape = (1, 5000, 512)  ← pre-computed
                self.pe[:, :5] extracts first 5 positions
                → shape = (1, 5, 512)

                self.pe[0, 0, :] = position 0 encoding (512-dim vector)
                self.pe[0, 1, :] = position 1 encoding (512-dim vector)
                self.pe[0, 2, :] = position 2 encoding (512-dim vector)
                self.pe[0, 3, :] = position 3 encoding (512-dim vector)
                self.pe[0, 4, :] = position 4 encoding (512-dim vector)

            Step 3: Add (broadcast addition)
                x + self.pe[:, :5]
                (1, 5, 512) + (1, 5, 512) → (1, 5, 512)
                         ↑
                    batch dimension broadcasts automatically

                Result:
                output[0, 0, :] = "I" embedding + position 0 encoding
                output[0, 1, :] = "love" embedding + position 1 encoding
                output[0, 2, :] = "eating" embedding + position 2 encoding
                output[0, 3, :] = "apples" embedding + position 3 encoding
                output[0, 4, :] = "<END>" embedding + position 4 encoding

                Now each word's embedding contains:
                ✓ Semantic information of the word (from word embedding)
                ✓ Position information (from positional encoding)

            Step 4: Dropout
                Randomly set some values to 0 (training only)
                Prevents overfitting
        """
        # ========== Step 1: Extract Positional Encodings ==========
        # x.size(1) is seq_len (sentence length)
        # Example: "I love eating apples" → seq_len = 5
        #
        # self.pe[:, :seq_len] means:
        # - [:, :seq_len] extract first seq_len positions' encodings
        # - Example: from (1, 5000, 512) extract (1, 5, 512)
        #
        # Why not extract all?
        # - Different sentences have different lengths
        # - Short sentence only needs first few positions
        # - Long sentence needs more positions
        # - As long as seq_len <= max_len (5000), we can handle it

        # ========== Step 2: Add Positional Encoding ==========
        # Addition broadcasts automatically:
        # x.shape                = (batch_size, seq_len, d_model)  e.g. (32, 50, 512)
        # self.pe[:, :seq_len]   = (1, seq_len, d_model)           e.g. (1, 50, 512)
        # Result                 = (batch_size, seq_len, d_model)  e.g. (32, 50, 512)
        #
        # Broadcasting process:
        # Each sample in batch gets the SAME positional encoding
        # → Because positional encoding only depends on position, not content
        #
        # .requires_grad_(False) purpose:
        # - Ensure positional encodings don't get gradients computed
        # - Because positional encodings are fixed math formulas, not learnable
        # - Saves memory and computation
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        # ========== Step 3: Dropout ==========
        # Training mode: randomly set 10% of values to 0
        # Evaluation mode: no change (automatically disabled)
        # Purpose: prevent overfitting (regularization technique)
        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    Visualize positional encoding (for understanding)

    This function is not required for the transformer,
    but helps understand the encoding pattern.

    Args:
        d_model: Model dimension
        max_len: Maximum length to visualize
    """
    import matplotlib.pyplot as plt

    # Create positional encoding
    pe = PositionalEncoding(d_model, max_len)

    # Extract the encoding matrix
    # shape: (1, max_len, d_model) -> (max_len, d_model)
    encoding = pe.pe.squeeze(0).numpy()

    # Plot heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.title(f'Positional Encoding (d_model={d_model})')
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    print("Positional encoding visualization saved to positional_encoding_visualization.png")
    plt.show()


if __name__ == "__main__":
    # Test code
    print("=== Testing Positional Encoding ===\n")

    d_model = 512
    batch_size = 2
    seq_len = 10

    # Create positional encoding module
    pe = PositionalEncoding(d_model)

    # Create dummy word embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Add positional encoding
    output = pe(x)
    print(f"Output shape: {output.shape}")

    # Visualize (optional)
    # visualize_positional_encoding(d_model=128, max_len=100)
