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

    Uses sine and cosine functions to encode position information.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
        pos = position index (0, 1, 2, ...)
        i = dimension index
        d_model = model dimension

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        # shape: (max_len, d_model)
        # This matrix is fixed (not learned), so we use register_buffer
        pe = torch.zeros(max_len, d_model)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        # unsqueeze(1) changes shape from (max_len,) to (max_len, 1)
        # Why? To enable broadcasting with (d_model,) vector
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the denominator: 10000^(2i/d_model)
        # First compute 2i/d_model part
        # torch.arange(0, d_model, 2) generates [0, 2, 4, 6, ..., d_model-2]
        # representing even dimension indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Why use exp(-log(...))?
        # Because 10000^(2i/d_model) = exp(log(10000^(2i/d_model))) = exp(2i/d_model * log(10000))
        # and exp(-x * log(10000)) is numerically more stable

        # Even dimensions use sin
        # position * div_term broadcasts: (max_len, 1) * (d_model/2,) -> (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Odd dimensions use cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        # Why? Because actual input is (batch_size, seq_len, d_model)
        # With this 1, PyTorch will automatically broadcast to any batch_size
        pe = pe.unsqueeze(0)

        # Register as buffer
        # register_buffer means:
        # 1. This tensor moves with the model (CPU/GPU)
        # 2. This tensor is saved in model's state_dict
        # 3. But it won't be treated as a parameter (won't be updated by optimizer)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input word embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            output: Input with positional encoding added, shape (batch_size, seq_len, d_model)
        """
        # Extract positional encodings for the sequence length
        # x.size(1) is seq_len
        # self.pe[:, :seq_len] has shape (1, seq_len, d_model)
        # Addition broadcasts to (batch_size, seq_len, d_model)

        # Note: Use .requires_grad_(False) to ensure positional encodings
        # don't get gradients computed (they're fixed, not learned)
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        # Apply dropout (regularization to prevent overfitting)
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
