"""
Position-wise Feedforward Network

Why is this needed?
- Attention only rearranges information, doesn't transform it
- FFN provides non-linear transformation, increasing model expressiveness

Architecture:
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)

Typically d_ff = 4 * d_model (e.g., 512 → 2048 → 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Network (FFN)

    Applies the same two-layer fully-connected network to each position
    independently.

    Why "position-wise"?
    - Each position (token) is processed independently
    - Unlike Attention which looks at the whole sequence, FFN only looks at
      the current position
    - But all positions share the same weights

    Architecture details:
        1. First layer: Expand dimensions (d_model → d_ff)
           - Provides larger representation space
           - Like "unpacking" information

        2. Activation: ReLU or GELU
           - Introduces non-linearity
           - ReLU: max(0, x)
           - GELU: smoother version, used by GPT

        3. Dropout: Regularization
           - Randomly drops some neurons
           - Prevents overfitting

        4. Second layer: Compress back (d_ff → d_model)
           - Back to original dimension
           - Like "summarizing" information

    Args:
        d_model: Model dimension (input/output dimension)
        d_ff: Hidden dimension of feedforward network (typically 4x d_model)
        dropout: Dropout rate (default 0.1)
        activation: Activation function type, 'relu' or 'gelu' (default 'relu')
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # First linear layer: expand dimension
        # d_model → d_ff (e.g., 512 → 2048)
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear layer: compress back to original dimension
        # d_ff → d_model (e.g., 2048 → 512)
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout layer: prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Select activation function
        self.activation = activation
        if activation not in ['relu', 'gelu']:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
                   (same dimension as input)

        Computation flow:
            x → Linear1 → Activation → Dropout → Linear2

        Example:
            Assume x.shape = (2, 10, 512)  # batch=2, seq_len=10, d_model=512

            1. linear1(x) → (2, 10, 2048)  # expand to d_ff
            2. activation → (2, 10, 2048)   # non-linear transform
            3. dropout    → (2, 10, 2048)   # random dropout
            4. linear2    → (2, 10, 512)    # compress back to d_model
        """
        # Step 1: First linear layer + activation
        # shape: (batch_size, seq_len, d_model) → (batch_size, seq_len, d_ff)
        if self.activation == 'relu':
            # ReLU: max(0, x)
            # Pros: simple, fast
            # Cons: may have "dying ReLU" problem (some neurons never activate)
            hidden = F.relu(self.linear1(x))
        else:  # gelu
            # GELU: Gaussian Error Linear Unit
            # Pros: smoother, often better performance
            # Cons: slightly slower
            # Used by GPT, BERT
            hidden = F.gelu(self.linear1(x))

        # Step 2: Dropout
        # Training: randomly sets some values to 0
        # Evaluation: automatically turned off
        hidden = self.dropout(hidden)

        # Step 3: Second linear layer
        # shape: (batch_size, seq_len, d_ff) → (batch_size, seq_len, d_model)
        output = self.linear2(hidden)

        return output


class GatedFeedForward(nn.Module):
    """
    Gated Feedforward Network (advanced version, optional)

    This is an improved version of FFN using gating mechanism (similar to LSTM gates).
    Some modern Transformer variants (e.g., GLU) use this architecture.

    Architecture:
        output = Linear2(GELU(Linear1_1(x)) ⊙ Linear1_2(x))

    where ⊙ is element-wise multiplication

    Why is it better?
    - Gating mechanism lets model learn "which information to pass through"
    - Often better performance than standard FFN, but 2x parameters

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension of feedforward network
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Two parallel linear layers (for gating)
        self.linear1_1 = nn.Linear(d_model, d_ff)
        self.linear1_2 = nn.Linear(d_model, d_ff)

        # Output linear layer
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for gated feedforward network

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute two branches
        # Branch 1: activated values
        activated = F.gelu(self.linear1_1(x))

        # Branch 2: gate values (decides which information passes through)
        gate = self.linear1_2(x)

        # Element-wise multiplication (gating mechanism)
        # gate values determine which parts of activated to keep
        gated = activated * gate

        # Dropout and final linear layer
        gated = self.dropout(gated)
        output = self.linear2(gated)

        return output


if __name__ == "__main__":
    # Test code
    print("=== Testing Position-wise Feedforward Network ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    # Create FFN
    ffn = PositionwiseFeedForward(d_model, d_ff)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = ffn(x)
    print(f"Output shape: {output.shape}")

    # Check parameter count
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Linear1 params: {d_model * d_ff + d_ff:,}")
    print(f"Linear2 params: {d_ff * d_model + d_model:,}")

    # Test gated version
    print("\n=== Testing Gated Feedforward Network ===\n")
    gated_ffn = GatedFeedForward(d_model, d_ff)
    output_gated = gated_ffn(x)
    print(f"Gated FFN output shape: {output_gated.shape}")

    gated_params = sum(p.numel() for p in gated_ffn.parameters())
    print(f"Gated FFN total parameters: {gated_params:,}")
    print(f"(approximately {gated_params / total_params:.1f}x standard FFN)")
