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

    【Why Do We Need FFN? Isn't Attention Enough?】
    Problem: Attention only "rearranges" information, doesn't "transform" it!

    Analogy:
        Attention is like: Searching for books in a library (gathering information)
        FFN is like: Reading and thinking about the books (processing information)

    Technical perspective:
        - Attention is essentially weighted averaging (linear combination)
        - No non-linear transformation
        - Cannot learn complex patterns

    FFN provides:
        - Non-linear transformation (ReLU/GELU)
        - Larger representation space (512 → 2048 → 512)
        - Increases model expressiveness

    Architecture:
    ```
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
    Example: Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)
    ```

    【What Does "Position-wise" Mean?】
    - Processes each position (token) independently
    - Unlike Attention which looks at the whole sequence
    - But all positions share the same weights (like CNN)

    【Why d_ff = 4 * d_model?】
    - Original paper's setting ("Attention is All You Need")
    - Empirically works well
    - Provides sufficient capacity for learning complex features

    【ReLU vs GELU】
    - ReLU: max(0, x)
      * Simple, fast
      * May have "dying ReLU" problem
      * Used in standard Transformer

    - GELU: Gaussian Error Linear Unit
      * Smoother, often better performance
      * Slightly slower
      * Used by GPT, BERT

    Args:
        d_model: Model dimension (input/output dimension, e.g., 512)
        d_ff: FFN hidden dimension (typically 4x d_model, e.g., 2048)
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

        # ========== Component 1: First Linear Layer (Expand) ==========
        # d_model → d_ff (e.g., 512 → 2048)
        #
        # This is a fully connected layer
        # Parameters: d_model * d_ff + d_ff
        # Example: 512 * 2048 + 2048 = 1,050,624 parameters
        #
        # Purpose: Expand each token's representation from 512 to 2048 dims
        # → Provides larger space for model to learn complex features
        self.linear1 = nn.Linear(d_model, d_ff)

        # ========== Component 2: Second Linear Layer (Compress) ==========
        # d_ff → d_model (e.g., 2048 → 512)
        #
        # Parameters: d_ff * d_model + d_model
        # Example: 2048 * 512 + 512 = 1,049,088 parameters
        #
        # Purpose: Compress 2048-dim representation back to 512 dims
        # → Return to original dimension, can connect to next layer (or residual)
        self.linear2 = nn.Linear(d_ff, d_model)

        # ========== Component 3: Dropout Layer (Regularization) ==========
        # Dropout is a regularization technique:
        # - Training: randomly set some neuron outputs to 0 (with probability dropout)
        # - Testing: no change (automatically disabled)
        #
        # Why effective?
        # - Prevents model from over-relying on specific neurons
        # - Forces model to learn more robust features
        # - Similar to "training with partial information, so model learns to use what remains"
        #
        # dropout=0.1 means 10% of neurons are randomly disabled
        self.dropout = nn.Dropout(dropout)

        # ========== Component 4: Activation Function Selection ==========
        # Store activation function type ('relu' or 'gelu')
        # Actual activation computation happens in forward method
        self.activation = activation

        # Validate activation function type
        # Only allow 'relu' or 'gelu'
        # Others (like 'sigmoid', 'tanh') don't work well in Transformer
        if activation not in ['relu', 'gelu']:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
               (typically output from Attention)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
                   (same dimension as input)

        Complete flow:
            x → Linear1 → Activation → Dropout → Linear2

        Concrete example ("I love eating apples"):
            Assume input x.shape = (2, 5, 512)
            - batch_size = 2 (two sentences)
            - seq_len = 5 (5 words each)
            - d_model = 512 (each word in 512-dim vector)

            Step 1: Linear1 (Expand)
                Input: (2, 5, 512)
                linear1(x) → (2, 5, 2048)
                → Each word expanded from 512 to 2048 dims

            Step 2: Activation (Non-linear transformation)
                Input: (2, 5, 2048)
                activation → (2, 5, 2048)
                → Introduces non-linearity, allows learning complex patterns
                → Shape unchanged, but values transformed

                ReLU example:
                    Input: [-2, -1, 0, 1, 2]
                    Output: [0, 0, 0, 1, 2]  ← negatives become 0, positives unchanged

                GELU example:
                    Input: [-2, -1, 0, 1, 2]
                    Output: [-0.05, -0.16, 0, 0.84, 1.95]  ← smoother

            Step 3: Dropout (Regularization)
                Training: randomly set 10% of values to 0
                Input: [1, 2, 3, 4, 5]
                Output: [0, 2, 3, 0, 5]  ← random (different each time)

                Testing: no change
                Input: [1, 2, 3, 4, 5]
                Output: [1, 2, 3, 4, 5]  ← exactly the same

            Step 4: Linear2 (Compress)
                Input: (2, 5, 2048)
                linear2 → (2, 5, 512)
                → Compressed from 2048 back to 512 dims

            Final output: (2, 5, 512)
            → Same shape as input, can connect to residual or next layer
        """
        # ========== Step 1: First Linear Layer + Activation ==========
        # shape: (batch_size, seq_len, d_model) → (batch_size, seq_len, d_ff)
        # Example: (2, 5, 512) → (2, 5, 2048)

        if self.activation == 'relu':
            # ReLU (Rectified Linear Unit): max(0, x)
            # Mathematical definition:
            #   f(x) = x  if x > 0
            #   f(x) = 0  if x ≤ 0
            #
            # Advantages:
            # ✓ Simple, fast computation
            # ✓ No gradient vanishing (positive part has gradient 1)
            #
            # Disadvantages:
            # ✗ "Dying ReLU" problem: if a neuron's input is always negative
            #   its output is always 0, gradient is always 0, can never learn
            #
            # Use case: Standard Transformer (original paper)
            hidden = F.relu(self.linear1(x))

        else:  # gelu
            # GELU (Gaussian Error Linear Unit)
            # Mathematical definition (approximate):
            #   f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
            #
            # Intuition:
            # - Similar to ReLU but smoother
            # - Negative values don't become exactly 0, just close to 0
            # - Has a probabilistic interpretation (based on Gaussian distribution)
            #
            # Advantages:
            # ✓ Smoother (no sharp corner at x=0)
            # ✓ Empirically better performance
            # ✓ No "dying ReLU" problem
            #
            # Disadvantages:
            # ✗ Slightly slower (more complex formula)
            #
            # Use case: GPT, BERT (modern models)
            hidden = F.gelu(self.linear1(x))

        # Now hidden.shape = (batch_size, seq_len, d_ff)
        # Example: (2, 5, 2048)

        # ========== Step 2: Dropout ==========
        # Dropout only active during training!
        #
        # Training mode (model.train()):
        # - Randomly set some values to 0
        # - Probability determined by dropout parameter (here 0.1 = 10%)
        # - Remaining values are scaled up (multiply by 1/(1-dropout)) to maintain expected value
        #
        # Evaluation mode (model.eval()):
        # - Completely unchanged
        # - All neurons active
        #
        # Why different in training vs testing?
        # - Training: want model not to over-rely on certain neurons → randomly disable some
        # - Testing: want model to use full power → all neurons active
        hidden = self.dropout(hidden)

        # ========== Step 3: Second Linear Layer ==========
        # shape: (batch_size, seq_len, d_ff) → (batch_size, seq_len, d_model)
        # Example: (2, 5, 2048) → (2, 5, 512)
        #
        # Purpose:
        # - Compress back to original dimension
        # - Can connect to residual connection (x + FFN(x))
        # - Can connect to next Encoder Layer
        output = self.linear2(hidden)

        # Final output shape: (batch_size, seq_len, d_model)
        # Example: (2, 5, 512)
        # → Same shape as input x
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
