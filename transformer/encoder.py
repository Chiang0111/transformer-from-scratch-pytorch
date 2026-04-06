"""
Transformer Encoder Layer

Integrates all components:
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Position-wise Feedforward Network
4. Residual Connection + Layer Normalization

Architecture:
    Input
     ↓
    [Multi-Head Attention] → Add & Norm
     ↓
    [Feedforward Network]  → Add & Norm
     ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    【What Is This?】
    This is the core building block of Transformer!
    A complete Encoder consists of multiple EncoderLayers stacked (typically 6 layers)

    【Complete Architecture】
    One EncoderLayer contains two sub-layers:

    Sub-layer 1: Multi-Head Self-Attention
        → Lets each word "see" the entire sentence, gather relevant information

    Sub-layer 2: Position-wise FFN
        → Independent non-linear transformation for each word

    Each sub-layer is followed by:
        - Residual Connection: x + SubLayer(x)
        - Layer Normalization: stabilize training

    【Architecture Diagram】
        Input x (batch, seq_len, d_model)
         ↓
        ┌─────────────────────────────┐
        │  Multi-Head Self-Attention  │  ← Sub-layer 1: Gather information
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + Attention(x), then normalize
         ↓
        ┌─────────────────────────────┐
        │  Feedforward Network (FFN)  │  ← Sub-layer 2: Process information
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x), then normalize
         ↓
        Output (batch, seq_len, d_model)

    【Why This Order?】
    1. Attention first: gather information
       - Each word looks at entire sentence
       - Finds which words are relevant
       - Collects related information

    2. FFN next: process information
       - Non-linear transformation on collected information
       - Extract more complex features
       - Increase model expressiveness

    3. Residual + Norm at each step: stabilize training
       - Residual: allows gradients to flow back directly, solves gradient vanishing
       - Norm: stabilizes numerical range, speeds up convergence

    【What Is Residual Connection?】
    Simple idea: output = input + transform(input)

    Without residual:
        output = F(x)
        Problem: if F is complex (many layers), gradients vanish

    With residual:
        output = x + F(x)
        Benefits:
        ✓ Gradients can flow directly through x (shortcut)
        ✓ F only needs to learn "modifications", not "reconstruction"
        ✓ Training more stable, faster

    Analogy:
        Without residual: "Rewrite an entire article" (hard)
        With residual: "Make edits to existing article" (easy)

    【What Is Layer Normalization?】
    Purpose: Normalize each layer's output to a stable range

    Formula:
        output = (x - mean) / std * gamma + beta

    Where:
        mean, std: computed per sample (not across batch)
        gamma, beta: learnable parameters

    Why needed?
    - Stabilize training (prevent value explosion or vanishing)
    - Speed up convergence
    - Keep each layer's input distribution stable

    【Complete Flow Example】
    Assume input: "I love eating apples" (5 words)
    x.shape = (1, 5, 512)  # batch=1, seq_len=5, d_model=512

    Step 1: Self-Attention
        - Each word sees entire sentence
        - "eating" attends to "apples" (what is eaten?)
        - "love" attends to "I" (who loves?)
        → attn_output.shape = (1, 5, 512)

    Step 2: Add & Norm (first time)
        - x = x + attn_output  (residual)
        - x = LayerNorm(x)     (normalize)
        → x.shape = (1, 5, 512)

    Step 3: FFN
        - Process each word independently
        - 512 → 2048 → 512 (expand→transform→compress)
        → ff_output.shape = (1, 5, 512)

    Step 4: Add & Norm (second time)
        - x = x + ff_output  (residual)
        - x = LayerNorm(x)   (normalize)
        → x.shape = (1, 5, 512)

    Final output: (1, 5, 512) ← same shape as input

    Args:
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: FFN hidden dimension (e.g., 2048, typically 4x d_model)
        dropout: Dropout rate (default 0.1)
        activation: FFN activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # ========== Component 1: Multi-Head Self-Attention ==========
        # This is the first sub-layer, responsible for "gathering information"
        #
        # Self-Attention means:
        # - Query, Key, Value all come from the same input (self-attending)
        # - Each word in sentence can see entire sentence
        # - Find which words are relevant
        #
        # Multi-Head means:
        # - Use multiple attention heads (num_heads)
        # - Each head can learn different attention patterns
        # - Example: head1 focuses on subject-verb, head2 on modifier relations...
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== Component 2: Position-wise Feedforward Network ==========
        # This is the second sub-layer, responsible for "processing information"
        #
        # Purpose:
        # - Independent non-linear transformation for each position
        # - Extract more complex features
        # - Increase model expressiveness
        #
        # Architecture: Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== Components 3 & 4: Two Layer Normalization Layers ==========
        # Why two?
        # - Because we have two sub-layers (Attention and FFN)
        # - Each sub-layer needs a LayerNorm
        #
        # LayerNorm purpose:
        # - Normalize: standardize each sample's each position to mean=0, std=1
        # - Stabilize training: prevent value explosion or vanishing
        # - Speed up convergence: make gradients more stable
        #
        # LayerNorm vs BatchNorm difference:
        # - BatchNorm: normalizes same feature across batch (for CNN)
        # - LayerNorm: normalizes all features of one sample (for NLP)
        # - Why LayerNorm in NLP? Because sentence lengths vary, batch hard to align
        self.norm1 = nn.LayerNorm(d_model)  # For first sub-layer (Attention)
        self.norm2 = nn.LayerNorm(d_model)  # For second sub-layer (FFN)

        # ========== Components 5 & 6: Two Dropout Layers ==========
        # Why two?
        # - Because we have two sub-layers (Attention and FFN)
        # - Each sub-layer's output needs Dropout (before residual)
        #
        # Where to apply Dropout?
        # - After sub-layer output
        # - Before residual connection
        # - Flow: SubLayer(x) → Dropout → x + ·
        #
        # Why Dropout here?
        # - Prevent overfitting
        # - Make model not over-rely on certain paths
        self.dropout1 = nn.Dropout(dropout)  # For first sub-layer (Attention)
        self.dropout2 = nn.Dropout(dropout)  # For second sub-layer (FFN)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder Layer forward pass

        Args:
            x: Input sequence of shape (batch_size, seq_len, d_model)
               (typically word embeddings + positional encoding)
            mask: Padding mask of shape (batch_size, 1, 1, seq_len) or None
                 Used to ignore <PAD> tokens

        Returns:
            output: Encoder layer output of shape (batch_size, seq_len, d_model)
                   (same dimension as input)

        Complete flow:
            1. Self-Attention: each token attends to entire sequence, gather relevant info
            2. Add & Norm: residual connection + layer normalization
            3. FFN: process each token independently, non-linear transformation
            4. Add & Norm: residual connection + layer normalization

        Concrete example ("I love eating apples"):
            Input x.shape = (1, 5, 512)  # batch=1, seq_len=5, d_model=512

            Sub-layer 1: Self-Attention
                - "eating" attends to "apples" (what is eaten?)
                - "love" attends to "I" (who loves?)
                - Each word gathers relevant information
                → attn_output.shape = (1, 5, 512)

            Add & Norm 1:
                - x = x + attn_output  (residual)
                - x = LayerNorm(x)     (normalize)
                → x.shape = (1, 5, 512)

            Sub-layer 2: FFN
                - Each word processed independently
                - 512 → 2048 → 512 (expand→transform→compress)
                → ff_output.shape = (1, 5, 512)

            Add & Norm 2:
                - x = x + ff_output  (residual)
                - x = LayerNorm(x)   (normalize)
                → x.shape = (1, 5, 512)

            Final output: (1, 5, 512)
        """
        # ========== Sub-layer 1: Multi-Head Self-Attention ==========

        # Step 1: Self-Attention
        # Q = K = V = x (all three inputs are x, hence "self" attention)
        #
        # What does this do?
        # - Each token in sequence can "see" entire sequence
        # - Compute attention scores between each token pair
        # - Gather relevant information based on attention scores
        #
        # Concrete example ("I love eating apples"):
        # - "eating" attends to:
        #   * "apples" (high attention) ← what is eaten?
        #   * "I" (low attention)
        #   * "love" (low attention)
        # - "love" attends to:
        #   * "I" (high attention) ← who loves?
        #   * "eating" (medium attention) ← loves to do what?
        #   * "apples" (low attention)
        #
        # mask purpose:
        # - If sentence has padding (like "I love eating apples<PAD><PAD>")
        # - mask tells model: don't attend to <PAD>
        # - Avoids model learning meaningless padding information
        attn_output = self.self_attention(x, x, x, mask)
        # attn_output.shape = (batch_size, seq_len, d_model)

        # Step 2: Dropout + Residual Connection
        #
        # First Dropout:
        # - Training: randomly set some values to 0
        # - Prevents overfitting
        attn_output = self.dropout1(attn_output)

        # Then Residual Connection:
        # x = x + attn_output
        #     ↑        ↑
        #  original  attention output
        #  input
        #
        # Why add original input x?
        #
        # 1. Gradient Flow:
        #    Without residual: gradients pass through many layers, may vanish
        #    With residual: gradients can flow directly through x (shortcut)
        #    → Training more stable
        #
        # 2. Different Learning Objective:
        #    Without residual: model must learn to "reconstruct" entire output
        #    With residual: model only needs to learn "modifications" (delta)
        #    → Learning easier
        #
        # 3. Preserve Information:
        #    Even if attn_output learns poorly, x is still there
        #    → Won't completely lose information
        #
        # Analogy:
        # - Without residual: "Rewrite an entire article"
        # - With residual: "Make edits to existing article"
        x = x + attn_output  # This is the residual connection!
        # x.shape = (batch_size, seq_len, d_model) ← dimension unchanged

        # Step 3: Layer Normalization
        #
        # Purpose: Normalize each sample's each position
        # Formula: output = (x - mean) / std * gamma + beta
        #
        # Why needed?
        #
        # 1. Stabilize numerical range:
        #    After multiple operations, values may become very large or small
        #    → Normalize back to mean≈0, std≈1
        #    → Prevent value explosion or vanishing
        #
        # 2. Speed up convergence:
        #    Each layer's input distribution stable
        #    → Optimizer easier to find good update direction
        #    → Training faster
        #
        # 3. Layer independence:
        #    Even if previous layer's output changes, LayerNorm adjusts it back
        #    → Each layer can learn more independently
        x = self.norm1(x)
        # x.shape = (batch_size, seq_len, d_model) ← dimension unchanged, but values normalized

        # ========== Sub-layer 2: Position-wise Feedforward ==========

        # Step 4: Feedforward Network
        #
        # What does this do?
        # - Independent non-linear transformation for each position
        # - Unlike Attention which sees entire sequence, FFN only sees current position
        # - But all positions share same FFN weights
        #
        # Architecture:
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # Why need FFN?
        # - Attention only "rearranges" information (weighted average)
        # - FFN provides "non-linear transformation"
        # - Allows model to learn more complex features
        #
        # Analogy:
        # - Attention: Finding books in library (gathering information)
        # - FFN: Reading and thinking (processing information)
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, seq_len, d_model)

        # Step 5: Dropout + Residual Connection (second residual)
        #
        # Flow similar to above:
        # 1. Dropout: prevents overfitting
        # 2. Residual: x + FFN(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # Second residual connection
        # x.shape = (batch_size, seq_len, d_model)

        # Step 6: Layer Normalization (second normalization)
        #
        # Normalize again, reason same as above
        x = self.norm2(x)
        # x.shape = (batch_size, seq_len, d_model)

        # Final output:
        # - shape same as input: (batch_size, seq_len, d_model)
        # - But content has been transformed by Attention and FFN
        # - Can continue to next EncoderLayer, or as final output
        return x


class Encoder(nn.Module):
    """
    Complete Transformer Encoder

    【What Is This?】
    This is the complete Encoder!
    Consists of multiple EncoderLayers stacked together (original paper uses 6)

    【Architecture Diagram】
        Input (batch, seq_len, d_model)
         ↓
        ┌─────────────────┐
        │  EncoderLayer 1 │  ← Layer 1: Learn basic features
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 2 │  ← Layer 2: Learn mid-level features
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 3 │  ← Layer 3: Learn high-level features
        └─────────────────┘
         ↓
           ... (more layers)
         ↓
        ┌─────────────────┐
        │  EncoderLayer N │  ← Layer N: Learn most abstract features
        └─────────────────┘
         ↓
        Layer Normalization  ← Final normalization
         ↓
        Output (batch, seq_len, d_model)

    【Why Stack Multiple Layers? What Does Depth Mean?】

    Analogy 1: Reading comprehension levels
        Layer 1: Understand words ("cat", "sits", "mat")
        Layer 2: Understand phrases ("sits on the mat")
        Layer 3: Understand sentences ("The cat sits on the mat")
        Layer N: Understand semantics (describing a scene)

    Analogy 2: Deep CNN (computer vision)
        Shallow layers: learn edges, textures (simple features)
        Middle layers: learn shapes, parts (combined features)
        Deep layers: learn objects, scenes (abstract concepts)

    Transformer depth is similar:
        Layer 1: Local patterns
                 - Phrases ("New York", "Apple Inc")
                 - Basic grammar (subject-verb, verb-object)

        Layers 2-3: Mid-level patterns
                    - Phrase structures ("Apple Inc in New York")
                    - Semantic roles (agent, patient)

        Layers 4-6: High-level semantics
                    - Sentence-level semantics
                    - Abstract relationships (causal, comparative)

    【Are Each Layer's Parameters Independent?】
    Yes! Each layer has its own parameters (not shared)

    - Advantage: Each layer can learn different patterns
    - Disadvantage: More parameters (6 layers ≈ 6x parameters)

    If parameters were shared (like Universal Transformer):
    - Advantage: Fewer parameters
    - Disadvantage: Limited expressiveness (each layer does similar things)

    【Why 6 Layers in Original Paper?】
    - This is an empirical value (from experiments)
    - Balances performance and computational cost
    - Deeper (12, 24 layers) may perform better, but:
      * Higher computational cost
      * May overfit
      * Needs more data

    Modern model layer counts:
    - BERT-base: 12 layers
    - BERT-large: 24 layers
    - GPT-3: 96 layers!

    【Input and Output】
    Input:
        - Typically word embeddings + positional encoding
        - shape: (batch_size, seq_len, d_model)

    Output:
        - Encoded representation
        - shape: (batch_size, seq_len, d_model) ← dimension unchanged
        - But content has been transformed through multiple layers, rich with context

    Args:
        num_layers: Number of encoder layers (original paper uses 6, BERT uses 12)
        d_model: Model dimension (e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
        d_ff: FFN hidden dimension (e.g., 2048)
        dropout: Dropout rate (default 0.1)
        activation: FFN activation function ('relu' or 'gelu')
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # ========== Create Multiple EncoderLayers ==========
        # Use nn.ModuleList to store multiple layers
        #
        # Why nn.ModuleList?
        # - Automatically registers all sub-modules' (layers') parameters
        # - Lets PyTorch know these layers are part of the model
        # - So optimizer can find and update these parameters
        #
        # Why not regular Python list?
        # - Regular list: PyTorch doesn't know there are parameters inside, can't train
        # - nn.ModuleList: PyTorch auto-registers parameters, can train
        #
        # List comprehension:
        # [EncoderLayer(...) for _ in range(num_layers)]
        # Creates num_layers EncoderLayers
        # Each EncoderLayer has same "structure" but independent "parameters" (random init)
        #
        # Example with num_layers=6:
        # self.layers[0] ← Layer 1 (parameters A)
        # self.layers[1] ← Layer 2 (parameters B, different from A)
        # self.layers[2] ← Layer 3 (parameters C, different from A, B)
        # ...
        # self.layers[5] ← Layer 6 (parameters F)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # ========== Final Layer Normalization ==========
        # Why one more LayerNorm at the end?
        #
        # 1. Stabilize final output:
        #    - After multiple layer operations, value range may be unstable
        #    - Final LayerNorm ensures output distribution is stable
        #
        # 2. Easier for downstream processing:
        #    - If connecting to Decoder, Decoder's input is stable
        #    - If connecting to classifier, classifier's input is stable
        #
        # 3. Empirically better performance:
        #    - Original paper and BERT both add LayerNorm at the end
        #    - This is an empirical choice
        #
        # Note:
        # - This LayerNorm's parameters are independent
        # - Not any EncoderLayer's internal norm1 or norm2
        self.norm = nn.LayerNorm(d_model)

        # Store layer count (for external query)
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder forward pass

        Args:
            x: Input sequence of shape (batch_size, seq_len, d_model)
               (typically word embeddings + positional encoding)
            mask: Padding mask of shape (batch_size, 1, 1, seq_len) or None
                 (used to ignore <PAD> tokens)

        Returns:
            output: Encoder output of shape (batch_size, seq_len, d_model)
                   (encoded representation)

        Complete flow:
            Input → Layer1 → Layer2 → ... → LayerN → Norm → Output

        Concrete example ("I love eating apples"):
            Assume num_layers = 6, d_model = 512

            Input:
                x.shape = (1, 5, 512)
                x[0, 0, :] = "I" embedding + positional encoding
                x[0, 1, :] = "love" embedding + positional encoding
                x[0, 2, :] = "eating" embedding + positional encoding
                x[0, 3, :] = "apples" embedding + positional encoding
                x[0, 4, :] = "<END>" embedding + positional encoding

            Layer 1:
                - Attention: each word starts attending to related words
                - FFN: extract basic features
                → x.shape = (1, 5, 512)

            Layer 2:
                - Attention: learns more complex relationships based on Layer 1
                - FFN: extract mid-level features
                → x.shape = (1, 5, 512)

            Layers 3-6:
                - Progressively extract more abstract features
                - Final layer contains richest contextual information
                → x.shape = (1, 5, 512)

            Final LayerNorm:
                - Normalize final output
                → x.shape = (1, 5, 512)

            Output:
                x[0, 0, :] = "I" encoding (contains entire sentence context)
                x[0, 1, :] = "love" encoding (contains entire sentence context)
                ...
                Each word's representation contains information from whole sentence!

        【Why Does Each Layer Output Same Shape?】
        - Each layer's input and output dimensions are both d_model
        - This allows:
          1. Using residual connections (x + SubLayer(x))
          2. Stacking arbitrary number of layers
          3. Flexible composition

        【Information Accumulates Across Layers】
        - Layer 1 output → becomes Layer 2 input
        - Layer 2 output → becomes Layer 3 input
        - ...
        - Final layer contains information from all previous layers (cumulative effect)
        """
        # ========== Pass Through Each Encoder Layer Sequentially ==========
        # for loop executes in order:
        # x = layer_1(x, mask)
        # x = layer_2(x, mask)  ← input is layer_1's output
        # x = layer_3(x, mask)  ← input is layer_2's output
        # ...
        # x = layer_N(x, mask)  ← input is layer_{N-1}'s output
        #
        # Note:
        # - Each layer's input is previous layer's output
        # - x is continuously updated (overwritten)
        # - mask stays constant (each layer uses same mask)
        for layer in self.layers:
            x = layer(x, mask)
            # x.shape always (batch_size, seq_len, d_model)

        # ========== Final Layer Normalization ==========
        # Normalize final output
        # Ensures output distribution is stable
        x = self.norm(x)

        # Final output:
        # - shape: (batch_size, seq_len, d_model)
        # - Same shape as input, but content has been transformed through multiple layers
        # - Each token's representation contains entire sequence's contextual information
        # - This output can be:
        #   * Connected to Decoder (in Seq2Seq tasks)
        #   * Connected to classifier (in classification tasks)
        #   * Used for downstream tasks (like BERT's pre-trained representations)
        return x


if __name__ == "__main__":
    # Test code
    print("=== Testing Encoder Layer ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # Create single encoder layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Forward pass
    output = encoder_layer(x)
    print(f"Output shape: {output.shape}")

    # Test with mask
    # Assume first 7 tokens are real, last 3 are padding
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0  # Last 3 positions masked
    output_with_mask = encoder_layer(x, mask)
    print(f"Output with mask shape: {output_with_mask.shape}")

    print("\n=== Testing Full Encoder (6 layers) ===\n")

    num_layers = 6
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = encoder(x, mask)
    print(f"Full encoder output shape: {output_full.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nFull encoder ({num_layers} layers) total parameters: {total_params:,}")
    print(f"Approximately {total_params / 1e6:.1f}M parameters")
