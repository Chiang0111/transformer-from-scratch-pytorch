"""
Transformer Decoder Layer

Integrates all components:
1. Masked Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Cross-Attention to Encoder Output
4. Residual Connection + Layer Normalization
5. Position-wise Feedforward Network
6. Residual Connection + Layer Normalization

Architecture:
    Input
     ↓
    [Masked Self-Attention] → Add & Norm
     ↓
    [Cross-Attention to Encoder] → Add & Norm
     ↓
    [Feedforward Network] → Add & Norm
     ↓
    Output
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    【What Is This?】
    This is the core building block of the Transformer Decoder!
    A complete Decoder consists of multiple DecoderLayers stacked (typically 6 layers)

    【Key Difference from Encoder】
    Encoder has 2 sub-layers, Decoder has 3 sub-layers:

    Encoder:
        1. Self-Attention (看整个输入句子)
        2. Feedforward

    Decoder:
        1. Masked Self-Attention (只能看已生成的部分，不能看未来)
        2. Cross-Attention (看 Encoder 的输出，获取源语言信息)
        3. Feedforward

    【Complete Architecture】
        Input x (batch, tgt_len, d_model)
         ↓
        ┌─────────────────────────────────────┐
        │  Masked Multi-Head Self-Attention   │  ← Sub-layer 1: Look at generated tokens
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + MaskedAttention(x), then normalize
         ↓
        ┌─────────────────────────────────────┐
        │  Multi-Head Cross-Attention         │  ← Sub-layer 2: Get info from encoder
        │  (Q from decoder, K,V from encoder) │
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + CrossAttention(x, enc), then normalize
         ↓
        ┌─────────────────────────────────────┐
        │  Feedforward Network (FFN)          │  ← Sub-layer 3: Process information
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x), then normalize
         ↓
        Output (batch, tgt_len, d_model)

    【What Is Masked Self-Attention?】
    Problem: In language generation, we generate one word at a time
    - When generating word 3, we've only seen words 0, 1, 2
    - We CANNOT see words 4, 5, 6... (they don't exist yet!)

    Solution: Use causal mask (also called look-ahead mask)
    - When processing position i, can only attend to positions ≤ i
    - Prevents "cheating" by looking at future tokens

    Concrete example: "I love eating apples"
        Position 0 ("I"):      can see: ["I"]
        Position 1 ("love"):   can see: ["I", "love"]
        Position 2 ("eating"): can see: ["I", "love", "eating"]
        Position 3 ("apples"): can see: ["I", "love", "eating", "apples"]

    Mask matrix (1 = can see, 0 = cannot see):
        [[1, 0, 0, 0],  ← position 0 sees only position 0
         [1, 1, 0, 0],  ← position 1 sees positions 0-1
         [1, 1, 1, 0],  ← position 2 sees positions 0-2
         [1, 1, 1, 1]]  ← position 3 sees positions 0-3

    【What Is Cross-Attention?】
    Purpose: Let decoder "look at" the encoder's output
    - In translation: decoder looks at source sentence (English)
                     to generate target sentence (French)

    Mechanism: Different from self-attention!
    - Self-Attention: Q, K, V all from same input (decoder)
    - Cross-Attention: Q from decoder, K and V from encoder

    Analogy:
        Self-Attention: "What have I said so far?"
        Cross-Attention: "What did the original sentence say?"

    Concrete example (English→French translation):
        Encoder input: "I love eating apples"
        Decoder generating: "J'aime manger ..."

        When generating "manger" (eating):
        - Masked Self-Attention: looks at ["J'aime"]
        - Cross-Attention: looks at ["I", "love", "eating", "apples"]
                          finds "eating" is most relevant!
        → Helps generate correct word "manger"

    【Why This Order?】
    1. Masked Self-Attention first: understand what we've generated
       - Look at the target sequence generated so far
       - Build context from previously generated words

    2. Cross-Attention next: get information from source
       - Look at encoder output (source sentence)
       - Find which source words are relevant now

    3. FFN last: process combined information
       - Combine info from "what we've generated" and "source sentence"
       - Non-linear transformation
       - Prepare for next layer or final prediction

    【Complete Flow Example】
    Assume: English→French translation
    Source: "I love apples" (already encoded by Encoder)
    Target so far: "J'aime" (French for "I love")
    Now generating: next word (should be "les" or something)

    Input:
        x = embeddings of ["J'", "aime", "<CURRENT>"]
        x.shape = (1, 3, 512)  # batch=1, tgt_len=3, d_model=512
        encoder_output = encoded "I love apples"
        encoder_output.shape = (1, 3, 512)  # batch=1, src_len=3, d_model=512

    Step 1: Masked Self-Attention
        - "J'" sees: ["J'"]
        - "aime" sees: ["J'", "aime"]
        - "<CURRENT>" sees: ["J'", "aime", "<CURRENT>"]
        → Each token knows what came before it
        → masked_attn_output.shape = (1, 3, 512)

    Step 2: Add & Norm (first)
        - x = x + masked_attn_output (residual)
        - x = LayerNorm(x) (normalize)
        → x.shape = (1, 3, 512)

    Step 3: Cross-Attention
        - Q from decoder: "what am I looking for in source?"
        - K, V from encoder: "here's the source sentence"
        - "<CURRENT>" might attend highly to "apples"
        → cross_attn_output.shape = (1, 3, 512)

    Step 4: Add & Norm (second)
        - x = x + cross_attn_output (residual)
        - x = LayerNorm(x) (normalize)
        → x.shape = (1, 3, 512)

    Step 5: FFN
        - Process each position independently
        - 512 → 2048 → 512 (expand→transform→compress)
        → ff_output.shape = (1, 3, 512)

    Step 6: Add & Norm (third)
        - x = x + ff_output (residual)
        - x = LayerNorm(x) (normalize)
        → x.shape = (1, 3, 512)

    Final output: (1, 3, 512) ← same shape as input

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

        # ========== Component 1: Masked Multi-Head Self-Attention ==========
        # This is the first sub-layer, responsible for "看已生成的内容"
        #
        # Self-Attention means:
        # - Query, Key, Value all come from decoder input (self-attending)
        # - Each generated word can see previously generated words
        # - CANNOT see future words (masked)
        #
        # Why masked?
        # - During inference, future words don't exist yet!
        # - During training, we mask to simulate this condition
        # - This is auto-regressive generation (one word at a time)
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== Component 2: Multi-Head Cross-Attention ==========
        # This is the second sub-layer, responsible for "看源语言信息"
        #
        # Cross-Attention means:
        # - Query from decoder (what am I looking for?)
        # - Key and Value from encoder (here's the source info)
        # - Different from self-attention where Q, K, V all from same source
        #
        # Why needed?
        # - Decoder needs to know what the source sentence says!
        # - Example: translating "eating" → need to look back at English "eating"
        # - This is how decoder "attends to" the input sequence
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # ========== Component 3: Position-wise Feedforward Network ==========
        # This is the third sub-layer, responsible for "处理信息"
        #
        # Same as Encoder's FFN:
        # - Independent non-linear transformation for each position
        # - Extract more complex features
        # - Increase model expressiveness
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== Components 4, 5, 6: Three Layer Normalization Layers ==========
        # Why three?
        # - Because we have three sub-layers (Masked Self-Attn, Cross-Attn, FFN)
        # - Each sub-layer needs a LayerNorm
        #
        # Purpose: same as Encoder
        # - Normalize: standardize each sample to mean=0, std=1
        # - Stabilize training: prevent value explosion or vanishing
        # - Speed up convergence: make gradients more stable
        self.norm1 = nn.LayerNorm(d_model)  # For first sub-layer (Masked Self-Attention)
        self.norm2 = nn.LayerNorm(d_model)  # For second sub-layer (Cross-Attention)
        self.norm3 = nn.LayerNorm(d_model)  # For third sub-layer (FFN)

        # ========== Components 7, 8, 9: Three Dropout Layers ==========
        # Why three?
        # - Because we have three sub-layers
        # - Each sub-layer's output needs Dropout (before residual)
        #
        # Purpose: same as Encoder
        # - Prevent overfitting
        # - Make model not over-rely on certain paths
        self.dropout1 = nn.Dropout(dropout)  # For Masked Self-Attention
        self.dropout2 = nn.Dropout(dropout)  # For Cross-Attention
        self.dropout3 = nn.Dropout(dropout)  # For FFN

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder Layer forward pass

        Args:
            x: Target sequence input of shape (batch_size, tgt_len, d_model)
               (typically word embeddings + positional encoding of target)

            encoder_output: Encoder's output of shape (batch_size, src_len, d_model)
                           (encoded source sequence, e.g., English sentence)

            tgt_mask: Target mask of shape (batch_size, 1, tgt_len, tgt_len)
                     Combines:
                     1. Padding mask: ignore <PAD> tokens in target
                     2. Causal mask: prevent looking at future tokens
                     Used for masked self-attention

            src_mask: Source padding mask of shape (batch_size, 1, 1, src_len)
                     Ignore <PAD> tokens in source sequence
                     Used for cross-attention

        Returns:
            output: Decoder layer output of shape (batch_size, tgt_len, d_model)
                   (same dimension as input)

        Complete flow:
            1. Masked Self-Attention: each target token attends to previous tokens only
            2. Add & Norm: residual connection + layer normalization
            3. Cross-Attention: target attends to source (encoder output)
            4. Add & Norm: residual connection + layer normalization
            5. FFN: process each token independently, non-linear transformation
            6. Add & Norm: residual connection + layer normalization

        Concrete example (English→French translation):
            Source (English): "I love apples"
            Target (French): "J'aime les pommes"

            encoder_output = encoded("I love apples")
            encoder_output.shape = (1, 3, 512)

            During training, target input = "J'aime les pommes"
            x.shape = (1, 4, 512)  # batch=1, tgt_len=4, d_model=512

            Sub-layer 1: Masked Self-Attention
                - "J'" sees: ["J'"] only
                - "aime" sees: ["J'", "aime"]
                - "les" sees: ["J'", "aime", "les"]
                - "pommes" sees: ["J'", "aime", "les", "pommes"]
                → Each word knows context before it
                → masked_attn_output.shape = (1, 4, 512)

            Add & Norm 1:
                - x = x + masked_attn_output (residual)
                - x = LayerNorm(x) (normalize)
                → x.shape = (1, 4, 512)

            Sub-layer 2: Cross-Attention
                - Q from decoder: ["J'", "aime", "les", "pommes"]
                - K, V from encoder: ["I", "love", "apples"]
                - "pommes" highly attends to "apples"
                - "aime" highly attends to "love"
                → Gets source information
                → cross_attn_output.shape = (1, 4, 512)

            Add & Norm 2:
                - x = x + cross_attn_output (residual)
                - x = LayerNorm(x) (normalize)
                → x.shape = (1, 4, 512)

            Sub-layer 3: FFN
                - Process each word independently
                - 512 → 2048 → 512 (expand→transform→compress)
                → ff_output.shape = (1, 4, 512)

            Add & Norm 3:
                - x = x + ff_output (residual)
                - x = LayerNorm(x) (normalize)
                → x.shape = (1, 4, 512)

            Final output: (1, 4, 512)
        """
        # ========== Sub-layer 1: Masked Multi-Head Self-Attention ==========

        # Step 1: Masked Self-Attention
        # Q = K = V = x (all three inputs are decoder input, hence "self" attention)
        # But with tgt_mask to prevent looking at future positions
        #
        # What does this do?
        # - Each target token attends to itself and previous tokens
        # - CANNOT attend to future tokens (they don't exist during generation!)
        # - Builds context from what's been generated so far
        #
        # Concrete example ("J'aime les pommes"):
        # - "J'" (pos 0) attends to:
        #   * "J'" (pos 0) ✓ (can see)
        #   * "aime" (pos 1) ✗ (future, masked)
        #   * "les" (pos 2) ✗ (future, masked)
        #   * "pommes" (pos 3) ✗ (future, masked)
        #
        # - "les" (pos 2) attends to:
        #   * "J'" (pos 0) ✓ (past, can see)
        #   * "aime" (pos 1) ✓ (past, can see)
        #   * "les" (pos 2) ✓ (current, can see)
        #   * "pommes" (pos 3) ✗ (future, masked)
        #
        # tgt_mask purpose:
        # 1. Causal mask: prevent seeing future (lower triangular matrix)
        # 2. Padding mask: ignore <PAD> tokens if target has padding
        masked_attn_output = self.self_attention(x, x, x, tgt_mask)
        # masked_attn_output.shape = (batch_size, tgt_len, d_model)

        # Step 2: Dropout + Residual Connection
        #
        # Dropout:
        # - Training: randomly set some values to 0
        # - Prevents overfitting
        masked_attn_output = self.dropout1(masked_attn_output)

        # Residual Connection:
        # x = x + MaskedAttention(x)
        #     ↑            ↑
        #  original   masked attention output
        #  input
        #
        # Why add original input x?
        # 1. Gradient Flow: gradients can flow directly through x (shortcut)
        # 2. Easier Learning: model only needs to learn "modifications" (delta)
        # 3. Preserve Information: even if attention learns poorly, x is still there
        x = x + masked_attn_output  # First residual connection
        # x.shape = (batch_size, tgt_len, d_model)

        # Step 3: Layer Normalization
        #
        # Normalize each sample's each position
        # Formula: output = (x - mean) / std * gamma + beta
        #
        # Why needed?
        # 1. Stabilize numerical range: prevent value explosion/vanishing
        # 2. Speed up convergence: stable input distribution
        # 3. Layer independence: each layer can learn more independently
        x = self.norm1(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== Sub-layer 2: Multi-Head Cross-Attention ==========

        # Step 4: Cross-Attention
        #
        # This is DIFFERENT from self-attention!
        # - Query (Q): from decoder (x) - "what am I looking for?"
        # - Key (K): from encoder (encoder_output) - "what's available in source?"
        # - Value (V): from encoder (encoder_output) - "actual source content"
        #
        # What does this do?
        # - Decoder asks: "which part of source sentence is relevant now?"
        # - Encoder provides: information from source sentence
        # - Attention mechanism finds the match
        #
        # Concrete example (English→French):
        # Source (encoder_output): "I love apples"
        # Target (x): "J'aime ..."
        #
        # When decoder processes "aime" (French for "love"):
        # - Q from "aime": "I'm the word 'love' in French, what's my English source?"
        # - K from encoder: ["I", "love", "apples"]
        # - Attention weights: [0.1, 0.8, 0.1]  ← "aime" highly attends to "love"!
        # - V weighted by attention → mostly gets information from "love"
        #
        # This is how decoder "knows" what source word to translate!
        #
        # Arguments:
        # - query=x: from decoder (target)
        # - key=encoder_output: from encoder (source)
        # - value=encoder_output: from encoder (source)
        # - mask=src_mask: ignore <PAD> in source
        cross_attn_output = self.cross_attention(
            query=x,                    # Q from decoder
            key=encoder_output,         # K from encoder
            value=encoder_output,       # V from encoder
            mask=src_mask               # Ignore source padding
        )
        # cross_attn_output.shape = (batch_size, tgt_len, d_model)
        # Note: output length = query length (tgt_len), not key length!

        # Step 5: Dropout + Residual Connection (second residual)
        #
        # Same pattern as before:
        # 1. Dropout: prevents overfitting
        # 2. Residual: x + CrossAttention(x, encoder)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = x + cross_attn_output  # Second residual connection
        # x.shape = (batch_size, tgt_len, d_model)

        # Step 6: Layer Normalization (second normalization)
        #
        # Normalize again, reason same as above
        x = self.norm2(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== Sub-layer 3: Position-wise Feedforward ==========

        # Step 7: Feedforward Network
        #
        # Same as Encoder's FFN:
        # - Independent non-linear transformation for each position
        # - Unlike Attention which sees other positions, FFN only sees current position
        # - But all positions share same FFN weights
        #
        # Architecture:
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # Why needed?
        # - Attention only "rearranges" information (weighted average)
        # - FFN provides "non-linear transformation"
        # - Allows model to learn more complex features
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, tgt_len, d_model)

        # Step 8: Dropout + Residual Connection (third residual)
        #
        # Flow similar to above:
        # 1. Dropout: prevents overfitting
        # 2. Residual: x + FFN(x)
        ff_output = self.dropout3(ff_output)
        x = x + ff_output  # Third residual connection
        # x.shape = (batch_size, tgt_len, d_model)

        # Step 9: Layer Normalization (third normalization)
        #
        # Final normalization
        x = self.norm3(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # Final output:
        # - shape same as input: (batch_size, tgt_len, d_model)
        # - But content has been transformed by:
        #   * Masked Self-Attention (context from previous words)
        #   * Cross-Attention (information from source)
        #   * FFN (non-linear processing)
        # - Can continue to next DecoderLayer, or output final prediction
        return x


class Decoder(nn.Module):
    """
    Complete Transformer Decoder

    【What Is This?】
    This is the complete Decoder!
    Consists of multiple DecoderLayers stacked together (original paper uses 6)

    【Architecture Diagram】
        Input (batch, tgt_len, d_model)
         ↓
        ┌─────────────────┐
        │  DecoderLayer 1 │  ← Layer 1: Learn basic patterns
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 2 │  ← Layer 2: Learn mid-level patterns
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 3 │  ← Layer 3: Learn high-level patterns
        └─────────────────┘
         ↓
           ... (more layers)
         ↓
        ┌─────────────────┐
        │  DecoderLayer N │  ← Layer N: Learn most abstract patterns
        └─────────────────┘
         ↓
        Layer Normalization  ← Final normalization
         ↓
        Output (batch, tgt_len, d_model)

    【Decoder vs Encoder: What's the Difference?】

    Encoder:
    - Purpose: Understand source sentence
    - Input: Source tokens (e.g., English)
    - Attention: Bi-directional (can see entire sentence)
    - Output: Rich representation of source
    - Used for: Encoding input for translation, classification, etc.

    Decoder:
    - Purpose: Generate target sentence
    - Input: Target tokens (e.g., French)
    - Attention: Uni-directional (can only see previous tokens)
    - Output: Representation for next token prediction
    - Used for: Auto-regressive generation (one token at a time)

    【Why Stack Multiple Layers?】
    Same reasoning as Encoder:

    Layer 1: Local patterns
             - Simple word relationships
             - Basic grammar

    Layers 2-3: Mid-level patterns
                - Phrase structures
                - Semantic roles

    Layers 4-6: High-level semantics
                - Sentence-level meaning
                - Complex dependencies between source and target

    Each layer builds on the previous, learning increasingly abstract features.

    【How Decoder Uses Encoder Output】
    Every DecoderLayer receives encoder_output via cross-attention:
    - Encoder runs once: encodes entire source sentence
    - Decoder runs multiple times: one token at a time during generation
    - Each decoder layer looks at encoder output to get source info

    Think of it like this:
    - Encoder: "Here's what the English sentence means" (runs once)
    - Decoder: "Let me check the English while generating French"
              (checks encoder output at every generation step)

    【Training vs Inference】

    Training (teacher forcing):
    - Input: entire target sentence "J'aime les pommes"
    - Use causal mask to simulate generation
    - All positions processed in parallel (efficient!)
    - Loss computed on all positions at once

    Inference (auto-regressive generation):
    - Start with: "<START>"
    - Generate: "J'" → "J' aime" → "J' aime les" → "J' aime les pommes"
    - One token at a time (slower)
    - Use previous output as next input

    Args:
        num_layers: Number of decoder layers (original paper uses 6)
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

        # ========== Create Multiple DecoderLayers ==========
        # Use nn.ModuleList to store multiple layers
        #
        # Why nn.ModuleList?
        # - Automatically registers all sub-modules' (layers') parameters
        # - Lets PyTorch know these layers are part of the model
        # - So optimizer can find and update these parameters
        #
        # Why not regular Python list?
        # - Regular list: PyTorch doesn't know there are parameters inside
        # - nn.ModuleList: PyTorch auto-registers parameters
        #
        # List comprehension:
        # [DecoderLayer(...) for _ in range(num_layers)]
        # Creates num_layers DecoderLayers
        # Each DecoderLayer has same structure but independent parameters
        #
        # Example with num_layers=6:
        # self.layers[0] ← Layer 1 (parameters A)
        # self.layers[1] ← Layer 2 (parameters B, different from A)
        # self.layers[2] ← Layer 3 (parameters C, different from A, B)
        # ...
        # self.layers[5] ← Layer 6 (parameters F)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
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
        #    - If connecting to output projection (e.g., vocabulary prediction)
        #    - Stable input helps the linear layer learn better
        #
        # 3. Empirically better performance:
        #    - Original paper uses this
        #    - Standard practice in Transformer models
        #
        # Note:
        # - This LayerNorm's parameters are independent
        # - Not any DecoderLayer's internal norm1, norm2, or norm3
        self.norm = nn.LayerNorm(d_model)

        # Store layer count (for external query)
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decoder forward pass

        Args:
            x: Target sequence input of shape (batch_size, tgt_len, d_model)
               (typically word embeddings + positional encoding of target)

            encoder_output: Encoder's output of shape (batch_size, src_len, d_model)
                           (encoded source sequence)

            tgt_mask: Target mask of shape (batch_size, 1, tgt_len, tgt_len)
                     Combines causal mask + padding mask for target

            src_mask: Source padding mask of shape (batch_size, 1, 1, src_len)
                     Ignore <PAD> tokens in source sequence

        Returns:
            output: Decoder output of shape (batch_size, tgt_len, d_model)
                   (decoded representation, ready for final projection)

        Complete flow:
            Input → Layer1 → Layer2 → ... → LayerN → Norm → Output

        Concrete example (English→French translation):
            Source: "I love apples"
            Target: "J'aime les pommes"

            Assume num_layers = 6, d_model = 512

            encoder_output:
                shape = (1, 3, 512)
                Encoded representation of "I love apples"

            Input x:
                shape = (1, 4, 512)
                x[0, 0, :] = "J'" embedding + positional encoding
                x[0, 1, :] = "aime" embedding + positional encoding
                x[0, 2, :] = "les" embedding + positional encoding
                x[0, 3, :] = "pommes" embedding + positional encoding

            Layer 1:
                - Masked Self-Attention: build context from previous French words
                - Cross-Attention: look at English source
                - FFN: extract basic features
                → x.shape = (1, 4, 512)

            Layer 2:
                - More complex relationships between French and English
                - Build on Layer 1's features
                → x.shape = (1, 4, 512)

            Layers 3-6:
                - Progressively extract more abstract features
                - Final layer contains richest contextual information
                → x.shape = (1, 4, 512)

            Final LayerNorm:
                - Normalize final output
                → x.shape = (1, 4, 512)

            Output:
                x[0, 0, :] = "J'" encoding (knows: this is start, next is verb)
                x[0, 1, :] = "aime" encoding (knows: subject is "I", object upcoming)
                x[0, 2, :] = "les" encoding (knows: article, noun coming)
                x[0, 3, :] = "pommes" encoding (knows: this is object, end sentence)

                Each word's representation contains:
                ✓ Context from previous French words (masked self-attention)
                ✓ Information from English source (cross-attention)
                ✓ Rich features from multiple layers (depth)

        【Why Does Each Layer Output Same Shape?】
        - Each layer's input and output dimensions are both d_model
        - This allows:
          1. Using residual connections (x + SubLayer(x))
          2. Stacking arbitrary number of layers
          3. Flexible composition
          4. Same encoder_output can be used by all layers

        【Information Flow Across Layers】
        - Layer 1 output → becomes Layer 2 input
        - Layer 2 output → becomes Layer 3 input
        - ...
        - Final layer contains information from all previous layers
        - Each layer adds richer understanding of source-target relationship
        """
        # ========== Pass Through Each Decoder Layer Sequentially ==========
        # for loop executes in order:
        # x = layer_1(x, encoder_output, tgt_mask, src_mask)
        # x = layer_2(x, encoder_output, tgt_mask, src_mask)
        # x = layer_3(x, encoder_output, tgt_mask, src_mask)
        # ...
        # x = layer_N(x, encoder_output, tgt_mask, src_mask)
        #
        # Important notes:
        # - Each layer's input is previous layer's output (x updated)
        # - encoder_output stays constant (same for all layers)
        # - tgt_mask stays constant (same causal mask for all layers)
        # - src_mask stays constant (same padding mask for all layers)
        #
        # Why encoder_output doesn't change?
        # - Encoder runs once, produces fixed representation of source
        # - Every decoder layer uses this same source representation
        # - Like a reference book: decoder keeps consulting it, but book doesn't change
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            # x.shape always (batch_size, tgt_len, d_model)

        # ========== Final Layer Normalization ==========
        # Normalize final output
        # Ensures output distribution is stable
        x = self.norm(x)

        # Final output:
        # - shape: (batch_size, tgt_len, d_model)
        # - Same shape as input, but content has been transformed
        # - Each target token's representation contains:
        #   * Context from previous target tokens (via masked self-attention)
        #   * Information from source sequence (via cross-attention)
        #   * Rich features from multiple layers (via depth)
        # - This output can be:
        #   * Connected to output projection (linear layer to vocabulary)
        #   * Used for next token prediction
        #   * Used for other downstream tasks
        return x


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create causal mask (also called look-ahead mask) for decoder self-attention

    【What Is This?】
    A lower triangular matrix that prevents positions from attending to future positions.
    Used in decoder's masked self-attention.

    【Why Needed?】
    During auto-regressive generation, we generate one token at a time:
    - When generating token 3, we've only seen tokens 0, 1, 2
    - We CANNOT see tokens 4, 5, 6... (they don't exist yet!)
    - During training, we simulate this by masking future positions

    【Mask Format】
    Returns a matrix where:
    - 1 = can attend (position is visible)
    - 0 = cannot attend (position is masked)

    【Example】
    For size=4 (sentence with 4 tokens):

    [[1, 0, 0, 0],   ← Position 0 can only see position 0
     [1, 1, 0, 0],   ← Position 1 can see positions 0-1
     [1, 1, 1, 0],   ← Position 2 can see positions 0-2
     [1, 1, 1, 1]]   ← Position 3 can see positions 0-3

    Visual representation:
    ```
           pos 0  1  2  3
    pos 0:  ✓  ✗  ✗  ✗
    pos 1:  ✓  ✓  ✗  ✗
    pos 2:  ✓  ✓  ✓  ✗
    pos 3:  ✓  ✓  ✓  ✓
    ```

    This is called "causal" because:
    - Information flows causally (past → present)
    - Cannot flow backwards (future → present)
    - Respects temporal order

    Args:
        size: Sequence length (number of tokens)

    Returns:
        mask: Causal mask of shape (1, 1, size, size)
              Shape explanation:
              - First 1: batch dimension (broadcasts across batches)
              - Second 1: head dimension (broadcasts across attention heads)
              - (size, size): actual mask matrix (query_len × key_len)

    Usage in decoder:
        tgt_len = 5
        causal_mask = create_causal_mask(tgt_len)
        # causal_mask.shape = (1, 1, 5, 5)

        # Use in attention:
        attention(query, key, value, mask=causal_mask)
        # Each position can only attend to itself and previous positions
    """
    # ========== Create Lower Triangular Matrix ==========
    # torch.tril creates a lower triangular matrix
    # torch.ones(size, size) creates all-ones matrix
    # tril sets upper triangle to 0, keeps lower triangle as 1
    #
    # Example for size=4:
    # torch.ones(4, 4):
    # [[1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1]]
    #
    # torch.tril(...):
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    #
    # This is exactly the causal mask we want!
    mask = torch.tril(torch.ones(size, size))
    # mask.shape = (size, size)

    # ========== Add Batch and Head Dimensions ==========
    # Reshape from (size, size) to (1, 1, size, size)
    # Why?
    # - Attention expects mask shape: (batch, heads, query_len, key_len)
    # - (1, 1, size, size) broadcasts to (batch, heads, size, size)
    #
    # Broadcasting example:
    # mask.shape = (1, 1, 4, 4)
    # attention scores.shape = (32, 8, 4, 4)  # batch=32, heads=8
    # → mask broadcasts to (32, 8, 4, 4) automatically
    #
    # .unsqueeze(0) adds dimension at position 0: (size, size) → (1, size, size)
    # .unsqueeze(0) again adds dimension at position 0: (1, size, size) → (1, 1, size, size)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape = (1, 1, size, size)

    return mask


if __name__ == "__main__":
    # Test code
    print("=== Testing Causal Mask ===\n")

    # Create a small causal mask to visualize
    causal_mask = create_causal_mask(5)
    print(f"Causal mask shape: {causal_mask.shape}")
    print("Causal mask (5x5):")
    print(causal_mask.squeeze().numpy())
    print("\n(1 = can see, 0 = cannot see)")

    print("\n=== Testing Decoder Layer ===\n")

    batch_size = 2
    src_len = 6  # Source sequence length (e.g., English)
    tgt_len = 5  # Target sequence length (e.g., French)
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # Create decoder layer
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

    # Create dummy inputs
    x = torch.randn(batch_size, tgt_len, d_model)  # Target input
    encoder_output = torch.randn(batch_size, src_len, d_model)  # Encoder output

    print(f"Target input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")

    # Create masks
    tgt_mask = create_causal_mask(tgt_len)  # Causal mask for target
    src_mask = torch.ones(batch_size, 1, 1, src_len)  # No padding in source (all 1s)

    # Forward pass
    output = decoder_layer(x, encoder_output, tgt_mask, src_mask)
    print(f"Decoder layer output shape: {output.shape}")

    print("\n=== Testing Full Decoder (6 layers) ===\n")

    num_layers = 6
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = decoder(x, encoder_output, tgt_mask, src_mask)
    print(f"Full decoder output shape: {output_full.shape}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nFull decoder ({num_layers} layers) total parameters: {total_params:,}")
    print(f"Approximately {total_params / 1e6:.1f}M parameters")
