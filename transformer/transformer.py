"""
Complete Transformer Model

This is the top-level model that combines:
1. Source and Target Embeddings
2. Positional Encoding
3. Encoder Stack (bidirectional attention)
4. Decoder Stack (masked self-attention + cross-attention)
5. Final Linear Projection to Vocabulary

Architecture:
    Source Sequence
         ↓
    [Embedding + Positional Encoding]
         ↓
    [Encoder Stack] → Memory (encoder outputs)
         ↓
    Target Sequence
         ↓
    [Embedding + Positional Encoding]
         ↓
    [Decoder Stack] ← Memory
         ↓
    [Linear Projection]
         ↓
    Output Logits (vocabulary probabilities)
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder, create_causal_mask
from .positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    """
    Token Embedding Layer

    【What Is This?】
    Converts discrete tokens (word IDs) into continuous vector representations
    that the neural network can process.

    【Why Do We Need This?】
    Neural networks can't directly process discrete symbols (like word IDs).
    They need continuous vectors. Token embeddings map each word in the vocabulary
    to a learned dense vector.

    【How It Works】
    - Input: Token IDs [batch_size, seq_len] (integers)
    - Output: Embeddings [batch_size, seq_len, d_model] (floats)

    Example:
        vocab_size = 10000 (10,000 different words)
        d_model = 512 (each word → 512-dimensional vector)

        Token ID 42 → [0.23, -0.45, 0.12, ..., 0.67]  (512 numbers)
        Token ID 99 → [0.81, 0.34, -0.23, ..., -0.12]  (512 numbers)

    【Scaling Factor】
    We multiply embeddings by sqrt(d_model) to prevent them from being too small
    compared to positional encodings, which ensures stable training.

    This scaling is mentioned in the original Transformer paper (Vaswani et al., 2017).
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize Token Embedding Layer

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            d_model: Dimension of model (embedding size)

        Example:
            vocab_size=10000 means we have 10,000 unique words
            d_model=512 means each word becomes a 512-dimensional vector
        """
        super().__init__()

        # Embedding lookup table: vocab_size × d_model
        # Each row is the embedding vector for one token
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Scaling factor: sqrt(d_model)
        # Prevents embeddings from being overwhelmed by positional encodings
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings

        Args:
            x: Token IDs, shape [batch_size, seq_len]

        Returns:
            Scaled embeddings, shape [batch_size, seq_len, d_model]

        Example:
            Input: [[5, 42, 99]]  # batch=1, seq_len=3
            Output: [[[0.23, -0.45, ...],   # embedding for token 5
                      [0.81, 0.34, ...],    # embedding for token 42
                      [-0.12, 0.67, ...]]]  # embedding for token 99
                    shape: (1, 3, 512)
        """
        # Step 1: Look up embeddings for each token
        # x: (batch, seq_len) → (batch, seq_len, d_model)
        embedded = self.embedding(x)

        # Step 2: Scale by sqrt(d_model)
        # This scaling helps balance the magnitude of embeddings and positional encodings
        return embedded * self.scale


class Transformer(nn.Module):
    """
    Complete Transformer Model

    【What Is This?】
    This is the full Transformer architecture from "Attention Is All You Need" (2017).
    It combines all the components we've built into a complete sequence-to-sequence model.

    【Main Components】

    1. Source Embedding
       - Converts source tokens to vectors
       - Adds positional information

    2. Target Embedding
       - Converts target tokens to vectors
       - Adds positional information

    3. Encoder
       - Processes source sequence
       - Uses bidirectional self-attention (can see entire input)
       - Produces "memory" (encoded representation)

    4. Decoder
       - Generates target sequence
       - Uses masked self-attention (can only see previous tokens)
       - Uses cross-attention to encoder memory
       - Produces contextualized representations

    5. Output Projection
       - Maps decoder output to vocabulary logits
       - Each position gets a probability distribution over all possible tokens

    【Complete Data Flow】

    Training (with known target):
        Source: "I love AI"
            ↓ [Embedding + Position]
        Encoder → Memory: [encoded representation of "I love AI"]
            ↓
        Target: "<start> 我 爱"
            ↓ [Embedding + Position]
        Decoder (+ Memory) → Output: "我 爱 AI"
            ↓ [Linear Projection]
        Logits: [probability distributions]

    Inference (generate target):
        Source: "I love AI"
            ↓ [Embedding + Position]
        Encoder → Memory
            ↓
        Target: "<start>"
            ↓ [Embedding + Position]
        Decoder (+ Memory) → "我"
            ↓
        Target: "<start> 我"
            ↓ [Embedding + Position]
        Decoder (+ Memory) → "爱"
            ↓
        ... (continue until <end> token)

    【Key Design Decisions】

    1. Shared vs Separate Embeddings:
       - We use separate embeddings for source and target
       - They might have different vocabularies (e.g., English vs Chinese)
       - But both have the same d_model dimension

    2. Weight Tying:
       - Optionally, we can share weights between target embedding and output projection
       - This reduces parameters and often improves performance
       - Not implemented here for clarity

    3. Positional Encoding:
       - Added to embeddings before encoder/decoder
       - Uses sinusoidal functions (sin/cos with different frequencies)
       - Allows model to understand word order

    【Shape Tracking】
    Throughout the model, we maintain these shapes:

    - Source tokens: (batch, src_len)
    - Target tokens: (batch, tgt_len)
    - Source embeddings: (batch, src_len, d_model)
    - Target embeddings: (batch, tgt_len, d_model)
    - Encoder output (memory): (batch, src_len, d_model)
    - Decoder output: (batch, tgt_len, d_model)
    - Final logits: (batch, tgt_len, tgt_vocab_size)

    The d_model dimension (typically 512) is preserved throughout the entire model!
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        activation: str = "relu"
    ):
        """
        Initialize Complete Transformer Model

        Args:
            src_vocab_size: Size of source vocabulary
                Example: 10000 for 10,000 English words

            tgt_vocab_size: Size of target vocabulary
                Example: 8000 for 8,000 Chinese characters

            d_model: Dimension of model (embedding size)
                Default: 512 (from original paper)
                This is the "width" of the model

            num_heads: Number of attention heads
                Default: 8 (from original paper)
                d_model must be divisible by num_heads

            num_encoder_layers: Number of encoder layers to stack
                Default: 6 (from original paper)
                More layers = deeper model, can learn more complex patterns

            num_decoder_layers: Number of decoder layers to stack
                Default: 6 (from original paper)
                Typically same as num_encoder_layers

            d_ff: Dimension of feedforward network
                Default: 2048 (from original paper)
                Usually 4× the d_model
                This is the "expansion factor" in FFN

            dropout: Dropout probability for regularization
                Default: 0.1 (10% dropout)
                Prevents overfitting by randomly dropping connections

            max_seq_length: Maximum sequence length
                Default: 5000
                Determines size of positional encoding table
                Must be ≥ longest sequence in your data

            activation: Activation function in FFN
                Options: "relu" or "gelu"
                Default: "relu" (from original paper)

        【Parameter Count Example】
        With default settings (src_vocab=10000, tgt_vocab=8000):
        - Source embeddings: 10000 × 512 = 5.1M parameters
        - Target embeddings: 8000 × 512 = 4.1M parameters
        - Encoder: ~6M parameters per layer × 6 = 36M
        - Decoder: ~9M parameters per layer × 6 = 54M
        - Output projection: 512 × 8000 = 4.1M
        - Total: ~103M parameters

        This is a medium-sized model by modern standards.
        For CPU training, we typically use smaller values (d_model=256, layers=2-4).
        """
        super().__init__()

        # ============================================================
        # 1. Embedding Layers
        # ============================================================
        # Convert token IDs to continuous vectors

        # Source embedding (e.g., English words → vectors)
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)

        # Target embedding (e.g., Chinese characters → vectors)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # ============================================================
        # 2. Positional Encoding
        # ============================================================
        # Add position information to embeddings
        # Shared between encoder and decoder since position encoding is universal

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )

        # ============================================================
        # 3. Encoder Stack
        # ============================================================
        # Process source sequence with bidirectional attention

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation
        )

        # ============================================================
        # 4. Decoder Stack
        # ============================================================
        # Generate target sequence with masked self-attention + cross-attention

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation
        )

        # ============================================================
        # 5. Output Projection
        # ============================================================
        # Map decoder output to vocabulary logits
        # Projects from d_model dimensions to vocabulary size

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # ============================================================
        # 6. Initialize Parameters
        # ============================================================
        # Use Xavier/Glorot initialization for better training stability
        # This is important for deep networks!

        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize model parameters with Xavier uniform initialization

        【Why Initialize?】
        Proper initialization is crucial for training deep networks:
        - Too small → vanishing gradients
        - Too large → exploding gradients
        - Xavier initialization balances both

        【What Gets Initialized?】
        - Linear layers (weights and biases)
        - Embedding layers

        LayerNorm and other components have their own default initialization.
        """
        for p in self.parameters():
            if p.dim() > 1:
                # Multi-dimensional parameters (weights) → Xavier uniform
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode source sequence

        【What Does This Do?】
        Processes the source sequence (e.g., English sentence) through:
        1. Token embedding
        2. Positional encoding
        3. Encoder stack

        Result: "Memory" - encoded representation of the source

        Args:
            src: Source token IDs, shape [batch_size, src_len]
                Example: [[5, 42, 99, 103]] (4 tokens)

            src_mask: Optional mask for source, shape [batch_size, 1, 1, src_len]
                Used to mask padding tokens
                1 = attend, 0 = ignore

        Returns:
            Encoder output (memory), shape [batch_size, src_len, d_model]

        【Example Flow】
        Input tokens: [5, 42, 99]
            ↓ [Token Embedding]
        Embeddings: [[0.23, ...], [0.81, ...], [-0.12, ...]]  (each 512-dim)
            ↓ [Add Positional Encoding]
        Position-aware: [[0.25, ...], [0.79, ...], [-0.15, ...]]
            ↓ [Encoder Stack - 6 layers of attention + FFN]
        Memory: [[0.67, ...], [0.34, ...], [0.12, ...]]  (encoded representation)
        """
        # Step 1: Convert token IDs to embeddings
        # src: (batch, src_len) → (batch, src_len, d_model)
        src_embedded = self.src_embedding(src)

        # Step 2: Add positional encoding
        # Tells the model where each word is in the sequence
        src_encoded = self.positional_encoding(src_embedded)

        # Step 3: Pass through encoder stack
        # Each layer applies self-attention + FFN
        memory = self.encoder(src_encoded, src_mask)

        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode target sequence given encoder memory

        【What Does This Do?】
        Processes the target sequence (e.g., Chinese characters) through:
        1. Token embedding
        2. Positional encoding
        3. Decoder stack (with access to encoder memory)

        Result: Contextualized representations ready for output projection

        Args:
            tgt: Target token IDs, shape [batch_size, tgt_len]
                Example: [[1, 203, 456]] (3 tokens including <start>)

            memory: Encoder output, shape [batch_size, src_len, d_model]
                The encoded source sequence

            tgt_mask: Optional causal mask for target, shape [batch_size, 1, tgt_len, tgt_len]
                Prevents attending to future positions
                Lower triangular matrix: [[1,0,0], [1,1,0], [1,1,1]]

            src_mask: Optional mask for source, shape [batch_size, 1, 1, src_len]
                Used in cross-attention to mask source padding

        Returns:
            Decoder output, shape [batch_size, tgt_len, d_model]

        【Example Flow - Translation】
        Target tokens: [<start>, 我, 爱]
            ↓ [Token Embedding]
        Embeddings: [[0.45, ...], [0.23, ...], [0.67, ...]]
            ↓ [Add Positional Encoding]
        Position-aware: [[0.47, ...], [0.21, ...], [0.69, ...]]
            ↓ [Decoder Stack - 6 layers]
              Each layer:
              - Masked self-attention (look at previous words)
              - Cross-attention (look at source: "I love AI")
              - Feedforward
        Output: [[0.89, ...], [0.34, ...], [0.56, ...]]
        """
        # Step 1: Convert token IDs to embeddings
        # tgt: (batch, tgt_len) → (batch, tgt_len, d_model)
        tgt_embedded = self.tgt_embedding(tgt)

        # Step 2: Add positional encoding
        tgt_encoded = self.positional_encoding(tgt_embedded)

        # Step 3: Pass through decoder stack
        # Each layer uses:
        # - Masked self-attention on target
        # - Cross-attention to source (memory)
        # - Feedforward network
        output = self.decoder(tgt_encoded, memory, tgt_mask, src_mask)

        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Complete forward pass through the Transformer

        【What Does This Do?】
        This is the main entry point for the model. It performs:
        1. Encode source sequence → memory
        2. Decode target sequence using memory → contextualized output
        3. Project to vocabulary → logits

        Used during training with teacher forcing (providing correct targets).

        Args:
            src: Source token IDs, shape [batch_size, src_len]
                Example: [[5, 42, 99, 103]] (English words)

            tgt: Target token IDs, shape [batch_size, tgt_len]
                Example: [[1, 203, 456, 789]] (Chinese characters)

            src_mask: Optional source mask, shape [batch_size, 1, 1, src_len]
                Masks padding in source

            tgt_mask: Optional target mask, shape [batch_size, 1, tgt_len, tgt_len]
                Masks padding and future positions in target
                Typically a causal mask (lower triangular)

        Returns:
            Output logits, shape [batch_size, tgt_len, tgt_vocab_size]
            These are raw scores before softmax

        【Complete Example - Machine Translation】

        Source: "I love AI" → [5, 42, 99]
        Target: "<start> 我 爱 AI" → [1, 203, 456, 789]

        Step 1: Encode source
            [5, 42, 99]
                ↓ embedding + positional
            [[0.23, ...], [0.81, ...], [-0.12, ...]]
                ↓ encoder (6 layers of self-attention + FFN)
            memory: [[0.67, ...], [0.34, ...], [0.12, ...]]

        Step 2: Decode target
            [1, 203, 456, 789]
                ↓ embedding + positional
            [[0.45, ...], [0.23, ...], [0.67, ...], [0.89, ...]]
                ↓ decoder (6 layers of masked self-attn + cross-attn + FFN)
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]

        Step 3: Project to vocabulary
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]
                ↓ linear (d_model → tgt_vocab_size)
            logits: [batch, 4, 8000]
                Position 0: probability distribution over 8000 words for "我"
                Position 1: probability distribution over 8000 words for "爱"
                Position 2: probability distribution over 8000 words for "AI"
                Position 3: probability distribution over 8000 words for <end>

        【Training vs Inference】

        Training (teacher forcing):
            - We have the correct target sequence
            - Pass entire target through at once
            - Use causal mask to prevent cheating
            - Fast and parallelizable

        Inference (autoregressive):
            - Generate one token at a time
            - Use previous predictions as input
            - Slower but necessary for generation
            - See generate() method below
        """
        # Step 1: Encode source sequence
        # src: (batch, src_len) → memory: (batch, src_len, d_model)
        memory = self.encode(src, src_mask)

        # Step 2: Decode target sequence using memory
        # tgt: (batch, tgt_len) → output: (batch, tgt_len, d_model)
        output = self.decode(tgt, memory, tgt_mask, src_mask)

        # Step 3: Project to vocabulary
        # output: (batch, tgt_len, d_model) → logits: (batch, tgt_len, tgt_vocab_size)
        logits = self.output_projection(output)

        return logits

    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        start_token: int = 1,
        end_token: int = 2
    ) -> torch.Tensor:
        """
        Generate target sequence autoregressively (one token at a time)

        【What Does This Do?】
        This is used during inference to generate translations/responses.
        Unlike training (where we have the full target), here we:
        1. Start with just <start> token
        2. Generate next token
        3. Append it to sequence
        4. Repeat until <end> token or max_len

        This is called "autoregressive generation" - each token depends on all previous tokens.

        Args:
            src: Source token IDs, shape [batch_size, src_len]
                The input sentence to translate

            src_mask: Optional source mask, shape [batch_size, 1, 1, src_len]

            max_len: Maximum length to generate
                Stop even if <end> token not reached

            start_token: ID of <start> token
                Typically 1 or <bos> (beginning of sequence)

            end_token: ID of <end> token
                Typically 2 or <eos> (end of sequence)

        Returns:
            Generated token IDs, shape [batch_size, generated_len]

        【Generation Process Example】

        Input: "I love AI" → [5, 42, 99]

        Step 0: Encode source
            memory = encode([5, 42, 99])

        Step 1: Start with <start>
            tgt = [1]  # [<start>]
            logits = decode([1], memory)
            next_token = argmax(logits[-1]) = 203  # "我"
            tgt = [1, 203]

        Step 2: Generate next token
            tgt = [1, 203]  # [<start>, 我]
            logits = decode([1, 203], memory)
            next_token = argmax(logits[-1]) = 456  # "爱"
            tgt = [1, 203, 456]

        Step 3: Continue...
            tgt = [1, 203, 456]  # [<start>, 我, 爱]
            logits = decode([1, 203, 456], memory)
            next_token = argmax(logits[-1]) = 789  # "AI"
            tgt = [1, 203, 456, 789]

        Step 4: End
            tgt = [1, 203, 456, 789]  # [<start>, 我, 爱, AI]
            logits = decode([1, 203, 456, 789], memory)
            next_token = argmax(logits[-1]) = 2  # <end>
            STOP! Return [1, 203, 456, 789, 2]

        Final output (removing <start>): "我 爱 AI"

        【Why Is This Slow?】
        - Must run decoder forward pass N times for sequence of length N
        - Can't parallelize - each token depends on previous
        - This is inherent to autoregressive models
        - Modern optimization: KV-caching (not implemented here)
        """
        # Set model to evaluation mode
        # Disables dropout and other training-specific behaviors
        self.eval()

        batch_size = src.size(0)
        device = src.device

        # Step 1: Encode source sequence once
        # We only need to do this once since source doesn't change
        memory = self.encode(src, src_mask)

        # Step 2: Initialize target sequence with <start> token
        # Shape: (batch_size, 1) - just one token per batch item
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Step 3: Generate tokens one by one
        with torch.no_grad():  # Don't compute gradients during inference
            for _ in range(max_len - 1):  # -1 because we already have <start>

                # Create causal mask for current target sequence
                # Ensures we only attend to previous positions
                tgt_len = tgt.size(1)
                tgt_mask = create_causal_mask(tgt_len).to(device)

                # Decode current target sequence
                # tgt: (batch, current_len) → output: (batch, current_len, d_model)
                output = self.decode(tgt, memory, tgt_mask, src_mask)

                # Project last position to vocabulary
                # output[:, -1, :]: (batch, d_model) → logits: (batch, tgt_vocab_size)
                logits = self.output_projection(output[:, -1, :])

                # Get most likely next token
                # For each batch item, select token with highest score
                # logits: (batch, tgt_vocab_size) → next_token: (batch, 1)
                next_token = logits.argmax(dim=-1, keepdim=True)

                # Append next token to sequence
                # tgt: (batch, current_len) → (batch, current_len + 1)
                tgt = torch.cat([tgt, next_token], dim=1)

                # Check if all sequences in batch have generated <end> token
                # If so, we can stop early
                if (next_token == end_token).all():
                    break

        return tgt

    def count_parameters(self) -> int:
        """
        Count total trainable parameters in the model

        【Why Does This Matter?】
        - Larger models are more powerful but slower to train
        - Helps estimate memory requirements
        - Useful for comparing model sizes

        Returns:
            Total number of trainable parameters

        Example output:
            ~103M parameters for default configuration
            ~25M parameters for small model (d_model=256, layers=2)
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_seq_length: int = 5000
) -> Transformer:
    """
    Factory function to create a Transformer model with standard configuration

    【Why Use This?】
    Convenience function for creating common model configurations.
    Ensures encoder and decoder have the same number of layers (common practice).

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of layers (same for encoder and decoder)
        d_ff: Feedforward dimension
        dropout: Dropout rate
        max_seq_length: Maximum sequence length

    Returns:
        Initialized Transformer model

    【Common Configurations】

    Original Paper (Transformer Base):
        d_model=512, num_heads=8, num_layers=6, d_ff=2048
        ~60M parameters

    Transformer Big:
        d_model=1024, num_heads=16, num_layers=6, d_ff=4096
        ~210M parameters

    Small (CPU-friendly):
        d_model=256, num_heads=4, num_layers=2, d_ff=1024
        ~10M parameters (good for learning and small datasets)

    Tiny (very fast):
        d_model=128, num_heads=4, num_layers=2, d_ff=512
        ~3M parameters (demo purposes)
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )

    return model
