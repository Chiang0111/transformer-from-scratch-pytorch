"""
Attention Mechanisms

This module implements:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,      # Type hint: specifies expected type
    key: torch.Tensor,        # Python won't enforce this at runtime
    value: torch.Tensor,      # But helps with IDE autocomplete & documentation
    mask: Optional[torch.Tensor] = None  # Optional means can be None or Tensor
) -> tuple[torch.Tensor, torch.Tensor]:  # Return type: tuple of 2 Tensors
    """
    Scaled Dot-Product Attention - The Core of Transformer

    This is THE most important function in the entire Transformer architecture.
    Everything else is built around this mechanism.

    【What is Attention? Library Analogy】
    Imagine you're in a library looking for information:

    Query (Q):  "What am I looking for?"
                (The question you want answered)

    Key (K):    "What information is available?"
                (Index cards describing each book)

    Value (V):  "The actual content"
                (The books themselves)

    Process:
    1. Compare your query against all keys (Q·K^T)
    2. Find which keys match best (softmax)
    3. Retrieve corresponding values weighted by match (attention·V)

    【Concrete Example: "I love eating apples"】
    For the word "eating":

    Query (eating):  "I need to find the object (what is being eaten?)"
    Keys:
      - "I":      "I'm a pronoun, subject" → low match
      - "love":   "I'm a verb"             → low match
      - "eating": "I'm the current word"   → medium match
      - "apples": "I'm a noun, can be object" → HIGH MATCH!

    Result: "eating" pays high attention to "apples"

    【The Formula】
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Let's break this down:

    1. QK^T: Dot product between queries and keys
       - Measures similarity/relevance
       - Each query compares with all keys
       - Result: attention scores matrix

    2. /√d_k: Scaling factor (this is crucial!)
       - Without scaling, scores grow large with higher dimensions
       - Large scores → softmax saturates → gradients vanish
       - Example: if d_k=64, we divide by √64 = 8

    3. softmax(...): Convert scores to probability distribution
       - Each row sums to 1
       - High scores get close to 1, low scores close to 0
       - Creates attention weights

    4. (...) V: Weighted sum of values
       - Use attention weights to combine values
       - This is the actual "attending" step

    【Why Type Hints?】
    Type hints like `query: torch.Tensor` are annotations that:
    - Don't affect runtime (Python ignores them during execution)
    - Help developers understand expected types
    - Enable IDE autocomplete and error checking
    - Can be validated with tools like mypy

    Example:
    ```python
    def add(x: int, y: int) -> int:  # Hints only, not enforced
        return x + y

    add("hello", "world")  # Python allows this! Type hints are just hints.
    ```

    【Arguments】
    Args:
        query: Query tensor
               Shape: (batch_size, num_heads, seq_len_q, d_k)
               Example: (32, 8, 50, 64) = 32 samples, 8 heads, 50 tokens, 64 dims

        key: Key tensor
             Shape: (batch_size, num_heads, seq_len_k, d_k)
             Usually seq_len_k = seq_len_q (self-attention)

        value: Value tensor
               Shape: (batch_size, num_heads, seq_len_v, d_v)
               Usually seq_len_v = seq_len_k and d_v = d_k

        mask: Optional mask tensor
              Shape: (batch_size, 1, 1, seq_len_k) or
                     (batch_size, 1, seq_len_q, seq_len_k)
              Used for:
              - Padding mask: ignore <PAD> tokens
              - Causal mask: prevent looking at future tokens (decoder)

    【Returns】
    Returns:
        output: Attention-weighted output
                Shape: (batch_size, num_heads, seq_len_q, d_v)
                The result of attending to values

        attention_weights: Attention weight matrix
                          Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
                          How much each query attends to each key
                          Each row sums to 1 (probability distribution)

    【Return Type Annotation】
    `-> tuple[torch.Tensor, torch.Tensor]` means:
    - This function returns a tuple
    - The tuple contains exactly 2 elements
    - Both elements are torch.Tensor types
    """
    # ========== Step 1: Get d_k (dimension of keys) ==========
    # query.size(-1) gets the last dimension
    # Example: if query.shape = (32, 8, 50, 64), then d_k = 64
    #
    # Why do we need d_k?
    # - For scaling the attention scores (prevent gradient vanishing)
    # - This is the "scaled" part of "scaled dot-product attention"
    d_k = query.size(-1)

    # ========== Step 2: Compute Attention Scores (Q·K^T) ==========
    # Matrix multiplication: Query × Key^transpose
    #
    # key.transpose(-2, -1) swaps the last two dimensions:
    # Before: (batch, heads, seq_len_k, d_k)
    # After:  (batch, heads, d_k, seq_len_k)
    #
    # Matrix multiplication:
    # (batch, heads, seq_len_q, d_k) × (batch, heads, d_k, seq_len_k)
    # →  (batch, heads, seq_len_q, seq_len_k)
    #
    # What does scores[i,j,q,k] represent?
    # - How much query q attends to key k
    # - Higher value = more relevant
    #
    # Concrete example (seq_len=5, d_k=64):
    # Q[0] = [1, 2, ..., 64]  # query for "eating"
    # K[3] = [3, 1, ..., 32]  # key for "apples"
    # score = Q[0]·K[3] = 1*3 + 2*1 + ... = some large number
    scores = torch.matmul(query, key.transpose(-2, -1))
    # scores.shape = (batch_size, num_heads, seq_len_q, seq_len_k)

    # ========== Step 3: Scale by √d_k (Critical Step!) ==========
    # This is THE defining feature of "Scaled" Dot-Product Attention
    #
    # Why do we scale? Why is this necessary?
    #
    # Problem: Dot products grow with dimensionality
    # - If d_k = 64, dot product might be: 1*2 + 3*4 + ... (64 terms)
    # - Even with small numbers, sum can be very large
    # - Example: average dot product ≈ 0 but variance ≈ d_k
    #
    # What happens without scaling?
    # Let's say d_k = 64 and scores = [200, 180, 150, 120]
    # softmax([200, 180, 150, 120]) ≈ [0.99, 0.01, 0.00, 0.00]
    # → Nearly one-hot! (gradient vanishing)
    #
    # With scaling (divide by √64 = 8):
    # scores = [25, 22.5, 18.75, 15]
    # softmax([25, 22.5, 18.75, 15]) ≈ [0.65, 0.24, 0.08, 0.03]
    # → Smooth distribution (healthy gradients)
    #
    # Why √d_k specifically?
    # - Theoretical analysis shows variance of QK^T is proportional to d_k
    # - Dividing by √d_k normalizes variance to ≈1
    # - Keeps softmax input in reasonable range
    scores = scores / math.sqrt(d_k)
    # Now scores are scaled appropriately

    # ========== Step 4: Apply Mask (If Provided) ==========
    # Masks are used to "ignore" certain positions
    #
    # Two main types of masks:
    #
    # 1. Padding Mask:
    #    Sentence: "I love eating apples <PAD> <PAD>"
    #    Mask:     [1, 1, 1, 1, 0, 0]
    #    → Don't attend to <PAD> tokens (they're meaningless)
    #
    # 2. Causal Mask (for decoder):
    #    When predicting word i, can only see words 0...i-1
    #    Prevents "cheating" by looking at future words
    #    Example (position 2):
    #    Mask: [1, 1, 1, 0, 0]  # can see words 0,1,2 but not 3,4
    #
    # How masking works:
    # - Set masked positions to very negative value (-1e9)
    # - After softmax, exp(-1e9) ≈ 0
    # - These positions get near-zero attention weight
    #
    # Why -1e9 instead of -inf?
    # - -inf can cause NaN in some edge cases
    # - -1e9 is "negative enough" and numerically stable
    if mask is not None:
        # masked_fill: where mask==0, replace with -1e9
        # Example:
        # scores = [[10, 20, 30, 40, 50]]
        # mask   = [[1,  1,  1,  0,  0]]
        # result = [[10, 20, 30, -1e9, -1e9]]
        scores = scores.masked_fill(mask == 0, -1e9)

    # ========== Step 5: Apply Softmax → Attention Weights ==========
    # Softmax converts scores into a probability distribution
    #
    # Formula: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    #
    # Properties:
    # - All values between 0 and 1
    # - Sum of all values = 1 (probability distribution)
    # - Larger inputs → larger outputs (but normalized)
    #
    # dim=-1 means:
    # - Apply softmax over the last dimension (seq_len_k)
    # - Each row becomes a probability distribution
    # - attention_weights[i, j, q, :].sum() = 1
    #
    # Concrete example:
    # Input scores:  [2.0, 1.5, 0.5, -1e9]
    # After softmax: [0.58, 0.32, 0.10, 0.00]
    # Notice:
    # - Highest score (2.0) → highest weight (0.58)
    # - Masked position (-1e9) → nearly zero (0.00)
    # - Sum = 1.00
    attention_weights = F.softmax(scores, dim=-1)
    # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)

    # ========== Step 6: Weighted Sum of Values ==========
    # This is where we actually "attend" to the values!
    #
    # Matrix multiplication:
    # (batch, heads, seq_len_q, seq_len_k) × (batch, heads, seq_len_k, d_v)
    # →  (batch, heads, seq_len_q, d_v)
    #
    # What this does:
    # For each query position, combine all values weighted by attention
    #
    # Concrete example: "I love eating apples"
    # For query "eating":
    #   attention_weights = [0.1, 0.1, 0.2, 0.6]  # high attention on "apples"
    #   values:
    #     V["I"]      = [v1, v2, v3, ...]
    #     V["love"]   = [v4, v5, v6, ...]
    #     V["eating"] = [v7, v8, v9, ...]
    #     V["apples"] = [v10, v11, v12, ...]
    #
    #   output = 0.1*V["I"] + 0.1*V["love"] + 0.2*V["eating"] + 0.6*V["apples"]
    #          = mostly information from "apples"!
    #
    # This is the magic of attention:
    # - Dynamically select relevant information
    # - Different for each position
    # - Learned from data
    output = torch.matmul(attention_weights, value)
    # output.shape = (batch_size, num_heads, seq_len_q, d_v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism

    【Why Do We Need Multiple Heads?】
    Imagine a bank deciding whether to lend you money:

    Single Expert Problem:
    - One expert only looks at credit score
    - Misses other important factors
    - Limited perspective

    Multi-Expert Solution (Multi-Head):
    - Expert 1: Checks credit score
    - Expert 2: Analyzes income stability
    - Expert 3: Reviews assets
    - Expert 4: Examines employment history
    → Combine all opinions for better decision!

    Similarly, in language:

    Single Attention Head Problem:
    - Might only capture subject-verb relationships
    - Misses other important patterns
    - Example: "The bank can refuse to lend money"
      * Only seeing: "bank → refuse" (syntactic)
      * Missing: "refuse → lend" (semantic)

    Multi-Head Attention Solution:
    - Head 1: Subject-verb relationships
    - Head 2: Verb-object relationships
    - Head 3: Modifier relationships
    - Head 4: Positional relationships
    - Head 5-8: Other patterns...
    → Each head learns different aspects!

    【Architecture Overview】
    Flow:
    ```
    Input (batch, seq_len, d_model)
      ↓
    [Linear Projections: W_q, W_k, W_v]
      ↓
    [Split into num_heads]
      ↓
    [Scaled Dot-Product Attention] (in parallel for each head)
      ↓
    [Combine heads]
      ↓
    [Linear Projection: W_o]
      ↓
    Output (batch, seq_len, d_model)
    ```

    【Why This Design?】
    - Multiple heads = multiple perspectives
    - Each head has d_k = d_model / num_heads dimensions
    - Total computation ≈ same as single head with full dimension
    - But much more expressive!

    【Concrete Example】
    d_model = 512, num_heads = 8
    → Each head gets d_k = 512/8 = 64 dimensions
    → 8 different 64-dim "experts" working in parallel
    → Final combination creates rich 512-dim representation

    Args:
        d_model: Model dimension (input/output dimension, e.g., 512)
        num_heads: Number of attention heads (e.g., 8)
                   Note: d_model must be divisible by num_heads
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # ========== Validation: d_model must be divisible by num_heads ==========
        # Why this requirement?
        # - We split d_model equally among heads
        # - Each head gets d_k = d_model // num_heads dimensions
        # - If not divisible, we can't split evenly
        #
        # Example:
        # ✓ d_model=512, num_heads=8  → d_k=64 (works!)
        # ✗ d_model=512, num_heads=7  → d_k=73.14... (doesn't work!)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads

        # ========== Calculate dimension per head ==========
        # Each head operates on d_k dimensions
        # Example: 512 dimensions split across 8 heads = 64 dims per head
        self.d_k = d_model // num_heads  # Integer division

        # ========== Define 4 Linear Layers (Learnable Projections) ==========

        # W_q, W_k, W_v: Project input to Query, Key, Value
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, d_model)
        #
        # Why d_model → d_model and not d_model → d_k?
        # - We project to full d_model first
        # - Then split into num_heads pieces of size d_k each
        # - This is more efficient than doing num_heads separate projections
        #
        # Why are these learnable?
        # - The model learns what questions to ask (W_q)
        # - The model learns what keys to produce (W_k)
        # - The model learns what values to return (W_v)
        #
        # Example with d_model=512:
        # - W_q has 512×512 = 262,144 parameters
        # - W_k has 512×512 = 262,144 parameters
        # - W_v has 512×512 = 262,144 parameters
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection

        # W_o: Output projection
        # After combining all heads, project back to d_model
        # Input: (batch, seq_len, d_model)
        # Output: (batch, seq_len, d_model)
        #
        # Why do we need this?
        # - Integrates information from all heads
        # - Allows model to learn how to combine head outputs
        # - Without it, we'd just be concatenating, not learning to combine
        #
        # Total parameters: 512×512 = 262,144
        self.W_o = nn.Linear(d_model, d_model)  # Output projection

        # Total parameters for Multi-Head Attention:
        # W_q + W_k + W_v + W_o = 4 × (d_model × d_model)
        # Example: 4 × (512 × 512) = 1,048,576 parameters

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split input into multiple attention heads

        Input shape:  (batch_size, seq_len, d_model)
        Output shape: (batch_size, num_heads, seq_len, d_k)

        【What This Does】
        Takes a tensor with d_model dimensions and splits it into num_heads pieces,
        each with d_k dimensions.

        【Visual Example】
        d_model = 512, num_heads = 8, d_k = 64

        Input:
        (batch, seq_len, 512)
        [......512 dimensions......]

        After split:
        (batch, seq_len, 8, 64)
        [64][64][64][64][64][64][64][64]
         ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
        h0  h1  h2  h3  h4  h5  h6  h7

        After transpose:
        (batch, 8, seq_len, 64)
        Now organized by head, so each head can work independently!

        【Why Transpose?】
        We want the batch operations to process all heads in parallel:
        - Before: (batch, seq_len, num_heads, d_k)
          Hard to process heads independently
        - After: (batch, num_heads, seq_len, d_k)
          Easy! Each head is a separate "batch" item
        """
        batch_size, seq_len, d_model = x.size()

        # ========== Step 1: Reshape to split dimensions ==========
        # view() reshapes the tensor without copying data
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)
        #
        # Example with batch=2, seq_len=5, d_model=512, num_heads=8:
        # (2, 5, 512) → (2, 5, 8, 64)
        #
        # The 512 dimensions are split into 8 groups of 64:
        # [0:64]    → head 0
        # [64:128]  → head 1
        # [128:192] → head 2
        # ...
        # [448:512] → head 7
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # ========== Step 2: Transpose to group by head ==========
        # transpose(1, 2) swaps dimensions 1 and 2
        # Before: (batch_size, seq_len, num_heads, d_k)
        # After:  (batch_size, num_heads, seq_len, d_k)
        #
        # Why is this better?
        # - Now "num_heads" dimension comes before "seq_len"
        # - Can process all heads in parallel using batch operations
        # - Each head independently processes its seq_len × d_k data
        #
        # Think of it as having 8 separate attention mechanisms
        # running in parallel, each working on 64-dimensional space
        return x.transpose(1, 2)
        # Output shape: (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple attention heads back together

        Input shape:  (batch_size, num_heads, seq_len, d_k)
        Output shape: (batch_size, seq_len, d_model)

        This is the inverse of split_heads.

        【Visual Example】
        num_heads = 8, d_k = 64, d_model = 512

        Input (after attention):
        (batch, 8, seq_len, 64)
        Head 0: [64 dims]
        Head 1: [64 dims]
        ...
        Head 7: [64 dims]

        After transpose:
        (batch, seq_len, 8, 64)

        After view (concatenate):
        (batch, seq_len, 512)
        [h0: 64][h1: 64]...[h7: 64] = 512 dimensions

        【Why .contiguous()?】
        Technical detail about PyTorch memory layout:
        - transpose() doesn't copy data, just changes the view
        - view() requires memory to be contiguous
        - contiguous() creates a contiguous copy if needed
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # ========== Step 1: Transpose back ==========
        # Swap dimensions 1 and 2
        # (batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, num_heads, d_k)
        #
        # This brings seq_len back to dimension 1
        # Now: each position has num_heads pieces of d_k dimensions
        x = x.transpose(1, 2)

        # ========== Step 2: Merge heads (concatenate) ==========
        # .contiguous() ensures memory is laid out correctly
        # Why needed?
        # - transpose() creates a VIEW of the data (doesn't copy)
        # - The actual memory is still in the original order
        # - view() requires contiguous memory
        # - contiguous() makes a copy if necessary
        #
        # .view() reshapes the tensor
        # (batch_size, seq_len, num_heads, d_k) → (batch_size, seq_len, d_model)
        #
        # Example: (2, 5, 8, 64) → (2, 5, 512)
        # The 8 chunks of 64 dimensions are concatenated:
        # [head0: 64 dims][head1: 64 dims]...[head7: 64 dims] = 512 dims
        #
        # Each position now has information from all 8 heads combined!
        return x.contiguous().view(batch_size, seq_len, self.d_model)
        # Output shape: (batch_size, seq_len, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-head attention forward pass

        Args:
            query: Query tensor, shape (batch_size, seq_len, d_model)
            key:   Key tensor, shape (batch_size, seq_len, d_model)
            value: Value tensor, shape (batch_size, seq_len, d_model)
            mask:  Optional mask, shape (batch_size, 1, 1, seq_len) or
                   (batch_size, 1, seq_len, seq_len)

        Returns:
            output: Multi-head attention output, shape (batch_size, seq_len, d_model)

        【Complete Flow Example: "I love eating apples"】
        Assume: batch=1, seq_len=5, d_model=512, num_heads=8, d_k=64

        Input:
            query = key = value (self-attention)
            shape: (1, 5, 512)
            Each word is a 512-dim vector

        Step 1: Linear Projections
            Q = W_q(query) = (1, 5, 512)
            K = W_k(key) = (1, 5, 512)
            V = W_v(value) = (1, 5, 512)
            (Model learns what questions/keys/values to create)

        Step 2: Split into 8 Heads
            Q = (1, 8, 5, 64)  # 8 different 64-dim queries per word
            K = (1, 8, 5, 64)  # 8 different 64-dim keys per word
            V = (1, 8, 5, 64)  # 8 different 64-dim values per word

            Head 0 might focus on: subject-verb relationships
            Head 1 might focus on: verb-object relationships
            ...
            Head 7 might focus on: positional patterns

        Step 3: Attention (per head)
            For "eating" in head 1 (verb-object head):
            - Query("eating"): "I need an object"
            - Keys: ["I", "love", "eating", "apples", "<END>"]
            - Attention weights: [0.1, 0.1, 0.2, 0.6, 0.0]
            - Output: mainly value of "apples"!

        Step 4: Combine Heads
            (1, 8, 5, 64) → (1, 5, 512)
            Concatenate all 8 heads' outputs:
            [head0: 64][head1: 64]...[head7: 64] = 512 dims

        Step 5: Output Projection
            W_o learns how to combine information from all heads
            (1, 5, 512) → (1, 5, 512)

        Final output:
            Each word now contains information from multiple perspectives!
        """
        batch_size = query.size(0)

        # ========== Step 1: Apply Linear Projections ==========
        # Transform input into Query, Key, Value representations
        #
        # Why do we need these projections?
        # - Learn task-specific transformations
        # - Create appropriate "questions", "keys", and "values"
        # - Different weights for Q, K, V allow different roles
        #
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        #
        # Example: "eating" with d_model=512
        # Original vector: [x1, x2, ..., x512]
        # After W_q: [q1, q2, ..., q512]  ("what object do I need?")
        # After W_k: [k1, k2, ..., k512]  ("I'm a verb")
        # After W_v: [v1, v2, ..., v512]  ("here's my meaning")
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)

        # ========== Step 2: Split into Multiple Heads ==========
        # Divide the d_model dimensions among num_heads heads
        # Each head gets d_k = d_model / num_heads dimensions
        #
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, num_heads, seq_len, d_k)
        #
        # Example: (1, 5, 512) → (1, 8, 5, 64)
        # Now we have 8 parallel attention mechanisms!
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # ========== Step 3: Apply Scaled Dot-Product Attention ==========
        # Run attention independently for each head
        # All heads process in parallel (batch operations)
        #
        # Input: Q, K, V all of shape (batch_size, num_heads, seq_len, d_k)
        # Output: attn_output of shape (batch_size, num_heads, seq_len, d_k)
        #
        # Each head:
        # - Computes its own attention weights
        # - Focuses on different aspects of the input
        # - Produces its own d_k-dimensional output per position
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: (batch_size, num_heads, seq_len, d_k)

        # We ignore the attention weights here (the _ part)
        # But they can be useful for visualization or debugging

        # ========== Step 4: Combine Heads ==========
        # Merge all heads back together
        #
        # Input: (batch_size, num_heads, seq_len, d_k)
        # Output: (batch_size, seq_len, d_model)
        #
        # Example: (1, 8, 5, 64) → (1, 5, 512)
        # Concatenates the 8 heads:
        # [head0_output: 64][head1_output: 64]...[head7_output: 64] = 512
        #
        # Now each position has information from all 8 perspectives!
        output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)

        # ========== Step 5: Final Linear Projection (W_o) ==========
        # Learn how to best combine information from all heads
        #
        # Input: (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        #
        # Why is this needed?
        # - Simple concatenation may not be optimal
        # - W_o learns how to integrate multi-head outputs
        # - Allows model to weight different heads differently
        #
        # Example: Maybe head 1's output is very important, head 7 less so
        # W_o can learn these relative importances
        output = self.W_o(output)  # (batch_size, seq_len, d_model)

        # Final output shape: (batch_size, seq_len, d_model)
        # Same shape as input, but now enriched with multi-head attention!
        return output
