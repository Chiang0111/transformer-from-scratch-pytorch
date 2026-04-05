"""
Transformer 從零實作

一個模組化、生產級的 Transformer 架構實作。
"""

from .attention import scaled_dot_product_attention, MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .feedforward import PositionwiseFeedForward, GatedFeedForward
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder, create_causal_mask

__version__ = "0.1.0"

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "GatedFeedForward",
    "EncoderLayer",
    "Encoder",
    "DecoderLayer",
    "Decoder",
    "create_causal_mask",
]
