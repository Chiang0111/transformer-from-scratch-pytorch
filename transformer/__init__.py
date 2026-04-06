"""
Transformer 從零實現

模組化、可用於生產環境的 Transformer 架構實現。
"""

from .attention import scaled_dot_product_attention, MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .feedforward import PositionwiseFeedForward, GatedFeedForward
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder, create_causal_mask
from .transformer import Transformer, TokenEmbedding, create_transformer

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
    "Transformer",
    "TokenEmbedding",
    "create_transformer",
]
