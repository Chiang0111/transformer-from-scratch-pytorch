"""
Transformer Encoder Layer

整合所有元件：
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Position-wise Feedforward Network
4. Residual Connection + Layer Normalization

架構：
    輸入
     ↓
    [Multi-Head Attention] → Add & Norm
     ↓
    [Feedforward Network]  → Add & Norm
     ↓
    輸出
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    一個完整的 Encoder 層，包含：
    1. Multi-Head Self-Attention（多頭自注意力）
    2. Add & Norm（殘差連接 + 層歸一化）
    3. Position-wise FFN（位置前饋網路）
    4. Add & Norm（殘差連接 + 層歸一化）

    完整流程：
        x → [Self-Attention] → Add(x, ·) → LayerNorm →
        → [FFN] → Add(·, ·) → LayerNorm → output

    為什麼這個順序？
    - 先 Attention：收集資訊
    - 再 FFN：處理資訊
    - 每步都有 Residual + Norm：穩定訓練

    參數：
        d_model: 模型維度
        num_heads: 注意力頭數量
        d_ff: FFN 隱藏層維度
        dropout: Dropout 比例
        activation: FFN 激活函數（'relu' 或 'gelu'）
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

        # 元件 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # 元件 2: Position-wise Feedforward Network
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # 兩個 Layer Normalization 層
        # 為什麼需要兩個？因為有兩個子層（Attention 和 FFN）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 兩個 Dropout 層（用於殘差連接之後）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder Layer 的前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入序列
            mask: shape (batch_size, 1, 1, seq_len) 或 None
                 Padding mask，用來忽略 <PAD> token

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   Encoder layer 的輸出

        流程詳解：
            1. Self-Attention: 讓每個 token 看到整個序列
            2. Add & Norm: 加上輸入 + 標準化
            3. FFN: 對每個 token 獨立處理
            4. Add & Norm: 再次加上輸入 + 標準化
        """
        # ===== 子層 1: Multi-Head Self-Attention =====

        # Step 1: Self-Attention
        # Q = K = V = x（自注意力）
        # 讓序列中的每個 token 都能「看到」整個序列
        attn_output = self.self_attention(x, x, x, mask)

        # Step 2: Residual Connection（殘差連接）
        # 為什麼加上 x？
        # - 讓梯度可以直接流回前面的層
        # - 讓模型學習「修改」而不是「重建」
        # - 類比：告訴模型「在原本的基礎上做調整」
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # 這就是殘差連接！

        # Step 3: Layer Normalization
        # 為什麼要標準化？
        # - 穩定訓練（防止數值爆炸或消失）
        # - 加快收斂速度
        # - 讓每層的輸出分佈穩定
        x = self.norm1(x)

        # ===== 子層 2: Position-wise Feedforward =====

        # Step 4: Feedforward Network
        # 對每個位置獨立處理
        # 提供非線性轉換
        ff_output = self.feed_forward(x)

        # Step 5: Residual Connection + Dropout
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # 第二個殘差連接

        # Step 6: Layer Normalization
        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    """
    完整的 Transformer Encoder

    堆疊多個 EncoderLayer。

    為什麼要堆疊多層？
    - 每層可以學習不同層次的特徵
    - 第 1 層：可能學習局部模式（如詞組）
    - 第 2 層：可能學習中層模式（如短語）
    - 第 N 層：可能學習高層語義

    類比深度 CNN：
    - 淺層：學習邊緣、紋理
    - 中層：學習形狀、部件
    - 深層：學習物體、場景

    參數：
        num_layers: Encoder layer 的數量（原論文用 6）
        d_model: 模型維度
        num_heads: 注意力頭數量
        d_ff: FFN 隱藏層維度
        dropout: Dropout 比例
        activation: FFN 激活函數
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

        # 建立 num_layers 個相同結構的 EncoderLayer
        # 注意：每層的權重是獨立的（不共享）
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # 最後的 Layer Normalization
        # 為什麼最後還要一個？
        # - 確保整個 Encoder 的輸出分佈穩定
        # - 方便後續處理（如接 Decoder）
        self.norm = nn.LayerNorm(d_model)

        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encoder 的前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入序列（通常是 word embeddings + positional encoding）
            mask: shape (batch_size, 1, 1, seq_len) 或 None
                 Padding mask

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   Encoder 的輸出

        流程：
            輸入 → Layer1 → Layer2 → ... → LayerN → Norm → 輸出
        """
        # 依序通過每個 Encoder Layer
        for layer in self.layers:
            x = layer(x, mask)

        # 最後的 Layer Normalization
        x = self.norm(x)

        return x


if __name__ == "__main__":
    # 測試程式碼
    print("=== 測試 Encoder Layer ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 建立單個 Encoder Layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # 建立假輸入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入 shape: {x.shape}")

    # 前向傳播
    output = encoder_layer(x)
    print(f"輸出 shape: {output.shape}")

    # 測試帶 mask 的情況
    # 假設序列中前 7 個是真實 token，後 3 個是 padding
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0  # 後 3 個位置被遮罩
    output_with_mask = encoder_layer(x, mask)
    print(f"帶 mask 的輸出 shape: {output_with_mask.shape}")

    print("\n=== 測試完整 Encoder（6 層）===\n")

    num_layers = 6
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = encoder(x, mask)
    print(f"完整 Encoder 輸出 shape: {output_full.shape}")

    # 計算參數量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n完整 Encoder（{num_layers} 層）總參數量: {total_params:,}")
    print(f"約 {total_params / 1e6:.1f}M 參數")
