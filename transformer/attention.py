"""
Attention 機制實作

包含：
1. Scaled Dot-Product Attention (縮放點積注意力)
2. Multi-Head Attention (多頭注意力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    縮放點積注意力機制

    這是 Transformer 的核心！理解這個函數就理解了 80% 的 Transformer。

    參數說明：
        query: shape (batch_size, num_heads, seq_len_q, d_k)
               查詢向量 - 代表「我想要找什麼資訊」
        key:   shape (batch_size, num_heads, seq_len_k, d_k)
               鍵向量 - 代表「每個位置有什麼資訊」
        value: shape (batch_size, num_heads, seq_len_v, d_v)
               值向量 - 代表「實際的資訊內容」
        mask:  shape (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, num_heads, seq_len_q, seq_len_k)
               遮罩 - 用來屏蔽某些位置（例如 padding 或未來的詞）

    回傳：
        output: shape (batch_size, num_heads, seq_len_q, d_v)
                注意力加權後的輸出
        attention_weights: shape (batch_size, num_heads, seq_len_q, seq_len_k)
                注意力權重（可視覺化用）

    公式：Attention(Q,K,V) = softmax(QK^T / √d_k) * V
    """
    # 步驟 1: 獲取 key 的維度 d_k（用於縮放）
    # 為什麼要縮放？因為點積結果的方差會隨著 d_k 增大而增大
    # 如果不縮放，softmax 會趨向極端值（接近 0 或 1），導致梯度消失
    d_k = query.size(-1)

    # 步驟 2: 計算注意力分數 (Q·K^T)
    # torch.matmul 會自動處理批次矩陣乘法
    # key.transpose(-2, -1) 將最後兩個維度轉置 (seq_len_k, d_k) -> (d_k, seq_len_k)
    # 結果 shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 步驟 3: 縮放（除以 √d_k）
    # 這是 "scaled" dot-product 的關鍵
    scores = scores / math.sqrt(d_k)

    # 步驟 4: 應用遮罩（如果有的話）
    # 遮罩的作用：
    # - Padding mask: 不要 attend 到 <PAD> token
    # - Causal mask: 不要 attend 到未來的 token（用於 decoder）
    if mask is not None:
        # 將遮罩為 0 的位置設為 -inf，這樣 softmax 後會變成 0
        # 為什麼用 -1e9 而不是 -inf？因為 -inf 可能導致 NaN
        scores = scores.masked_fill(mask == 0, -1e9)

    # 步驟 5: Softmax 轉成機率分佈
    # dim=-1 表示在最後一個維度（seq_len_k）上做 softmax
    # 結果：每個 query 位置對所有 key 位置的注意力權重總和為 1
    attention_weights = F.softmax(scores, dim=-1)

    # 步驟 6: 用注意力權重加權 values
    # 這是「提取資訊」的步驟
    # attention_weights @ value 相當於加權平均
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制

    為什麼需要多頭？
    - 單一注意力頭可能只關注一種模式（例如：語法關係）
    - 多頭可以同時關注不同的模式（語法、語義、位置等）
    - 類比：多個專家從不同角度分析同一個問題

    參數說明：
        d_model: 模型維度（輸入/輸出的維度）
        num_heads: 注意力頭的數量
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # 檢查：d_model 必須能被 num_heads 整除
        # 因為我們要把 d_model 平均分配給每個頭
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        # 每個頭的維度
        self.d_k = d_model // num_heads

        # 為 Q, K, V 建立線性投影層
        # 為什麼需要投影？因為我們要學習「從不同角度看資料」
        # 這些權重是可學習的參數
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 最後的輸出投影層
        # 將多個頭的結果合併回 d_model 維度
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將輸入分割成多個注意力頭

        輸入 shape: (batch_size, seq_len, d_model)
        輸出 shape: (batch_size, num_heads, seq_len, d_k)

        這個函數做了什麼？
        1. 將 d_model 分成 num_heads 個 d_k
        2. 重新排列維度，讓每個頭可以獨立計算
        """
        batch_size, seq_len, d_model = x.size()

        # 重塑：(batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 轉置：(batch_size, num_heads, seq_len, d_k)
        # 為什麼轉置？因為我們要對每個頭獨立做批次運算
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將多個注意力頭的結果合併

        輸入 shape: (batch_size, num_heads, seq_len, d_k)
        輸出 shape: (batch_size, seq_len, d_model)

        這是 split_heads 的逆操作
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # 轉置回來：(batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # 合併最後兩個維度：(batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多頭注意力的前向傳播

        參數：
            query, key, value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)

        回傳：
            output: shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 步驟 1: 線性投影 Q, K, V
        # 這些投影是可學習的，讓模型學會「從哪些角度看資料」
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)

        # 步驟 2: 分割成多個頭
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # 步驟 3: 計算 scaled dot-product attention
        # 每個頭獨立計算注意力
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output shape: (batch_size, num_heads, seq_len, d_k)

        # 步驟 4: 合併多個頭
        output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)

        # 步驟 5: 最後的線性投影
        # 這個投影將多頭的資訊整合在一起
        output = self.W_o(output)  # (batch_size, seq_len, d_model)

        return output
