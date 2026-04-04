"""
位置編碼（Positional Encoding）

為什麼需要？
- Transformer 的 Attention 機制對詞的順序不敏感
- 需要額外加入位置資訊，讓模型知道每個詞的位置

實作方式：
- 使用正弦/餘弦函數產生位置編碼
- 加到詞嵌入（word embedding）上
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置編碼模組

    使用正弦和餘弦函數來編碼位置資訊。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
        pos = 位置索引 (0, 1, 2, ...)
        i = 維度索引
        d_model = 模型維度

    參數：
        d_model: 模型維度
        max_len: 最大序列長度（預設 5000）
        dropout: dropout 比例（預設 0.1）
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 建立位置編碼矩陣
        # shape: (max_len, d_model)
        # 這個矩陣是固定的，不需要學習（所以用 register_buffer）
        pe = torch.zeros(max_len, d_model)

        # 建立位置索引：[0, 1, 2, ..., max_len-1]
        # unsqueeze(1) 將 shape 從 (max_len,) 變成 (max_len, 1)
        # 為什麼？因為等下要跟 (d_model,) 的向量做廣播運算
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 計算分母部分：10000^(2i/d_model)
        # 先算 2i/d_model 部分
        # torch.arange(0, d_model, 2) 產生 [0, 2, 4, 6, ..., d_model-2]
        # 代表偶數維度的索引
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 為什麼用 exp(-log(...))？
        # 因為 10000^(2i/d_model) = exp(log(10000^(2i/d_model))) = exp(2i/d_model * log(10000))
        # 而 exp(-x * log(10000)) 在數值上更穩定

        # 偶數維度用 sin
        # position * div_term 會廣播：(max_len, 1) * (d_model/2,) -> (max_len, d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇數維度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 維度：(max_len, d_model) -> (1, max_len, d_model)
        # 為什麼？因為實際使用時輸入是 (batch_size, seq_len, d_model)
        # 有了這個 1，PyTorch 會自動廣播到任意 batch_size
        pe = pe.unsqueeze(0)

        # 將位置編碼註冊為 buffer
        # register_buffer 的作用：
        # 1. 這個張量會跟著模型移動（CPU/GPU）
        # 2. 這個張量會被保存在模型的 state_dict 中
        # 3. 但不會被當作參數（不會被優化器更新）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        將位置編碼加到輸入上

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入的詞嵌入（word embeddings）

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   加上位置編碼後的結果
        """
        # 取出對應序列長度的位置編碼
        # x.size(1) 是 seq_len
        # self.pe[:, :seq_len] 的 shape 是 (1, seq_len, d_model)
        # 加法時會自動廣播到 (batch_size, seq_len, d_model)

        # 注意：這裡用 .requires_grad_(False) 確保位置編碼不會被計算梯度
        # 因為位置編碼是固定的，不需要學習
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        # 加入 dropout（正則化，防止過擬合）
        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    視覺化位置編碼（用於理解）

    這個函數不是必要的，但可以幫助你理解位置編碼的模式

    參數：
        d_model: 模型維度
        max_len: 要視覺化的最大長度
    """
    import matplotlib.pyplot as plt

    # 建立位置編碼
    pe = PositionalEncoding(d_model, max_len)

    # 取出位置編碼矩陣
    # shape: (1, max_len, d_model) -> (max_len, d_model)
    encoding = pe.pe.squeeze(0).numpy()

    # 繪製熱力圖
    plt.figure(figsize=(15, 5))
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.title(f'Positional Encoding (d_model={d_model})')
    plt.tight_layout()
    plt.savefig('positional_encoding_visualization.png', dpi=150)
    print("位置編碼視覺化已儲存至 positional_encoding_visualization.png")
    plt.show()


if __name__ == "__main__":
    # 測試程式碼
    print("=== 測試位置編碼 ===\n")

    d_model = 512
    batch_size = 2
    seq_len = 10

    # 建立位置編碼模組
    pe = PositionalEncoding(d_model)

    # 建立假的詞嵌入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入 shape: {x.shape}")

    # 加入位置編碼
    output = pe(x)
    print(f"輸出 shape: {output.shape}")

    # 視覺化（可選）
    # visualize_positional_encoding(d_model=128, max_len=100)
