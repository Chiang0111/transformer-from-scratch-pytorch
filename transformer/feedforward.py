"""
位置前饋網路（Position-wise Feedforward Network）

為什麼需要？
- Attention 只是重新組合資訊，沒有「轉換」資訊
- FFN 提供非線性轉換，增加模型的表達能力

架構：
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)

通常 d_ff = 4 * d_model（例如：512 → 2048 → 512）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    位置前饋網路（FFN）

    對序列中的每個位置獨立應用相同的兩層全連接網路。

    為什麼叫 "position-wise"？
    - 對每個位置（token）獨立處理
    - 不像 Attention 會看整個序列，FFN 只看當前位置
    - 但所有位置共享相同的權重

    架構詳解：
        1. 第一層：擴展維度（d_model → d_ff）
           - 提供更大的表示空間
           - 類似「展開」資訊

        2. 激活函數：ReLU 或 GELU
           - 引入非線性
           - ReLU: max(0, x)
           - GELU: 更平滑的版本，GPT 用這個

        3. Dropout：正則化
           - 隨機丟棄一些神經元
           - 防止過擬合

        4. 第二層：壓縮回原維度（d_ff → d_model）
           - 回到原始維度
           - 類似「摘要」資訊

    參數：
        d_model: 模型維度（輸入/輸出維度）
        d_ff: 前饋網路的隱藏層維度（通常是 d_model 的 4 倍）
        dropout: Dropout 比例（預設 0.1）
        activation: 激活函數類型，'relu' 或 'gelu'（預設 'relu'）
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()

        # 第一層線性轉換：擴展維度
        # d_model → d_ff（例如：512 → 2048）
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二層線性轉換：壓縮回原維度
        # d_ff → d_model（例如：2048 → 512）
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout 層：防止過擬合
        self.dropout = nn.Dropout(dropout)

        # 選擇激活函數
        self.activation = activation
        if activation not in ['relu', 'gelu']:
            raise ValueError(f"activation 必須是 'relu' 或 'gelu'，但得到 '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入張量

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   輸出張量（維度與輸入相同）

        計算流程：
            x → Linear1 → Activation → Dropout → Linear2

        範例：
            假設 x.shape = (2, 10, 512)  # batch=2, seq_len=10, d_model=512

            1. linear1(x) → (2, 10, 2048)  # 擴展到 d_ff
            2. activation → (2, 10, 2048)   # 非線性轉換
            3. dropout    → (2, 10, 2048)   # 隨機丟棄
            4. linear2    → (2, 10, 512)    # 壓縮回 d_model
        """
        # 步驟 1: 第一層線性轉換 + 激活函數
        # shape: (batch_size, seq_len, d_model) → (batch_size, seq_len, d_ff)
        if self.activation == 'relu':
            # ReLU: max(0, x)
            # 優點：簡單、快速
            # 缺點：可能有 "dying ReLU" 問題（某些神經元永遠不激活）
            hidden = F.relu(self.linear1(x))
        else:  # gelu
            # GELU: Gaussian Error Linear Unit
            # 優點：更平滑、效果通常更好
            # 缺點：計算稍慢
            # GPT、BERT 都用 GELU
            hidden = F.gelu(self.linear1(x))

        # 步驟 2: Dropout
        # 訓練時：隨機將一些值設為 0
        # 測試時：自動關閉
        hidden = self.dropout(hidden)

        # 步驟 3: 第二層線性轉換
        # shape: (batch_size, seq_len, d_ff) → (batch_size, seq_len, d_model)
        output = self.linear2(hidden)

        return output


class GatedFeedForward(nn.Module):
    """
    門控前饋網路（進階版本，可選）

    這是 FFN 的改進版本，使用門控機制（類似 LSTM 的門）。
    某些現代 Transformer 變體（如 GLU）使用這種架構。

    架構：
        output = Linear2(GELU(Linear1_1(x)) ⊙ Linear1_2(x))

    其中 ⊙ 是逐元素相乘

    為什麼更好？
    - 門控機制讓模型學習「哪些資訊要通過」
    - 通常比標準 FFN 效果更好，但參數量多一倍

    參數：
        d_model: 模型維度
        d_ff: 前饋網路的隱藏層維度
        dropout: Dropout 比例
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 兩個並行的線性層（用於門控）
        self.linear1_1 = nn.Linear(d_model, d_ff)
        self.linear1_2 = nn.Linear(d_model, d_ff)

        # 輸出線性層
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        門控前饋網路的前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)

        回傳：
            output: shape (batch_size, seq_len, d_model)
        """
        # 計算兩個分支
        # 分支 1: 激活後的值
        activated = F.gelu(self.linear1_1(x))

        # 分支 2: 門控值（決定哪些資訊通過）
        gate = self.linear1_2(x)

        # 逐元素相乘（門控機制）
        # gate 的值決定 activated 中哪些值要保留
        gated = activated * gate

        # Dropout 和最後的線性層
        gated = self.dropout(gated)
        output = self.linear2(gated)

        return output


if __name__ == "__main__":
    # 測試程式碼
    print("=== 測試位置前饋網路 ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    # 建立 FFN
    ffn = PositionwiseFeedForward(d_model, d_ff)

    # 建立假輸入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入 shape: {x.shape}")

    # 前向傳播
    output = ffn(x)
    print(f"輸出 shape: {output.shape}")

    # 檢查參數數量
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n總參數量: {total_params:,}")
    print(f"Linear1 參數: {d_model * d_ff + d_ff:,}")
    print(f"Linear2 參數: {d_ff * d_model + d_model:,}")

    # 測試門控版本
    print("\n=== 測試門控前饋網路 ===\n")
    gated_ffn = GatedFeedForward(d_model, d_ff)
    output_gated = gated_ffn(x)
    print(f"門控 FFN 輸出 shape: {output_gated.shape}")

    gated_params = sum(p.numel() for p in gated_ffn.parameters())
    print(f"門控 FFN 總參數量: {gated_params:,}")
    print(f"（約是標準 FFN 的 {gated_params / total_params:.1f} 倍）")
