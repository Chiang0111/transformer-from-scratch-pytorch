"""
逐位置前饋網路（Position-wise Feedforward Network）

為什麼需要這個？
- 注意力只重新排列資訊，不轉換它
- FFN 提供非線性轉換，增加模型表達能力

架構：
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)

通常 d_ff = 4 * d_model（例如 512 → 2048 → 512）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前饋網路（Position-wise Feedforward Network, FFN）

    【為什麼需要 FFN？注意力不夠嗎？】
    問題：注意力只「重新排列」資訊，不「轉換」它！

    類比：
        注意力像是：在圖書館搜尋書籍（收集資訊）
        FFN 像是：閱讀並思考書籍（處理資訊）

    技術視角：
        - 注意力本質上是加權平均（線性組合）
        - 沒有非線性轉換
        - 無法學習複雜模式

    FFN 提供：
        - 非線性轉換（ReLU/GELU）
        - 更大的表示空間（512 → 2048 → 512）
        - 增加模型表達能力

    架構：
    ```
    Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
    範例：Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)
    ```

    【「逐位置」是什麼意思？】
    - 獨立處理每個位置（詞元）
    - 不像注意力會查看整個序列
    - 但所有位置共享相同權重（像 CNN）

    【為什麼 d_ff = 4 * d_model？】
    - 原論文的設定（"Attention is All You Need"）
    - 經驗上效果良好
    - 提供足夠容量學習複雜特徵

    【ReLU vs GELU】
    - ReLU：max(0, x)
      * 簡單、快速
      * 可能有「死亡 ReLU」問題
      * 標準 Transformer 使用

    - GELU：高斯誤差線性單元
      * 更平滑，通常效能更好
      * 稍慢
      * GPT、BERT 使用

    Args:
        d_model: 模型維度（輸入/輸出維度，例如 512）
        d_ff: FFN 隱藏維度（通常是 4x d_model，例如 2048）
        dropout: Dropout 機率（預設 0.1）
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

        # ========== 組件 1：第一個線性層（擴展）==========
        # d_model → d_ff（例如 512 → 2048）
        #
        # 這是一個全連接層
        # 參數量：d_model * d_ff + d_ff
        # 範例：512 * 2048 + 2048 = 1,050,624 個參數
        #
        # 目的：將每個詞元的表示從 512 維擴展到 2048 維
        # → 提供更大空間讓模型學習複雜特徵
        self.linear1 = nn.Linear(d_model, d_ff)

        # ========== 組件 2：第二個線性層（壓縮）==========
        # d_ff → d_model（例如 2048 → 512）
        #
        # 參數量：d_ff * d_model + d_model
        # 範例：2048 * 512 + 512 = 1,049,088 個參數
        #
        # 目的：將 2048 維表示壓縮回 512 維
        # → 返回原始維度，可連接到下一層（或殘差）
        self.linear2 = nn.Linear(d_ff, d_model)

        # ========== 組件 3：Dropout 層（正則化）==========
        # Dropout 是一種正則化技術：
        # - 訓練時：隨機將某些神經元輸出設為 0（機率為 dropout）
        # - 測試時：不改變（自動停用）
        #
        # 為什麼有效？
        # - 防止模型過度依賴特定神經元
        # - 強制模型學習更穩健的特徵
        # - 類似「用部分資訊訓練，讓模型學會使用剩餘資訊」
        #
        # dropout=0.1 代表隨機停用 10% 的神經元
        self.dropout = nn.Dropout(dropout)

        # ========== 組件 4：激活函數選擇 ==========
        # 儲存激活函數類型（'relu' 或 'gelu'）
        # 實際激活計算發生在 forward 方法中
        self.activation = activation

        # 驗證激活函數類型
        # 只允許 'relu' 或 'gelu'
        # 其他（如 'sigmoid'、'tanh'）在 Transformer 中效果不好
        if activation not in ['relu', 'gelu']:
            raise ValueError(f"activation must be 'relu' or 'gelu', got '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入張量，形狀 (batch_size, seq_len, d_model)
               （通常是注意力的輸出）

        Returns:
            output: 輸出張量，形狀 (batch_size, seq_len, d_model)
                   （與輸入維度相同）

        完整流程：
            x → Linear1 → Activation → Dropout → Linear2

        具體範例（"I love eating apples"）：
            假設輸入 x.shape = (2, 5, 512)
            - batch_size = 2（兩個句子）
            - seq_len = 5（每個 5 個詞）
            - d_model = 512（每個詞是 512 維向量）

            步驟 1：Linear1（擴展）
                輸入：(2, 5, 512)
                linear1(x) → (2, 5, 2048)
                → 每個詞從 512 維擴展到 2048 維

            步驟 2：Activation（非線性轉換）
                輸入：(2, 5, 2048)
                activation → (2, 5, 2048)
                → 引入非線性，允許學習複雜模式
                → 形狀不變，但值被轉換

                ReLU 範例：
                    輸入：[-2, -1, 0, 1, 2]
                    輸出：[0, 0, 0, 1, 2]  ← 負值變 0，正值不變

                GELU 範例：
                    輸入：[-2, -1, 0, 1, 2]
                    輸出：[-0.05, -0.16, 0, 0.84, 1.95]  ← 更平滑

            步驟 3：Dropout（正則化）
                訓練時：隨機將 10% 的值設為 0
                輸入：[1, 2, 3, 4, 5]
                輸出：[0, 2, 3, 0, 5]  ← 隨機（每次不同）

                測試時：不改變
                輸入：[1, 2, 3, 4, 5]
                輸出：[1, 2, 3, 4, 5]  ← 完全相同

            步驟 4：Linear2（壓縮）
                輸入：(2, 5, 2048)
                linear2 → (2, 5, 512)
                → 從 2048 壓縮回 512 維

            最終輸出：(2, 5, 512)
            → 與輸入形狀相同，可連接到殘差或下一層
        """
        # ========== 步驟 1：第一個線性層 + 激活 ==========
        # 形狀：(batch_size, seq_len, d_model) → (batch_size, seq_len, d_ff)
        # 範例：(2, 5, 512) → (2, 5, 2048)

        if self.activation == 'relu':
            # ReLU（整流線性單元）：max(0, x)
            # 數學定義：
            #   f(x) = x  若 x > 0
            #   f(x) = 0  若 x ≤ 0
            #
            # 優點：
            # ✓ 簡單、計算快速
            # ✓ 無梯度消失（正數部分梯度為 1）
            #
            # 缺點：
            # ✗ 「死亡 ReLU」問題：若神經元輸入總是負數
            #   其輸出總是 0，梯度總是 0，永遠無法學習
            #
            # 使用場景：標準 Transformer（原論文）
            hidden = F.relu(self.linear1(x))

        else:  # gelu
            # GELU（高斯誤差線性單元）
            # 數學定義（近似）：
            #   f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
            #
            # 直觀理解：
            # - 類似 ReLU 但更平滑
            # - 負值不會變成正好 0，只是接近 0
            # - 有機率解釋（基於高斯分布）
            #
            # 優點：
            # ✓ 更平滑（x=0 處無尖角）
            # ✓ 經驗上效能更好
            # ✓ 無「死亡 ReLU」問題
            #
            # 缺點：
            # ✗ 稍慢（公式更複雜）
            #
            # 使用場景：GPT、BERT（現代模型）
            hidden = F.gelu(self.linear1(x))

        # 現在 hidden.shape = (batch_size, seq_len, d_ff)
        # 範例：(2, 5, 2048)

        # ========== 步驟 2：Dropout ==========
        # Dropout 只在訓練時激活！
        #
        # 訓練模式（model.train()）：
        # - 隨機將某些值設為 0
        # - 機率由 dropout 參數決定（這裡 0.1 = 10%）
        # - 剩餘值會放大（乘以 1/(1-dropout)）以維持期望值
        #
        # 評估模式（model.eval()）：
        # - 完全不變
        # - 所有神經元都激活
        #
        # 為什麼訓練與測試不同？
        # - 訓練：希望模型不過度依賴某些神經元 → 隨機停用一些
        # - 測試：希望模型使用全部能力 → 所有神經元激活
        hidden = self.dropout(hidden)

        # ========== 步驟 3：第二個線性層 ==========
        # 形狀：(batch_size, seq_len, d_ff) → (batch_size, seq_len, d_model)
        # 範例：(2, 5, 2048) → (2, 5, 512)
        #
        # 目的：
        # - 壓縮回原始維度
        # - 可連接到殘差連接（x + FFN(x)）
        # - 可連接到下一個編碼器層
        output = self.linear2(hidden)

        # 最終輸出形狀：(batch_size, seq_len, d_model)
        # 範例：(2, 5, 512)
        # → 與輸入 x 形狀相同
        return output


class GatedFeedForward(nn.Module):
    """
    門控前饋網路（進階版本，可選）

    這是使用門控機制的改進版 FFN（類似 LSTM 門）。
    一些現代 Transformer 變體（例如 GLU）使用此架構。

    架構：
        output = Linear2(GELU(Linear1_1(x)) ⊙ Linear1_2(x))

    其中 ⊙ 是逐元素乘法

    為什麼更好？
    - 門控機制讓模型學習「哪些資訊要通過」
    - 通常效能優於標準 FFN，但參數量是 2 倍

    Args:
        d_model: 模型維度
        d_ff: 前饋網路的隱藏維度
        dropout: Dropout 機率
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 兩個平行的線性層（用於門控）
        self.linear1_1 = nn.Linear(d_model, d_ff)
        self.linear1_2 = nn.Linear(d_model, d_ff)

        # 輸出線性層
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        門控前饋網路的前向傳播

        Args:
            x: 輸入張量，形狀 (batch_size, seq_len, d_model)

        Returns:
            output: 輸出張量，形狀 (batch_size, seq_len, d_model)
        """
        # 計算兩個分支
        # 分支 1：激活值
        activated = F.gelu(self.linear1_1(x))

        # 分支 2：門控值（決定哪些資訊通過）
        gate = self.linear1_2(x)

        # 逐元素乘法（門控機制）
        # gate 值決定保留 activated 的哪些部分
        gated = activated * gate

        # Dropout 和最終線性層
        gated = self.dropout(gated)
        output = self.linear2(gated)

        return output


if __name__ == "__main__":
    # 測試程式碼
    print("=== 測試逐位置前饋網路 ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    # 創建 FFN
    ffn = PositionwiseFeedForward(d_model, d_ff)

    # 創建假輸入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入形狀：{x.shape}")

    # 前向傳播
    output = ffn(x)
    print(f"輸出形狀：{output.shape}")

    # 檢查參數量
    total_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n總參數量：{total_params:,}")
    print(f"Linear1 參數量：{d_model * d_ff + d_ff:,}")
    print(f"Linear2 參數量：{d_ff * d_model + d_model:,}")

    # 測試門控版本
    print("\n=== 測試門控前饋網路 ===\n")
    gated_ffn = GatedFeedForward(d_model, d_ff)
    output_gated = gated_ffn(x)
    print(f"門控 FFN 輸出形狀：{output_gated.shape}")

    gated_params = sum(p.numel() for p in gated_ffn.parameters())
    print(f"門控 FFN 總參數量：{gated_params:,}")
    print(f"（約為標準 FFN 的 {gated_params / total_params:.1f} 倍）")
