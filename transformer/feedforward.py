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
    位置前饋網路（Position-wise Feedforward Network，簡稱 FFN）

    【為什麼需要 FFN？Attention 還不夠嗎？】
    問題：Attention 只是「重新組合」資訊，沒有「轉換」資訊！

    類比理解：
        Attention 像是「資訊搜尋」：
        - Q: 我要找什麼？
        - K: 有哪些資訊？
        - V: 找到後給我對應的值
        - 結果：把相關的資訊收集起來（加權平均）

        但這只是「組合」，沒有「轉換」！
        就像你在圖書館找到了很多書，但還沒「讀」它們、沒「思考」它們

    FFN 就是「思考」的過程：
        - 對每個 token，獨立地進行非線性轉換
        - 提供更複雜的特徵提取能力
        - 增加模型的表達能力

    【為什麼叫 "position-wise"（逐位置）？】
    - 對序列中的每個位置（token）獨立處理
    - 不像 Attention 會看整個序列，FFN 只看當前位置
    - 但所有位置共享相同的權重（就像 CNN 的卷積核）

    具體例子：
        句子："我愛吃蘋果"（5 個字）

        Attention 的處理方式：
        - "我" 可以看到 "我"、"愛"、"吃"、"蘋"、"果"（全局資訊）
        - "愛" 可以看到 "我"、"愛"、"吃"、"蘋"、"果"（全局資訊）
        - ...（每個詞都能看到所有詞）

        FFN 的處理方式：
        - "我" 只處理 "我" 自己（獨立）
        - "愛" 只處理 "愛" 自己（獨立）
        - "吃" 只處理 "吃" 自己（獨立）
        - ...（每個詞獨立處理）
        - 但所有詞都用同樣的 FFN 權重

    【架構詳解：兩層 MLP】
        架構：Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
        例如：Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)

        1. 第一層線性轉換：擴展維度（d_model → d_ff）
           - 目的：提供更大的表示空間
           - 類比：把資訊「展開」，給模型更多空間去學習複雜特徵
           - 通常 d_ff = 4 * d_model（例如 512 → 2048）

        2. 激活函數：ReLU 或 GELU
           - 引入非線性（這是關鍵！）
           - 為什麼需要非線性？
             * 沒有非線性，多層線性變換 = 一層線性變換（無意義）
             * 非線性讓模型可以學習複雜的模式
           - ReLU: max(0, x)
             * 優點：簡單、快速、梯度不會消失（正數部分）
             * 缺點：可能有 "dying ReLU"（某些神經元永遠不激活）
           - GELU: Gaussian Error Linear Unit
             * 優點：更平滑、通常效果更好
             * GPT、BERT 都用 GELU

        3. Dropout：正則化
           - 訓練時：隨機將一些神經元的輸出設為 0
           - 目的：防止過擬合
           - 迫使模型不依賴特定的神經元

        4. 第二層線性轉換：壓縮回原維度（d_ff → d_model）
           - 目的：回到原始維度，可以接下一層
           - 類比：把「展開」的資訊「摘要」回來

    【為什麼 d_ff = 4 * d_model？】
        - 這是原論文（Attention is All You Need）的設定
        - 經驗上效果不錯
        - 提供足夠的容量（capacity）讓模型學習複雜特徵
        - 例如：512 → 2048 → 512

    參數：
        d_model: 模型維度（輸入/輸出維度，如 512）
        d_ff: 前饋網路的隱藏層維度（通常是 d_model 的 4 倍，如 2048）
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

        # ========== 元件 1: 第一層線性轉換（擴展維度）==========
        # d_model → d_ff（例如：512 → 2048）
        #
        # 這是一個全連接層（Fully Connected Layer）
        # 參數量：d_model * d_ff + d_ff
        # 例如：512 * 2048 + 2048 = 1,050,624 個參數
        #
        # 作用：把每個 token 的表示從 512 維「展開」到 2048 維
        # → 提供更大的空間讓模型學習複雜特徵
        self.linear1 = nn.Linear(d_model, d_ff)

        # ========== 元件 2: 第二層線性轉換（壓縮回原維度）==========
        # d_ff → d_model（例如：2048 → 512）
        #
        # 參數量：d_ff * d_model + d_model
        # 例如：2048 * 512 + 512 = 1,049,088 個參數
        #
        # 作用：把 2048 維的表示「摘要」回 512 維
        # → 回到原始維度，可以接下一層（或殘差連接）
        self.linear2 = nn.Linear(d_ff, d_model)

        # ========== 元件 3: Dropout 層（正則化）==========
        # Dropout 是一種正則化技術：
        # - 訓練時：隨機將一些神經元的輸出設為 0（以機率 dropout）
        # - 測試時：不做任何改變（自動關閉）
        #
        # 為什麼有效？
        # - 防止模型過度依賴特定的神經元
        # - 迫使模型學習更魯棒的特徵
        # - 類似「訓練時隨機遮住一些資訊，讓模型學會用剩下的資訊」
        #
        # dropout=0.1 表示有 10% 的神經元會被隨機關閉
        self.dropout = nn.Dropout(dropout)

        # ========== 元件 4: 激活函數選擇 ==========
        # 儲存激活函數的類型（'relu' 或 'gelu'）
        # 實際的激活計算在 forward 方法中進行
        self.activation = activation

        # 驗證激活函數類型
        # 只允許 'relu' 或 'gelu'
        # 其他的（如 'sigmoid', 'tanh'）在 Transformer 中效果不好
        if activation not in ['relu', 'gelu']:
            raise ValueError(f"activation 必須是 'relu' 或 'gelu'，但得到 '{activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入張量（通常是 Attention 的輸出）

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   輸出張量（維度與輸入相同）

        完整流程：
            x → Linear1 → Activation → Dropout → Linear2

        具體範例（以 "我愛吃蘋果" 為例）：
            假設輸入 x.shape = (2, 5, 512)
            - batch_size = 2（兩個句子）
            - seq_len = 5（每個句子 5 個字）
            - d_model = 512（每個字用 512 維向量表示）

            步驟 1: Linear1（擴展）
                輸入：(2, 5, 512)
                linear1(x) → (2, 5, 2048)
                → 每個字從 512 維擴展到 2048 維

            步驟 2: Activation（非線性轉換）
                輸入：(2, 5, 2048)
                activation → (2, 5, 2048)
                → 引入非線性，讓模型可以學習複雜模式
                → shape 不變，但值改變了

                ReLU 例子：
                    輸入: [-2, -1, 0, 1, 2]
                    輸出: [0, 0, 0, 1, 2]  ← 負數變 0，正數不變

                GELU 例子：
                    輸入: [-2, -1, 0, 1, 2]
                    輸出: [-0.05, -0.16, 0, 0.84, 1.95]  ← 更平滑

            步驟 3: Dropout（正則化）
                訓練時：隨機將 10% 的值設為 0
                輸入: [1, 2, 3, 4, 5]
                輸出: [0, 2, 3, 0, 5]  ← 隨機（每次不同）

                測試時：不做任何改變
                輸入: [1, 2, 3, 4, 5]
                輸出: [1, 2, 3, 4, 5]  ← 完全相同

            步驟 4: Linear2（壓縮）
                輸入：(2, 5, 2048)
                linear2 → (2, 5, 512)
                → 從 2048 維壓縮回 512 維

            最終輸出：(2, 5, 512)
            → 與輸入 shape 相同，可以接殘差連接或下一層
        """
        # ========== 步驟 1: 第一層線性轉換 + 激活函數 ==========
        # shape: (batch_size, seq_len, d_model) → (batch_size, seq_len, d_ff)
        # 例如：(2, 5, 512) → (2, 5, 2048)

        if self.activation == 'relu':
            # ReLU（Rectified Linear Unit）: max(0, x)
            # 數學定義：
            #   f(x) = x  if x > 0
            #   f(x) = 0  if x ≤ 0
            #
            # 優點：
            # ✓ 計算簡單、速度快
            # ✓ 不會有梯度消失問題（正數部分梯度恆為 1）
            #
            # 缺點：
            # ✗ "Dying ReLU" 問題：如果某個神經元的輸入總是負數
            #   它的輸出永遠是 0，梯度永遠是 0，再也學不到東西
            #
            # 適用：標準 Transformer（原論文用這個）
            hidden = F.relu(self.linear1(x))

        else:  # gelu
            # GELU（Gaussian Error Linear Unit）
            # 數學定義（近似）：
            #   f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
            #
            # 直覺理解：
            # - 類似 ReLU，但更平滑
            # - 負數不會完全變 0，而是接近 0
            # - 有一點「機率」的概念（基於高斯分布）
            #
            # 優點：
            # ✓ 更平滑（沒有 ReLU 的硬轉折）
            # ✓ 實驗上通常效果更好
            # ✓ 不會有 "Dying ReLU" 問題
            #
            # 缺點：
            # ✗ 計算稍慢（公式更複雜）
            #
            # 適用：GPT、BERT 等現代模型都用這個
            hidden = F.gelu(self.linear1(x))

        # 現在 hidden.shape = (batch_size, seq_len, d_ff)
        # 例如：(2, 5, 2048)

        # ========== 步驟 2: Dropout ==========
        # Dropout 只在訓練時啟用！
        #
        # 訓練模式（model.train()）：
        # - 隨機將一些神經元的輸出設為 0
        # - 機率由 dropout 參數決定（這裡是 0.1 = 10%）
        # - 剩下的值會放大（乘以 1/(1-dropout)），保持期望值不變
        #
        # 評估模式（model.eval()）：
        # - 完全不做任何改變
        # - 所有神經元都保留
        #
        # 為什麼訓練和測試不一樣？
        # - 訓練：希望模型不要過度依賴某些神經元 → 隨機關閉一些
        # - 測試：希望模型發揮全部實力 → 所有神經元都用
        hidden = self.dropout(hidden)

        # ========== 步驟 3: 第二層線性轉換 ==========
        # shape: (batch_size, seq_len, d_ff) → (batch_size, seq_len, d_model)
        # 例如：(2, 5, 2048) → (2, 5, 512)
        #
        # 目的：
        # - 壓縮回原始維度
        # - 可以接殘差連接（x + FFN(x)）
        # - 可以接下一層 Encoder Layer
        output = self.linear2(hidden)

        # 最終輸出 shape: (batch_size, seq_len, d_model)
        # 例如：(2, 5, 512)
        # → 與輸入 x 的 shape 完全相同
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
