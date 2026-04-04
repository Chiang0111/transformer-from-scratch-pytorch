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
    位置編碼模組（Positional Encoding）

    【為什麼需要位置編碼？】
    問題：Attention 機制對詞的順序完全不敏感！

    具體例子：
        句子 A："我愛吃蘋果"
        句子 B："蘋果吃愛我"

    對於 Attention 來說：
        - 這兩個句子的 Query、Key、Value 矩陣完全一樣！
        - 因為 Attention 只看「哪些詞相關」，不看「詞的順序」
        - 就像把句子裡的詞都倒進一個袋子裡，順序全亂了

    這是大問題，因為：
        - "我愛吃蘋果" 和 "蘋果吃愛我" 意思完全不同
        - "The cat sat on the mat" 和 "The mat sat on the cat" 意思完全不同
        - 語言中，順序是核心！

    解決方案：位置編碼
        在每個詞的 embedding 上「加入」它的位置資訊
        → 讓模型知道：這個詞在句子的第幾個位置

    【為什麼用 sin/cos 函數？】
    有很多種方式可以編碼位置，為什麼選 sin/cos？

    1. 可學習的位置編碼（Learned Positional Embedding）
       缺點：只能處理訓練時見過的最大長度
       → 如果訓練時最長 100 詞，測試時來了 150 詞的句子就掛了

    2. 固定的 sin/cos 編碼（本實作）
       優點：
       ✓ 可以處理任意長度的序列（超出訓練長度也 OK）
       ✓ 不需要學習（減少參數量）
       ✓ 有數學特性：相對位置可以用線性組合表示
         （模型可以學到「往前 3 個位置」這種相對關係）

    【sin/cos 編碼的直覺理解】
    想像一個時鐘：
        - 秒針轉得最快（高頻率）
        - 分針轉得較慢（中頻率）
        - 時針轉得最慢（低頻率）

    位置編碼也是類似的概念：
        - 低維度：高頻率波動（像秒針，變化快）
        - 中維度：中頻率波動（像分針）
        - 高維度：低頻率波動（像時針，變化慢）

    這樣的組合可以唯一標識每個位置！

    【公式詳解】
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
        pos     = 位置索引 (0, 1, 2, 3, ...)
        i       = 維度索引 (0, 1, 2, 3, ...)
        d_model = 模型維度（如 512）
        2i      = 偶數維度用 sin
        2i+1    = 奇數維度用 cos

    具體例子（假設 d_model=512）：
        位置 0 的編碼：
            維度 0 (偶數): sin(0 / 10000^(0/512))   = sin(0) = 0
            維度 1 (奇數): cos(0 / 10000^(0/512))   = cos(0) = 1
            維度 2 (偶數): sin(0 / 10000^(2/512))   = sin(0) = 0
            維度 3 (奇數): cos(0 / 10000^(2/512))   = cos(0) = 1
            ...

        位置 1 的編碼：
            維度 0: sin(1 / 10000^(0/512))   = sin(1) ≈ 0.841
            維度 1: cos(1 / 10000^(0/512))   = cos(1) ≈ 0.540
            維度 2: sin(1 / 10000^(2/512))   ≈ 0.841
            維度 3: cos(1 / 10000^(2/512))   ≈ 0.540
            ...

    【為什麼 10000？】
        - 這是一個經驗值（原論文使用）
        - 目的：讓不同維度有不同的頻率
        - 10000^(2i/d_model) 隨著 i 增加而增加
        - → 頻率逐漸降低（從快速波動到緩慢波動）

    參數：
        d_model: 模型維度（如 512）
        max_len: 最大序列長度（預設 5000，可以處理 5000 個詞的句子）
        dropout: dropout 比例（預設 0.1，防止過擬合）
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ========== 步驟 1: 初始化位置編碼矩陣 ==========
        # 建立一個全 0 矩陣，稍後填入 sin/cos 值
        # shape: (max_len, d_model)
        # 例如：(5000, 512) → 可以處理最長 5000 個詞的句子
        #
        # 為什麼是 (max_len, d_model)？
        # - 每一行：代表一個位置（第 0 個詞、第 1 個詞...）
        # - 每一列：代表一個維度（512 個維度）
        # - 矩陣中的每個值：該位置在該維度的編碼
        #
        # 這個矩陣是固定的，不需要學習！（所以用 register_buffer，不是 nn.Parameter）
        pe = torch.zeros(max_len, d_model)

        # ========== 步驟 2: 建立位置索引 ==========
        # 建立位置索引：[0, 1, 2, ..., max_len-1]
        # 例如：[0, 1, 2, ..., 4999]
        position = torch.arange(0, max_len, dtype=torch.float)  # shape: (max_len,)

        # unsqueeze(1) 將 shape 從 (max_len,) 變成 (max_len, 1)
        # 例如：(5000,) → (5000, 1)
        #
        # 為什麼要 unsqueeze？
        # 因為等下要跟 div_term（shape 是 (d_model/2,)）做廣播運算
        # 廣播規則：(5000, 1) * (256,) → (5000, 256)
        position = position.unsqueeze(1)  # shape: (max_len, 1)

        # ========== 步驟 3: 計算頻率項（分母部分）==========
        # 目標：計算 10000^(2i/d_model)
        #
        # torch.arange(0, d_model, 2) 產生偶數索引
        # 例如 d_model=512 時：[0, 2, 4, 6, ..., 510]
        # → shape: (256,)，共 256 個值（d_model/2）
        #
        # 為什麼只算偶數？
        # 因為公式中：
        # - 偶數維度 (2i) 用 sin
        # - 奇數維度 (2i+1) 用 cos
        # 但它們用的頻率項（分母）是一樣的！
        # 所以只需要計算 d_model/2 個頻率

        # 數學推導：
        # 原公式分母：10000^(2i/d_model)
        # 轉換：10000^(2i/d_model) = exp(log(10000^(2i/d_model)))
        #                           = exp((2i/d_model) * log(10000))
        # 而我們要算的是 1 / 10000^(2i/d_model)
        # = exp(-(2i/d_model) * log(10000))
        #
        # 為什麼這樣轉換？
        # - 直接算 10000^x 當 x 很大時會數值溢出
        # - 用 exp(-x * log(10000)) 數值更穩定
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term shape: (d_model/2,) 例如 (256,)
        #
        # 具體數值例子（d_model=512）：
        # i=0:   exp(-(0/512) * log(10000)) = exp(0) = 1.0
        # i=2:   exp(-(2/512) * log(10000)) ≈ 0.912
        # i=4:   exp(-(4/512) * log(10000)) ≈ 0.832
        # ...
        # i=510: exp(-(510/512) * log(10000)) ≈ 0.0001  （頻率很低）
        #
        # → 頻率從高到低遞減（就像時鐘的秒針、分針、時針）

        # ========== 步驟 4: 計算 sin 和 cos 編碼 ==========
        # 偶數維度用 sin
        # position * div_term 會廣播運算
        # (max_len, 1) * (d_model/2,) → (max_len, d_model/2)
        # 例如：(5000, 1) * (256,) → (5000, 256)
        #
        # pe[:, 0::2] 是什麼意思？
        # - [:, 0::2] 表示「所有行，從第 0 列開始每隔 2 列取一個」
        # - 也就是第 0, 2, 4, 6, ... 列（偶數列）
        # - shape: (max_len, d_model/2)
        #
        # 具體例子（位置 pos=1, 維度 i=0）：
        # sin(1 * 1.0) = sin(1) ≈ 0.841
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇數維度用 cos
        # pe[:, 1::2] 表示「所有行，從第 1 列開始每隔 2 列取一個」
        # 也就是第 1, 3, 5, 7, ... 列（奇數列）
        #
        # 具體例子（位置 pos=1, 維度 i=0）：
        # cos(1 * 1.0) = cos(1) ≈ 0.540
        pe[:, 1::2] = torch.cos(position * div_term)

        # 現在 pe 的 shape: (max_len, d_model)
        # 例如：(5000, 512)
        # 每一行是一個位置的完整編碼（512 維向量）

        # ========== 步驟 5: 增加 batch 維度 ==========
        # 將 shape 從 (max_len, d_model) 變成 (1, max_len, d_model)
        # 例如：(5000, 512) → (1, 5000, 512)
        #
        # 為什麼要加這個 1？
        # 因為實際使用時，輸入的 shape 是 (batch_size, seq_len, d_model)
        # 例如：(32, 50, 512)  ← 32 個樣本，每個 50 個詞，每個詞 512 維
        #
        # 加法運算：
        # (32, 50, 512) + (1, 5000, 512)
        # PyTorch 會自動廣播：
        # (32, 50, 512) + (1, 50, 512) → (32, 50, 512)
        #             ↑
        #     只取前 50 個位置的編碼
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        # ========== 步驟 6: 註冊為 buffer ==========
        # 將位置編碼註冊為 buffer（而不是 parameter）
        #
        # register_buffer vs nn.Parameter 的差異：
        #
        # nn.Parameter:
        # ✓ 會被優化器更新（訓練時會改變）
        # ✓ 用於需要學習的權重（如 W_q, W_k, W_v）
        #
        # register_buffer:
        # ✓ 不會被優化器更新（固定不變）
        # ✓ 會跟著模型移動（model.to('cuda') 時自動移到 GPU）
        # ✓ 會被保存在 state_dict 中（checkpoint 包含這個）
        # ✓ 用於固定的常數（如位置編碼）
        #
        # 為什麼位置編碼用 buffer？
        # - 位置編碼是數學公式計算出來的，是固定的
        # - 不需要訓練去學習
        # - 但需要跟著模型移動到 GPU/CPU
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

        完整流程範例：
            假設輸入是 "我愛吃蘋果"（5 個字）
            batch_size = 2, seq_len = 5, d_model = 512

            步驟 1: 輸入（word embeddings）
                x.shape = (2, 5, 512)
                x[0, 0, :] = "我" 的 embedding（512 維向量）
                x[0, 1, :] = "愛" 的 embedding（512 維向量）
                x[0, 2, :] = "吃" 的 embedding（512 維向量）
                x[0, 3, :] = "蘋" 的 embedding（512 維向量）
                x[0, 4, :] = "果" 的 embedding（512 維向量）

                問題：這些 embedding 裡沒有位置資訊！
                     "我愛吃蘋果" 和 "蘋果吃愛我" 的 embedding 完全一樣

            步驟 2: 取出位置編碼
                self.pe.shape = (1, 5000, 512)  ← 預先計算好的
                self.pe[:, :5] 取出前 5 個位置的編碼
                → shape = (1, 5, 512)

                self.pe[0, 0, :] = 位置 0 的編碼（512 維向量）
                self.pe[0, 1, :] = 位置 1 的編碼（512 維向量）
                self.pe[0, 2, :] = 位置 2 的編碼（512 維向量）
                self.pe[0, 3, :] = 位置 3 的編碼（512 維向量）
                self.pe[0, 4, :] = 位置 4 的編碼（512 維向量）

            步驟 3: 相加（廣播）
                x + self.pe[:, :5]
                (2, 5, 512) + (1, 5, 512) → (2, 5, 512)
                         ↑
                    batch 維度自動廣播

                結果：
                output[0, 0, :] = "我" 的 embedding + 位置 0 的編碼
                output[0, 1, :] = "愛" 的 embedding + 位置 1 的編碼
                output[0, 2, :] = "吃" 的 embedding + 位置 2 的編碼
                output[0, 3, :] = "蘋" 的 embedding + 位置 3 的編碼
                output[0, 4, :] = "果" 的 embedding + 位置 4 的編碼

                現在每個詞的 embedding 包含：
                ✓ 詞本身的語義資訊（來自 word embedding）
                ✓ 它在句子中的位置資訊（來自 positional encoding）

            步驟 4: Dropout
                隨機將一些值設為 0（訓練時）
                防止過擬合
        """
        # ========== 步驟 1: 取出對應長度的位置編碼 ==========
        # x.size(1) 是 seq_len（句子的長度）
        # 例如："我愛吃蘋果" → seq_len = 5
        #
        # self.pe[:, :seq_len] 表示：
        # - [:, :seq_len] 取出前 seq_len 個位置的編碼
        # - 例如從 (1, 5000, 512) 取出 (1, 5, 512)
        #
        # 為什麼不是全部取？
        # - 因為不同句子長度不同
        # - 短句子只需要前面幾個位置的編碼
        # - 長句子需要更多位置的編碼
        # - 只要 seq_len <= max_len (5000)，都可以處理

        # ========== 步驟 2: 加上位置編碼 ==========
        # 加法會自動廣播：
        # x.shape                = (batch_size, seq_len, d_model)  例如 (2, 5, 512)
        # self.pe[:, :seq_len]   = (1, seq_len, d_model)           例如 (1, 5, 512)
        # 結果                   = (batch_size, seq_len, d_model)  例如 (2, 5, 512)
        #
        # 廣播過程：
        # batch 中的每個樣本都加上同樣的位置編碼
        # → 因為位置編碼只跟「位置」有關，跟「內容」無關
        #
        # .requires_grad_(False) 的作用：
        # - 確保位置編碼不會被計算梯度
        # - 因為位置編碼是固定的數學公式，不需要學習
        # - 節省記憶體和計算量
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        # ========== 步驟 3: Dropout ==========
        # 訓練模式：隨機將一些值設為 0
        # 評估模式：不做任何改變
        # 目的：防止過擬合（正則化技巧）
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
