"""
位置編碼（Positional Encoding）

為什麼需要這個？
- Transformer 的注意力機制是順序無關的
- 我們需要明確注入位置資訊

實現方式：
- 使用正弦函數生成位置編碼
- 加到詞嵌入上
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置編碼模組（Positional Encoding Module）

    【為什麼需要位置編碼？】
    問題：注意力機制完全是順序無關的！

    具體範例：
        句子 A："I love eating apples"
        句子 B："apples eating love I"

    對注意力機制而言：
        - 這兩個句子有完全相同的 Q、K、V 矩陣！
        - 因為注意力只看「哪些詞相關」
        - 它不在乎「順序」
        - 就像把詞丟進袋子裡，失去所有順序資訊

    這是個大問題，因為：
        - "I love eating apples" 和 "apples eating love I" 意思完全不同
        - "The cat sat on the mat" vs "The mat sat on the cat" - 完全不同！
        - 在語言中，順序很重要！

    解決方案：位置編碼
        為每個詞的嵌入添加位置資訊
        → 讓模型知道：這個詞在位置 0、1、2 等等

    【為什麼使用 sin/cos 函數？】
    有很多方法可以編碼位置。為什麼選擇 sin/cos？

    選項 1：可學習的位置嵌入
        缺點：只能處理訓練時見過的長度
        → 如果訓練時最長 100 詞，在 150 詞的句子上會失敗

    選項 2：固定的 sin/cos 編碼（本實現）
        優點：
        ✓ 可以處理任何序列長度（超出訓練範圍也能泛化）
        ✓ 無需學習參數（減少模型大小）
        ✓ 數學性質：相對位置可以表示為線性組合
          （模型可以學習「前進 3 個位置」）

    【時鐘類比】
    想像一個時鐘：
        - 秒針：轉得快（高頻率）
        - 分針：轉得較慢（中等頻率）
        - 時針：轉得最慢（低頻率）
        - 組合起來，它們可以唯一標識任何時刻！

    位置編碼類似：
        - 低維度：高頻波（像秒針）
        - 中間維度：中頻波（像分針）
        - 高維度：低頻波（像時針）
        - 組合起來，它們可以唯一標識每個位置！

    【公式說明】
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中：
        pos     = 位置索引（0, 1, 2, 3, ...）
        i       = 維度索引（0, 1, 2, 3, ...）
        d_model = 模型維度（例如 512）
        2i      = 偶數維度使用 sin
        2i+1    = 奇數維度使用 cos

    具體範例（假設 d_model=512）：
        位置 0 的編碼：
            維度 0（偶數）：sin(0 / 10000^(0/512))   = sin(0) = 0
            維度 1（奇數）：cos(0 / 10000^(0/512))   = cos(0) = 1
            維度 2（偶數）：sin(0 / 10000^(2/512))   = sin(0) = 0
            維度 3（奇數）：cos(0 / 10000^(2/512))   = cos(0) = 1
            ...

        位置 1 的編碼：
            維度 0：sin(1 / 10000^(0/512))   = sin(1) ≈ 0.841
            維度 1：cos(1 / 10000^(0/512))   = cos(1) ≈ 0.540
            維度 2：sin(1 / 10000^(2/512))   ≈ 0.841
            維度 3：cos(1 / 10000^(2/512))   ≈ 0.540
            ...

    【為什麼是 10000？】
        - 這是經驗值（原論文使用）
        - 目的：為不同維度創建不同頻率
        - 10000^(2i/d_model) 隨 i 增加
        - → 頻率降低（從快速振盪到慢速振盪）

    Args:
        d_model: 模型維度（例如 512）
        max_len: 最大序列長度（預設 5000，可處理 5000 詞的句子）
        dropout: Dropout 機率（預設 0.1，防止過擬合）
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ========== 步驟 1：初始化位置編碼矩陣 ==========
        # 創建全零矩陣，稍後填入 sin/cos 值
        # 形狀：(max_len, d_model)
        # 範例：(5000, 512) → 可處理最多 5000 詞的句子
        #
        # 為什麼是 (max_len, d_model)？
        # - 每一行：代表一個位置（詞 0、詞 1、詞 2...）
        # - 每一列：代表一個維度（512 個維度）
        # - 每個值：該位置在該維度的編碼
        #
        # 這個矩陣是固定的（不學習！），所以使用 register_buffer，不是 nn.Parameter
        pe = torch.zeros(max_len, d_model)

        # ========== 步驟 2：創建位置索引 ==========
        # 創建位置索引：[0, 1, 2, ..., max_len-1]
        # 範例：[0, 1, 2, ..., 4999]
        position = torch.arange(0, max_len, dtype=torch.float)  # 形狀：(max_len,)

        # unsqueeze(1) 將形狀從 (max_len,) 變為 (max_len, 1)
        # 範例：(5000,) → (5000, 1)
        #
        # 為什麼要 unsqueeze？
        # 因為我們要與 div_term（形狀：(d_model/2,)）相乘
        # 廣播規則：(5000, 1) * (256,) → (5000, 256)
        position = position.unsqueeze(1)  # 形狀：(max_len, 1)

        # ========== 步驟 3：計算頻率項（分母）==========
        # 目標：計算 10000^(2i/d_model)
        #
        # torch.arange(0, d_model, 2) 生成偶數索引
        # d_model=512 的範例：[0, 2, 4, 6, ..., 510]
        # → 形狀：(256,)，即 d_model/2
        #
        # 為什麼只需要偶數索引？
        # 因為在公式中：
        # - 偶數維度（2i）使用 sin
        # - 奇數維度（2i+1）使用 cos
        # 但它們共享相同的頻率項（分母）！
        # 所以我們只需要計算 d_model/2 個頻率值

        # 數學推導：
        # 原始分母：10000^(2i/d_model)
        # 轉換為 exp 形式：10000^(2i/d_model) = exp(log(10000^(2i/d_model)))
        #                                          = exp((2i/d_model) * log(10000))
        # 但我們需要 1 / 10000^(2i/d_model)
        # = exp(-(2i/d_model) * log(10000))
        #
        # 為什麼要這樣轉換？
        # - 直接計算 10000^x 當 x 很大時可能溢位
        # - exp(-x * log(10000)) 在數值上更穩定
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term 形狀：(d_model/2,) 範例：(256,)
        #
        # 具體範例值（d_model=512）：
        # i=0:   exp(-(0/512) * log(10000)) = exp(0) = 1.0
        # i=2:   exp(-(2/512) * log(10000)) ≈ 0.912
        # i=4:   exp(-(4/512) * log(10000)) ≈ 0.832
        # ...
        # i=510: exp(-(510/512) * log(10000)) ≈ 0.0001  （非常低的頻率）
        #
        # → 頻率從高到低遞減（像時鐘指針）

        # ========== 步驟 4：計算 sin 和 cos 編碼 ==========
        # 偶數維度使用 sin
        # position * div_term 廣播：
        # (max_len, 1) * (d_model/2,) → (max_len, d_model/2)
        # 範例：(5000, 1) * (256,) → (5000, 256)
        #
        # pe[:, 0::2] 是什麼意思？
        # - [:, 0::2] = 「所有行，從列 0 開始，步長為 2」
        # - 即列 0, 2, 4, 6, ...（偶數列）
        # - 形狀：(max_len, d_model/2)
        #
        # 具體範例（位置 pos=1，維度 i=0）：
        # sin(1 * 1.0) = sin(1) ≈ 0.841
        pe[:, 0::2] = torch.sin(position * div_term)

        # 奇數維度使用 cos
        # pe[:, 1::2] 意思是：
        # - 「所有行，從列 1 開始，步長為 2」
        # - 即列 1, 3, 5, 7, ...（奇數列）
        #
        # 具體範例（位置 pos=1，維度 i=0）：
        # cos(1 * 1.0) = cos(1) ≈ 0.540
        pe[:, 1::2] = torch.cos(position * div_term)

        # 現在 pe 形狀：(max_len, d_model)
        # 範例：(5000, 512)
        # 每一行是一個位置的完整編碼（512 維向量）

        # ========== 步驟 5：添加批次維度 ==========
        # 將形狀從 (max_len, d_model) 變為 (1, max_len, d_model)
        # 範例：(5000, 512) → (1, 5000, 512)
        #
        # 為什麼要添加這個 1？
        # 因為實際輸入形狀是 (batch_size, seq_len, d_model)
        # 範例：(32, 50, 512)  ← 32 個樣本，每個 50 詞，每詞 512 維
        #
        # 相加時：
        # (32, 50, 512) + (1, 5000, 512)
        # PyTorch 廣播：
        # (32, 50, 512) + (1, 50, 512) → (32, 50, 512)
        #             ↑
        #     只取前 50 個位置的編碼
        pe = pe.unsqueeze(0)  # 形狀：(1, max_len, d_model)

        # ========== 步驟 6：註冊為 Buffer ==========
        # 將位置編碼註冊為 buffer（不是 parameter）
        #
        # register_buffer vs nn.Parameter 的差異：
        #
        # nn.Parameter：
        # ✓ 訓練時由優化器更新（權重改變）
        # ✓ 用於可學習權重（如 W_q、W_k、W_v）
        #
        # register_buffer：
        # ✓ 不被優化器更新（固定常數）
        # ✓ 隨模型移動（model.to('cuda') 自動移到 GPU）
        # ✓ 儲存在 state_dict（包含在檢查點中）
        # ✓ 用於固定常數（如位置編碼）
        #
        # 為什麼對位置編碼使用 buffer？
        # - 位置編碼由數學公式計算，是固定的
        # - 訓練時無需學習
        # - 但需要隨模型移動到 GPU/CPU
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        為輸入添加位置編碼

        Args:
            x: 輸入詞嵌入，形狀 (batch_size, seq_len, d_model)

        Returns:
            output: 添加位置編碼後的輸入，形狀 (batch_size, seq_len, d_model)

        完整流程範例："I love eating apples"
            假設：batch_size=1, seq_len=5, d_model=512

            步驟 1：輸入（詞嵌入）
                x.shape = (1, 5, 512)
                x[0, 0, :] = "I" 的嵌入（512 維向量）
                x[0, 1, :] = "love" 的嵌入（512 維向量）
                x[0, 2, :] = "eating" 的嵌入（512 維向量）
                x[0, 3, :] = "apples" 的嵌入（512 維向量）
                x[0, 4, :] = "<END>" 的嵌入（512 維向量）

                問題：這些嵌入沒有位置資訊！
                     "I love eating apples" 和 "apples eating love I"
                     有相同的嵌入

            步驟 2：提取位置編碼
                self.pe.shape = (1, 5000, 512)  ← 預先計算
                self.pe[:, :5] 提取前 5 個位置
                → 形狀 = (1, 5, 512)

                self.pe[0, 0, :] = 位置 0 編碼（512 維向量）
                self.pe[0, 1, :] = 位置 1 編碼（512 維向量）
                self.pe[0, 2, :] = 位置 2 編碼（512 維向量）
                self.pe[0, 3, :] = 位置 3 編碼（512 維向量）
                self.pe[0, 4, :] = 位置 4 編碼（512 維向量）

            步驟 3：相加（廣播相加）
                x + self.pe[:, :5]
                (1, 5, 512) + (1, 5, 512) → (1, 5, 512)
                         ↑
                    批次維度自動廣播

                結果：
                output[0, 0, :] = "I" 嵌入 + 位置 0 編碼
                output[0, 1, :] = "love" 嵌入 + 位置 1 編碼
                output[0, 2, :] = "eating" 嵌入 + 位置 2 編碼
                output[0, 3, :] = "apples" 嵌入 + 位置 3 編碼
                output[0, 4, :] = "<END>" 嵌入 + 位置 4 編碼

                現在每個詞的嵌入包含：
                ✓ 詞的語義資訊（來自詞嵌入）
                ✓ 位置資訊（來自位置編碼）

            步驟 4：Dropout
                隨機將某些值設為 0（僅訓練時）
                防止過擬合
        """
        # ========== 步驟 1：提取位置編碼 ==========
        # x.size(1) 是 seq_len（句子長度）
        # 範例："I love eating apples" → seq_len = 5
        #
        # self.pe[:, :seq_len] 意思是：
        # - [:, :seq_len] 提取前 seq_len 個位置的編碼
        # - 範例：從 (1, 5000, 512) 提取 (1, 5, 512)
        #
        # 為什麼不提取全部？
        # - 不同句子有不同長度
        # - 短句子只需要前幾個位置
        # - 長句子需要更多位置
        # - 只要 seq_len <= max_len (5000)，我們就能處理

        # ========== 步驟 2：添加位置編碼 ==========
        # 相加自動廣播：
        # x.shape                = (batch_size, seq_len, d_model)  例如 (32, 50, 512)
        # self.pe[:, :seq_len]   = (1, seq_len, d_model)           例如 (1, 50, 512)
        # 結果                   = (batch_size, seq_len, d_model)  例如 (32, 50, 512)
        #
        # 廣播過程：
        # 批次中的每個樣本獲得相同的位置編碼
        # → 因為位置編碼只依賴位置，不依賴內容
        #
        # .requires_grad_(False) 的目的：
        # - 確保位置編碼不計算梯度
        # - 因為位置編碼是固定的數學公式，不可學習
        # - 節省記憶體和計算
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        # ========== 步驟 3：Dropout ==========
        # 訓練模式：隨機將 10% 的值設為 0
        # 評估模式：不改變（自動停用）
        # 目的：防止過擬合（正則化技術）
        return self.dropout(x)


def visualize_positional_encoding(d_model: int = 512, max_len: int = 100):
    """
    視覺化位置編碼（用於理解）

    此函數對 transformer 不是必需的，
    但有助於理解編碼模式。

    Args:
        d_model: 模型維度
        max_len: 要視覺化的最大長度
    """
    import matplotlib.pyplot as plt

    # 創建位置編碼
    pe = PositionalEncoding(d_model, max_len)

    # 提取編碼矩陣
    # 形狀：(1, max_len, d_model) -> (max_len, d_model)
    encoding = pe.pe.squeeze(0).numpy()

    # 繪製熱圖
    plt.figure(figsize=(15, 5))
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('位置')
    plt.ylabel('維度')
    plt.colorbar()
    plt.title(f'位置編碼（d_model={d_model}）')
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

    # 創建位置編碼模組
    pe = PositionalEncoding(d_model)

    # 創建假的詞嵌入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入形狀：{x.shape}")

    # 添加位置編碼
    output = pe(x)
    print(f"輸出形狀：{output.shape}")

    # 視覺化（可選）
    # visualize_positional_encoding(d_model=128, max_len=100)
