"""
完整 Transformer 模型

這是整合以下所有元件的頂層模型：
1. 來源和目標嵌入層（Embeddings）
2. 位置編碼（Positional Encoding）
3. 編碼器堆疊（雙向注意力）
4. 解碼器堆疊（遮罩自注意力 + 交叉注意力）
5. 最終線性投影到詞彙表

架構：
    來源序列
         ↓
    [嵌入層 + 位置編碼]
         ↓
    [編碼器堆疊] → 記憶（編碼器輸出）
         ↓
    目標序列
         ↓
    [嵌入層 + 位置編碼]
         ↓
    [解碼器堆疊] ← 記憶
         ↓
    [線性投影]
         ↓
    輸出邏輯值（詞彙表機率）
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .encoder import Encoder
from .decoder import Decoder, create_causal_mask
from .positional_encoding import PositionalEncoding


class TokenEmbedding(nn.Module):
    """
    標記嵌入層

    【這是什麼？】
    將離散的標記（單字 ID）轉換成神經網路可以處理的連續向量表示。

    【為什麼需要這個？】
    神經網路無法直接處理離散符號（如單字 ID）。
    它們需要連續向量。標記嵌入將詞彙表中的每個單字映射到一個學習得到的稠密向量。

    【運作方式】
    - 輸入：標記 ID [batch_size, seq_len]（整數）
    - 輸出：嵌入向量 [batch_size, seq_len, d_model]（浮點數）

    範例：
        vocab_size = 10000（10,000 個不同的單字）
        d_model = 512（每個單字 → 512 維向量）

        標記 ID 42 → [0.23, -0.45, 0.12, ..., 0.67]（512 個數字）
        標記 ID 99 → [0.81, 0.34, -0.23, ..., -0.12]（512 個數字）

    【縮放因子】
    我們將嵌入向量乘以 sqrt(d_model)，防止它們相對於位置編碼來說太小，
    這確保了訓練的穩定性。

    這個縮放技巧在原始 Transformer 論文（Vaswani et al., 2017）中有提到。
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化標記嵌入層

        Args:
            vocab_size: 詞彙表大小（唯一標記的數量）
            d_model: 模型維度（嵌入向量大小）

        範例：
            vocab_size=10000 表示我們有 10,000 個唯一的單字
            d_model=512 表示每個單字變成一個 512 維的向量
        """
        super().__init__()

        # 嵌入查詢表：vocab_size × d_model
        # 每一列是一個標記的嵌入向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # 縮放因子：sqrt(d_model)
        # 防止嵌入向量被位置編碼壓過
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        將標記 ID 轉換為嵌入向量

        Args:
            x: 標記 ID，形狀 [batch_size, seq_len]

        Returns:
            縮放後的嵌入向量，形狀 [batch_size, seq_len, d_model]

        範例：
            輸入：[[5, 42, 99]]  # batch=1, seq_len=3
            輸出：[[[0.23, -0.45, ...],   # 標記 5 的嵌入向量
                      [0.81, 0.34, ...],    # 標記 42 的嵌入向量
                      [-0.12, 0.67, ...]]]  # 標記 99 的嵌入向量
                    形狀：(1, 3, 512)
        """
        # 步驟 1：為每個標記查詢嵌入向量
        # x: (batch, seq_len) → (batch, seq_len, d_model)
        embedded = self.embedding(x)

        # 步驟 2：乘以 sqrt(d_model) 進行縮放
        # 這個縮放有助於平衡嵌入向量和位置編碼的大小
        return embedded * self.scale


class Transformer(nn.Module):
    """
    完整 Transformer 模型

    【這是什麼？】
    這是來自「Attention Is All You Need」（2017）的完整 Transformer 架構。
    它將我們建構的所有元件整合成一個完整的序列到序列模型。

    【主要元件】

    1. 來源嵌入層
       - 將來源標記轉換為向量
       - 加入位置資訊

    2. 目標嵌入層
       - 將目標標記轉換為向量
       - 加入位置資訊

    3. 編碼器
       - 處理來源序列
       - 使用雙向自注意力（可以看到整個輸入）
       - 產生「記憶」（編碼表示）

    4. 解碼器
       - 生成目標序列
       - 使用遮罩自注意力（只能看到先前的標記）
       - 使用交叉注意力存取編碼器記憶
       - 產生上下文化的表示

    5. 輸出投影
       - 將解碼器輸出映射到詞彙表邏輯值
       - 每個位置得到所有可能標記的機率分布

    【完整資料流程】

    訓練（已知目標）：
        來源：「I love AI」
            ↓ [嵌入層 + 位置]
        編碼器 → 記憶：[「I love AI」的編碼表示]
            ↓
        目標：「<start> 我 爱」
            ↓ [嵌入層 + 位置]
        解碼器（+ 記憶）→ 輸出：「我 爱 AI」
            ↓ [線性投影]
        邏輯值：[機率分布]

    推論（生成目標）：
        來源：「I love AI」
            ↓ [嵌入層 + 位置]
        編碼器 → 記憶
            ↓
        目標：「<start>」
            ↓ [嵌入層 + 位置]
        解碼器（+ 記憶）→ 「我」
            ↓
        目標：「<start> 我」
            ↓ [嵌入層 + 位置]
        解碼器（+ 記憶）→ 「愛」
            ↓
        ...（繼續直到 <end> 標記）

    【關鍵設計決策】

    1. 共享 vs 分離嵌入層：
       - 我們對來源和目標使用分離的嵌入層
       - 它們可能有不同的詞彙表（例如，英文 vs 中文）
       - 但兩者都有相同的 d_model 維度

    2. 權重綁定：
       - 可選地，我們可以在目標嵌入層和輸出投影之間共享權重
       - 這減少了參數並通常提升效能
       - 為了清晰起見，這裡沒有實作

    3. 位置編碼：
       - 在編碼器/解碼器之前加入到嵌入向量
       - 使用正弦函數（不同頻率的 sin/cos）
       - 讓模型能夠理解單字順序

    【形狀追蹤】
    在整個模型中，我們維護這些形狀：

    - 來源標記：(batch, src_len)
    - 目標標記：(batch, tgt_len)
    - 來源嵌入向量：(batch, src_len, d_model)
    - 目標嵌入向量：(batch, tgt_len, d_model)
    - 編碼器輸出（記憶）：(batch, src_len, d_model)
    - 解碼器輸出：(batch, tgt_len, d_model)
    - 最終邏輯值：(batch, tgt_len, tgt_vocab_size)

    d_model 維度（通常是 512）在整個模型中保持不變！
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 5000,
        activation: str = "relu"
    ):
        """
        初始化完整 Transformer 模型

        Args:
            src_vocab_size: 來源詞彙表大小
                範例：10000 表示 10,000 個英文單字

            tgt_vocab_size: 目標詞彙表大小
                範例：8000 表示 8,000 個中文字

            d_model: 模型維度（嵌入向量大小）
                預設：512（來自原始論文）
                這是模型的「寬度」

            num_heads: 注意力頭的數量
                預設：8（來自原始論文）
                d_model 必須能被 num_heads 整除

            num_encoder_layers: 要堆疊的編碼器層數
                預設：6（來自原始論文）
                更多層 = 更深的模型，可以學習更複雜的模式

            num_decoder_layers: 要堆疊的解碼器層數
                預設：6（來自原始論文）
                通常與 num_encoder_layers 相同

            d_ff: 前饋網路的維度
                預設：2048（來自原始論文）
                通常是 d_model 的 4 倍
                這是 FFN 中的「擴展因子」

            dropout: 用於正規化的 Dropout 機率
                預設：0.1（10% dropout）
                透過隨機丟棄連接來防止過擬合

            max_seq_length: 最大序列長度
                預設：5000
                決定位置編碼表的大小
                必須 ≥ 資料中最長的序列

            activation: FFN 中的激活函數
                選項：「relu」或「gelu」
                預設：「relu」（來自原始論文）

        【參數數量範例】
        使用預設設定（src_vocab=10000, tgt_vocab=8000）：
        - 來源嵌入層：10000 × 512 = 5.1M 參數
        - 目標嵌入層：8000 × 512 = 4.1M 參數
        - 編碼器：每層約 6M 參數 × 6 = 36M
        - 解碼器：每層約 9M 參數 × 6 = 54M
        - 輸出投影：512 × 8000 = 4.1M
        - 總計：約 103M 參數

        以現代標準來說，這是中等大小的模型。
        對於 CPU 訓練，我們通常使用較小的值（d_model=256, layers=2-4）。
        """
        super().__init__()

        # ============================================================
        # 1. 嵌入層
        # ============================================================
        # 將標記 ID 轉換為連續向量

        # 來源嵌入層（例如：英文單字 → 向量）
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)

        # 目標嵌入層（例如：中文字 → 向量）
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # ============================================================
        # 2. 位置編碼
        # ============================================================
        # 將位置資訊加入嵌入向量
        # 在編碼器和解碼器之間共享，因為位置編碼是通用的

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )

        # ============================================================
        # 3. 編碼器堆疊
        # ============================================================
        # 使用雙向注意力處理來源序列

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation
        )

        # ============================================================
        # 4. 解碼器堆疊
        # ============================================================
        # 使用遮罩自注意力 + 交叉注意力生成目標序列

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout,
            activation=activation
        )

        # ============================================================
        # 5. 輸出投影
        # ============================================================
        # 將解碼器輸出映射到詞彙表邏輯值
        # 從 d_model 維度投影到詞彙表大小

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # ============================================================
        # 6. 初始化參數
        # ============================================================
        # 使用 Xavier/Glorot 初始化以獲得更好的訓練穩定性
        # 這對深度網路很重要！

        self._init_parameters()

    def _init_parameters(self):
        """
        使用 Xavier uniform 初始化模型參數

        【為什麼要初始化？】
        正確的初始化對訓練深度網路至關重要：
        - 太小 → 梯度消失
        - 太大 → 梯度爆炸
        - Xavier 初始化在兩者之間取得平衡

        【哪些會被初始化？】
        - 線性層（權重和偏差）
        - 嵌入層

        LayerNorm 和其他元件有自己的預設初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                # 多維參數（權重）→ Xavier uniform
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼來源序列

        【這會做什麼？】
        透過以下步驟處理來源序列（例如：英文句子）：
        1. 標記嵌入
        2. 位置編碼
        3. 編碼器堆疊

        結果：「記憶」- 來源的編碼表示

        Args:
            src: 來源標記 ID，形狀 [batch_size, src_len]
                範例：[[5, 42, 99, 103]]（4 個標記）

            src_mask: 可選的來源遮罩，形狀 [batch_size, 1, 1, src_len]
                用於遮罩填充標記
                1 = 注意，0 = 忽略

        Returns:
            編碼器輸出（記憶），形狀 [batch_size, src_len, d_model]

        【範例流程】
        輸入標記：[5, 42, 99]
            ↓ [標記嵌入]
        嵌入向量：[[0.23, ...], [0.81, ...], [-0.12, ...]]（每個 512 維）
            ↓ [加入位置編碼]
        位置感知：[[0.25, ...], [0.79, ...], [-0.15, ...]]
            ↓ [編碼器堆疊 - 6 層注意力 + FFN]
        記憶：[[0.67, ...], [0.34, ...], [0.12, ...]]（編碼表示）
        """
        # 步驟 1：將標記 ID 轉換為嵌入向量
        # src: (batch, src_len) → (batch, src_len, d_model)
        src_embedded = self.src_embedding(src)

        # 步驟 2：加入位置編碼
        # 告訴模型每個單字在序列中的位置
        src_encoded = self.positional_encoding(src_embedded)

        # 步驟 3：通過編碼器堆疊
        # 每層應用自注意力 + FFN
        memory = self.encoder(src_encoded, src_mask)

        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        給定編碼器記憶解碼目標序列

        【這會做什麼？】
        透過以下步驟處理目標序列（例如：中文字）：
        1. 標記嵌入
        2. 位置編碼
        3. 解碼器堆疊（可存取編碼器記憶）

        結果：準備好進行輸出投影的上下文化表示

        Args:
            tgt: 目標標記 ID，形狀 [batch_size, tgt_len]
                範例：[[1, 203, 456]]（3 個標記，包括 <start>）

            memory: 編碼器輸出，形狀 [batch_size, src_len, d_model]
                編碼的來源序列

            tgt_mask: 可選的因果遮罩，形狀 [batch_size, 1, tgt_len, tgt_len]
                防止注意到未來位置
                下三角矩陣：[[1,0,0], [1,1,0], [1,1,1]]

            src_mask: 可選的來源遮罩，形狀 [batch_size, 1, 1, src_len]
                在交叉注意力中用於遮罩來源填充

        Returns:
            解碼器輸出，形狀 [batch_size, tgt_len, d_model]

        【範例流程 - 翻譯】
        目標標記：[<start>, 我, 愛]
            ↓ [標記嵌入]
        嵌入向量：[[0.45, ...], [0.23, ...], [0.67, ...]]
            ↓ [加入位置編碼]
        位置感知：[[0.47, ...], [0.21, ...], [0.69, ...]]
            ↓ [解碼器堆疊 - 6 層]
              每層：
              - 遮罩自注意力（查看先前的單字）
              - 交叉注意力（查看來源：「I love AI」）
              - 前饋網路
        輸出：[[0.89, ...], [0.34, ...], [0.56, ...]]
        """
        # 步驟 1：將標記 ID 轉換為嵌入向量
        # tgt: (batch, tgt_len) → (batch, tgt_len, d_model)
        tgt_embedded = self.tgt_embedding(tgt)

        # 步驟 2：加入位置編碼
        tgt_encoded = self.positional_encoding(tgt_embedded)

        # 步驟 3：通過解碼器堆疊
        # 每層使用：
        # - 目標的遮罩自注意力
        # - 到來源（記憶）的交叉注意力
        # - 前饋網路
        output = self.decoder(tgt_encoded, memory, tgt_mask, src_mask)

        return output

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整的 Transformer 前向傳遞

        【這會做什麼？】
        這是模型的主要進入點。它執行：
        1. 編碼來源序列 → 記憶
        2. 使用記憶解碼目標序列 → 上下文化輸出
        3. 投影到詞彙表 → 邏輯值

        在訓練期間使用教師強制（提供正確的目標）。

        Args:
            src: 來源標記 ID，形狀 [batch_size, src_len]
                範例：[[5, 42, 99, 103]]（英文單字）

            tgt: 目標標記 ID，形狀 [batch_size, tgt_len]
                範例：[[1, 203, 456, 789]]（中文字）

            src_mask: 可選的來源遮罩，形狀 [batch_size, 1, 1, src_len]
                遮罩來源中的填充

            tgt_mask: 可選的目標遮罩，形狀 [batch_size, 1, tgt_len, tgt_len]
                遮罩目標中的填充和未來位置
                通常是因果遮罩（下三角）

        Returns:
            輸出邏輯值，形狀 [batch_size, tgt_len, tgt_vocab_size]
            這些是 softmax 前的原始分數

        【完整範例 - 機器翻譯】

        來源：「I love AI」→ [5, 42, 99]
        目標：「<start> 我 愛 AI」→ [1, 203, 456, 789]

        步驟 1：編碼來源
            [5, 42, 99]
                ↓ 嵌入 + 位置
            [[0.23, ...], [0.81, ...], [-0.12, ...]]
                ↓ 編碼器（6 層自注意力 + FFN）
            記憶：[[0.67, ...], [0.34, ...], [0.12, ...]]

        步驟 2：解碼目標
            [1, 203, 456, 789]
                ↓ 嵌入 + 位置
            [[0.45, ...], [0.23, ...], [0.67, ...], [0.89, ...]]
                ↓ 解碼器（6 層遮罩自注意力 + 交叉注意力 + FFN）
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]

        步驟 3：投影到詞彙表
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]
                ↓ 線性（d_model → tgt_vocab_size）
            邏輯值：[batch, 4, 8000]
                位置 0：「我」在 8000 個單字上的機率分布
                位置 1：「愛」在 8000 個單字上的機率分布
                位置 2：「AI」在 8000 個單字上的機率分布
                位置 3：<end> 在 8000 個單字上的機率分布

        【訓練 vs 推論】

        訓練（教師強制）：
            - 我們有正確的目標序列
            - 一次通過整個目標
            - 使用因果遮罩防止作弊
            - 快速且可平行化

        推論（自迴歸）：
            - 一次生成一個標記
            - 使用先前的預測作為輸入
            - 較慢但生成所必需
            - 參見下面的 generate() 方法
        """
        # 步驟 1：編碼來源序列
        # src: (batch, src_len) → memory: (batch, src_len, d_model)
        memory = self.encode(src, src_mask)

        # 步驟 2：使用記憶解碼目標序列
        # tgt: (batch, tgt_len) → output: (batch, tgt_len, d_model)
        output = self.decode(tgt, memory, tgt_mask, src_mask)

        # 步驟 3：投影到詞彙表
        # output: (batch, tgt_len, d_model) → logits: (batch, tgt_len, tgt_vocab_size)
        logits = self.output_projection(output)

        return logits

    def generate(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        start_token: int = 1,
        end_token: int = 2
    ) -> torch.Tensor:
        """
        自迴歸生成目標序列（一次一個標記）

        【這會做什麼？】
        這在推論期間用於生成翻譯/回應。
        與訓練不同（我們有完整的目標），這裡我們：
        1. 從 <start> 標記開始
        2. 生成下一個標記
        3. 將它附加到序列
        4. 重複直到 <end> 標記或達到 max_len

        這稱為「自迴歸生成」- 每個標記依賴於所有先前的標記。

        Args:
            src: 來源標記 ID，形狀 [batch_size, src_len]
                要翻譯的輸入句子

            src_mask: 可選的來源遮罩，形狀 [batch_size, 1, 1, src_len]

            max_len: 要生成的最大長度
                即使未達到 <end> 標記也會停止

            start_token: <start> 標記的 ID
                通常是 1 或 <bos>（序列開始）

            end_token: <end> 標記的 ID
                通常是 2 或 <eos>（序列結束）

        Returns:
            生成的標記 ID，形狀 [batch_size, generated_len]

        【生成過程範例】

        輸入：「I love AI」→ [5, 42, 99]

        步驟 0：編碼來源
            memory = encode([5, 42, 99])

        步驟 1：從 <start> 開始
            tgt = [1]  # [<start>]
            logits = decode([1], memory)
            next_token = argmax(logits[-1]) = 203  # 「我」
            tgt = [1, 203]

        步驟 2：生成下一個標記
            tgt = [1, 203]  # [<start>, 我]
            logits = decode([1, 203], memory)
            next_token = argmax(logits[-1]) = 456  # 「愛」
            tgt = [1, 203, 456]

        步驟 3：繼續...
            tgt = [1, 203, 456]  # [<start>, 我, 愛]
            logits = decode([1, 203, 456], memory)
            next_token = argmax(logits[-1]) = 789  # 「AI」
            tgt = [1, 203, 456, 789]

        步驟 4：結束
            tgt = [1, 203, 456, 789]  # [<start>, 我, 愛, AI]
            logits = decode([1, 203, 456, 789], memory)
            next_token = argmax(logits[-1]) = 2  # <end>
            停止！返回 [1, 203, 456, 789, 2]

        最終輸出（移除 <start>）：「我 愛 AI」

        【為什麼這很慢？】
        - 對於長度為 N 的序列，必須執行 N 次解碼器前向傳遞
        - 無法平行化 - 每個標記依賴於先前的標記
        - 這是自迴歸模型固有的特性
        - 現代優化：KV 快取（這裡未實作）
        """
        # 將模型設為評估模式
        # 停用 dropout 和其他訓練特定的行為
        self.eval()

        batch_size = src.size(0)
        device = src.device

        # 步驟 1：編碼來源序列一次
        # 我們只需要做一次，因為來源不會改變
        memory = self.encode(src, src_mask)

        # 步驟 2：使用 <start> 標記初始化目標序列
        # 形狀：(batch_size, 1) - 每個批次項目只有一個標記
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # 步驟 3：逐個生成標記
        with torch.no_grad():  # 推論期間不計算梯度
            for _ in range(max_len - 1):  # -1 因為我們已經有 <start>

                # 為當前目標序列建立因果遮罩
                # 確保我們只注意到先前的位置
                tgt_len = tgt.size(1)
                tgt_mask = create_causal_mask(tgt_len).to(device)

                # 解碼當前目標序列
                # tgt: (batch, current_len) → output: (batch, current_len, d_model)
                output = self.decode(tgt, memory, tgt_mask, src_mask)

                # 將最後位置投影到詞彙表
                # output[:, -1, :]: (batch, d_model) → logits: (batch, tgt_vocab_size)
                logits = self.output_projection(output[:, -1, :])

                # 取得最可能的下一個標記
                # 對於每個批次項目，選擇分數最高的標記
                # logits: (batch, tgt_vocab_size) → next_token: (batch, 1)
                next_token = logits.argmax(dim=-1, keepdim=True)

                # 將下一個標記附加到序列
                # tgt: (batch, current_len) → (batch, current_len + 1)
                tgt = torch.cat([tgt, next_token], dim=1)

                # 檢查批次中的所有序列是否都已生成 <end> 標記
                # 如果是，我們可以提前停止
                if (next_token == end_token).all():
                    break

        return tgt

    def count_parameters(self) -> int:
        """
        計算模型中的總可訓練參數數量

        【為什麼這很重要？】
        - 較大的模型更強大但訓練較慢
        - 有助於估計記憶體需求
        - 對比較模型大小很有用

        Returns:
            總可訓練參數數量

        範例輸出：
            預設配置約 103M 參數
            小型模型約 25M 參數（d_model=256, layers=2）
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_seq_length: int = 5000
) -> Transformer:
    """
    建立標準配置 Transformer 模型的工廠函數

    【為什麼使用這個？】
    用於建立常見模型配置的便利函數。
    確保編碼器和解碼器具有相同的層數（常見做法）。

    Args:
        src_vocab_size: 來源詞彙表大小
        tgt_vocab_size: 目標詞彙表大小
        d_model: 模型維度
        num_heads: 注意力頭數量
        num_layers: 層數（編碼器和解碼器相同）
        d_ff: 前饋網路維度
        dropout: Dropout 率
        max_seq_length: 最大序列長度

    Returns:
        已初始化的 Transformer 模型

    【常見配置】

    原始論文（Transformer Base）：
        d_model=512, num_heads=8, num_layers=6, d_ff=2048
        約 60M 參數

    Transformer Big：
        d_model=1024, num_heads=16, num_layers=6, d_ff=4096
        約 210M 參數

    小型（CPU 友善）：
        d_model=256, num_heads=4, num_layers=2, d_ff=1024
        約 10M 參數（適合學習和小型資料集）

    微型（非常快）：
        d_model=128, num_heads=4, num_layers=2, d_ff=512
        約 3M 參數（示範用途）
    """
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )

    return model
