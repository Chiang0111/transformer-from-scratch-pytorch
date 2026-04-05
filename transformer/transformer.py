"""
完整的 Transformer 模型

這是組合所有元件的頂層模型：
1. 源序列與目標序列的嵌入層
2. 位置編碼
3. Encoder 堆疊（雙向注意力）
4. Decoder 堆疊（遮罩自注意力 + 交叉注意力）
5. 最終的線性投影到詞彙表

架構：
    源序列（Source Sequence）
         ↓
    [嵌入 + 位置編碼]
         ↓
    [Encoder 堆疊] → Memory（編碼器輸出）
         ↓
    目標序列（Target Sequence）
         ↓
    [嵌入 + 位置編碼]
         ↓
    [Decoder 堆疊] ← Memory
         ↓
    [線性投影]
         ↓
    輸出 Logits（詞彙表機率分佈）
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
    詞元嵌入層（Token Embedding Layer）

    【這是什麼？】
    將離散的詞元（單字 ID）轉換成連續的向量表示，
    讓神經網路可以處理。

    【為什麼需要這個？】
    神經網路無法直接處理離散符號（如單字 ID）。
    它們需要連續向量。詞元嵌入將詞彙表中的每個單字
    映射到一個學習到的密集向量。

    【如何運作？】
    - 輸入：詞元 ID [batch_size, seq_len]（整數）
    - 輸出：嵌入 [batch_size, seq_len, d_model]（浮點數）

    範例：
        vocab_size = 10000（10,000 個不同單字）
        d_model = 512（每個單字 → 512 維向量）

        詞元 ID 42 → [0.23, -0.45, 0.12, ..., 0.67]（512 個數字）
        詞元 ID 99 → [0.81, 0.34, -0.23, ..., -0.12]（512 個數字）

    【縮放因子】
    我們將嵌入乘以 sqrt(d_model) 以防止它們相對於位置編碼太小，
    這確保了訓練的穩定性。

    這個縮放在原始 Transformer 論文（Vaswani et al., 2017）中有提到。
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化詞元嵌入層

        Args:
            vocab_size: 詞彙表大小（唯一詞元的數量）
            d_model: 模型維度（嵌入大小）

        範例：
            vocab_size=10000 表示我們有 10,000 個唯一單字
            d_model=512 表示每個單字變成 512 維向量
        """
        super().__init__()

        # 嵌入查找表：vocab_size × d_model
        # 每一行是一個詞元的嵌入向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # 縮放因子：sqrt(d_model)
        # 防止嵌入被位置編碼淹沒
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        將詞元 ID 轉換為嵌入

        Args:
            x: 詞元 ID，形狀 [batch_size, seq_len]

        Returns:
            縮放後的嵌入，形狀 [batch_size, seq_len, d_model]

        範例：
            輸入：[[5, 42, 99]]  # batch=1, seq_len=3
            輸出：[[[0.23, -0.45, ...],   # 詞元 5 的嵌入
                   [0.81, 0.34, ...],     # 詞元 42 的嵌入
                   [-0.12, 0.67, ...]]]   # 詞元 99 的嵌入
                  形狀：(1, 3, 512)
        """
        # 步驟 1：查找每個詞元的嵌入
        # x: (batch, seq_len) → (batch, seq_len, d_model)
        embedded = self.embedding(x)

        # 步驟 2：乘以 sqrt(d_model) 進行縮放
        # 這個縮放有助於平衡嵌入和位置編碼的幅度
        return embedded * self.scale


class Transformer(nn.Module):
    """
    完整的 Transformer 模型

    【這是什麼？】
    這是來自 "Attention Is All You Need" (2017) 的完整 Transformer 架構。
    它將我們建構的所有元件組合成一個完整的序列到序列模型。

    【主要元件】

    1. 源序列嵌入（Source Embedding）
       - 將源詞元轉換為向量
       - 加入位置資訊

    2. 目標序列嵌入（Target Embedding）
       - 將目標詞元轉換為向量
       - 加入位置資訊

    3. Encoder
       - 處理源序列
       - 使用雙向自注意力（可以看到整個輸入）
       - 產生「記憶」（編碼表示）

    4. Decoder
       - 生成目標序列
       - 使用遮罩自注意力（只能看到之前的詞元）
       - 使用交叉注意力到編碼器記憶
       - 產生上下文化的表示

    5. 輸出投影（Output Projection）
       - 將解碼器輸出映射到詞彙表 logits
       - 每個位置得到所有可能詞元的機率分佈

    【完整資料流】

    訓練（已知目標）：
        源：「I love AI」
            ↓ [嵌入 + 位置]
        Encoder → Memory：[「I love AI」的編碼表示]
            ↓
        目標：「<start> 我 愛」
            ↓ [嵌入 + 位置]
        Decoder（+ Memory）→ 輸出：「我 愛 AI」
            ↓ [線性投影]
        Logits：[機率分佈]

    推論（生成目標）：
        源：「I love AI」
            ↓ [嵌入 + 位置]
        Encoder → Memory
            ↓
        目標：「<start>」
            ↓ [嵌入 + 位置]
        Decoder（+ Memory）→ 「我」
            ↓
        目標：「<start> 我」
            ↓ [嵌入 + 位置]
        Decoder（+ Memory）→ 「愛」
            ↓
        ...（繼續直到 <end> 詞元）

    【關鍵設計決策】

    1. 共享 vs 獨立嵌入：
       - 我們為源和目標使用獨立嵌入
       - 它們可能有不同的詞彙表（例如英文 vs 中文）
       - 但兩者都有相同的 d_model 維度

    2. 權重綁定（Weight Tying）：
       - 可選地，我們可以在目標嵌入和輸出投影之間共享權重
       - 這減少了參數並經常改善性能
       - 這裡為了清晰起見沒有實作

    3. 位置編碼：
       - 在 encoder/decoder 之前加到嵌入
       - 使用正弦函數（sin/cos，不同頻率）
       - 允許模型理解單字順序

    【形狀追蹤】
    在整個模型中，我們維護這些形狀：

    - 源詞元：(batch, src_len)
    - 目標詞元：(batch, tgt_len)
    - 源嵌入：(batch, src_len, d_model)
    - 目標嵌入：(batch, tgt_len, d_model)
    - Encoder 輸出（memory）：(batch, src_len, d_model)
    - Decoder 輸出：(batch, tgt_len, d_model)
    - 最終 logits：(batch, tgt_len, tgt_vocab_size)

    d_model 維度（通常 512）在整個模型中保持不變！
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
        初始化完整的 Transformer 模型

        Args:
            src_vocab_size: 源詞彙表大小
                範例：10000 表示 10,000 個英文單字

            tgt_vocab_size: 目標詞彙表大小
                範例：8000 表示 8,000 個中文字符

            d_model: 模型維度（嵌入大小）
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

            dropout: 正則化的 Dropout 機率
                預設：0.1（10% dropout）
                通過隨機丟棄連接來防止過擬合

            max_seq_length: 最大序列長度
                預設：5000
                決定位置編碼表的大小
                必須 ≥ 資料中最長的序列

            activation: FFN 中的激活函數
                選項：「relu」或「gelu」
                預設：「relu」（來自原始論文）

        【參數數量範例】
        使用預設設定（src_vocab=10000, tgt_vocab=8000）：
        - 源嵌入：10000 × 512 = 5.1M 參數
        - 目標嵌入：8000 × 512 = 4.1M 參數
        - Encoder：每層 ~6M 參數 × 6 = 36M
        - Decoder：每層 ~9M 參數 × 6 = 54M
        - 輸出投影：512 × 8000 = 4.1M
        - 總計：~103M 參數

        按現代標準這是中等大小的模型。
        對於 CPU 訓練，我們通常使用較小的值（d_model=256, layers=2-4）。
        """
        super().__init__()

        # ============================================================
        # 1. 嵌入層
        # ============================================================
        # 將詞元 ID 轉換為連續向量

        # 源嵌入（例如英文單字 → 向量）
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)

        # 目標嵌入（例如中文字符 → 向量）
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # ============================================================
        # 2. 位置編碼
        # ============================================================
        # 為嵌入加入位置資訊
        # 在編碼器和解碼器之間共享，因為位置編碼是通用的

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )

        # ============================================================
        # 3. Encoder 堆疊
        # ============================================================
        # 用雙向注意力處理源序列

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout,
            activation=activation
        )

        # ============================================================
        # 4. Decoder 堆疊
        # ============================================================
        # 用遮罩自注意力 + 交叉注意力生成目標序列

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
        # 將解碼器輸出映射到詞彙表 logits
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
        適當的初始化對訓練深度網路至關重要：
        - 太小 → 梯度消失
        - 太大 → 梯度爆炸
        - Xavier 初始化平衡兩者

        【初始化什麼？】
        - 線性層（權重和偏置）
        - 嵌入層

        LayerNorm 和其他元件有它們自己的預設初始化。
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
        編碼源序列

        【這做什麼？】
        通過以下步驟處理源序列（例如英文句子）：
        1. 詞元嵌入
        2. 位置編碼
        3. Encoder 堆疊

        結果：「記憶」- 源的編碼表示

        Args:
            src: 源詞元 ID，形狀 [batch_size, src_len]
                範例：[[5, 42, 99, 103]]（4 個詞元）

            src_mask: 可選的源遮罩，形狀 [batch_size, 1, 1, src_len]
                用於遮罩填充詞元
                1 = 注意，0 = 忽略

        Returns:
            Encoder 輸出（記憶），形狀 [batch_size, src_len, d_model]

        【範例流程】
        輸入詞元：[5, 42, 99]
            ↓ [詞元嵌入]
        嵌入：[[0.23, ...], [0.81, ...], [-0.12, ...]]（每個 512 維）
            ↓ [加入位置編碼]
        位置感知：[[0.25, ...], [0.79, ...], [-0.15, ...]]
            ↓ [Encoder 堆疊 - 6 層注意力 + FFN]
        記憶：[[0.67, ...], [0.34, ...], [0.12, ...]]（編碼表示）
        """
        # 步驟 1：將詞元 ID 轉換為嵌入
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

        【這做什麼？】
        通過以下步驟處理目標序列（例如中文字符）：
        1. 詞元嵌入
        2. 位置編碼
        3. Decoder 堆疊（可以訪問編碼器記憶）

        結果：準備好進行輸出投影的上下文化表示

        Args:
            tgt: 目標詞元 ID，形狀 [batch_size, tgt_len]
                範例：[[1, 203, 456]]（3 個詞元，包括 <start>）

            memory: Encoder 輸出，形狀 [batch_size, src_len, d_model]
                編碼的源序列

            tgt_mask: 可選的因果遮罩，形狀 [batch_size, 1, tgt_len, tgt_len]
                防止注意到未來位置
                下三角矩陣：[[1,0,0], [1,1,0], [1,1,1]]

            src_mask: 可選的源遮罩，形狀 [batch_size, 1, 1, src_len]
                在交叉注意力中用於遮罩源填充

        Returns:
            Decoder 輸出，形狀 [batch_size, tgt_len, d_model]

        【範例流程 - 翻譯】
        目標詞元：[<start>, 我, 愛]
            ↓ [詞元嵌入]
        嵌入：[[0.45, ...], [0.23, ...], [0.67, ...]]
            ↓ [加入位置編碼]
        位置感知：[[0.47, ...], [0.21, ...], [0.69, ...]]
            ↓ [Decoder 堆疊 - 6 層]
              每層：
              - 遮罩自注意力（看之前的單字）
              - 交叉注意力（看源：「I love AI」）
              - 前饋網路
        輸出：[[0.89, ...], [0.34, ...], [0.56, ...]]
        """
        # 步驟 1：將詞元 ID 轉換為嵌入
        # tgt: (batch, tgt_len) → (batch, tgt_len, d_model)
        tgt_embedded = self.tgt_embedding(tgt)

        # 步驟 2：加入位置編碼
        tgt_encoded = self.positional_encoding(tgt_embedded)

        # 步驟 3：通過解碼器堆疊
        # 每層使用：
        # - 目標上的遮罩自注意力
        # - 到源（記憶）的交叉注意力
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
        通過 Transformer 的完整前向傳播

        【這做什麼？】
        這是模型的主要入口點。它執行：
        1. 編碼源序列 → 記憶
        2. 使用記憶解碼目標序列 → 上下文化輸出
        3. 投影到詞彙表 → logits

        在訓練期間使用教師強迫（teacher forcing，提供正確目標）。

        Args:
            src: 源詞元 ID，形狀 [batch_size, src_len]
                範例：[[5, 42, 99, 103]]（英文單字）

            tgt: 目標詞元 ID，形狀 [batch_size, tgt_len]
                範例：[[1, 203, 456, 789]]（中文字符）

            src_mask: 可選的源遮罩，形狀 [batch_size, 1, 1, src_len]
                遮罩源中的填充

            tgt_mask: 可選的目標遮罩，形狀 [batch_size, 1, tgt_len, tgt_len]
                遮罩目標中的填充和未來位置
                通常是因果遮罩（下三角）

        Returns:
            輸出 logits，形狀 [batch_size, tgt_len, tgt_vocab_size]
            這些是 softmax 之前的原始分數

        【完整範例 - 機器翻譯】

        源：「I love AI」→ [5, 42, 99]
        目標：「<start> 我 愛 AI」→ [1, 203, 456, 789]

        步驟 1：編碼源
            [5, 42, 99]
                ↓ 嵌入 + 位置
            [[0.23, ...], [0.81, ...], [-0.12, ...]]
                ↓ encoder（6 層自注意力 + FFN）
            memory：[[0.67, ...], [0.34, ...], [0.12, ...]]

        步驟 2：解碼目標
            [1, 203, 456, 789]
                ↓ 嵌入 + 位置
            [[0.45, ...], [0.23, ...], [0.67, ...], [0.89, ...]]
                ↓ decoder（6 層遮罩自注意力 + 交叉注意力 + FFN）
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]

        步驟 3：投影到詞彙表
            [[0.89, ...], [0.34, ...], [0.56, ...], [0.78, ...]]
                ↓ linear（d_model → tgt_vocab_size）
            logits：[batch, 4, 8000]
                位置 0：8000 個詞的機率分佈用於「我」
                位置 1：8000 個詞的機率分佈用於「愛」
                位置 2：8000 個詞的機率分佈用於「AI」
                位置 3：8000 個詞的機率分佈用於 <end>

        【訓練 vs 推論】

        訓練（教師強迫）：
            - 我們有正確的目標序列
            - 一次通過整個目標
            - 使用因果遮罩防止作弊
            - 快速且可並行化

        推論（自回歸）：
            - 一次生成一個詞元
            - 使用之前的預測作為輸入
            - 較慢但對生成是必要的
            - 參見下面的 generate() 方法
        """
        # 步驟 1：編碼源序列
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
        自回歸生成目標序列（一次一個詞元）

        【這做什麼？】
        這在推論期間用於生成翻譯/回應。
        與訓練不同（我們有完整目標），這裡我們：
        1. 只從 <start> 詞元開始
        2. 生成下一個詞元
        3. 將它附加到序列
        4. 重複直到 <end> 詞元或 max_len

        這稱為「自回歸生成」- 每個詞元依賴於所有之前的詞元。

        Args:
            src: 源詞元 ID，形狀 [batch_size, src_len]
                要翻譯的輸入句子

            src_mask: 可選的源遮罩，形狀 [batch_size, 1, 1, src_len]

            max_len: 生成的最大長度
                即使未達到 <end> 詞元也停止

            start_token: <start> 詞元的 ID
                通常是 1 或 <bos>（序列開始）

            end_token: <end> 詞元的 ID
                通常是 2 或 <eos>（序列結束）

        Returns:
            生成的詞元 ID，形狀 [batch_size, generated_len]

        【生成過程範例】

        輸入：「I love AI」→ [5, 42, 99]

        步驟 0：編碼源
            memory = encode([5, 42, 99])

        步驟 1：從 <start> 開始
            tgt = [1]  # [<start>]
            logits = decode([1], memory)
            next_token = argmax(logits[-1]) = 203  # 「我」
            tgt = [1, 203]

        步驟 2：生成下一個詞元
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
        - 必須對長度為 N 的序列運行解碼器前向傳播 N 次
        - 無法並行化 - 每個詞元依賴於之前的
        - 這是自回歸模型的固有特性
        - 現代優化：KV-caching（這裡未實作）
        """
        # 設置模型為評估模式
        # 禁用 dropout 和其他訓練特定行為
        self.eval()

        batch_size = src.size(0)
        device = src.device

        # 步驟 1：編碼源序列一次
        # 我們只需要做一次，因為源不會改變
        memory = self.encode(src, src_mask)

        # 步驟 2：用 <start> 詞元初始化目標序列
        # 形狀：(batch_size, 1) - 每個批次項目只有一個詞元
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # 步驟 3：一個一個生成詞元
        with torch.no_grad():  # 推論期間不計算梯度
            for _ in range(max_len - 1):  # -1 因為我們已經有 <start>

                # 為當前目標序列創建因果遮罩
                # 確保我們只注意到之前的位置
                tgt_len = tgt.size(1)
                tgt_mask = create_causal_mask(tgt_len).to(device)

                # 解碼當前目標序列
                # tgt: (batch, current_len) → output: (batch, current_len, d_model)
                output = self.decode(tgt, memory, tgt_mask, src_mask)

                # 將最後位置投影到詞彙表
                # output[:, -1, :]: (batch, d_model) → logits: (batch, tgt_vocab_size)
                logits = self.output_projection(output[:, -1, :])

                # 獲取最可能的下一個詞元
                # 對於每個批次項目，選擇分數最高的詞元
                # logits: (batch, tgt_vocab_size) → next_token: (batch, 1)
                next_token = logits.argmax(dim=-1, keepdim=True)

                # 將下一個詞元附加到序列
                # tgt: (batch, current_len) → (batch, current_len + 1)
                tgt = torch.cat([tgt, next_token], dim=1)

                # 檢查批次中的所有序列是否都生成了 <end> 詞元
                # 如果是，我們可以提前停止
                if (next_token == end_token).all():
                    break

        return tgt

    def count_parameters(self) -> int:
        """
        計算模型中可訓練參數的總數

        【為什麼這很重要？】
        - 更大的模型更強大但訓練更慢
        - 有助於估計記憶體需求
        - 用於比較模型大小

        Returns:
            可訓練參數的總數

        範例輸出：
            預設配置約 103M 參數
            小模型約 25M 參數（d_model=256, layers=2）
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
    使用標準配置創建 Transformer 模型的工廠函數

    【為什麼使用這個？】
    創建常見模型配置的便利函數。
    確保編碼器和解碼器有相同數量的層（常見做法）。

    Args:
        src_vocab_size: 源詞彙表大小
        tgt_vocab_size: 目標詞彙表大小
        d_model: 模型維度
        num_heads: 注意力頭數量
        num_layers: 層數（編碼器和解碼器相同）
        d_ff: 前饋維度
        dropout: Dropout 率
        max_seq_length: 最大序列長度

    Returns:
        初始化的 Transformer 模型

    【常見配置】

    原始論文（Transformer Base）：
        d_model=512, num_heads=8, num_layers=6, d_ff=2048
        約 60M 參數

    Transformer Big：
        d_model=1024, num_heads=16, num_layers=6, d_ff=4096
        約 210M 參數

    小型（CPU 友善）：
        d_model=256, num_heads=4, num_layers=2, d_ff=1024
        約 10M 參數（適合學習和小資料集）

    極小（非常快）：
        d_model=128, num_heads=4, num_layers=2, d_ff=512
        約 3M 參數（演示用途）
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
