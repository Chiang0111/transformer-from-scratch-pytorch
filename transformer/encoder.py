"""
Transformer 編碼器層（Encoder Layer）

整合所有組件：
1. 多頭自注意力（Multi-Head Self-Attention）
2. 殘差連接 + 層標準化（Residual Connection + Layer Normalization）
3. 逐位置前饋網路（Position-wise Feedforward Network）
4. 殘差連接 + 層標準化（Residual Connection + Layer Normalization）

架構：
    輸入
     ↓
    [多頭注意力] → Add & Norm
     ↓
    [前饋網路]  → Add & Norm
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
    Transformer 編碼器層（Encoder Layer）

    【這是什麼？】
    這是 Transformer 的核心組件！
    一個完整的編碼器（Encoder）由多個 EncoderLayer 堆疊而成（通常是 6 層）

    【完整架構】
    一個 EncoderLayer 包含兩個子層（sub-layer）：

    子層 1：多頭自注意力（Multi-Head Self-Attention）
        → 讓每個詞「看到」整個句子，蒐集相關資訊

    子層 2：逐位置前饋網路（Position-wise FFN）
        → 對每個詞進行獨立的非線性轉換

    每個子層後面都有：
        - 殘差連接（Residual Connection）：x + SubLayer(x)
        - 層標準化（Layer Normalization）：穩定訓練

    【架構圖】
        輸入 x (batch, seq_len, d_model)
         ↓
        ┌─────────────────────────────┐
        │  多頭自注意力機制            │  ← 子層 1：蒐集資訊
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + Attention(x)，然後標準化
         ↓
        ┌─────────────────────────────┐
        │  前饋網路 (FFN)              │  ← 子層 2：處理資訊
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x)，然後標準化
         ↓
        輸出 (batch, seq_len, d_model)

    【為什麼是這個順序？】
    1. 先做注意力：蒐集資訊
       - 每個詞看整個句子
       - 找出哪些詞是相關的
       - 收集相關資訊

    2. 再做前饋網路：處理資訊
       - 對收集到的資訊做非線性轉換
       - 提取更複雜的特徵
       - 增加模型表達能力

    3. 每一步都有殘差 + 標準化：穩定訓練
       - 殘差：讓梯度可以直接回傳，解決梯度消失
       - 標準化：穩定數值範圍，加快收斂

    【什麼是殘差連接（Residual Connection）？】
    簡單概念：輸出 = 輸入 + 轉換(輸入)

    沒有殘差：
        輸出 = F(x)
        問題：如果 F 很複雜（很多層），梯度會消失

    有殘差：
        輸出 = x + F(x)
        好處：
        ✓ 梯度可以直接通過 x 回傳（捷徑）
        ✓ F 只需要學「修改」，不用學「重建」
        ✓ 訓練更穩定、更快

    比喻：
        沒有殘差：「重寫整篇文章」（困難）
        有殘差：「在現有文章上做修改」（簡單）

    【什麼是層標準化（Layer Normalization）？】
    目的：將每一層的輸出標準化到穩定範圍

    公式：
        輸出 = (x - 平均值) / 標準差 * gamma + beta

    其中：
        平均值、標準差：每個樣本自己算（不跨批次）
        gamma、beta：可學習參數

    為什麼需要？
    - 穩定訓練（防止數值爆炸或消失）
    - 加快收斂
    - 讓每一層的輸入分佈保持穩定

    【完整流程範例】
    假設輸入：「我 愛 吃 蘋果」（4 個詞）
    x.shape = (1, 4, 512)  # batch=1, seq_len=4, d_model=512

    步驟 1：自注意力
        - 每個詞看整個句子
        - 「吃」會注意到「蘋果」（吃什麼？）
        - 「愛」會注意到「我」（誰愛？）
        → attn_output.shape = (1, 4, 512)

    步驟 2：Add & Norm（第一次）
        - x = x + attn_output  (殘差)
        - x = LayerNorm(x)     (標準化)
        → x.shape = (1, 4, 512)

    步驟 3：前饋網路
        - 獨立處理每個詞
        - 512 → 2048 → 512（擴展→轉換→壓縮）
        → ff_output.shape = (1, 4, 512)

    步驟 4：Add & Norm（第二次）
        - x = x + ff_output  (殘差)
        - x = LayerNorm(x)   (標準化)
        → x.shape = (1, 4, 512)

    最終輸出：(1, 4, 512) ← 與輸入形狀相同

    參數說明：
        d_model: 模型維度（例如 512）
        num_heads: 注意力頭數（例如 8）
        d_ff: 前饋網路隱藏層維度（例如 2048，通常是 d_model 的 4 倍）
        dropout: Dropout 比率（預設 0.1）
        activation: 前饋網路的激活函數（'relu' 或 'gelu'）
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

        # ========== 組件 1：多頭自注意力機制 ==========
        # 這是第一個子層，負責「蒐集資訊」
        #
        # Self-Attention（自注意力）的意思是：
        # - Query、Key、Value 都來自同一個輸入（自己看自己）
        # - 句子中的每個詞都能看到整個句子
        # - 找出哪些詞是相關的
        #
        # Multi-Head（多頭）的意思是：
        # - 使用多個注意力頭（num_heads 個）
        # - 每個頭可以學習不同的注意力模式
        # - 例如：第 1 個頭關注主謂關係，第 2 個頭關注修飾關係...
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 組件 2：逐位置前饋網路 ==========
        # 這是第二個子層，負責「處理資訊」
        #
        # 目的：
        # - 對每個位置做獨立的非線性轉換
        # - 提取更複雜的特徵
        # - 增加模型的表達能力
        #
        # 架構：Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== 組件 3 & 4：兩個層標準化層 ==========
        # 為什麼要兩個？
        # - 因為我們有兩個子層（Attention 和 FFN）
        # - 每個子層都需要一個 LayerNorm
        #
        # LayerNorm 的作用：
        # - 標準化：將每個樣本的每個位置標準化為 mean=0, std=1
        # - 穩定訓練：防止數值爆炸或消失
        # - 加快收斂：讓梯度更穩定
        #
        # LayerNorm vs BatchNorm 的差異：
        # - BatchNorm：對同一特徵在批次間標準化（用於 CNN）
        # - LayerNorm：對一個樣本的所有特徵標準化（用於 NLP）
        # - 為什麼 NLP 用 LayerNorm？因為句子長度不同，批次難對齊
        self.norm1 = nn.LayerNorm(d_model)  # 用於第一個子層（Attention）
        self.norm2 = nn.LayerNorm(d_model)  # 用於第二個子層（FFN）

        # ========== 組件 5 & 6：兩個 Dropout 層 ==========
        # 為什麼要兩個？
        # - 因為我們有兩個子層（Attention 和 FFN）
        # - 每個子層的輸出都需要 Dropout（在殘差連接之前）
        #
        # Dropout 在哪裡使用？
        # - 子層輸出之後
        # - 殘差連接之前
        # - 流程：SubLayer(x) → Dropout → x + ·
        #
        # 為什麼要在這裡 Dropout？
        # - 防止過擬合
        # - 讓模型不要過度依賴某些路徑
        self.dropout1 = nn.Dropout(dropout)  # 用於第一個子層（Attention）
        self.dropout2 = nn.Dropout(dropout)  # 用於第二個子層（FFN）

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼器層的前向傳播

        參數：
            x: 輸入序列，形狀為 (batch_size, seq_len, d_model)
               （通常是詞嵌入 + 位置編碼）
            mask: 填充遮罩，形狀為 (batch_size, 1, 1, seq_len) 或 None
                 用於忽略 <PAD> 標記

        回傳：
            output: 編碼器層輸出，形狀為 (batch_size, seq_len, d_model)
                   （與輸入維度相同）

        完整流程：
            1. 自注意力：每個 token 關注整個序列，蒐集相關資訊
            2. Add & Norm：殘差連接 + 層標準化
            3. FFN：獨立處理每個 token，進行非線性轉換
            4. Add & Norm：殘差連接 + 層標準化

        具體範例（「我 愛 吃 蘋果」）：
            輸入 x.shape = (1, 4, 512)  # batch=1, seq_len=4, d_model=512

            子層 1：自注意力
                - 「吃」關注到「蘋果」（吃什麼？）
                - 「愛」關注到「我」（誰愛？）
                - 每個詞蒐集相關資訊
                → attn_output.shape = (1, 4, 512)

            Add & Norm 1:
                - x = x + attn_output  (殘差)
                - x = LayerNorm(x)     (標準化)
                → x.shape = (1, 4, 512)

            子層 2：FFN
                - 獨立處理每個詞
                - 512 → 2048 → 512（擴展→轉換→壓縮）
                → ff_output.shape = (1, 4, 512)

            Add & Norm 2:
                - x = x + ff_output  (殘差)
                - x = LayerNorm(x)   (標準化)
                → x.shape = (1, 4, 512)

            最終輸出：(1, 4, 512)
        """
        # ========== 子層 1：多頭自注意力 ==========

        # 步驟 1：自注意力
        # Q = K = V = x（三個輸入都是 x，所以叫「自」注意力）
        #
        # 這一步在做什麼？
        # - 序列中的每個 token 都能「看到」整個序列
        # - 計算每對 token 之間的注意力分數
        # - 根據注意力分數蒐集相關資訊
        #
        # 具體範例（「我 愛 吃 蘋果」）：
        # - 「吃」會關注到：
        #   * 「蘋果」（高注意力） ← 吃什麼？
        #   * 「我」（低注意力）
        #   * 「愛」（低注意力）
        # - 「愛」會關注到：
        #   * 「我」（高注意力） ← 誰愛？
        #   * 「吃」（中注意力） ← 愛做什麼？
        #   * 「蘋果」（低注意力）
        #
        # mask 的作用：
        # - 如果句子有填充（例如「我 愛 吃 蘋果 <PAD> <PAD>」）
        # - mask 告訴模型：不要關注 <PAD>
        # - 避免模型學習無意義的填充資訊
        attn_output = self.self_attention(x, x, x, mask)
        # attn_output.shape = (batch_size, seq_len, d_model)

        # 步驟 2：Dropout + 殘差連接
        #
        # 先做 Dropout：
        # - 訓練時：隨機將一些值設為 0
        # - 防止過擬合
        attn_output = self.dropout1(attn_output)

        # 然後做殘差連接：
        # x = x + attn_output
        #     ↑        ↑
        #  原始輸入  注意力輸出
        #
        # 為什麼要加上原始輸入 x？
        #
        # 1. 梯度流動：
        #    沒有殘差：梯度要經過很多層，可能會消失
        #    有殘差：梯度可以直接通過 x 回傳（捷徑）
        #    → 訓練更穩定
        #
        # 2. 學習目標不同：
        #    沒有殘差：模型要學會「重建」整個輸出
        #    有殘差：模型只需要學「修改」（delta）
        #    → 學習更容易
        #
        # 3. 保留資訊：
        #    即使 attn_output 學得不好，x 還在
        #    → 不會完全丟失資訊
        #
        # 比喻：
        # - 沒有殘差：「重寫整篇文章」
        # - 有殘差：「在現有文章上做修改」
        x = x + attn_output  # 這就是殘差連接！
        # x.shape = (batch_size, seq_len, d_model) ← 維度不變

        # 步驟 3：層標準化
        #
        # 目的：標準化每個樣本的每個位置
        # 公式：輸出 = (x - 平均值) / 標準差 * gamma + beta
        #
        # 為什麼需要？
        #
        # 1. 穩定數值範圍：
        #    經過多次運算後，數值可能變得很大或很小
        #    → 標準化回 mean≈0, std≈1
        #    → 防止數值爆炸或消失
        #
        # 2. 加快收斂：
        #    每一層的輸入分佈穩定
        #    → 優化器更容易找到好的更新方向
        #    → 訓練更快
        #
        # 3. 層之間獨立：
        #    即使上一層的輸出變化，LayerNorm 會把它調回來
        #    → 每一層可以更獨立地學習
        x = self.norm1(x)
        # x.shape = (batch_size, seq_len, d_model) ← 維度不變，但數值被標準化了

        # ========== 子層 2：逐位置前饋網路 ==========

        # 步驟 4：前饋網路
        #
        # 這一步在做什麼？
        # - 對每個位置做獨立的非線性轉換
        # - 與注意力不同（注意力看整個序列），FFN 只看當前位置
        # - 但所有位置共享同一個 FFN 的權重
        #
        # 架構：
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # 為什麼需要 FFN？
        # - 注意力只是「重新排列」資訊（加權平均）
        # - FFN 提供「非線性轉換」
        # - 讓模型能學習更複雜的特徵
        #
        # 比喻：
        # - 注意力：在圖書館找書（蒐集資訊）
        # - FFN：閱讀和思考（處理資訊）
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, seq_len, d_model)

        # 步驟 5：Dropout + 殘差連接（第二次殘差）
        #
        # 流程與上面類似：
        # 1. Dropout：防止過擬合
        # 2. 殘差：x + FFN(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # 第二次殘差連接
        # x.shape = (batch_size, seq_len, d_model)

        # 步驟 6：層標準化（第二次標準化）
        #
        # 再次標準化，原因同上
        x = self.norm2(x)
        # x.shape = (batch_size, seq_len, d_model)

        # 最終輸出：
        # - 形狀與輸入相同：(batch_size, seq_len, d_model)
        # - 但內容已經被注意力和 FFN 轉換過了
        # - 可以繼續傳給下一個 EncoderLayer，或作為最終輸出
        return x


class Encoder(nn.Module):
    """
    完整的 Transformer 編碼器

    【這是什麼？】
    這是完整的編碼器！
    由多個 EncoderLayer 堆疊而成（原始論文使用 6 層）

    【架構圖】
        輸入 (batch, seq_len, d_model)
         ↓
        ┌─────────────────┐
        │  EncoderLayer 1 │  ← 層 1：學習基本特徵
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 2 │  ← 層 2：學習中階特徵
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 3 │  ← 層 3：學習高階特徵
        └─────────────────┘
         ↓
           ...（更多層）
         ↓
        ┌─────────────────┐
        │  EncoderLayer N │  ← 層 N：學習最抽象的特徵
        └─────────────────┘
         ↓
        層標準化  ← 最終標準化
         ↓
        輸出 (batch, seq_len, d_model)

    【為什麼要堆疊多層？深度代表什麼？】

    比喻 1：閱讀理解的層次
        層 1：理解單字（「貓」、「坐」、「墊子」）
        層 2：理解片語（「坐在墊子上」）
        層 3：理解句子（「貓坐在墊子上」）
        層 N：理解語義（描述一個場景）

    比喻 2：深度卷積神經網路（電腦視覺）
        淺層：學習邊緣、紋理（簡單特徵）
        中層：學習形狀、部件（組合特徵）
        深層：學習物體、場景（抽象概念）

    Transformer 的深度也類似：
        層 1：局部模式
                 - 片語（「紐約」、「蘋果公司」）
                 - 基本文法（主謂、動賓）

        層 2-3：中階模式
                    - 片語結構（「蘋果公司在紐約」）
                    - 語義角色（施事者、受事者）

        層 4-6：高階語義
                    - 句子級別的語義
                    - 抽象關係（因果、比較）

    【每一層的參數是獨立的嗎？】
    是的！每一層都有自己的參數（不共享）

    - 優點：每一層可以學習不同的模式
    - 缺點：參數更多（6 層 ≈ 6 倍參數）

    如果參數共享（像 Universal Transformer）：
    - 優點：參數更少
    - 缺點：表達能力受限（每一層做類似的事）

    【為什麼原始論文用 6 層？】
    - 這是經驗值（來自實驗）
    - 平衡效能和計算成本
    - 更深（12、24 層）可能效果更好，但：
      * 計算成本更高
      * 可能過擬合
      * 需要更多資料

    現代模型的層數：
    - BERT-base：12 層
    - BERT-large：24 層
    - GPT-3：96 層！

    【輸入和輸出】
    輸入：
        - 通常是詞嵌入 + 位置編碼
        - 形狀：(batch_size, seq_len, d_model)

    輸出：
        - 編碼後的表示
        - 形狀：(batch_size, seq_len, d_model) ← 維度不變
        - 但內容已經過多層轉換，蘊含豐富的上下文資訊

    參數說明：
        num_layers: 編碼器層數（原始論文用 6，BERT 用 12）
        d_model: 模型維度（例如 512）
        num_heads: 注意力頭數（例如 8）
        d_ff: 前饋網路隱藏層維度（例如 2048）
        dropout: Dropout 比率（預設 0.1）
        activation: 前饋網路的激活函數（'relu' 或 'gelu'）
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

        # ========== 建立多個編碼器層 ==========
        # 使用 nn.ModuleList 來儲存多個層
        #
        # 為什麼用 nn.ModuleList？
        # - 自動註冊所有子模組（層）的參數
        # - 讓 PyTorch 知道這些層是模型的一部分
        # - 這樣優化器才能找到並更新這些參數
        #
        # 為什麼不用普通的 Python list？
        # - 普通 list：PyTorch 不知道裡面有參數，無法訓練
        # - nn.ModuleList：PyTorch 會自動註冊參數，可以訓練
        #
        # 列表推導式：
        # [EncoderLayer(...) for _ in range(num_layers)]
        # 建立 num_layers 個 EncoderLayer
        # 每個 EncoderLayer 有相同的「結構」但獨立的「參數」（隨機初始化）
        #
        # 例如 num_layers=6：
        # self.layers[0] ← 層 1（參數 A）
        # self.layers[1] ← 層 2（參數 B，與 A 不同）
        # self.layers[2] ← 層 3（參數 C，與 A、B 不同）
        # ...
        # self.layers[5] ← 層 6（參數 F）
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # ========== 最終的層標準化 ==========
        # 為什麼最後還要再加一個 LayerNorm？
        #
        # 1. 穩定最終輸出：
        #    - 經過多層運算後，數值範圍可能不穩定
        #    - 最終的 LayerNorm 確保輸出分佈穩定
        #
        # 2. 方便下游處理：
        #    - 如果要接到 Decoder，Decoder 的輸入會更穩定
        #    - 如果要接到分類器，分類器的輸入會更穩定
        #
        # 3. 實驗效果更好：
        #    - 原始論文和 BERT 都在最後加了 LayerNorm
        #    - 這是經驗性的選擇
        #
        # 注意：
        # - 這個 LayerNorm 的參數是獨立的
        # - 不是任何一個 EncoderLayer 內部的 norm1 或 norm2
        self.norm = nn.LayerNorm(d_model)

        # 儲存層數（方便外部查詢）
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼器的前向傳播

        參數：
            x: 輸入序列，形狀為 (batch_size, seq_len, d_model)
               （通常是詞嵌入 + 位置編碼）
            mask: 填充遮罩，形狀為 (batch_size, 1, 1, seq_len) 或 None
                 （用於忽略 <PAD> 標記）

        回傳：
            output: 編碼器輸出，形狀為 (batch_size, seq_len, d_model)
                   （編碼後的表示）

        完整流程：
            輸入 → 層1 → 層2 → ... → 層N → Norm → 輸出

        具體範例（「我 愛 吃 蘋果」）：
            假設 num_layers = 6, d_model = 512

            輸入：
                x.shape = (1, 4, 512)
                x[0, 0, :] = 「我」的嵌入 + 位置編碼
                x[0, 1, :] = 「愛」的嵌入 + 位置編碼
                x[0, 2, :] = 「吃」的嵌入 + 位置編碼
                x[0, 3, :] = 「蘋果」的嵌入 + 位置編碼

            層 1：
                - 注意力：每個詞開始關注相關的詞
                - FFN：提取基本特徵
                → x.shape = (1, 4, 512)

            層 2：
                - 注意力：基於層 1 學習更複雜的關係
                - FFN：提取中階特徵
                → x.shape = (1, 4, 512)

            層 3-6：
                - 逐步提取更抽象的特徵
                - 最後一層包含最豐富的上下文資訊
                → x.shape = (1, 4, 512)

            最終層標準化：
                - 標準化最終輸出
                → x.shape = (1, 4, 512)

            輸出：
                x[0, 0, :] = 「我」的編碼（包含整個句子的上下文）
                x[0, 1, :] = 「愛」的編碼（包含整個句子的上下文）
                x[0, 2, :] = 「吃」的編碼（包含整個句子的上下文）
                x[0, 3, :] = 「蘋果」的編碼（包含整個句子的上下文）
                每個詞的表示都包含了整個句子的資訊！

        【為什麼每一層的輸出形狀都相同？】
        - 每一層的輸入和輸出維度都是 d_model
        - 這樣可以：
          1. 使用殘差連接（x + SubLayer(x)）
          2. 堆疊任意數量的層
          3. 靈活組合

        【資訊在層之間累積】
        - 層 1 的輸出 → 成為層 2 的輸入
        - 層 2 的輸出 → 成為層 3 的輸入
        - ...
        - 最後一層包含所有前面層的資訊（累積效應）
        """
        # ========== 依序通過每個編碼器層 ==========
        # for 迴圈依序執行：
        # x = layer_1(x, mask)
        # x = layer_2(x, mask)  ← 輸入是 layer_1 的輸出
        # x = layer_3(x, mask)  ← 輸入是 layer_2 的輸出
        # ...
        # x = layer_N(x, mask)  ← 輸入是 layer_{N-1} 的輸出
        #
        # 注意：
        # - 每一層的輸入都是上一層的輸出
        # - x 會不斷被更新（覆蓋）
        # - mask 保持不變（每一層使用相同的 mask）
        for layer in self.layers:
            x = layer(x, mask)
            # x.shape 始終是 (batch_size, seq_len, d_model)

        # ========== 最終的層標準化 ==========
        # 標準化最終輸出
        # 確保輸出分佈穩定
        x = self.norm(x)

        # 最終輸出：
        # - 形狀：(batch_size, seq_len, d_model)
        # - 與輸入形狀相同，但內容已經過多層轉換
        # - 每個 token 的表示都包含了整個序列的上下文資訊
        # - 這個輸出可以：
        #   * 接到解碼器（在 Seq2Seq 任務中）
        #   * 接到分類器（在分類任務中）
        #   * 用於下游任務（如 BERT 的預訓練表示）
        return x


if __name__ == "__main__":
    # 測試代碼
    print("=== 測試編碼器層 ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 建立單個編碼器層
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # 建立測試輸入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入形狀：{x.shape}")

    # 前向傳播
    output = encoder_layer(x)
    print(f"輸出形狀：{output.shape}")

    # 測試帶遮罩的情況
    # 假設前 7 個 token 是真實的，後 3 個是填充
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0  # 最後 3 個位置被遮罩
    output_with_mask = encoder_layer(x, mask)
    print(f"帶遮罩的輸出形狀：{output_with_mask.shape}")

    print("\n=== 測試完整編碼器（6 層）===\n")

    num_layers = 6
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = encoder(x, mask)
    print(f"完整編碼器輸出形狀：{output_full.shape}")

    # 計算參數數量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n完整編碼器（{num_layers} 層）總參數量：{total_params:,}")
    print(f"約 {total_params / 1e6:.1f}M 參數")
