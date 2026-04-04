"""
Transformer Encoder Layer

整合所有元件：
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Position-wise Feedforward Network
4. Residual Connection + Layer Normalization

架構：
    輸入
     ↓
    [Multi-Head Attention] → Add & Norm
     ↓
    [Feedforward Network]  → Add & Norm
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
    Transformer Encoder Layer（編碼器層）

    【這是什麼？】
    這是 Transformer 的核心構建塊！
    一個完整的 Encoder 由多個 EncoderLayer 堆疊而成（通常 6 層）

    【完整架構】
    一個 EncoderLayer 包含兩個子層（sub-layer）：

    子層 1: Multi-Head Self-Attention（多頭自注意力）
        → 讓每個詞「看到」整個句子，收集相關資訊
    子層 2: Position-wise FFN（位置前饋網路）
        → 對每個詞進行獨立的非線性轉換

    每個子層後面都有：
        - Residual Connection（殘差連接）：x + SubLayer(x)
        - Layer Normalization（層歸一化）：穩定訓練

    【架構圖】
        輸入 x (batch, seq_len, d_model)
         ↓
        ┌─────────────────────────────┐
        │  Multi-Head Self-Attention  │  ← 子層 1：收集資訊
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + Attention(x)，然後標準化
         ↓
        ┌─────────────────────────────┐
        │  Feedforward Network (FFN)  │  ← 子層 2：處理資訊
        └─────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x)，然後標準化
         ↓
        輸出 (batch, seq_len, d_model)

    【為什麼這個順序？】
    1. 先 Attention：收集資訊
       - 讓每個詞看到整個句子
       - 找出哪些詞是相關的
       - 把相關的資訊收集起來

    2. 再 FFN：處理資訊
       - 對收集到的資訊進行非線性轉換
       - 提取更複雜的特徵
       - 增加模型的表達能力

    3. 每步都有 Residual + Norm：穩定訓練
       - Residual：讓梯度可以直接流回去，解決梯度消失
       - Norm：穩定數值範圍，加快收斂

    【Residual Connection（殘差連接）是什麼？】
    想法很簡單：輸出 = 輸入 + 變換(輸入)

    沒有殘差連接：
        output = F(x)
        問題：如果 F 很複雜（多層），梯度會消失

    有殘差連接：
        output = x + F(x)
        好處：
        ✓ 梯度可以直接通過 x 流回去（捷徑）
        ✓ F 只需要學習「修改」，不需要學習「重建」
        ✓ 訓練更穩定、更快

    類比：
        沒有殘差：「重新寫一篇文章」（難）
        有殘差：「在原文基礎上修改」（易）

    【Layer Normalization（層歸一化）是什麼？】
    目的：把每一層的輸出標準化到穩定的範圍

    公式：
        output = (x - mean) / std * gamma + beta

    其中：
        mean, std：對每個樣本計算（不是整個 batch）
        gamma, beta：可學習的參數

    為什麼需要？
    - 穩定訓練（防止數值爆炸或消失）
    - 加快收斂速度
    - 讓每層的輸入分佈保持穩定

    【完整流程範例】
    假設輸入："我愛吃蘋果"（5 個字）
    x.shape = (1, 5, 512)  # batch=1, seq_len=5, d_model=512

    步驟 1: Self-Attention
        - 讓每個字看到整個句子
        - "吃" 可以注意到 "蘋果"（它吃什麼？）
        - "愛" 可以注意到 "我"（誰愛？）
        → attn_output.shape = (1, 5, 512)

    步驟 2: Add & Norm（第一次）
        - x = x + attn_output  ← 殘差連接
        - x = LayerNorm(x)     ← 層歸一化
        → x.shape = (1, 5, 512)

    步驟 3: FFN
        - 對每個字獨立處理
        - 512 → 2048 → 512（展開→轉換→壓縮）
        → ff_output.shape = (1, 5, 512)

    步驟 4: Add & Norm（第二次）
        - x = x + ff_output  ← 殘差連接
        - x = LayerNorm(x)   ← 層歸一化
        → x.shape = (1, 5, 512)

    最終輸出：(1, 5, 512) ← 與輸入 shape 相同

    參數：
        d_model: 模型維度（如 512）
        num_heads: 注意力頭數量（如 8）
        d_ff: FFN 隱藏層維度（如 2048，通常是 d_model 的 4 倍）
        dropout: Dropout 比例（預設 0.1）
        activation: FFN 激活函數（'relu' 或 'gelu'）
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

        # ========== 元件 1: Multi-Head Self-Attention ==========
        # 這是第一個子層，負責「收集資訊」
        #
        # Self-Attention 的意思：
        # - Query、Key、Value 都來自同一個輸入（自己注意自己）
        # - 讓句子中的每個詞都能看到整個句子
        # - 找出哪些詞是相關的
        #
        # Multi-Head 的意思：
        # - 使用多個注意力頭（num_heads 個）
        # - 每個頭可以學習不同的注意力模式
        # - 例如：頭1 關注主謂關係，頭2 關注修飾關係...
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 元件 2: Position-wise Feedforward Network ==========
        # 這是第二個子層，負責「處理資訊」
        #
        # 作用：
        # - 對每個位置獨立進行非線性轉換
        # - 提取更複雜的特徵
        # - 增加模型的表達能力
        #
        # 架構：Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== 元件 3 & 4: 兩個 Layer Normalization 層 ==========
        # 為什麼需要兩個？
        # - 因為有兩個子層（Attention 和 FFN）
        # - 每個子層後面都需要一個 LayerNorm
        #
        # LayerNorm 的作用：
        # - 標準化：把每個樣本的每個位置標準化到 mean=0, std=1
        # - 穩定訓練：防止數值爆炸或消失
        # - 加快收斂：讓梯度更穩定
        #
        # LayerNorm vs BatchNorm 的區別：
        # - BatchNorm: 對一個 batch 的同一個特徵標準化（用於 CNN）
        # - LayerNorm: 對一個樣本的所有特徵標準化（用於 NLP）
        # - 為什麼 NLP 用 LayerNorm？因為句子長度不同，batch 不好對齊
        self.norm1 = nn.LayerNorm(d_model)  # 用於第一個子層（Attention）
        self.norm2 = nn.LayerNorm(d_model)  # 用於第二個子層（FFN）

        # ========== 元件 5 & 6: 兩個 Dropout 層 ==========
        # 為什麼需要兩個？
        # - 因為有兩個子層（Attention 和 FFN）
        # - 每個子層的輸出都需要 Dropout（在殘差連接之前）
        #
        # Dropout 放在哪裡？
        # - 在子層的輸出之後
        # - 在殘差連接之前
        # - 流程：SubLayer(x) → Dropout → x + ·
        #
        # 為什麼在這裡 Dropout？
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
        Encoder Layer 的前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入序列（通常是 word embeddings + positional encoding）
            mask: shape (batch_size, 1, 1, seq_len) 或 None
                 Padding mask，用來忽略 <PAD> token

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   Encoder layer 的輸出（維度與輸入相同）

        完整流程：
            1. Self-Attention: 讓每個 token 看到整個序列，收集相關資訊
            2. Add & Norm: 殘差連接 + 層歸一化
            3. FFN: 對每個 token 獨立處理，進行非線性轉換
            4. Add & Norm: 殘差連接 + 層歸一化

        具體範例（"我愛吃蘋果"）：
            輸入 x.shape = (1, 5, 512)  # batch=1, seq_len=5, d_model=512

            子層 1: Self-Attention
                - "吃" 注意到 "蘋果"（吃什麼？）
                - "愛" 注意到 "我"（誰愛？）
                - 每個詞收集相關資訊
                → attn_output.shape = (1, 5, 512)

            Add & Norm 1:
                - x = x + attn_output  （殘差連接）
                - x = LayerNorm(x)     （標準化）
                → x.shape = (1, 5, 512)

            子層 2: FFN
                - 每個詞獨立處理
                - 512 → 2048 → 512（展開→轉換→壓縮）
                → ff_output.shape = (1, 5, 512)

            Add & Norm 2:
                - x = x + ff_output  （殘差連接）
                - x = LayerNorm(x)   （標準化）
                → x.shape = (1, 5, 512)

            最終輸出：(1, 5, 512)
        """
        # ========== 子層 1: Multi-Head Self-Attention ==========

        # Step 1: Self-Attention（自注意力）
        # Q = K = V = x（三個輸入都是 x，所以叫「自」注意力）
        #
        # 這一步做什麼？
        # - 讓序列中的每個 token 都能「看到」整個序列
        # - 計算每個 token 對其他 token 的注意力分數
        # - 根據注意力分數，收集相關資訊
        #
        # 具體例子（"我愛吃蘋果"）：
        # - "吃" 這個字會注意到：
        #   * "蘋果" (高注意力) ← 吃什麼？
        #   * "我" (低注意力)
        #   * "愛" (低注意力)
        # - "愛" 這個字會注意到：
        #   * "我" (高注意力) ← 誰愛？
        #   * "吃" (中等注意力) ← 愛做什麼？
        #   * "蘋果" (低注意力)
        #
        # mask 的作用：
        # - 如果句子有 padding（如 "我愛吃蘋果<PAD><PAD>"）
        # - mask 告訴模型：不要注意 <PAD>
        # - 避免模型學到無意義的 padding 資訊
        attn_output = self.self_attention(x, x, x, mask)
        # attn_output.shape = (batch_size, seq_len, d_model)

        # Step 2: Dropout + Residual Connection（殘差連接）
        #
        # 先 Dropout：
        # - 訓練時：隨機將一些值設為 0
        # - 防止過擬合
        attn_output = self.dropout1(attn_output)

        # 再殘差連接：
        # x = x + attn_output
        #     ↑        ↑
        #   原始輸入  注意力的輸出
        #
        # 為什麼要加上原始輸入 x？
        #
        # 1. 梯度流動（Gradient Flow）：
        #    沒有殘差：梯度要經過很多層，可能消失
        #    有殘差：梯度可以直接通過 x 這條捷徑流回去
        #    → 訓練更穩定
        #
        # 2. 學習目標不同：
        #    沒有殘差：模型要學習「重建」整個輸出
        #    有殘差：模型只需要學習「修改」（增量）
        #    → 學習更容易
        #
        # 3. 保留原始資訊：
        #    即使 attn_output 學得不好，x 還在
        #    → 不會完全丟失資訊
        #
        # 類比：
        # - 沒有殘差：「重新寫一篇文章」
        # - 有殘差：「在原文基礎上修改、補充」
        x = x + attn_output  # 這就是殘差連接！
        # x.shape = (batch_size, seq_len, d_model) ← 維度不變

        # Step 3: Layer Normalization（層歸一化）
        #
        # 作用：把每個樣本的每個位置標準化
        # 公式：output = (x - mean) / std * gamma + beta
        #
        # 為什麼需要？
        #
        # 1. 穩定數值範圍：
        #    經過多層運算，數值可能變很大或很小
        #    → 標準化回 mean≈0, std≈1
        #    → 防止數值爆炸或消失
        #
        # 2. 加快收斂：
        #    每層的輸入分佈穩定
        #    → 優化器更容易找到好的更新方向
        #    → 訓練更快
        #
        # 3. 讓每層獨立：
        #    即使前面的層輸出變化，LayerNorm 會調整回來
        #    → 每層可以更獨立地學習
        x = self.norm1(x)
        # x.shape = (batch_size, seq_len, d_model) ← 維度不變，但值被標準化了

        # ========== 子層 2: Position-wise Feedforward ==========

        # Step 4: Feedforward Network（前饋網路）
        #
        # 這一步做什麼？
        # - 對每個位置獨立進行非線性轉換
        # - 不像 Attention 會看整個序列，FFN 只看當前位置
        # - 但所有位置共享同樣的 FFN 權重
        #
        # 架構：
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # 為什麼需要 FFN？
        # - Attention 只是「重新組合」資訊（加權平均）
        # - FFN 提供「非線性轉換」
        # - 讓模型可以學習更複雜的特徵
        #
        # 類比：
        # - Attention：在圖書館找書（收集資訊）
        # - FFN：讀書、思考（處理資訊）
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, seq_len, d_model)

        # Step 5: Dropout + Residual Connection（第二個殘差連接）
        #
        # 流程與前面類似：
        # 1. Dropout：防止過擬合
        # 2. 殘差連接：x + FFN(x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output  # 第二個殘差連接
        # x.shape = (batch_size, seq_len, d_model)

        # Step 6: Layer Normalization（第二個層歸一化）
        #
        # 再次標準化，原因同前
        x = self.norm2(x)
        # x.shape = (batch_size, seq_len, d_model)

        # 最終輸出：
        # - shape 與輸入完全相同：(batch_size, seq_len, d_model)
        # - 但內容已經被 Attention 和 FFN 轉換過了
        # - 可以繼續接下一個 EncoderLayer，或作為最終輸出
        return x


class Encoder(nn.Module):
    """
    完整的 Transformer Encoder（編碼器）

    【這是什麼？】
    這是完整的 Encoder！
    由多個 EncoderLayer 堆疊而成（原論文用 6 層）

    【架構圖】
        輸入 (batch, seq_len, d_model)
         ↓
        ┌─────────────────┐
        │  EncoderLayer 1 │  ← 第 1 層：學習基礎特徵
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 2 │  ← 第 2 層：學習中層特徵
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  EncoderLayer 3 │  ← 第 3 層：學習高層特徵
        └─────────────────┘
         ↓
           ... (更多層)
         ↓
        ┌─────────────────┐
        │  EncoderLayer N │  ← 第 N 層：學習最抽象的特徵
        └─────────────────┘
         ↓
        Layer Normalization  ← 最後的標準化
         ↓
        輸出 (batch, seq_len, d_model)

    【為什麼要堆疊多層？深度的意義是什麼？】

    類比 1: 閱讀理解的層次
        第 1 層：理解單詞（"貓"、"坐"、"墊子"）
        第 2 層：理解短語（"坐在墊子上"）
        第 3 層：理解句子（"貓坐在墊子上"）
        第 N 層：理解語義（這是在描述一個場景）

    類比 2: 深度 CNN（電腦視覺）
        淺層：學習邊緣、紋理（簡單特徵）
        中層：學習形狀、部件（組合特徵）
        深層：學習物體、場景（抽象概念）

    Transformer 的深度也類似：
        第 1 層：可能學習局部模式
                 - 詞組（"紐約"、"蘋果公司"）
                 - 基本語法關係（主謂、動賓）

        第 2-3 層：可能學習中層模式
                   - 短語結構（"在紐約的蘋果公司"）
                   - 語義角色（施事者、受事者）

        第 4-6 層：可能學習高層語義
                   - 句子級語義
                   - 抽象關係（因果、比較）
                   - 上下文依賴

    【每層的參數是獨立的嗎？】
    是的！每層都有自己的參數（不共享）

    - 好處：每層可以學習不同的模式
    - 缺點：參數量多（6 層 ≈ 6 倍參數）

    如果參數共享（像 Universal Transformer）：
    - 好處：參數少
    - 缺點：表達能力受限（每層做類似的事）

    【為什麼原論文用 6 層？】
    - 這是一個經驗值（實驗出來的）
    - 6 層在效果和計算成本之間取得平衡
    - 更深（12 層、24 層）可能效果更好，但：
      * 計算成本更高
      * 可能過擬合
      * 需要更多數據

    現代模型的層數：
    - BERT-base: 12 層
    - BERT-large: 24 層
    - GPT-3: 96 層！

    【輸入和輸出】
    輸入：
        - 通常是 word embeddings + positional encoding
        - shape: (batch_size, seq_len, d_model)

    輸出：
        - 編碼後的表示
        - shape: (batch_size, seq_len, d_model) ← 維度不變
        - 但內容已經過多層轉換，包含豐富的上下文資訊

    參數：
        num_layers: Encoder layer 的數量（原論文用 6，BERT 用 12）
        d_model: 模型維度（如 512）
        num_heads: 注意力頭數量（如 8）
        d_ff: FFN 隱藏層維度（如 2048）
        dropout: Dropout 比例（預設 0.1）
        activation: FFN 激活函數（'relu' 或 'gelu'）
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

        # ========== 建立多個 EncoderLayer ==========
        # 使用 nn.ModuleList 來儲存多個層
        #
        # 為什麼用 nn.ModuleList？
        # - 自動註冊所有子模組（子層）的參數
        # - 讓 PyTorch 知道這些層是模型的一部分
        # - 這樣 optimizer 才能找到並更新這些參數
        #
        # 為什麼不用普通的 Python list？
        # - 普通 list：PyTorch 不知道裡面有參數，無法訓練
        # - nn.ModuleList：PyTorch 會自動註冊參數，可以訓練
        #
        # 列表推導式（List Comprehension）：
        # [EncoderLayer(...) for _ in range(num_layers)]
        # 建立 num_layers 個 EncoderLayer
        # 每個 EncoderLayer 的「結構」相同，但「參數」獨立（隨機初始化）
        #
        # 例如 num_layers=6：
        # self.layers[0] ← 第 1 層（參數 A）
        # self.layers[1] ← 第 2 層（參數 B，與 A 不同）
        # self.layers[2] ← 第 3 層（參數 C，與 A、B 不同）
        # ...
        # self.layers[5] ← 第 6 層（參數 F）
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])

        # ========== 最後的 Layer Normalization ==========
        # 為什麼最後還要一個 LayerNorm？
        #
        # 1. 穩定最終輸出：
        #    - 經過多層運算後，數值範圍可能不穩定
        #    - 最後的 LayerNorm 確保輸出分佈穩定
        #
        # 2. 方便後續處理：
        #    - 如果接 Decoder，Decoder 的輸入會是穩定的
        #    - 如果接分類器，分類器的輸入會是穩定的
        #
        # 3. 實驗上效果更好：
        #    - 原論文和 BERT 都在最後加了一個 LayerNorm
        #    - 這是一個經驗上的選擇
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
        Encoder 的前向傳播

        參數：
            x: shape (batch_size, seq_len, d_model)
               輸入序列（通常是 word embeddings + positional encoding）
            mask: shape (batch_size, 1, 1, seq_len) 或 None
                 Padding mask（用來忽略 <PAD> token）

        回傳：
            output: shape (batch_size, seq_len, d_model)
                   Encoder 的輸出（編碼後的表示）

        完整流程：
            輸入 → Layer1 → Layer2 → ... → LayerN → Norm → 輸出

        具體範例（"我愛吃蘋果"）：
            假設 num_layers = 6, d_model = 512

            輸入：
                x.shape = (1, 5, 512)
                x[0, 0, :] = "我" 的 embedding + positional encoding
                x[0, 1, :] = "愛" 的 embedding + positional encoding
                x[0, 2, :] = "吃" 的 embedding + positional encoding
                x[0, 3, :] = "蘋" 的 embedding + positional encoding
                x[0, 4, :] = "果" 的 embedding + positional encoding

            Layer 1:
                - Attention: 每個字開始注意到相關的字
                - FFN: 提取基礎特徵
                → x.shape = (1, 5, 512)

            Layer 2:
                - Attention: 在 Layer 1 的基礎上，學習更複雜的關係
                - FFN: 提取中層特徵
                → x.shape = (1, 5, 512)

            Layer 3-6:
                - 逐層提取更抽象的特徵
                - 最後一層包含最豐富的上下文資訊
                → x.shape = (1, 5, 512)

            最後的 LayerNorm:
                - 標準化最終輸出
                → x.shape = (1, 5, 512)

            輸出：
                x[0, 0, :] = "我" 的編碼（包含整個句子的上下文）
                x[0, 1, :] = "愛" 的編碼（包含整個句子的上下文）
                ...
                每個字的表示都包含了整個句子的資訊！

        【為什麼每層的輸出 shape 都一樣？】
        - 每層的輸入和輸出維度都是 d_model
        - 這樣才能：
          1. 使用殘差連接（x + SubLayer(x)）
          2. 堆疊任意多層
          3. 靈活組合

        【每層學到的資訊是累積的】
        - Layer 1 的輸出 → 作為 Layer 2 的輸入
        - Layer 2 的輸出 → 作為 Layer 3 的輸入
        - ...
        - 最後一層包含所有前面層的資訊（累積效果）
        """
        # ========== 依序通過每個 Encoder Layer ==========
        # for loop 會依序執行：
        # x = layer_1(x, mask)
        # x = layer_2(x, mask)  ← 輸入是 layer_1 的輸出
        # x = layer_3(x, mask)  ← 輸入是 layer_2 的輸出
        # ...
        # x = layer_N(x, mask)  ← 輸入是 layer_{N-1} 的輸出
        #
        # 注意：
        # - 每層的輸入是上一層的輸出
        # - x 會被不斷更新（覆蓋）
        # - mask 保持不變（每層都用同樣的 mask）
        for layer in self.layers:
            x = layer(x, mask)
            # x.shape 始終是 (batch_size, seq_len, d_model)

        # ========== 最後的 Layer Normalization ==========
        # 標準化最終輸出
        # 確保輸出分佈穩定
        x = self.norm(x)

        # 最終輸出：
        # - shape: (batch_size, seq_len, d_model)
        # - 與輸入 shape 相同，但內容已經過多層轉換
        # - 每個 token 的表示包含了整個序列的上下文資訊
        # - 這個輸出可以：
        #   * 接 Decoder（在 Seq2Seq 任務中）
        #   * 接分類器（在分類任務中）
        #   * 用於下游任務（如 BERT 的預訓練表示）
        return x


if __name__ == "__main__":
    # 測試程式碼
    print("=== 測試 Encoder Layer ===\n")

    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 建立單個 Encoder Layer
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

    # 建立假輸入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"輸入 shape: {x.shape}")

    # 前向傳播
    output = encoder_layer(x)
    print(f"輸出 shape: {output.shape}")

    # 測試帶 mask 的情況
    # 假設序列中前 7 個是真實 token，後 3 個是 padding
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 7:] = 0  # 後 3 個位置被遮罩
    output_with_mask = encoder_layer(x, mask)
    print(f"帶 mask 的輸出 shape: {output_with_mask.shape}")

    print("\n=== 測試完整 Encoder（6 層）===\n")

    num_layers = 6
    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = encoder(x, mask)
    print(f"完整 Encoder 輸出 shape: {output_full.shape}")

    # 計算參數量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n完整 Encoder（{num_layers} 層）總參數量: {total_params:,}")
    print(f"約 {total_params / 1e6:.1f}M 參數")
