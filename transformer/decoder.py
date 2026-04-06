"""
Transformer 解碼器層（Decoder Layer）

整合所有組件：
1. 遮罩多頭自注意力（Masked Multi-Head Self-Attention）
2. 殘差連接 + 層標準化（Residual Connection + Layer Normalization）
3. 交叉注意力至編碼器輸出（Cross-Attention to Encoder Output）
4. 殘差連接 + 層標準化（Residual Connection + Layer Normalization）
5. 逐位置前饋網路（Position-wise Feedforward Network）
6. 殘差連接 + 層標準化（Residual Connection + Layer Normalization）

架構：
    輸入
     ↓
    [遮罩自注意力] → Add & Norm
     ↓
    [編碼器交叉注意力] → Add & Norm
     ↓
    [前饋網路] → Add & Norm
     ↓
    輸出
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Transformer 解碼器層（Decoder Layer）

    【這是什麼？】
    這是 Transformer 解碼器的核心組成單元！
    一個完整的解碼器（Decoder）由多個 DecoderLayer 堆疊而成（通常是 6 層）

    【與編碼器的關鍵差異】
    編碼器有 2 個子層，解碼器有 3 個子層：

    編碼器：
        1. 自注意力（看整個輸入句子）
        2. 前饋網路

    解碼器：
        1. 遮罩自注意力（只能看已生成的部分，不能看未來）
        2. 交叉注意力（看編碼器的輸出，獲取來源語言資訊）
        3. 前饋網路

    【完整架構】
        輸入 x (batch, tgt_len, d_model)
         ↓
        ┌─────────────────────────────────────┐
        │  遮罩多頭自注意力                    │  ← 子層 1：看已生成的 token
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + MaskedAttention(x)，然後標準化
         ↓
        ┌─────────────────────────────────────┐
        │  多頭交叉注意力                      │  ← 子層 2：從編碼器獲取資訊
        │  (Q 來自解碼器, K,V 來自編碼器)     │
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + CrossAttention(x, enc)，然後標準化
         ↓
        ┌─────────────────────────────────────┐
        │  前饋網路 (FFN)                      │  ← 子層 3：處理資訊
        └─────────────────────────────────────┘
         ↓
        Add & Norm  ← x + FFN(x)，然後標準化
         ↓
        輸出 (batch, tgt_len, d_model)

    【什麼是遮罩自注意力（Masked Self-Attention）？】
    問題：在語言生成中，我們一次生成一個詞
    - 當生成第 3 個詞時，我們只看過第 0、1、2 個詞
    - 我們「不能」看到第 4、5、6 個詞...（它們還不存在！）

    解決方案：使用因果遮罩（causal mask，也叫前瞻遮罩 look-ahead mask）
    - 處理位置 i 時，只能關注位置 ≤ i 的 token
    - 防止「作弊」去看未來的 token

    具體範例：「I love eating apples」
        位置 0 ("I"):      能看到：["I"]
        位置 1 ("love"):   能看到：["I", "love"]
        位置 2 ("eating"): 能看到：["I", "love", "eating"]
        位置 3 ("apples"): 能看到：["I", "love", "eating", "apples"]

    遮罩矩陣（1 = 能看到，0 = 不能看到）：
        [[1, 0, 0, 0],  ← 位置 0 只能看到位置 0
         [1, 1, 0, 0],  ← 位置 1 能看到位置 0-1
         [1, 1, 1, 0],  ← 位置 2 能看到位置 0-2
         [1, 1, 1, 1]]  ← 位置 3 能看到位置 0-3

    【什麼是交叉注意力（Cross-Attention）？】
    目的：讓解碼器「看到」編碼器的輸出
    - 在翻譯中：解碼器看來源句子（英文）
                來生成目標句子（法文）

    機制：與自注意力不同！
    - 自注意力：Q、K、V 全都來自同一個輸入（解碼器）
    - 交叉注意力：Q 來自解碼器，K 和 V 來自編碼器

    比喻：
        自注意力：「我到目前為止說了什麼？」
        交叉注意力：「原始句子說了什麼？」

    具體範例（英文→法文翻譯）：
        編碼器輸入：「I love eating apples」
        解碼器正在生成：「J'aime manger ...」

        當生成「manger」（eating）時：
        - 遮罩自注意力：看 ["J'aime"]
        - 交叉注意力：看 ["I", "love", "eating", "apples"]
                      發現「eating」最相關！
        → 幫助生成正確的詞「manger」

    【為什麼是這個順序？】
    1. 先做遮罩自注意力：理解我們已經生成的內容
       - 看到目前為止生成的目標序列
       - 從先前生成的詞建立上下文

    2. 接著做交叉注意力：從來源獲取資訊
       - 看編碼器輸出（來源句子）
       - 找出現在哪些來源詞是相關的

    3. 最後做 FFN：處理組合後的資訊
       - 結合「我們已生成的內容」和「來源句子」的資訊
       - 非線性轉換
       - 為下一層或最終預測做準備

    【完整流程範例】
    假設：英文→法文翻譯
    來源：「I love apples」（已由編碼器編碼）
    目前目標：「J'aime」（法文的「I love」）
    正在生成：下一個詞（應該是「les」或其他）

    輸入：
        x = 嵌入 ["J'", "aime", "<CURRENT>"]
        x.shape = (1, 3, 512)  # batch=1, tgt_len=3, d_model=512
        encoder_output = 已編碼的 "I love apples"
        encoder_output.shape = (1, 3, 512)  # batch=1, src_len=3, d_model=512

    步驟 1：遮罩自注意力
        - "J'" 看到：["J'"]
        - "aime" 看到：["J'", "aime"]
        - "<CURRENT>" 看到：["J'", "aime", "<CURRENT>"]
        → 每個 token 知道它之前的內容
        → masked_attn_output.shape = (1, 3, 512)

    步驟 2：Add & Norm（第一次）
        - x = x + masked_attn_output（殘差）
        - x = LayerNorm(x)（標準化）
        → x.shape = (1, 3, 512)

    步驟 3：交叉注意力
        - Q 來自解碼器：「我在來源中找什麼？」
        - K、V 來自編碼器：「這是來源句子」
        - "<CURRENT>" 可能高度關注「apples」
        → cross_attn_output.shape = (1, 3, 512)

    步驟 4：Add & Norm（第二次）
        - x = x + cross_attn_output（殘差）
        - x = LayerNorm(x)（標準化）
        → x.shape = (1, 3, 512)

    步驟 5：FFN
        - 獨立處理每個位置
        - 512 → 2048 → 512（擴展→轉換→壓縮）
        → ff_output.shape = (1, 3, 512)

    步驟 6：Add & Norm（第三次）
        - x = x + ff_output（殘差）
        - x = LayerNorm(x)（標準化）
        → x.shape = (1, 3, 512)

    最終輸出：(1, 3, 512) ← 與輸入形狀相同

    參數說明：
        d_model: 模型維度（例如 512）
        num_heads: 注意力頭數（例如 8）
        d_ff: FFN 隱藏層維度（例如 2048，通常是 d_model 的 4 倍）
        dropout: Dropout 比率（預設 0.1）
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

        # ========== 組件 1：遮罩多頭自注意力 ==========
        # 這是第一個子層，負責「看已生成的內容」
        #
        # 自注意力的意思是：
        # - Query、Key、Value 都來自解碼器輸入（自己看自己）
        # - 每個已生成的詞都能看到先前生成的詞
        # - 「不能」看到未來的詞（遮罩）
        #
        # 為什麼要遮罩？
        # - 在推論時，未來的詞還不存在！
        # - 在訓練時，我們用遮罩來模擬這個條件
        # - 這是自回歸生成（一次生成一個詞）
        self.self_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 組件 2：多頭交叉注意力 ==========
        # 這是第二個子層，負責「看來源語言資訊」
        #
        # 交叉注意力的意思是：
        # - Query 來自解碼器（我在找什麼？）
        # - Key 和 Value 來自編碼器（這是來源資訊）
        # - 與自注意力不同，自注意力的 Q、K、V 都來自同一個來源
        #
        # 為什麼需要？
        # - 解碼器需要知道來源句子說了什麼！
        # - 範例：翻譯「eating」→ 需要回頭看英文的「eating」
        # - 這是解碼器「關注」輸入序列的方式
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        # ========== 組件 3：逐位置前饋網路 ==========
        # 這是第三個子層，負責「處理資訊」
        #
        # 與編碼器的 FFN 相同：
        # - 對每個位置做獨立的非線性轉換
        # - 提取更複雜的特徵
        # - 增加模型表達能力
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout, activation
        )

        # ========== 組件 4、5、6：三個層標準化層 ==========
        # 為什麼要三個？
        # - 因為我們有三個子層（遮罩自注意力、交叉注意力、FFN）
        # - 每個子層都需要一個 LayerNorm
        #
        # 目的：與編碼器相同
        # - 標準化：將每個樣本標準化為 mean=0, std=1
        # - 穩定訓練：防止數值爆炸或消失
        # - 加快收斂：讓梯度更穩定
        self.norm1 = nn.LayerNorm(d_model)  # 用於第一個子層（遮罩自注意力）
        self.norm2 = nn.LayerNorm(d_model)  # 用於第二個子層（交叉注意力）
        self.norm3 = nn.LayerNorm(d_model)  # 用於第三個子層（FFN）

        # ========== 組件 7、8、9：三個 Dropout 層 ==========
        # 為什麼要三個？
        # - 因為我們有三個子層
        # - 每個子層的輸出都需要 Dropout（在殘差連接之前）
        #
        # 目的：與編碼器相同
        # - 防止過擬合
        # - 讓模型不要過度依賴某些路徑
        self.dropout1 = nn.Dropout(dropout)  # 用於遮罩自注意力
        self.dropout2 = nn.Dropout(dropout)  # 用於交叉注意力
        self.dropout3 = nn.Dropout(dropout)  # 用於 FFN

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解碼器層的前向傳播

        參數：
            x: 目標序列輸入，形狀為 (batch_size, tgt_len, d_model)
               （通常是目標的詞嵌入 + 位置編碼）

            encoder_output: 編碼器的輸出，形狀為 (batch_size, src_len, d_model)
                           （已編碼的來源序列，例如英文句子）

            tgt_mask: 目標遮罩，形狀為 (batch_size, 1, tgt_len, tgt_len)
                     結合：
                     1. 填充遮罩：忽略目標中的 <PAD> token
                     2. 因果遮罩：防止看到未來的 token
                     用於遮罩自注意力

            src_mask: 來源填充遮罩，形狀為 (batch_size, 1, 1, src_len)
                     忽略來源序列中的 <PAD> token
                     用於交叉注意力

        回傳：
            output: 解碼器層輸出，形狀為 (batch_size, tgt_len, d_model)
                   （與輸入維度相同）

        完整流程：
            1. 遮罩自注意力：每個目標 token 只關注先前的 token
            2. Add & Norm：殘差連接 + 層標準化
            3. 交叉注意力：目標關注來源（編碼器輸出）
            4. Add & Norm：殘差連接 + 層標準化
            5. FFN：獨立處理每個 token，進行非線性轉換
            6. Add & Norm：殘差連接 + 層標準化

        具體範例（英文→法文翻譯）：
            來源（英文）：「I love apples」
            目標（法文）：「J'aime les pommes」

            encoder_output = 已編碼的("I love apples")
            encoder_output.shape = (1, 3, 512)

            訓練時，目標輸入 = "J'aime les pommes"
            x.shape = (1, 4, 512)  # batch=1, tgt_len=4, d_model=512

            子層 1：遮罩自注意力
                - "J'" 看到：只有 ["J'"]
                - "aime" 看到：["J'", "aime"]
                - "les" 看到：["J'", "aime", "les"]
                - "pommes" 看到：["J'", "aime", "les", "pommes"]
                → 每個詞都知道它之前的上下文
                → masked_attn_output.shape = (1, 4, 512)

            Add & Norm 1:
                - x = x + masked_attn_output（殘差）
                - x = LayerNorm(x)（標準化）
                → x.shape = (1, 4, 512)

            子層 2：交叉注意力
                - Q 來自解碼器：["J'", "aime", "les", "pommes"]
                - K、V 來自編碼器：["I", "love", "apples"]
                - "pommes" 高度關注 "apples"
                - "aime" 高度關注 "love"
                → 獲取來源資訊
                → cross_attn_output.shape = (1, 4, 512)

            Add & Norm 2:
                - x = x + cross_attn_output（殘差）
                - x = LayerNorm(x)（標準化）
                → x.shape = (1, 4, 512)

            子層 3：FFN
                - 獨立處理每個詞
                - 512 → 2048 → 512（擴展→轉換→壓縮）
                → ff_output.shape = (1, 4, 512)

            Add & Norm 3:
                - x = x + ff_output（殘差）
                - x = LayerNorm(x)（標準化）
                → x.shape = (1, 4, 512)

            最終輸出：(1, 4, 512)
        """
        # ========== 子層 1：遮罩多頭自注意力 ==========

        # 步驟 1：遮罩自注意力
        # Q = K = V = x（三個輸入都是解碼器輸入，因此叫「自」注意力）
        # 但帶有 tgt_mask 來防止看到未來的位置
        #
        # 這一步在做什麼？
        # - 每個目標 token 關注自己和先前的 token
        # - 「不能」關注未來的 token（它們在生成時還不存在！）
        # - 從到目前為止已生成的內容建立上下文
        #
        # 具體範例（"J'aime les pommes"）：
        # - "J'"（位置 0）關注：
        #   * "J'"（位置 0）✓（能看到）
        #   * "aime"（位置 1）✗（未來，遮罩）
        #   * "les"（位置 2）✗（未來，遮罩）
        #   * "pommes"（位置 3）✗（未來，遮罩）
        #
        # - "les"（位置 2）關注：
        #   * "J'"（位置 0）✓（過去，能看到）
        #   * "aime"（位置 1）✓（過去，能看到）
        #   * "les"（位置 2）✓（當前，能看到）
        #   * "pommes"（位置 3）✗（未來，遮罩）
        #
        # tgt_mask 的目的：
        # 1. 因果遮罩：防止看到未來（下三角矩陣）
        # 2. 填充遮罩：如果目標有填充，忽略 <PAD> token
        masked_attn_output = self.self_attention(x, x, x, tgt_mask)
        # masked_attn_output.shape = (batch_size, tgt_len, d_model)

        # 步驟 2：Dropout + 殘差連接
        #
        # Dropout：
        # - 訓練時：隨機將一些值設為 0
        # - 防止過擬合
        masked_attn_output = self.dropout1(masked_attn_output)

        # 殘差連接：
        # x = x + MaskedAttention(x)
        #     ↑            ↑
        #  原始輸入    遮罩注意力輸出
        #
        # 為什麼要加上原始輸入 x？
        # 1. 梯度流動：梯度可以直接通過 x 回傳（捷徑）
        # 2. 學習更容易：模型只需要學「修改」（delta）
        # 3. 保留資訊：即使注意力學得不好，x 還在
        x = x + masked_attn_output  # 第一次殘差連接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步驟 3：層標準化
        #
        # 標準化每個樣本的每個位置
        # 公式：輸出 = (x - 平均值) / 標準差 * gamma + beta
        #
        # 為什麼需要？
        # 1. 穩定數值範圍：防止數值爆炸/消失
        # 2. 加快收斂：穩定的輸入分佈
        # 3. 層獨立性：每一層可以更獨立地學習
        x = self.norm1(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== 子層 2：多頭交叉注意力 ==========

        # 步驟 4：交叉注意力
        #
        # 這與自注意力「不同」！
        # - Query (Q)：來自解碼器 (x) -「我在找什麼？」
        # - Key (K)：來自編碼器 (encoder_output) -「來源中有什麼可用的？」
        # - Value (V)：來自編碼器 (encoder_output) -「實際的來源內容」
        #
        # 這一步在做什麼？
        # - 解碼器問：「來源句子的哪個部分現在是相關的？」
        # - 編碼器提供：來源句子的資訊
        # - 注意力機制找到匹配
        #
        # 具體範例（英文→法文）：
        # 來源（encoder_output）：「I love apples」
        # 目標（x）：「J'aime ...」
        #
        # 當解碼器處理「aime」（法文的「love」）時：
        # - Q 來自「aime」：「我是法文的『love』，我的英文來源是什麼？」
        # - K 來自編碼器：["I", "love", "apples"]
        # - 注意力權重：[0.1, 0.8, 0.1]  ←「aime」高度關注「love」！
        # - V 根據注意力加權 → 主要從「love」獲取資訊
        #
        # 這就是解碼器「知道」要翻譯哪個來源詞的方式！
        #
        # 參數：
        # - query=x：來自解碼器（目標）
        # - key=encoder_output：來自編碼器（來源）
        # - value=encoder_output：來自編碼器（來源）
        # - mask=src_mask：忽略來源中的 <PAD>
        cross_attn_output = self.cross_attention(
            query=x,                    # Q 來自解碼器
            key=encoder_output,         # K 來自編碼器
            value=encoder_output,       # V 來自編碼器
            mask=src_mask               # 忽略來源填充
        )
        # cross_attn_output.shape = (batch_size, tgt_len, d_model)
        # 注意：輸出長度 = query 長度（tgt_len），而不是 key 長度！

        # 步驟 5：Dropout + 殘差連接（第二次殘差）
        #
        # 與之前相同的模式：
        # 1. Dropout：防止過擬合
        # 2. 殘差：x + CrossAttention(x, encoder)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = x + cross_attn_output  # 第二次殘差連接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步驟 6：層標準化（第二次標準化）
        #
        # 再次標準化，原因同上
        x = self.norm2(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # ========== 子層 3：逐位置前饋網路 ==========

        # 步驟 7：前饋網路
        #
        # 與編碼器的 FFN 相同：
        # - 對每個位置做獨立的非線性轉換
        # - 與能看到其他位置的注意力不同，FFN 只看當前位置
        # - 但所有位置共享同一個 FFN 的權重
        #
        # 架構：
        # Linear(512 → 2048) → ReLU/GELU → Dropout → Linear(2048 → 512)
        #
        # 為什麼需要？
        # - 注意力只是「重新排列」資訊（加權平均）
        # - FFN 提供「非線性轉換」
        # - 讓模型能學習更複雜的特徵
        ff_output = self.feed_forward(x)
        # ff_output.shape = (batch_size, tgt_len, d_model)

        # 步驟 8：Dropout + 殘差連接（第三次殘差）
        #
        # 流程與上面類似：
        # 1. Dropout：防止過擬合
        # 2. 殘差：x + FFN(x)
        ff_output = self.dropout3(ff_output)
        x = x + ff_output  # 第三次殘差連接
        # x.shape = (batch_size, tgt_len, d_model)

        # 步驟 9：層標準化（第三次標準化）
        #
        # 最終標準化
        x = self.norm3(x)
        # x.shape = (batch_size, tgt_len, d_model)

        # 最終輸出：
        # - 形狀與輸入相同：(batch_size, tgt_len, d_model)
        # - 但內容已經被以下轉換過：
        #   * 遮罩自注意力（來自先前詞的上下文）
        #   * 交叉注意力（來自來源的資訊）
        #   * FFN（非線性處理）
        # - 可以繼續傳給下一個 DecoderLayer，或輸出最終預測
        return x


class Decoder(nn.Module):
    """
    完整的 Transformer 解碼器

    【這是什麼？】
    這是完整的解碼器！
    由多個 DecoderLayer 堆疊而成（原始論文使用 6 層）

    【架構圖】
        輸入 (batch, tgt_len, d_model)
         ↓
        ┌─────────────────┐
        │  DecoderLayer 1 │  ← 層 1：學習基本模式
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 2 │  ← 層 2：學習中階模式
        └─────────────────┘
         ↓
        ┌─────────────────┐
        │  DecoderLayer 3 │  ← 層 3：學習高階模式
        └─────────────────┘
         ↓
           ...（更多層）
         ↓
        ┌─────────────────┐
        │  DecoderLayer N │  ← 層 N：學習最抽象的模式
        └─────────────────┘
         ↓
        層標準化  ← 最終標準化
         ↓
        輸出 (batch, tgt_len, d_model)

    【解碼器 vs 編碼器：有什麼差異？】

    編碼器：
    - 目的：理解來源句子
    - 輸入：來源 token（例如英文）
    - 注意力：雙向的（可以看到整個句子）
    - 輸出：來源的豐富表示
    - 用於：為翻譯、分類等編碼輸入

    解碼器：
    - 目的：生成目標句子
    - 輸入：目標 token（例如法文）
    - 注意力：單向的（只能看到先前的 token）
    - 輸出：用於下一個 token 預測的表示
    - 用於：自回歸生成（一次一個 token）

    【為什麼要堆疊多層？】
    與編碼器的理由相同：

    層 1：局部模式
             - 簡單的詞關係
             - 基本文法

    層 2-3：中階模式
                - 片語結構
                - 語義角色

    層 4-6：高階語義
                - 句子級別的意義
                - 來源和目標之間的複雜依賴關係

    每一層都建立在前一層之上，學習越來越抽象的特徵。

    【解碼器如何使用編碼器輸出】
    每個 DecoderLayer 都透過交叉注意力接收 encoder_output：
    - 編碼器執行一次：編碼整個來源句子
    - 解碼器執行多次：生成時一次一個 token
    - 每個解碼器層都看編碼器輸出來獲取來源資訊

    這樣想：
    - 編碼器：「這是英文句子的意思」（執行一次）
    - 解碼器：「讓我在生成法文時檢查英文」
              （在每個生成步驟都檢查編碼器輸出）

    【訓練 vs 推論】

    訓練（教師強制 teacher forcing）：
    - 輸入：整個目標句子「J'aime les pommes」
    - 使用因果遮罩來模擬生成
    - 所有位置並行處理（高效！）
    - 一次計算所有位置的損失

    推論（自回歸生成）：
    - 從「<START>」開始
    - 生成：「J'」→「J' aime」→「J' aime les」→「J' aime les pommes」
    - 一次一個 token（較慢）
    - 使用先前的輸出作為下一個輸入

    參數說明：
        num_layers: 解碼器層數（原始論文用 6）
        d_model: 模型維度（例如 512）
        num_heads: 注意力頭數（例如 8）
        d_ff: FFN 隱藏層維度（例如 2048）
        dropout: Dropout 比率（預設 0.1）
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

        # ========== 建立多個解碼器層 ==========
        # 使用 nn.ModuleList 來儲存多個層
        #
        # 為什麼用 nn.ModuleList？
        # - 自動註冊所有子模組（層）的參數
        # - 讓 PyTorch 知道這些層是模型的一部分
        # - 這樣優化器才能找到並更新這些參數
        #
        # 為什麼不用普通的 Python list？
        # - 普通 list：PyTorch 不知道裡面有參數
        # - nn.ModuleList：PyTorch 會自動註冊參數
        #
        # 列表推導式：
        # [DecoderLayer(...) for _ in range(num_layers)]
        # 建立 num_layers 個 DecoderLayer
        # 每個 DecoderLayer 有相同的結構但獨立的參數
        #
        # 例如 num_layers=6：
        # self.layers[0] ← 層 1（參數 A）
        # self.layers[1] ← 層 2（參數 B，與 A 不同）
        # self.layers[2] ← 層 3（參數 C，與 A、B 不同）
        # ...
        # self.layers[5] ← 層 6（參數 F）
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation)
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
        #    - 如果要接到輸出投影（例如詞彙預測）
        #    - 穩定的輸入幫助線性層學習得更好
        #
        # 3. 實驗效果更好：
        #    - 原始論文使用這個
        #    - Transformer 模型的標準做法
        #
        # 注意：
        # - 這個 LayerNorm 的參數是獨立的
        # - 不是任何一個 DecoderLayer 內部的 norm1、norm2 或 norm3
        self.norm = nn.LayerNorm(d_model)

        # 儲存層數（方便外部查詢）
        self.num_layers = num_layers

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解碼器的前向傳播

        參數：
            x: 目標序列輸入，形狀為 (batch_size, tgt_len, d_model)
               （通常是目標的詞嵌入 + 位置編碼）

            encoder_output: 編碼器的輸出，形狀為 (batch_size, src_len, d_model)
                           （已編碼的來源序列）

            tgt_mask: 目標遮罩，形狀為 (batch_size, 1, tgt_len, tgt_len)
                     結合因果遮罩 + 目標的填充遮罩

            src_mask: 來源填充遮罩，形狀為 (batch_size, 1, 1, src_len)
                     忽略來源序列中的 <PAD> token

        回傳：
            output: 解碼器輸出，形狀為 (batch_size, tgt_len, d_model)
                   （已解碼的表示，準備好進行最終投影）

        完整流程：
            輸入 → 層1 → 層2 → ... → 層N → Norm → 輸出

        具體範例（英文→法文翻譯）：
            來源：「I love apples」
            目標：「J'aime les pommes」

            假設 num_layers = 6, d_model = 512

            encoder_output:
                shape = (1, 3, 512)
                「I love apples」的編碼表示

            輸入 x:
                shape = (1, 4, 512)
                x[0, 0, :] = "J'" 嵌入 + 位置編碼
                x[0, 1, :] = "aime" 嵌入 + 位置編碼
                x[0, 2, :] = "les" 嵌入 + 位置編碼
                x[0, 3, :] = "pommes" 嵌入 + 位置編碼

            層 1：
                - 遮罩自注意力：從先前的法文詞建立上下文
                - 交叉注意力：看英文來源
                - FFN：提取基本特徵
                → x.shape = (1, 4, 512)

            層 2：
                - 法文和英文之間更複雜的關係
                - 建立在層 1 的特徵之上
                → x.shape = (1, 4, 512)

            層 3-6：
                - 逐步提取更抽象的特徵
                - 最後一層包含最豐富的上下文資訊
                → x.shape = (1, 4, 512)

            最終層標準化：
                - 標準化最終輸出
                → x.shape = (1, 4, 512)

            輸出：
                x[0, 0, :] = "J'" 編碼（知道：這是開始，下一個是動詞）
                x[0, 1, :] = "aime" 編碼（知道：主詞是「I」，賓語即將出現）
                x[0, 2, :] = "les" 編碼（知道：冠詞，名詞即將出現）
                x[0, 3, :] = "pommes" 編碼（知道：這是賓語，結束句子）

                每個詞的表示都包含：
                ✓ 來自先前法文詞的上下文（遮罩自注意力）
                ✓ 來自英文來源的資訊（交叉注意力）
                ✓ 來自多層的豐富特徵（深度）

        【為什麼每一層的輸出形狀都相同？】
        - 每一層的輸入和輸出維度都是 d_model
        - 這樣可以：
          1. 使用殘差連接（x + SubLayer(x)）
          2. 堆疊任意數量的層
          3. 靈活組合
          4. 所有層都可以使用同一個 encoder_output

        【資訊在層之間的流動】
        - 層 1 的輸出 → 成為層 2 的輸入
        - 層 2 的輸出 → 成為層 3 的輸入
        - ...
        - 最後一層包含所有前面層的資訊
        - 每一層都增加對來源-目標關係的更豐富理解
        """
        # ========== 依序通過每個解碼器層 ==========
        # for 迴圈依序執行：
        # x = layer_1(x, encoder_output, tgt_mask, src_mask)
        # x = layer_2(x, encoder_output, tgt_mask, src_mask)
        # x = layer_3(x, encoder_output, tgt_mask, src_mask)
        # ...
        # x = layer_N(x, encoder_output, tgt_mask, src_mask)
        #
        # 重要注意事項：
        # - 每一層的輸入都是上一層的輸出（x 被更新）
        # - encoder_output 保持不變（所有層都使用相同的）
        # - tgt_mask 保持不變（所有層都使用相同的因果遮罩）
        # - src_mask 保持不變（所有層都使用相同的填充遮罩）
        #
        # 為什麼 encoder_output 不會改變？
        # - 編碼器執行一次，產生來源的固定表示
        # - 每個解碼器層都使用這個相同的來源表示
        # - 就像參考書：解碼器不斷查閱它，但書本身不會改變
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
            # x.shape 始終是 (batch_size, tgt_len, d_model)

        # ========== 最終的層標準化 ==========
        # 標準化最終輸出
        # 確保輸出分佈穩定
        x = self.norm(x)

        # 最終輸出：
        # - 形狀：(batch_size, tgt_len, d_model)
        # - 與輸入形狀相同，但內容已經被轉換過
        # - 每個目標 token 的表示都包含：
        #   * 來自先前目標 token 的上下文（透過遮罩自注意力）
        #   * 來自來源序列的資訊（透過交叉注意力）
        #   * 來自多層的豐富特徵（透過深度）
        # - 這個輸出可以：
        #   * 接到輸出投影（線性層到詞彙表）
        #   * 用於下一個 token 預測
        #   * 用於其他下游任務
        return x


def create_causal_mask(size: int) -> torch.Tensor:
    """
    為解碼器自注意力建立因果遮罩（也叫前瞻遮罩 look-ahead mask）

    【這是什麼？】
    一個下三角矩陣，防止位置關注到未來的位置。
    用於解碼器的遮罩自注意力。

    【為什麼需要？】
    在自回歸生成中，我們一次生成一個 token：
    - 當生成 token 3 時，我們只看過 token 0、1、2
    - 我們「不能」看到 token 4、5、6...（它們還不存在！）
    - 在訓練時，我們透過遮罩未來的位置來模擬這個情況

    【遮罩格式】
    回傳一個矩陣，其中：
    - 1 = 可以關注（位置可見）
    - 0 = 不能關注（位置被遮罩）

    【範例】
    對於 size=4（有 4 個 token 的句子）：

    [[1, 0, 0, 0],   ← 位置 0 只能看到位置 0
     [1, 1, 0, 0],   ← 位置 1 能看到位置 0-1
     [1, 1, 1, 0],   ← 位置 2 能看到位置 0-2
     [1, 1, 1, 1]]   ← 位置 3 能看到位置 0-3

    視覺化表示：
    ```
           位置 0  1  2  3
    位置 0:  ✓  ✗  ✗  ✗
    位置 1:  ✓  ✓  ✗  ✗
    位置 2:  ✓  ✓  ✓  ✗
    位置 3:  ✓  ✓  ✓  ✓
    ```

    這被稱為「因果」是因為：
    - 資訊因果流動（過去 → 現在）
    - 不能逆向流動（未來 → 現在）
    - 尊重時間順序

    參數：
        size: 序列長度（token 數量）

    回傳：
        mask: 因果遮罩，形狀為 (1, 1, size, size)
              形狀說明：
              - 第一個 1：批次維度（跨批次廣播）
              - 第二個 1：頭維度（跨注意力頭廣播）
              - (size, size)：實際的遮罩矩陣（query_len × key_len）

    在解碼器中的用法：
        tgt_len = 5
        causal_mask = create_causal_mask(tgt_len)
        # causal_mask.shape = (1, 1, 5, 5)

        # 在注意力中使用：
        attention(query, key, value, mask=causal_mask)
        # 每個位置只能關注自己和先前的位置
    """
    # ========== 建立下三角矩陣 ==========
    # torch.tril 建立下三角矩陣
    # torch.ones(size, size) 建立全 1 矩陣
    # tril 將上三角設為 0，保持下三角為 1
    #
    # size=4 的範例：
    # torch.ones(4, 4):
    # [[1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1],
    #  [1, 1, 1, 1]]
    #
    # torch.tril(...):
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    #
    # 這正是我們想要的因果遮罩！
    mask = torch.tril(torch.ones(size, size))
    # mask.shape = (size, size)

    # ========== 加入批次和頭維度 ==========
    # 從 (size, size) 重塑為 (1, 1, size, size)
    # 為什麼？
    # - 注意力期望遮罩形狀：(batch, heads, query_len, key_len)
    # - (1, 1, size, size) 會廣播到 (batch, heads, size, size)
    #
    # 廣播範例：
    # mask.shape = (1, 1, 4, 4)
    # attention scores.shape = (32, 8, 4, 4)  # batch=32, heads=8
    # → mask 自動廣播到 (32, 8, 4, 4)
    #
    # .unsqueeze(0) 在位置 0 加入維度：(size, size) → (1, size, size)
    # 再次 .unsqueeze(0) 在位置 0 加入維度：(1, size, size) → (1, 1, size, size)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape = (1, 1, size, size)

    return mask


if __name__ == "__main__":
    # 測試代碼
    print("=== 測試因果遮罩 ===\n")

    # 建立一個小的因果遮罩來視覺化
    causal_mask = create_causal_mask(5)
    print(f"因果遮罩形狀：{causal_mask.shape}")
    print("因果遮罩 (5x5)：")
    print(causal_mask.squeeze().numpy())
    print("\n(1 = 能看到，0 = 不能看到)")

    print("\n=== 測試解碼器層 ===\n")

    batch_size = 2
    src_len = 6  # 來源序列長度（例如英文）
    tgt_len = 5  # 目標序列長度（例如法文）
    d_model = 512
    num_heads = 8
    d_ff = 2048

    # 建立解碼器層
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

    # 建立測試輸入
    x = torch.randn(batch_size, tgt_len, d_model)  # 目標輸入
    encoder_output = torch.randn(batch_size, src_len, d_model)  # 編碼器輸出

    print(f"目標輸入形狀：{x.shape}")
    print(f"編碼器輸出形狀：{encoder_output.shape}")

    # 建立遮罩
    tgt_mask = create_causal_mask(tgt_len)  # 目標的因果遮罩
    src_mask = torch.ones(batch_size, 1, 1, src_len)  # 來源沒有填充（全是 1）

    # 前向傳播
    output = decoder_layer(x, encoder_output, tgt_mask, src_mask)
    print(f"解碼器層輸出形狀：{output.shape}")

    print("\n=== 測試完整解碼器（6 層）===\n")

    num_layers = 6
    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )

    output_full = decoder(x, encoder_output, tgt_mask, src_mask)
    print(f"完整解碼器輸出形狀：{output_full.shape}")

    # 計算參數數量
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n完整解碼器（{num_layers} 層）總參數量：{total_params:,}")
    print(f"約 {total_params / 1e6:.1f}M 參數")
