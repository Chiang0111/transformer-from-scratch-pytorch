"""
注意力機制（Attention Mechanisms）

本模組實現：
1. 縮放點積注意力（Scaled Dot-Product Attention）
2. 多頭注意力（Multi-Head Attention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,      # 型別提示：指定預期的資料型別
    key: torch.Tensor,        # Python 不會在執行時強制檢查
    value: torch.Tensor,      # 但幫助 IDE 自動完成與文件生成
    mask: Optional[torch.Tensor] = None  # Optional 代表可以是 None 或 Tensor
) -> tuple[torch.Tensor, torch.Tensor]:  # 回傳型別：包含 2 個 Tensor 的 tuple
    """
    縮放點積注意力 - Transformer 的核心

    這是整個 Transformer 架構中**最重要**的函式。
    所有其他組件都圍繞這個機制構建。

    【什麼是注意力？圖書館類比】
    想像你在圖書館查找資訊：

    Query (Q):  "我在找什麼？"
                （你想要回答的問題）

    Key (K):    "有哪些資訊可用？"
                （描述每本書的索引卡）

    Value (V):  "實際內容"
                （書籍本身）

    流程：
    1. 將你的查詢與所有鍵值比對（Q·K^T）
    2. 找出哪些鍵值最匹配（softmax）
    3. 根據匹配度加權提取對應的值（attention·V）

    【具體範例："I love eating apples"】
    對於單字 "eating"：

    Query (eating):  "我需要找到賓語（吃什麼？）"
    Keys:
      - "I":      "我是代名詞，主語" → 低匹配度
      - "love":   "我是動詞"         → 低匹配度
      - "eating": "我是當前單字"     → 中等匹配度
      - "apples": "我是名詞，可以是賓語" → 高匹配度！

    結果："eating" 高度關注 "apples"

    【公式】
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

    讓我們拆解：

    1. QK^T：查詢與鍵值的點積
       - 衡量相似度/相關性
       - 每個查詢與所有鍵值比較
       - 結果：注意力分數矩陣

    2. /√d_k：縮放因子（這很關鍵！）
       - 沒有縮放，分數會隨著維度增大而增大
       - 大分數 → softmax 飽和 → 梯度消失
       - 範例：若 d_k=64，我們除以 √64 = 8

    3. softmax(...)：將分數轉換為機率分布
       - 每一行總和為 1
       - 高分數接近 1，低分數接近 0
       - 產生注意力權重

    4. (...) V：值的加權和
       - 使用注意力權重組合值
       - 這是實際的「關注」步驟

    【為什麼需要型別提示？】
    型別提示如 `query: torch.Tensor` 是註解，用於：
    - 不影響執行（Python 執行時會忽略）
    - 幫助開發者理解預期的型別
    - 啟用 IDE 自動完成與錯誤檢查
    - 可以用 mypy 等工具驗證

    範例：
    ```python
    def add(x: int, y: int) -> int:  # 只是提示，不強制
        return x + y

    add("hello", "world")  # Python 允許！型別提示只是提示。
    ```

    【參數】
    Args:
        query: 查詢張量
               形狀：(batch_size, num_heads, seq_len_q, d_k)
               範例：(32, 8, 50, 64) = 32 樣本，8 個頭，50 個詞，64 維

        key: 鍵值張量
             形狀：(batch_size, num_heads, seq_len_k, d_k)
             通常 seq_len_k = seq_len_q（自注意力）

        value: 值張量
               形狀：(batch_size, num_heads, seq_len_v, d_v)
               通常 seq_len_v = seq_len_k 且 d_v = d_k

        mask: 可選的遮罩張量
              形狀：(batch_size, 1, 1, seq_len_k) 或
                    (batch_size, 1, seq_len_q, seq_len_k)
              用於：
              - Padding mask：忽略 <PAD> 詞元
              - Causal mask：防止查看未來詞元（解碼器）

    【回傳值】
    Returns:
        output: 注意力加權後的輸出
                形狀：(batch_size, num_heads, seq_len_q, d_v)
                關注值後的結果

        attention_weights: 注意力權重矩陣
                          形狀：(batch_size, num_heads, seq_len_q, seq_len_k)
                          每個查詢對每個鍵值的關注程度
                          每一行總和為 1（機率分布）

    【回傳型別註解】
    `-> tuple[torch.Tensor, torch.Tensor]` 代表：
    - 此函式回傳一個 tuple
    - tuple 包含正好 2 個元素
    - 兩個元素都是 torch.Tensor 型別
    """
    # ========== 步驟 1：取得 d_k（鍵值的維度）==========
    # query.size(-1) 取得最後一個維度
    # 範例：若 query.shape = (32, 8, 50, 64)，則 d_k = 64
    #
    # 為什麼需要 d_k？
    # - 用於縮放注意力分數（防止梯度消失）
    # - 這是「縮放點積注意力」中的「縮放」部分
    d_k = query.size(-1)

    # ========== 步驟 2：計算注意力分數（Q·K^T）==========
    # 矩陣乘法：Query × Key^轉置
    #
    # key.transpose(-2, -1) 交換最後兩個維度：
    # 之前：(batch, heads, seq_len_k, d_k)
    # 之後：(batch, heads, d_k, seq_len_k)
    #
    # 矩陣乘法：
    # (batch, heads, seq_len_q, d_k) × (batch, heads, d_k, seq_len_k)
    # →  (batch, heads, seq_len_q, seq_len_k)
    #
    # scores[i,j,q,k] 代表什麼？
    # - 查詢 q 對鍵值 k 的關注程度
    # - 數值越高 = 越相關
    #
    # 具體範例（seq_len=5, d_k=64）：
    # Q[0] = [1, 2, ..., 64]  # "eating" 的查詢
    # K[3] = [3, 1, ..., 32]  # "apples" 的鍵值
    # score = Q[0]·K[3] = 1*3 + 2*1 + ... = 某個大數字
    scores = torch.matmul(query, key.transpose(-2, -1))
    # scores.shape = (batch_size, num_heads, seq_len_q, seq_len_k)

    # ========== 步驟 3：用 √d_k 縮放（關鍵步驟！）==========
    # 這是「縮放」點積注意力的決定性特徵
    #
    # 為什麼要縮放？為什麼這是必要的？
    #
    # 問題：點積會隨著維度增長
    # - 若 d_k = 64，點積可能是：1*2 + 3*4 + ...（64 項）
    # - 即使數字很小，總和也可能很大
    # - 範例：平均點積 ≈ 0 但變異數 ≈ d_k
    #
    # 沒有縮放會發生什麼？
    # 假設 d_k = 64 且 scores = [200, 180, 150, 120]
    # softmax([200, 180, 150, 120]) ≈ [0.99, 0.01, 0.00, 0.00]
    # → 幾乎是 one-hot！（梯度消失）
    #
    # 使用縮放（除以 √64 = 8）：
    # scores = [25, 22.5, 18.75, 15]
    # softmax([25, 22.5, 18.75, 15]) ≈ [0.65, 0.24, 0.08, 0.03]
    # → 平滑分布（健康的梯度）
    #
    # 為什麼特別是 √d_k？
    # - 理論分析顯示 QK^T 的變異數與 d_k 成正比
    # - 除以 √d_k 將變異數標準化至 ≈1
    # - 保持 softmax 輸入在合理範圍內
    scores = scores / math.sqrt(d_k)
    # 現在分數已適當縮放

    # ========== 步驟 4：套用遮罩（如果提供）==========
    # 遮罩用於「忽略」某些位置
    #
    # 兩種主要的遮罩類型：
    #
    # 1. Padding Mask（填充遮罩）：
    #    句子："I love eating apples <PAD> <PAD>"
    #    遮罩：[1, 1, 1, 1, 0, 0]
    #    → 不關注 <PAD> 詞元（它們沒有意義）
    #
    # 2. Causal Mask（因果遮罩，用於解碼器）：
    #    預測第 i 個詞時，只能看到第 0...i-1 個詞
    #    防止「作弊」查看未來的詞
    #    範例（位置 2）：
    #    遮罩：[1, 1, 1, 0, 0]  # 可以看到詞 0,1,2 但看不到 3,4
    #
    # 遮罩如何運作：
    # - 將被遮罩的位置設為非常負的值（-1e9）
    # - softmax 後，exp(-1e9) ≈ 0
    # - 這些位置獲得接近零的注意力權重
    #
    # 為什麼用 -1e9 而不是 -inf？
    # - -inf 在某些邊緣情況下可能導致 NaN
    # - -1e9 已經「夠負」且數值穩定
    if mask is not None:
        # masked_fill：mask==0 的位置替換為 -1e9
        # 範例：
        # scores = [[10, 20, 30, 40, 50]]
        # mask   = [[1,  1,  1,  0,  0]]
        # result = [[10, 20, 30, -1e9, -1e9]]
        scores = scores.masked_fill(mask == 0, -1e9)

    # ========== 步驟 5：套用 Softmax → 注意力權重 ==========
    # Softmax 將分數轉換為機率分布
    #
    # 公式：softmax(x_i) = exp(x_i) / Σ exp(x_j)
    #
    # 性質：
    # - 所有值介於 0 和 1 之間
    # - 所有值的總和 = 1（機率分布）
    # - 較大的輸入 → 較大的輸出（但經過標準化）
    #
    # dim=-1 代表：
    # - 在最後一個維度（seq_len_k）上套用 softmax
    # - 每一行變成機率分布
    # - attention_weights[i, j, q, :].sum() = 1
    #
    # 具體範例：
    # 輸入分數：  [2.0, 1.5, 0.5, -1e9]
    # Softmax 後：[0.58, 0.32, 0.10, 0.00]
    # 注意：
    # - 最高分數（2.0）→ 最高權重（0.58）
    # - 被遮罩位置（-1e9）→ 接近零（0.00）
    # - 總和 = 1.00
    attention_weights = F.softmax(scores, dim=-1)
    # attention_weights.shape = (batch_size, num_heads, seq_len_q, seq_len_k)

    # ========== 步驟 6：值的加權和 ==========
    # 這裡是我們實際「關注」值的地方！
    #
    # 矩陣乘法：
    # (batch, heads, seq_len_q, seq_len_k) × (batch, heads, seq_len_k, d_v)
    # →  (batch, heads, seq_len_q, d_v)
    #
    # 這做了什麼：
    # 對每個查詢位置，用注意力權重組合所有值
    #
    # 具體範例："I love eating apples"
    # 對於查詢 "eating"：
    #   attention_weights = [0.1, 0.1, 0.2, 0.6]  # 對 "apples" 高度關注
    #   值：
    #     V["I"]      = [v1, v2, v3, ...]
    #     V["love"]   = [v4, v5, v6, ...]
    #     V["eating"] = [v7, v8, v9, ...]
    #     V["apples"] = [v10, v11, v12, ...]
    #
    #   output = 0.1*V["I"] + 0.1*V["love"] + 0.2*V["eating"] + 0.6*V["apples"]
    #          = 主要來自 "apples" 的資訊！
    #
    # 這就是注意力的魔力：
    # - 動態選擇相關資訊
    # - 每個位置都不同
    # - 從資料中學習
    output = torch.matmul(attention_weights, value)
    # output.shape = (batch_size, num_heads, seq_len_q, d_v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制（Multi-Head Attention Mechanism）

    【為什麼需要多個頭？】
    想像一家銀行決定是否借錢給你：

    單一專家問題：
    - 一位專家只看信用評分
    - 錯過其他重要因素
    - 視角有限

    多專家解決方案（多頭）：
    - 專家 1：檢查信用評分
    - 專家 2：分析收入穩定性
    - 專家 3：審查資產
    - 專家 4：檢視就業歷史
    → 結合所有意見做出更好的決策！

    同樣地，在語言中：

    單一注意力頭問題：
    - 可能只捕捉主謂關係
    - 錯過其他重要模式
    - 範例："The bank can refuse to lend money"
      * 只看到："bank → refuse"（句法）
      * 錯過："refuse → lend"（語義）

    多頭注意力解決方案：
    - 頭 1：主謂關係
    - 頭 2：動賓關係
    - 頭 3：修飾關係
    - 頭 4：位置關係
    - 頭 5-8：其他模式...
    → 每個頭學習不同的面向！

    【架構概覽】
    流程：
    ```
    輸入 (batch, seq_len, d_model)
      ↓
    [線性投影：W_q, W_k, W_v]
      ↓
    [分割成 num_heads 個頭]
      ↓
    [縮放點積注意力]（每個頭平行處理）
      ↓
    [組合所有頭]
      ↓
    [線性投影：W_o]
      ↓
    輸出 (batch, seq_len, d_model)
    ```

    【為什麼這樣設計？】
    - 多個頭 = 多個視角
    - 每個頭有 d_k = d_model / num_heads 維度
    - 總計算量 ≈ 與單頭全維度相同
    - 但表達能力更強！

    【具體範例】
    d_model = 512, num_heads = 8
    → 每個頭獲得 d_k = 512/8 = 64 維度
    → 8 個不同的 64 維「專家」平行工作
    → 最終組合創造豐富的 512 維表示

    Args:
        d_model: 模型維度（輸入/輸出維度，例如 512）
        num_heads: 注意力頭的數量（例如 8）
                   注意：d_model 必須能被 num_heads 整除
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # ========== 驗證：d_model 必須能被 num_heads 整除 ==========
        # 為什麼有這個要求？
        # - 我們將 d_model 平均分給所有頭
        # - 每個頭獲得 d_k = d_model // num_heads 維度
        # - 如果不能整除，就無法平均分割
        #
        # 範例：
        # ✓ d_model=512, num_heads=8  → d_k=64（可行！）
        # ✗ d_model=512, num_heads=7  → d_k=73.14...（不可行！）
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads

        # ========== 計算每個頭的維度 ==========
        # 每個頭操作 d_k 維度
        # 範例：512 維度分成 8 個頭 = 每個頭 64 維
        self.d_k = d_model // num_heads  # 整數除法

        # ========== 定義 4 個線性層（可學習的投影）==========

        # W_q, W_k, W_v：將輸入投影到 Query、Key、Value
        # 輸入：(batch, seq_len, d_model)
        # 輸出：(batch, seq_len, d_model)
        #
        # 為什麼是 d_model → d_model 而不是 d_model → d_k？
        # - 我們先投影到完整的 d_model
        # - 然後分割成 num_heads 個大小為 d_k 的部分
        # - 這比做 num_heads 次獨立投影更有效率
        #
        # 為什麼這些是可學習的？
        # - 模型學習要問什麼問題（W_q）
        # - 模型學習要產生什麼鍵值（W_k）
        # - 模型學習要回傳什麼值（W_v）
        #
        # d_model=512 的範例：
        # - W_q 有 512×512 = 262,144 個參數
        # - W_k 有 512×512 = 262,144 個參數
        # - W_v 有 512×512 = 262,144 個參數
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影

        # W_o：輸出投影
        # 組合所有頭後，投影回 d_model
        # 輸入：(batch, seq_len, d_model)
        # 輸出：(batch, seq_len, d_model)
        #
        # 為什麼需要這個？
        # - 整合所有頭的資訊
        # - 讓模型學習如何組合頭的輸出
        # - 沒有它，我們只是在拼接，而不是學習組合
        #
        # 總參數量：512×512 = 262,144
        self.W_o = nn.Linear(d_model, d_model)  # 輸出投影

        # 多頭注意力的總參數量：
        # W_q + W_k + W_v + W_o = 4 × (d_model × d_model)
        # 範例：4 × (512 × 512) = 1,048,576 個參數

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將輸入分割成多個注意力頭

        輸入形狀：(batch_size, seq_len, d_model)
        輸出形狀：(batch_size, num_heads, seq_len, d_k)

        【這做了什麼】
        取一個具有 d_model 維度的張量，並將其分割成 num_heads 個部分，
        每個部分有 d_k 維度。

        【視覺範例】
        d_model = 512, num_heads = 8, d_k = 64

        輸入：
        (batch, seq_len, 512)
        [......512 維度......]

        分割後：
        (batch, seq_len, 8, 64)
        [64][64][64][64][64][64][64][64]
         ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
        h0  h1  h2  h3  h4  h5  h6  h7

        轉置後：
        (batch, 8, seq_len, 64)
        現在按頭組織，所以每個頭可以獨立工作！

        【為什麼要轉置？】
        我們希望批次操作平行處理所有頭：
        - 之前：(batch, seq_len, num_heads, d_k)
          難以獨立處理各頭
        - 之後：(batch, num_heads, seq_len, d_k)
          簡單！每個頭都是獨立的「批次」項目
        """
        batch_size, seq_len, d_model = x.size()

        # ========== 步驟 1：重塑以分割維度 ==========
        # view() 重塑張量而不複製資料
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)
        #
        # batch=2, seq_len=5, d_model=512, num_heads=8 的範例：
        # (2, 5, 512) → (2, 5, 8, 64)
        #
        # 512 維度被分割成 8 組，每組 64：
        # [0:64]    → 頭 0
        # [64:128]  → 頭 1
        # [128:192] → 頭 2
        # ...
        # [448:512] → 頭 7
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # ========== 步驟 2：轉置以按頭分組 ==========
        # transpose(1, 2) 交換維度 1 和 2
        # 之前：(batch_size, seq_len, num_heads, d_k)
        # 之後：(batch_size, num_heads, seq_len, d_k)
        #
        # 為什麼這樣更好？
        # - 現在 "num_heads" 維度在 "seq_len" 之前
        # - 可以使用批次操作平行處理所有頭
        # - 每個頭獨立處理其 seq_len × d_k 的資料
        #
        # 可以想像成有 8 個獨立的注意力機制
        # 平行運行，每個都在 64 維空間中工作
        return x.transpose(1, 2)
        # 輸出形狀：(batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將多個注意力頭組合回來

        輸入形狀：(batch_size, num_heads, seq_len, d_k)
        輸出形狀：(batch_size, seq_len, d_model)

        這是 split_heads 的逆操作。

        【視覺範例】
        num_heads = 8, d_k = 64, d_model = 512

        輸入（注意力後）：
        (batch, 8, seq_len, 64)
        頭 0：[64 維]
        頭 1：[64 維]
        ...
        頭 7：[64 維]

        轉置後：
        (batch, seq_len, 8, 64)

        view 後（拼接）：
        (batch, seq_len, 512)
        [h0: 64][h1: 64]...[h7: 64] = 512 維度

        【為什麼要 .contiguous()？】
        PyTorch 記憶體佈局的技術細節：
        - transpose() 不複製資料，只改變視圖
        - view() 需要記憶體是連續的
        - contiguous() 在需要時創建連續副本
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # ========== 步驟 1：轉置回來 ==========
        # 交換維度 1 和 2
        # (batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, num_heads, d_k)
        #
        # 這將 seq_len 帶回維度 1
        # 現在：每個位置有 num_heads 個 d_k 維度的片段
        x = x.transpose(1, 2)

        # ========== 步驟 2：合併頭（拼接）==========
        # .contiguous() 確保記憶體佈局正確
        # 為什麼需要？
        # - transpose() 創建資料的視圖（不複製）
        # - 實際記憶體仍然是原始順序
        # - view() 需要連續記憶體
        # - contiguous() 在必要時製作副本
        #
        # .view() 重塑張量
        # (batch_size, seq_len, num_heads, d_k) → (batch_size, seq_len, d_model)
        #
        # 範例：(2, 5, 8, 64) → (2, 5, 512)
        # 8 個 64 維度的塊被拼接：
        # [頭0: 64 維][頭1: 64 維]...[頭7: 64 維] = 512 維
        #
        # 每個位置現在擁有所有 8 個頭組合的資訊！
        return x.contiguous().view(batch_size, seq_len, self.d_model)
        # 輸出形狀：(batch_size, seq_len, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多頭注意力前向傳播

        Args:
            query: Query 張量，形狀 (batch_size, seq_len, d_model)
            key:   Key 張量，形狀 (batch_size, seq_len, d_model)
            value: Value 張量，形狀 (batch_size, seq_len, d_model)
            mask:  可選遮罩，形狀 (batch_size, 1, 1, seq_len) 或
                   (batch_size, 1, seq_len, seq_len)

        Returns:
            output: 多頭注意力輸出，形狀 (batch_size, seq_len, d_model)

        【完整流程範例："I love eating apples"】
        假設：batch=1, seq_len=5, d_model=512, num_heads=8, d_k=64

        輸入：
            query = key = value（自注意力）
            形狀：(1, 5, 512)
            每個詞是一個 512 維向量

        步驟 1：線性投影
            Q = W_q(query) = (1, 5, 512)
            K = W_k(key) = (1, 5, 512)
            V = W_v(value) = (1, 5, 512)
            （模型學習要創建什麼問題/鍵值/值）

        步驟 2：分割成 8 個頭
            Q = (1, 8, 5, 64)  # 每個詞有 8 個不同的 64 維查詢
            K = (1, 8, 5, 64)  # 每個詞有 8 個不同的 64 維鍵值
            V = (1, 8, 5, 64)  # 每個詞有 8 個不同的 64 維值

            頭 0 可能專注於：主謂關係
            頭 1 可能專注於：動賓關係
            ...
            頭 7 可能專注於：位置模式

        步驟 3：注意力（每個頭）
            對於頭 1 中的 "eating"（動賓頭）：
            - Query("eating")："我需要一個賓語"
            - Keys：["I", "love", "eating", "apples", "<END>"]
            - 注意力權重：[0.1, 0.1, 0.2, 0.6, 0.0]
            - 輸出：主要是 "apples" 的值！

        步驟 4：組合頭
            (1, 8, 5, 64) → (1, 5, 512)
            拼接所有 8 個頭的輸出：
            [頭0: 64][頭1: 64]...[頭7: 64] = 512 維

        步驟 5：輸出投影
            W_o 學習如何組合所有頭的資訊
            (1, 5, 512) → (1, 5, 512)

        最終輸出：
            每個詞現在包含多個視角的資訊！
        """
        batch_size = query.size(0)

        # ========== 步驟 1：套用線性投影 ==========
        # 將輸入轉換為 Query、Key、Value 表示
        #
        # 為什麼需要這些投影？
        # - 學習特定任務的轉換
        # - 創建適當的「問題」、「鍵值」和「值」
        # - Q、K、V 的不同權重允許不同角色
        #
        # 輸入：(batch_size, seq_len, d_model)
        # 輸出：(batch_size, seq_len, d_model)
        #
        # 範例："eating" 與 d_model=512
        # 原始向量：[x1, x2, ..., x512]
        # W_q 後：[q1, q2, ..., q512]（「我需要什麼賓語？」）
        # W_k 後：[k1, k2, ..., k512]（「我是一個動詞」）
        # W_v 後：[v1, v2, ..., v512]（「這是我的意義」）
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)

        # ========== 步驟 2：分割成多個頭 ==========
        # 將 d_model 維度分給 num_heads 個頭
        # 每個頭獲得 d_k = d_model / num_heads 維度
        #
        # 輸入：(batch_size, seq_len, d_model)
        # 輸出：(batch_size, num_heads, seq_len, d_k)
        #
        # 範例：(1, 5, 512) → (1, 8, 5, 64)
        # 現在我們有 8 個平行的注意力機制！
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # ========== 步驟 3：套用縮放點積注意力 ==========
        # 為每個頭獨立運行注意力
        # 所有頭平行處理（批次操作）
        #
        # 輸入：Q、K、V 都是形狀 (batch_size, num_heads, seq_len, d_k)
        # 輸出：attn_output 形狀 (batch_size, num_heads, seq_len, d_k)
        #
        # 每個頭：
        # - 計算自己的注意力權重
        # - 專注於輸入的不同面向
        # - 為每個位置產生自己的 d_k 維輸出
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output 形狀：(batch_size, num_heads, seq_len, d_k)

        # 我們在這裡忽略注意力權重（_ 部分）
        # 但它們對於視覺化或除錯很有用

        # ========== 步驟 4：組合頭 ==========
        # 將所有頭合併回來
        #
        # 輸入：(batch_size, num_heads, seq_len, d_k)
        # 輸出：(batch_size, seq_len, d_model)
        #
        # 範例：(1, 8, 5, 64) → (1, 5, 512)
        # 拼接 8 個頭：
        # [頭0_輸出: 64][頭1_輸出: 64]...[頭7_輸出: 64] = 512
        #
        # 現在每個位置都有來自所有 8 個視角的資訊！
        output = self.combine_heads(attn_output)  # (batch_size, seq_len, d_model)

        # ========== 步驟 5：最終線性投影（W_o）==========
        # 學習如何最佳地組合所有頭的資訊
        #
        # 輸入：(batch_size, seq_len, d_model)
        # 輸出：(batch_size, seq_len, d_model)
        #
        # 為什麼需要這個？
        # - 簡單的拼接可能不是最優的
        # - W_o 學習如何整合多頭輸出
        # - 允許模型對不同的頭賦予不同權重
        #
        # 範例：也許頭 1 的輸出非常重要，頭 7 較不重要
        # W_o 可以學習這些相對重要性
        output = self.W_o(output)  # (batch_size, seq_len, d_model)

        # 最終輸出形狀：(batch_size, seq_len, d_model)
        # 與輸入形狀相同，但現在用多頭注意力豐富了！
        return output
