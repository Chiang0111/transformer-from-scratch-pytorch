"""
Attention 機制實作

包含：
1. Scaled Dot-Product Attention (縮放點積注意力)
2. Multi-Head Attention (多頭注意力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,      # ← 型別提示：告訴你這個參數應該是 Tensor
    key: torch.Tensor,        # ← 讓 IDE 可以自動完成，也讓程式碼更清楚
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None  # Optional 表示可以是 None
) -> tuple[torch.Tensor, torch.Tensor]:  # ← 回傳型別：這個函數會回傳 2 個 Tensor
    """
    縮放點積注意力機制（Scaled Dot-Product Attention）

    這是 Transformer 的核心！理解這個函數就理解了 80% 的 Transformer。

    【QKV 的比喻】圖書館查資料：
    - Query (Q)：你的問題（"我想找關於機器學習的書"）
    - Key (K)：每本書的標題/索引（用來比對相關性）
    - Value (V)：書的實際內容
    - Attention：找出最相關的書，然後讀取內容

    【具體例子】理解句子 "我愛吃蘋果"：
    當處理 "吃" 這個詞時：
    - Q ("吃") 問：「誰在吃？吃什麼？」
    - K 有：["我"的標籤, "愛"的標籤, "吃"的標籤, "蘋果"的標籤]
    - 計算相關性：["我"→0.3, "愛"→0.1, "吃"→0.1, "蘋果"→0.5]
    - V 加權平均：0.3×"我" + 0.1×"愛" + 0.1×"吃" + 0.5×"蘋果"
    - 結果："吃" 現在包含上下文，特別關注與 "蘋果" 的關係

    參數說明：
        query: shape (batch_size, num_heads, seq_len_q, d_k)
               查詢向量 - 代表「我想要找什麼資訊」
        key:   shape (batch_size, num_heads, seq_len_k, d_k)
               鍵向量 - 代表「每個位置有什麼資訊」
        value: shape (batch_size, num_heads, seq_len_v, d_v)
               值向量 - 代表「實際的資訊內容」
        mask:  shape (batch_size, 1, seq_len_q, seq_len_k) 或 (batch_size, num_heads, seq_len_q, seq_len_k)
               遮罩 - 用來屏蔽某些位置（例如 padding 或未來的詞）

    回傳（tuple 包含 2 個元素）：
        output: shape (batch_size, num_heads, seq_len_q, d_v)
                注意力加權後的輸出（每個 query 的新表示，包含上下文資訊）
        attention_weights: shape (batch_size, num_heads, seq_len_q, seq_len_k)
                注意力權重（可用來視覺化「誰在關注誰」）

    公式：Attention(Q,K,V) = softmax(QK^T / √d_k) * V
    """
    # ========== 步驟 1: 獲取維度 d_k（用於後續縮放）==========
    d_k = query.size(-1)  # -1 表示最後一個維度，這裡是每個頭的維度

    # ========== 步驟 2: 計算「相關性分數」(Q·K^T) ==========
    # 這一步在問：每個 query 和每個 key 有多相關？
    #
    # 矩陣乘法細節：
    # query.shape     = (batch, heads, seq_len_q, d_k)
    # key.transpose   = (batch, heads, d_k, seq_len_k)  ← 轉置最後兩維
    # 相乘結果 scores = (batch, heads, seq_len_q, seq_len_k)
    #
    # scores[b, h, i, j] 的意義：
    # 在第 b 個句子、第 h 個頭中，第 i 個 query 對第 j 個 key 的相關性分數
    #
    # 例如：scores[0, 0, 2, 4] = 句子0、頭0、詞2對詞4的相關性
    scores = torch.matmul(query, key.transpose(-2, -1))

    # ========== 步驟 3: 縮放（除以 √d_k）==========
    # 【為什麼要縮放？】非常重要！
    #
    # 問題：當 d_k 很大時（例如 64），點積的結果會變很大
    # 例如：假設 d_k = 64
    #   不縮放：scores = [100, 200, -50, 300]  ← 數值範圍很大
    #   softmax([100, 200, -50, 300]) ≈ [0, 1, 0, 0]  ← 變得極端！
    #   問題：梯度幾乎消失（gradient ≈ 0），模型學不到東西
    #
    #   縮放後：scores / √64 = [12.5, 25, -6.25, 37.5]  ← 數值變小
    #   softmax([12.5, 25, -6.25, 37.5]) ≈ [0.1, 0.3, 0.05, 0.55]  ← 分佈平滑
    #   優點：梯度正常，模型可以學習
    #
    # 數學原因：點積的方差 ∝ d_k，所以除以 √d_k 來標準化
    scores = scores / math.sqrt(d_k)

    # ========== 步驟 4: 應用遮罩（mask）==========
    # 【mask 的兩個主要用途】
    #
    # 用途 1：Padding Mask（忽略填充的 token）
    #   句子 1："我愛吃蘋果"           → 長度 4
    #   句子 2："我愛你" + <PAD><PAD>  → 長度 3，補 2 個 <PAD>
    #
    #   mask = [[1, 1, 1, 1, 1],    # 句子 1：全部有效
    #           [1, 1, 1, 0, 0]]    # 句子 2：後 2 個無效（<PAD>）
    #
    #   我們不想讓模型 attend 到 <PAD>，因為它們沒有意義
    #
    # 用途 2：Causal Mask（不能看未來）- 用於 Decoder
    #   翻譯時，生成第 3 個詞時，不能偷看第 4、5 個詞（還沒生成）
    #
    #   mask = [[1, 0, 0, 0, 0],  # 詞1只能看詞1
    #           [1, 1, 0, 0, 0],  # 詞2能看詞1,2
    #           [1, 1, 1, 0, 0],  # 詞3能看詞1,2,3（不能看4,5）
    #           [1, 1, 1, 1, 0],
    #           [1, 1, 1, 1, 1]]
    if mask is not None:
        # 把 mask=0 的位置設為很大的負數（-1e9）
        # 原因：softmax([-1e9, 1, 2]) ≈ [0, 0.27, 0.73]
        #       被 mask 的位置權重 ≈ 0，幾乎不影響結果
        #
        # 為什麼不用 -inf？因為 -inf 可能導致 NaN（0/0 的情況）
        scores = scores.masked_fill(mask == 0, -1e9)

    # ========== 步驟 5: Softmax（轉成機率分佈）==========
    # 【softmax 的作用】把分數轉成「加起來等於 1」的機率
    #
    # 例子：
    #   原始分數：[2.5, 1.0, 0.5, 3.0]  ← 可以是任意數值
    #   softmax： [0.29, 0.06, 0.04, 0.61]  ← 總和 = 1，都是正數
    #
    #   意義：
    #   「第 1 個詞的重要性是 29%」
    #   「第 2 個詞的重要性是 6%」
    #   「第 3 個詞的重要性是 4%」
    #   「第 4 個詞的重要性是 61%」 ← 最重要！
    #
    # dim=-1 表示在最後一個維度做 softmax（對每個 query 的所有 keys）
    # 結果：每個 query 對所有 keys 的注意力權重總和 = 1
    attention_weights = F.softmax(scores, dim=-1)

    # ========== 步驟 6: 用注意力權重加權 values（提取資訊）==========
    # 【這是最後一步，也是最重要的一步！】
    #
    # 回到例子："我愛吃蘋果"，理解 "吃" 這個詞
    #
    # attention_weights = [0.3, 0.1, 0.1, 0.5]  ← 對 ["我","愛","吃","蘋果"] 的注意力
    #
    # values = [
    #     [0.1, 0.2, ...],  # "我" 的資訊向量
    #     [0.3, 0.1, ...],  # "愛" 的資訊向量
    #     [0.5, 0.4, ...],  # "吃" 的資訊向量
    #     [0.2, 0.8, ...]   # "蘋果" 的資訊向量
    # ]
    #
    # output = 0.3 × "我" + 0.1 × "愛" + 0.1 × "吃" + 0.5 × "蘋果"
    #        = [0.21, 0.51, ...]
    #
    # 結果："吃" 的新表示，融合了上下文資訊，特別是 "蘋果"（權重 0.5）
    #
    # 這就是 Attention 的核心：根據相關性，從其他詞「提取」有用的資訊！
    output = torch.matmul(attention_weights, value)

    # 回傳兩個東西：
    # 1. output - 加權後的結果（融合了上下文的新表示）
    # 2. attention_weights - 注意力分佈（可以視覺化，看模型在關注什麼）
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制（Multi-Head Attention）

    【為什麼需要多個「頭」？】

    問題：單一注意力的局限
    ─────────────────────────
    想像你在讀句子 "The bank can refuse to lend money."

    單一注意力可能只關注一種關係：
    - "bank" ← "money" （銀行-錢的關係）✓

    但會錯過：
    - "bank" ← "refuse" （銀行-拒絕的動作）✗
    - "bank" ← "lend" （銀行-借貸的功能）✗

    解決方案：Multi-Head Attention
    ─────────────────────────
    讓 8 個「專家」（heads）同時分析，每個關注不同模式：

    頭 1：關注「主詞-動詞」關係
      "The bank" → "can refuse" ✓

    頭 2：關注「動詞-受詞」關係
      "refuse" → "to lend" ✓

    頭 3：關注「語義」關係
      "bank" → "money" ✓

    頭 4-8：...發現其他模式

    最後融合所有專家的意見 → 完整理解！

    【架構流程】
    ─────────────────────────
    輸入 (batch, seq_len, 512)
      ↓
    [線性投影 Q, K, V]  ← W_q, W_k, W_v（學習不同角度）
      ↓
    [Split Heads]  ← 512維 → 8個頭×64維
      ↓
    [每個頭獨立做 Attention]  ← 8個專家並行分析
      ↓
    [Combine Heads]  ← 8個頭×64維 → 512維
      ↓
    [輸出投影]  ← W_o（融合所有頭的資訊）
      ↓
    輸出 (batch, seq_len, 512)

    參數說明：
        d_model: 模型維度（例如：512）
        num_heads: 注意力頭的數量（例如：8）

    數學關係：
        d_model = num_heads × d_k
        512 = 8 × 64
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        # ========== 檢查：d_model 必須能被 num_heads 整除 ==========
        # 原因：我們要平均分割！
        # 例如：512 ÷ 8 = 64 ✓
        #       512 ÷ 7 = 73.14... ✗（無法整除，會出錯）
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"

        self.d_model = d_model        # 例如：512
        self.num_heads = num_heads    # 例如：8
        self.d_k = d_model // num_heads  # 512 // 8 = 64（每個頭的維度）

        # ========== 建立 4 個線性層（可學習的權重）==========
        #
        # 為什麼需要這些投影層？
        # - 它們是「可學習的」，訓練時會自動調整
        # - 讓模型學會「從哪些角度看資料最有用」
        #
        # W_q, W_k, W_v：學習如何產生 Query, Key, Value
        # - 不同的投影 = 不同的「視角」
        # - 訓練後，每個頭會自動專注於不同的模式
        #
        # 例如訓練後可能變成：
        # - 頭1的 W_q：學會提取「主詞」的查詢
        # - 頭2的 W_q：學會提取「動詞」的查詢
        # - ...

        self.W_q = nn.Linear(d_model, d_model)  # 512 → 512
        self.W_k = nn.Linear(d_model, d_model)  # 512 → 512
        self.W_v = nn.Linear(d_model, d_model)  # 512 → 512

        # W_o：最後的輸出投影
        # - 融合所有頭的資訊
        # - 學習「如何最好地組合 8 個專家的意見」
        self.W_o = nn.Linear(d_model, d_model)  # 512 → 512

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將輸入分割成多個注意力頭（Split Heads）

        目的：把一個大的向量「切」成多個小向量

        輸入 shape: (batch_size, seq_len, d_model)
        輸出 shape: (batch_size, num_heads, seq_len, d_k)

        【視覺化理解】
        ─────────────────────────
        輸入：(batch=2, seq=10, d_model=512)
          [句子1的10個詞，每個詞512維]
          [句子2的10個詞，每個詞512維]

        步驟1 - 重塑：(batch=2, seq=10, heads=8, d_k=64)
          [句子1的10個詞，每個詞分成8份，每份64維]
          [句子2的10個詞，每個詞分成8份，每份64維]

        步驟2 - 轉置：(batch=2, heads=8, seq=10, d_k=64)
          [句子1的頭1: 10個詞×64維]
          [句子1的頭2: 10個詞×64維]
          ...
          [句子1的頭8: 10個詞×64維]
          [句子2的頭1: 10個詞×64維]
          ...

        【為什麼要轉置？】
        - 原因：我們要對每個頭「獨立」做批次運算
        - 轉置後，8個頭在第2個維度，方便平行計算
        - PyTorch 會自動對 (batch, heads, seq, d_k) 的每個頭做運算
        """
        batch_size, seq_len, d_model = x.size()

        # 步驟 1: 重塑（reshape）
        # (batch, seq, 512) → (batch, seq, 8, 64)
        # 把 512 維切成 8 份，每份 64 維
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # 步驟 2: 轉置（transpose）
        # (batch, seq, 8, 64) → (batch, 8, seq, 64)
        # 把 heads 維度移到前面，方便獨立處理每個頭
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        將多個注意力頭的結果合併（Combine Heads）

        目的：把多個小向量「合併」回一個大向量（split_heads 的逆操作）

        輸入 shape: (batch_size, num_heads, seq_len, d_k)
        輸出 shape: (batch_size, seq_len, d_model)

        【視覺化理解】
        ─────────────────────────
        輸入：(batch=2, heads=8, seq=10, d_k=64)
          [句子1的頭1: 10個詞×64維]
          [句子1的頭2: 10個詞×64維]
          ...
          [句子1的頭8: 10個詞×64維]

        步驟1 - 轉置：(batch=2, seq=10, heads=8, d_k=64)
          [句子1的10個詞，每個詞有8份64維]

        步驟2 - 合併：(batch=2, seq=10, d_model=512)
          [句子1的10個詞，每個詞512維]
          把 8×64 = 512 維合併回去

        【.contiguous() 是什麼？】
        - 技術細節：轉置後記憶體不連續
        - .contiguous() 重新整理記憶體
        - 讓 .view() 可以正確運作
        - 暫時可以當作「必要的技術步驟」
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # 步驟 1: 轉置回來
        # (batch, 8, seq, 64) → (batch, seq, 8, 64)
        x = x.transpose(1, 2)

        # 步驟 2: 合併最後兩個維度
        # (batch, seq, 8, 64) → (batch, seq, 512)
        # 把 8 個頭的 64 維合併成 512 維
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多頭注意力的前向傳播（主流程）

        【完整流程範例】
        ─────────────────────────
        輸入："我愛吃蘋果"（4個詞）
          ↓
        embedding: (batch=1, seq=4, d_model=512)
          [[詞1的512維向量],
           [詞2的512維向量],
           [詞3的512維向量],
           [詞4的512維向量]]

        步驟1：線性投影
        ───────────────
        Q = W_q × 輸入  ← 學習「查詢角度」
        K = W_k × 輸入  ← 學習「鍵的角度」
        V = W_v × 輸入  ← 學習「值的角度」

        步驟2：Split（分成8個頭）
        ───────────────
        (1, 4, 512) → (1, 8, 4, 64)
        現在有8個專家，每個專家看4個詞的64維表示

        步驟3：每個頭獨立做Attention
        ───────────────
        頭1: scaled_dot_product_attention(Q₁, K₁, V₁)
        頭2: scaled_dot_product_attention(Q₂, K₂, V₂)
        ...
        頭8: scaled_dot_product_attention(Q₈, K₈, V₈)

        每個頭關注不同的模式：
        - 頭1可能關注：主詞-動詞關係
        - 頭2可能關注：動詞-受詞關係
        - 頭3可能關注：語義相似性
        - ...

        步驟4：Combine（合併8個頭）
        ───────────────
        (1, 8, 4, 64) → (1, 4, 512)

        步驟5：最後投影
        ───────────────
        W_o × 合併結果
        融合所有專家的意見

        輸出
        ───────────────
        (1, 4, 512) ← 每個詞都包含了多角度的上下文資訊！

        參數：
            query, key, value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, 1, 1, seq_len) 或 (batch_size, 1, seq_len, seq_len)

        回傳：
            output: shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # ===== 步驟 1: 線性投影 Q, K, V =====
        # 這些投影是「可學習的」，訓練時會自動調整
        # 讓模型學會「從哪些角度看資料最有用」
        #
        # 為什麼每個頭不用不同的 W？
        # - 因為我們會在 split_heads 時「切」成不同部分
        # - 每個頭會用到 512 維中「不同的 64 維」
        # - 所以實際上每個頭還是有不同的參數
        Q = self.W_q(query)  # (batch_size, seq_len, 512) → (batch_size, seq_len, 512)
        K = self.W_k(key)    # (batch_size, seq_len, 512) → (batch_size, seq_len, 512)
        V = self.W_v(value)  # (batch_size, seq_len, 512) → (batch_size, seq_len, 512)

        # ===== 步驟 2: 分割成多個頭 =====
        # 512 維 → 8 個頭 × 64 維
        # 每個頭現在有自己的 Q, K, V
        Q = self.split_heads(Q)  # (batch, seq, 512) → (batch, 8, seq, 64)
        K = self.split_heads(K)  # (batch, seq, 512) → (batch, 8, seq, 64)
        V = self.split_heads(V)  # (batch, seq, 512) → (batch, 8, seq, 64)

        # ===== 步驟 3: 每個頭獨立計算 Attention =====
        # 就是你之前學的 scaled_dot_product_attention 函數！
        # PyTorch 會自動對 8 個頭並行計算（GPU 加速）
        #
        # 輸入：(batch, 8, seq, 64)
        # 輸出：(batch, 8, seq, 64)
        #
        # 每個頭都做：Attention(Q, K, V) = softmax(QK^T/√d_k) × V
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # ===== 步驟 4: 合併多個頭 =====
        # 8 個頭 × 64 維 → 512 維
        # 把所有專家的意見拼在一起
        output = self.combine_heads(attn_output)  # (batch, 8, seq, 64) → (batch, seq, 512)

        # ===== 步驟 5: 最後的線性投影 =====
        # W_o 學習「如何最好地組合 8 個專家的意見」
        # 這是另一個可學習的參數
        output = self.W_o(output)  # (batch, seq, 512) → (batch, seq, 512)

        return output
