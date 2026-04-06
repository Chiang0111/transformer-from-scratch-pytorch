"""
注意力機制的單元測試

這個測試檔案專門測試 Transformer 中的注意力機制，包括：
1. Scaled Dot-Product Attention（縮放點積注意力）
2. Multi-Head Attention（多頭注意力）

測試目標：
- 驗證輸出張量的形狀（shape）是否正確
- 驗證注意力權重是否符合機率分佈（總和為 1）
- 驗證遮罩（mask）機制是否正確運作
- 驗證多頭機制的參數是否正確初始化
"""

import torch  # PyTorch 深度學習框架
import pytest  # Python 測試框架
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """
    測試縮放點積注意力（Scaled Dot-Product Attention）

    這是 Transformer 的基礎注意力機制，計算公式：
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    其中：
    - Q (Query): 查詢向量
    - K (Key): 鍵向量
    - V (Value): 值向量
    - d_k: 鍵向量的維度
    """

    def test_output_shape(self):
        """
        測試輸出形狀是否正確

        目的：確保注意力機制的輸出張量維度符合預期

        為什麼重要：
        - 形狀錯誤會導致後續層無法處理
        - 這是最基本的健全性檢查（sanity check）
        """
        # === 設定測試參數 ===
        batch_size = 2  # 批次大小：一次處理 2 個樣本
        num_heads = 4   # 注意力頭數：4 個平行的注意力機制
        seq_len = 10    # 序列長度：每個序列有 10 個詞元（token）
        d_k = 64        # 每個注意力頭的維度：64

        # === 建立隨機測試資料 ===
        # 在實際應用中，Q、K、V 來自於輸入的線性轉換
        # 這裡我們直接產生隨機張量來測試
        # 形狀：(batch_size, num_heads, seq_len, d_k)
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)  # 查詢矩陣
        K = torch.randn(batch_size, num_heads, seq_len, d_k)  # 鍵矩陣
        V = torch.randn(batch_size, num_heads, seq_len, d_k)  # 值矩陣

        # === 執行注意力計算 ===
        # scaled_dot_product_attention 會返回兩個值：
        # 1. output: 注意力加權後的輸出
        # 2. attn_weights: 注意力權重矩陣（用於可視化或除錯）
        output, attn_weights = scaled_dot_product_attention(Q, K, V)

        # === 驗證輸出形狀 ===
        # 輸出應該與 V 的形狀相同
        # 這是因為輸出是 V 的加權和，不改變維度
        assert output.shape == (batch_size, num_heads, seq_len, d_k), \
            f"輸出形狀錯誤：期望 {(batch_size, num_heads, seq_len, d_k)}，得到 {output.shape}"

        # 注意力權重的形狀應該是 (batch_size, num_heads, seq_len, seq_len)
        # 這代表每個查詢（seq_len）對所有鍵（seq_len）的注意力分數
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
            f"注意力權重形狀錯誤：期望 {(batch_size, num_heads, seq_len, seq_len)}，得到 {attn_weights.shape}"

    def test_attention_weights_sum_to_one(self):
        """
        測試注意力權重是否符合機率分佈特性（總和為 1）

        目的：驗證 softmax 函數是否正確應用

        為什麼重要：
        - 注意力權重必須是機率分佈（每個值在 0-1 之間，總和為 1）
        - 這確保輸出是值向量的凸組合（convex combination）
        - 如果總和不為 1，代表 softmax 應用在錯誤的維度上
        """
        # === 設定測試參數 ===
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64

        # === 建立測試資料 ===
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # === 執行注意力計算 ===
        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        # === 驗證機率分佈特性 ===
        # 對最後一個維度（鍵的維度）求和
        # 對於每個查詢，它對所有鍵的注意力權重總和應該為 1
        # dim=-1 表示對最後一個維度求和
        sum_weights = attn_weights.sum(dim=-1)

        # torch.allclose 檢查兩個張量是否在數值誤差範圍內相等
        # atol=1e-6 表示絕對誤差容忍度為 0.000001
        # 由於浮點數運算的精度限制，我們不能期望完全相等
        assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6), \
            "注意力權重的總和不為 1，softmax 可能應用在錯誤的維度上"

    def test_mask_works(self):
        """
        測試遮罩（mask）機制是否正確運作

        目的：驗證遮罩能夠阻止注意力分配到特定位置

        為什麼重要：
        - 遮罩用於處理填充（padding）詞元，防止模型關注無意義的內容
        - 在解碼器中，遮罩用於實現因果性（causal），防止看到未來的詞元
        - 遮罩失效會導致資訊洩漏，破壞模型的正確性
        """
        # === 設定較小的測試參數以便除錯 ===
        batch_size = 1  # 單一樣本，方便觀察
        num_heads = 1   # 單一注意力頭，方便觀察
        seq_len = 5     # 短序列，方便驗證
        d_k = 8         # 較小維度，計算更快

        # === 建立測試資料 ===
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # === 建立遮罩 ===
        # 遮罩的形狀：(batch_size, 1, seq_len, seq_len)
        # 中間的 1 是為了廣播（broadcasting）到所有注意力頭
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)

        # 設定遮罩規則：只允許關注前 3 個位置
        # mask 值為 1 表示允許關注，0 表示禁止關注
        # 在實際實作中，mask 為 0 的位置會被設為 -inf，經過 softmax 後變成 0
        mask[:, :, :, :3] = 1  # 前 3 個位置（索引 0, 1, 2）設為 1（允許）
        # 位置 3, 4 保持為 0（禁止）

        # === 執行帶遮罩的注意力計算 ===
        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # === 驗證遮罩效果 ===
        # 提取被遮罩位置（索引 3, 4）的注意力權重
        # 這些位置的權重應該接近 0
        masked_weights = attn_weights[:, :, :, 3:]

        # 驗證被遮罩的權重確實接近 0
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6), \
            "遮罩失效：被遮罩的位置仍有非零的注意力權重"


class TestMultiHeadAttention:
    """
    測試多頭注意力（Multi-Head Attention）

    多頭注意力的概念：
    - 將輸入投影到多個不同的表示空間（多個頭）
    - 每個頭獨立計算注意力
    - 最後將所有頭的輸出串接（concatenate）並投影回原始維度

    這允許模型同時關注不同位置的不同表示子空間
    """

    def test_output_shape(self):
        """
        測試多頭注意力的輸出形狀是否正確

        目的：確保多頭注意力正確地處理維度轉換

        關鍵點：
        - 輸入形狀：(batch_size, seq_len, d_model)
        - 輸出形狀：(batch_size, seq_len, d_model)
        - 雖然內部有多頭拆分，但輸出維度應與輸入相同
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10    # 序列長度
        d_model = 512   # 模型維度（Transformer 論文中的標準配置）
        num_heads = 8   # 8 個注意力頭（Transformer 論文中的標準配置）
        # 注意：d_model 必須能被 num_heads 整除
        # 每個頭的維度 d_k = d_model / num_heads = 512 / 8 = 64

        # === 建立多頭注意力模組 ===
        mha = MultiHeadAttention(d_model, num_heads)

        # === 建立輸入資料 ===
        # 在實際應用中，這是詞嵌入（word embeddings）加上位置編碼的結果
        # 形狀：(batch_size, seq_len, d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行自注意力（Self-Attention）===
        # 自注意力：Q = K = V = x
        # 這是編碼器中使用的注意力機制
        output = mha(x, x, x)

        # === 驗證輸出形狀 ===
        # 輸出應該與輸入形狀完全相同
        # 這確保多頭注意力可以堆疊（stack）使用
        assert output.shape == (batch_size, seq_len, d_model), \
            f"輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_different_sequence_lengths(self):
        """
        測試查詢（Q）和鍵值（K, V）序列長度不同的情況（交叉注意力）

        目的：驗證多頭注意力支援交叉注意力（Cross-Attention）

        應用場景：
        - 解碼器中的編碼器-解碼器注意力
        - Q 來自解碼器（目標序列）
        - K, V 來自編碼器（源序列）
        - 序列長度可以不同
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len_q = 10   # 查詢序列長度（例如：目標語言句子）
        seq_len_kv = 15  # 鍵值序列長度（例如：源語言句子）
        d_model = 256    # 使用較小的模型維度以加快測試
        num_heads = 4

        # === 建立多頭注意力模組 ===
        mha = MultiHeadAttention(d_model, num_heads)

        # === 建立不同長度的輸入 ===
        # query: 目標序列的表示
        query = torch.randn(batch_size, seq_len_q, d_model)
        # key, value: 源序列的表示
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)

        # === 執行交叉注意力 ===
        # 這模擬解碼器關注編碼器輸出的情況
        output = mha(query, key, value)

        # === 驗證輸出形狀 ===
        # 重要：輸出的序列長度應該與查詢（query）相同，而非鍵值
        # 這是因為我們為每個查詢位置計算注意力輸出
        assert output.shape == (batch_size, seq_len_q, d_model), \
            f"交叉注意力輸出形狀錯誤：應與查詢長度相同 {seq_len_q}，得到 {output.shape[1]}"

    def test_parameters_exist(self):
        """
        測試多頭注意力模組是否包含所有必要的可學習參數

        目的：驗證模組正確初始化所有權重矩陣

        多頭注意力需要的參數：
        - W_q: 查詢投影矩陣
        - W_k: 鍵投影矩陣
        - W_v: 值投影矩陣
        - W_o: 輸出投影矩陣
        每個都是線性層，包含 weight 和 bias
        """
        # === 設定測試參數 ===
        d_model = 512
        num_heads = 8

        # === 建立多頭注意力模組 ===
        mha = MultiHeadAttention(d_model, num_heads)

        # === 檢查參數數量 ===
        # 獲取所有可學習參數
        params = list(mha.parameters())

        # 預期的參數數量：
        # W_q: weight + bias = 2 個參數
        # W_k: weight + bias = 2 個參數
        # W_v: weight + bias = 2 個參數
        # W_o: weight + bias = 2 個參數
        # 總共：4 * 2 = 8 個參數
        assert len(params) == 8, \
            f"參數數量錯誤：期望 8 個（4 個線性層，每個有 weight 和 bias），得到 {len(params)} 個"

    def test_d_model_divisible_by_num_heads(self):
        """
        測試當 d_model 不能被 num_heads 整除時是否正確報錯

        目的：驗證參數驗證邏輯

        為什麼 d_model 必須能被 num_heads 整除：
        - 每個頭的維度 d_k = d_model / num_heads
        - d_k 必須是整數才能正確分配維度
        - 如果不能整除，會導致維度不匹配錯誤
        """
        # === 測試無效配置 ===
        # pytest.raises 用於驗證程式碼是否拋出預期的例外
        with pytest.raises(AssertionError):
            # 嘗試建立無效的多頭注意力
            # 512 不能被 7 整除（512 / 7 = 73.14...）
            # 這應該在初始化時就拋出 AssertionError
            MultiHeadAttention(d_model=512, num_heads=7)


if __name__ == "__main__":
    # 允許直接執行這個檔案來進行測試
    # 這對於除錯特定測試很有用
    # -v 參數：verbose 模式，顯示詳細的測試資訊
    pytest.main([__file__, "-v"])
