"""
Attention 機制的單元測試

測試目標：
1. 驗證輸出的 shape 是否正確
2. 驗證注意力權重是否加總為 1
3. 驗證 mask 是否正常工作
"""

import torch
import pytest
from transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """測試 Scaled Dot-Product Attention"""

    def test_output_shape(self):
        """測試輸出形狀是否正確"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64

        # 建立隨機的 Q, K, V
        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # 執行 attention
        output, attn_weights = scaled_dot_product_attention(Q, K, V)

        # 驗證 output shape
        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        # 驗證 attention weights shape
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """測試注意力權重是否加總為 1（機率分佈的特性）"""
        batch_size = 2
        num_heads = 4
        seq_len = 10
        d_k = 64

        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        _, attn_weights = scaled_dot_product_attention(Q, K, V)

        # 檢查每一行（每個 query 對所有 keys 的注意力）加總是否為 1
        sum_weights = attn_weights.sum(dim=-1)
        assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-6)

    def test_mask_works(self):
        """測試 mask 是否正常工作"""
        batch_size = 1
        num_heads = 1
        seq_len = 5
        d_k = 8

        Q = torch.randn(batch_size, num_heads, seq_len, d_k)
        K = torch.randn(batch_size, num_heads, seq_len, d_k)
        V = torch.randn(batch_size, num_heads, seq_len, d_k)

        # 建立一個 mask：只允許 attend 到前 3 個位置
        mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, :3] = 1  # 前 3 個位置為 1（允許）

        _, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 檢查被 mask 的位置（後 2 個）注意力權重是否接近 0
        masked_weights = attn_weights[:, :, :, 3:]
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)


class TestMultiHeadAttention:
    """測試 Multi-Head Attention"""

    def test_output_shape(self):
        """測試輸出形狀是否正確"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8

        # 建立 multi-head attention 模組
        mha = MultiHeadAttention(d_model, num_heads)

        # 建立輸入
        x = torch.randn(batch_size, seq_len, d_model)

        # Self-attention（Q=K=V）
        output = mha(x, x, x)

        # 驗證輸出形狀
        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_sequence_lengths(self):
        """測試 Q 和 K,V 的序列長度不同的情況（Cross-attention）"""
        batch_size = 2
        seq_len_q = 10
        seq_len_kv = 15
        d_model = 256
        num_heads = 4

        mha = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)

        output = mha(query, key, value)

        # 輸出的序列長度應該跟 query 一樣
        assert output.shape == (batch_size, seq_len_q, d_model)

    def test_parameters_exist(self):
        """測試模組是否有正確的可學習參數"""
        d_model = 512
        num_heads = 8

        mha = MultiHeadAttention(d_model, num_heads)

        # 檢查是否有 4 個線性層（W_q, W_k, W_v, W_o）
        params = list(mha.parameters())
        # 每個線性層有 weight 和 bias，所以總共 8 個參數
        assert len(params) == 8

    def test_d_model_divisible_by_num_heads(self):
        """測試 d_model 不能被 num_heads 整除時是否會報錯"""
        with pytest.raises(AssertionError):
            # 512 不能被 7 整除，應該報錯
            MultiHeadAttention(d_model=512, num_heads=7)


if __name__ == "__main__":
    # 可以直接執行這個檔案來測試
    pytest.main([__file__, "-v"])
