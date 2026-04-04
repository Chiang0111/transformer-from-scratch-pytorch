"""
位置前饋網路的單元測試

測試目標：
1. 驗證輸出形狀正確
2. 驗證維度擴展和壓縮
3. 驗證不同激活函數
4. 驗證門控版本
"""

import torch
import pytest
from transformer.feedforward import PositionwiseFeedForward, GatedFeedForward


class TestPositionwiseFeedForward:
    """測試位置前饋網路"""

    def test_output_shape(self):
        """測試輸出形狀是否正確"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        ffn = PositionwiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        # 輸出形狀應該與輸入相同
        assert output.shape == (batch_size, seq_len, d_model)

    def test_relu_activation(self):
        """測試 ReLU 激活函數"""
        d_model = 256
        d_ff = 1024

        ffn = PositionwiseFeedForward(d_model, d_ff, activation='relu')
        x = torch.randn(1, 5, d_model)

        output = ffn(x)
        assert output.shape == (1, 5, d_model)

    def test_gelu_activation(self):
        """測試 GELU 激活函數"""
        d_model = 256
        d_ff = 1024

        ffn = PositionwiseFeedForward(d_model, d_ff, activation='gelu')
        x = torch.randn(1, 5, d_model)

        output = ffn(x)
        assert output.shape == (1, 5, d_model)

    def test_invalid_activation(self):
        """測試無效的激活函數應該報錯"""
        with pytest.raises(ValueError):
            PositionwiseFeedForward(d_model=512, d_ff=2048, activation='sigmoid')

    def test_parameters_count(self):
        """測試參數數量是否正確"""
        d_model = 512
        d_ff = 2048

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # 計算預期的參數量
        # Linear1: d_model * d_ff + d_ff (weight + bias)
        # Linear2: d_ff * d_model + d_model (weight + bias)
        expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

        total_params = sum(p.numel() for p in ffn.parameters())
        assert total_params == expected_params

    def test_dropout_in_training_mode(self):
        """測試訓練模式下 dropout 是否工作"""
        d_model = 256
        d_ff = 1024
        dropout = 0.5

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        ffn.train()  # 設為訓練模式

        x = torch.randn(1, 10, d_model)

        # 執行多次，結果應該不同（因為 dropout）
        output1 = ffn(x)
        output2 = ffn(x)

        # 由於 dropout，兩次結果應該不完全相同
        assert not torch.allclose(output1, output2)

    def test_no_dropout_in_eval_mode(self):
        """測試評估模式下 dropout 應該關閉"""
        d_model = 256
        d_ff = 1024
        dropout = 0.5

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        ffn.eval()  # 設為評估模式

        x = torch.randn(1, 10, d_model)

        # 執行多次，結果應該相同（因為 dropout 關閉）
        output1 = ffn(x)
        output2 = ffn(x)

        # 評估模式下，兩次結果應該完全相同
        assert torch.allclose(output1, output2)

    def test_variable_sequence_lengths(self):
        """測試不同序列長度"""
        d_model = 256
        d_ff = 1024
        batch_size = 2

        ffn = PositionwiseFeedForward(d_model, d_ff)

        # 測試不同的序列長度
        for seq_len in [5, 10, 50, 100]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            assert output.shape == (batch_size, seq_len, d_model)


class TestGatedFeedForward:
    """測試門控前饋網路"""

    def test_output_shape(self):
        """測試輸出形狀是否正確"""
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        gated_ffn = GatedFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = gated_ffn(x)

        # 輸出形狀應該與輸入相同
        assert output.shape == (batch_size, seq_len, d_model)

    def test_parameters_count(self):
        """測試門控版本的參數數量（應該約是標準版本的 2 倍）"""
        d_model = 512
        d_ff = 2048

        standard_ffn = PositionwiseFeedForward(d_model, d_ff)
        gated_ffn = GatedFeedForward(d_model, d_ff)

        standard_params = sum(p.numel() for p in standard_ffn.parameters())
        gated_params = sum(p.numel() for p in gated_ffn.parameters())

        # 門控版本有兩個並行的 linear1，所以參數量應該更多
        assert gated_params > standard_params

        # 具體來說，應該多一個 d_model * d_ff + d_ff
        expected_diff = d_model * d_ff + d_ff
        assert gated_params - standard_params == expected_diff

    def test_gating_mechanism(self):
        """測試門控機制是否工作"""
        d_model = 256
        d_ff = 1024

        gated_ffn = GatedFeedForward(d_model, d_ff)
        x = torch.randn(1, 10, d_model)

        # 只是確保能正常執行
        output = gated_ffn(x)
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
