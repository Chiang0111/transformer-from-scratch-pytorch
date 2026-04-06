"""
位置前饋網路的單元測試

這個測試檔案專門測試 Transformer 中的前饋神經網路（Feed-Forward Network）

前饋網路的結構：
- 標準版本：Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
- 門控版本：使用門控機制（Gated Linear Unit），類似 GLU

在 Transformer 中的角色：
- 每個編碼器和解碼器層都包含一個前饋網路
- 在注意力層之後應用
- 為模型添加非線性轉換能力
- d_ff 通常是 d_model 的 4 倍（如 d_model=512, d_ff=2048）

測試目標：
1. 驗證輸出形狀正確（不改變序列長度和模型維度）
2. 驗證不同激活函數（ReLU、GELU）都能正常運作
3. 驗證 dropout 在訓練和評估模式下的不同行為
4. 驗證門控版本的實作正確
"""

import torch  # PyTorch 深度學習框架
import pytest  # Python 測試框架
from transformer.feedforward import PositionwiseFeedForward, GatedFeedForward


class TestPositionwiseFeedForward:
    """
    測試標準的位置前饋網路（Position-wise Feed-Forward Network）

    「位置」（Position-wise）的意思：
    - 對序列中的每個位置獨立應用相同的前饋網路
    - 不同位置之間不共享資訊（這部分由注意力層處理）
    - 參數在所有位置之間共享
    """

    def test_output_shape(self):
        """
        測試輸出形狀是否正確

        目的：確保前饋網路不改變張量的整體結構

        為什麼重要：
        - 前饋網路應該只是「內部擴展再壓縮」
        - 輸入和輸出維度必須相同才能使用殘差連接（residual connection）
        - 序列長度不應該改變
        """
        # === 設定測試參數 ===
        batch_size = 2     # 批次大小
        seq_len = 10       # 序列長度
        d_model = 512      # 模型維度（輸入和輸出維度）
        d_ff = 2048        # 前饋網路的隱藏層維度（通常是 d_model 的 4 倍）

        # === 建立前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff)

        # === 建立輸入張量 ===
        # 形狀：(batch_size, seq_len, d_model)
        # 這是注意力層的輸出
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行前向傳播 ===
        # 內部流程：
        # 1. x: (batch, seq, 512) → Linear1 → (batch, seq, 2048)
        # 2. 應用激活函數（ReLU 或 GELU）
        # 3. 應用 dropout
        # 4. (batch, seq, 2048) → Linear2 → (batch, seq, 512)
        output = ffn(x)

        # === 驗證輸出形狀 ===
        # 輸出形狀必須與輸入完全相同
        assert output.shape == (batch_size, seq_len, d_model), \
            f"前饋網路改變了張量形狀：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_relu_activation(self):
        """
        測試使用 ReLU 激活函數是否正常運作

        ReLU (Rectified Linear Unit)：
        - 公式：ReLU(x) = max(0, x)
        - 特點：簡單、快速、有效
        - 原始 Transformer 論文使用的激活函數
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024

        # === 建立使用 ReLU 的前饋網路 ===
        # activation='relu' 明確指定使用 ReLU
        ffn = PositionwiseFeedForward(d_model, d_ff, activation='relu')

        # === 建立測試輸入 ===
        x = torch.randn(1, 5, d_model)

        # === 執行前向傳播 ===
        output = ffn(x)

        # === 驗證輸出形狀 ===
        assert output.shape == (1, 5, d_model), \
            "ReLU 激活函數導致輸出形狀錯誤"

    def test_gelu_activation(self):
        """
        測試使用 GELU 激活函數是否正常運作

        GELU (Gaussian Error Linear Unit)：
        - 公式：GELU(x) = x * Φ(x)，其中 Φ 是標準常態分佈的累積分佈函數
        - 特點：平滑的非線性，在許多現代模型中表現更好
        - 比 ReLU 更複雜但效果通常更好
        - 在 BERT、GPT 等模型中廣泛使用
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024

        # === 建立使用 GELU 的前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff, activation='gelu')

        # === 建立測試輸入 ===
        x = torch.randn(1, 5, d_model)

        # === 執行前向傳播 ===
        output = ffn(x)

        # === 驗證輸出形狀 ===
        assert output.shape == (1, 5, d_model), \
            "GELU 激活函數導致輸出形狀錯誤"

    def test_invalid_activation(self):
        """
        測試使用無效的激活函數名稱時是否正確報錯

        目的：驗證輸入驗證邏輯

        為什麼重要：
        - 使用者可能拼錯激活函數名稱
        - 應該在初始化時就發現錯誤，而非在訓練時
        - 明確的錯誤訊息有助於除錯
        """
        # === 測試無效的激活函數 ===
        # pytest.raises 用於驗證程式碼是否拋出預期的例外
        with pytest.raises(ValueError):
            # 嘗試使用不支援的激活函數
            # 'sigmoid' 不在支援的激活函數清單中
            # 這應該拋出 ValueError
            PositionwiseFeedForward(d_model=512, d_ff=2048, activation='sigmoid')

    def test_parameters_count(self):
        """
        測試參數數量是否正確

        目的：驗證網路結構是否如預期建立

        參數計算：
        - Linear1: (d_model × d_ff) 個權重 + d_ff 個偏置
        - Linear2: (d_ff × d_model) 個權重 + d_model 個偏置
        - 總計：d_model×d_ff + d_ff + d_ff×d_model + d_model
        """
        # === 設定測試參數 ===
        d_model = 512
        d_ff = 2048

        # === 建立前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff)

        # === 計算預期的參數量 ===
        # Linear1 的參數：
        # - 權重矩陣：d_model × d_ff = 512 × 2048 = 1,048,576
        # - 偏置向量：d_ff = 2048
        linear1_params = (d_model * d_ff) + d_ff

        # Linear2 的參數：
        # - 權重矩陣：d_ff × d_model = 2048 × 512 = 1,048,576
        # - 偏置向量：d_model = 512
        linear2_params = (d_ff * d_model) + d_model

        # 總參數量
        expected_params = linear1_params + linear2_params

        # === 計算實際參數量 ===
        # p.numel() 返回參數張量中的元素總數
        total_params = sum(p.numel() for p in ffn.parameters())

        # === 驗證參數量 ===
        assert total_params == expected_params, \
            f"參數量不正確：期望 {expected_params}，得到 {total_params}"

    def test_dropout_in_training_mode(self):
        """
        測試訓練模式下 dropout 是否正常運作

        Dropout 的作用：
        - 訓練時隨機將一部分神經元輸出設為 0
        - 防止過度擬合（overfitting）
        - 相當於訓練多個子網路的集成（ensemble）

        驗證策略：
        - 在訓練模式下，相同輸入應該產生不同輸出（因為 dropout 的隨機性）
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024
        dropout = 0.5  # 50% 的 dropout 率，確保能觀察到差異

        # === 建立前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        # === 設定為訓練模式 ===
        # 在訓練模式下，dropout 會被啟用
        ffn.train()

        # === 建立測試輸入 ===
        x = torch.randn(1, 10, d_model)

        # === 執行兩次前向傳播 ===
        # 由於 dropout 的隨機性，兩次結果應該不同
        output1 = ffn(x)
        output2 = ffn(x)

        # === 驗證兩次輸出不同 ===
        # 如果輸出完全相同，代表 dropout 沒有啟用
        assert not torch.allclose(output1, output2), \
            "訓練模式下 dropout 沒有啟用，兩次前向傳播的結果完全相同"

    def test_no_dropout_in_eval_mode(self):
        """
        測試評估模式下 dropout 是否被正確關閉

        為什麼評估時要關閉 dropout：
        - 推理時我們希望結果是確定性的
        - 評估模式使用所有神經元（相當於取平均）
        - 確保測試結果的可重現性

        驗證策略：
        - 在評估模式下，相同輸入應該產生相同輸出
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024
        dropout = 0.5  # 即使設定了 dropout，評估模式下也應該被關閉

        # === 建立前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        # === 設定為評估模式 ===
        # 在評估模式下，dropout 會被停用
        ffn.eval()

        # === 建立測試輸入 ===
        x = torch.randn(1, 10, d_model)

        # === 執行兩次前向傳播 ===
        # 評估模式下，結果應該完全相同
        output1 = ffn(x)
        output2 = ffn(x)

        # === 驗證兩次輸出相同 ===
        # 如果輸出不同，代表 dropout 沒有被正確關閉
        assert torch.allclose(output1, output2), \
            "評估模式下 dropout 沒有被關閉，兩次前向傳播的結果不同"

    def test_variable_sequence_lengths(self):
        """
        測試是否支援不同長度的序列

        目的：確保前饋網路可以處理任意長度的輸入

        為什麼重要：
        - 實際應用中序列長度是可變的
        - 前饋網路應該獨立處理每個位置，不依賴序列長度
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024
        batch_size = 2

        # === 建立前饋網路 ===
        ffn = PositionwiseFeedForward(d_model, d_ff)

        # === 測試多種序列長度 ===
        for seq_len in [5, 10, 50, 100]:
            # 建立該長度的輸入
            x = torch.randn(batch_size, seq_len, d_model)

            # 執行前向傳播
            output = ffn(x)

            # 驗證輸出形狀正確
            assert output.shape == (batch_size, seq_len, d_model), \
                f"序列長度 {seq_len} 時輸出形狀錯誤"


class TestGatedFeedForward:
    """
    測試門控前饋網路（Gated Feed-Forward Network）

    門控機制的概念：
    - 使用兩個平行的線性轉換
    - 一個分支提供主要資訊
    - 另一個分支提供「門控」信號，控制資訊流動
    - 公式：output = Linear2(GELU(Linear1a(x)) ⊙ Linear1b(x))
    - ⊙ 表示逐元素相乘（element-wise multiplication）

    優勢：
    - 更強的表達能力
    - 更好的控制資訊流動
    - 在某些任務上效果更好

    代價：
    - 參數量增加（約多一倍）
    - 計算量增加
    """

    def test_output_shape(self):
        """
        測試門控前饋網路的輸出形狀是否正確

        門控機制不應該改變輸入輸出的維度關係
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 512
        d_ff = 2048

        # === 建立門控前饋網路 ===
        gated_ffn = GatedFeedForward(d_model, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行前向傳播 ===
        output = gated_ffn(x)

        # === 驗證輸出形狀 ===
        # 輸出形狀應該與輸入相同
        assert output.shape == (batch_size, seq_len, d_model), \
            f"門控前饋網路輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_parameters_count(self):
        """
        測試門控版本的參數量是否約為標準版本的 2 倍

        為什麼參數會增加：
        - 標準版本：1 個 Linear1
        - 門控版本：2 個平行的 Linear1（一個用於資訊，一個用於門控）
        - Linear2 是相同的
        - 所以參數量應該多出一個 (d_model × d_ff + d_ff)
        """
        # === 設定測試參數 ===
        d_model = 512
        d_ff = 2048

        # === 建立兩種版本的前饋網路 ===
        standard_ffn = PositionwiseFeedForward(d_model, d_ff)
        gated_ffn = GatedFeedForward(d_model, d_ff)

        # === 計算參數量 ===
        standard_params = sum(p.numel() for p in standard_ffn.parameters())
        gated_params = sum(p.numel() for p in gated_ffn.parameters())

        # === 驗證參數量關係 ===
        # 門控版本應該比標準版本多
        assert gated_params > standard_params, \
            "門控版本的參數量應該比標準版本多"

        # 計算預期的差異
        # 門控版本多了一個 Linear1，參數量為 d_model × d_ff + d_ff
        expected_diff = d_model * d_ff + d_ff

        # 驗證差異是否符合預期
        actual_diff = gated_params - standard_params
        assert actual_diff == expected_diff, \
            f"參數量差異不正確：期望 {expected_diff}，得到 {actual_diff}"

    def test_gating_mechanism(self):
        """
        測試門控機制是否正常運作

        這是一個基本的煙霧測試（smoke test）：
        - 只確保程式碼能夠執行
        - 不驗證具體的數學正確性（那需要更複雜的測試）
        """
        # === 設定測試參數 ===
        d_model = 256
        d_ff = 1024

        # === 建立門控前饋網路 ===
        gated_ffn = GatedFeedForward(d_model, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(1, 10, d_model)

        # === 執行前向傳播 ===
        output = gated_ffn(x)

        # === 驗證輸出形狀 ===
        # 只要能正確執行並返回正確形狀，就算通過
        assert output.shape == x.shape, \
            "門控前饋網路無法正常執行"


if __name__ == "__main__":
    # 允許直接執行此檔案進行測試
    pytest.main([__file__, "-v"])
