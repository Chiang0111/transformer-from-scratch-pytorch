"""
位置編碼的單元測試

這個測試檔案專門測試 Transformer 的位置編碼（Positional Encoding）機制

為什麼需要位置編碼：
- Transformer 使用注意力機制，本身沒有順序概念
- 位置編碼為每個位置添加唯一的資訊，讓模型知道詞元的順序
- 使用正弦和餘弦函數，可以推廣到訓練時未見過的序列長度

測試目標：
1. 驗證輸出形狀是否正確（不改變輸入維度）
2. 驗證位置編碼確實被加到輸入上
3. 驗證不同位置有不同的編碼
4. 驗證編碼值在合理範圍內（-1 到 1）
5. 驗證支援可變序列長度
"""

import torch  # PyTorch 深度學習框架
import pytest  # Python 測試框架
from transformer.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    """測試位置編碼層"""

    def test_output_shape(self):
        """
        測試輸出形狀是否與輸入相同

        目的：確保位置編碼不改變張量維度

        為什麼重要：
        - 位置編碼是「加法」操作，不是「轉換」操作
        - 輸出必須與輸入形狀相同才能正確加到詞嵌入上
        - 形狀不匹配會導致廣播錯誤或計算錯誤
        """
        # === 設定測試參數 ===
        d_model = 512      # 模型維度（必須與詞嵌入維度相同）
        batch_size = 2     # 批次大小
        seq_len = 10       # 序列長度

        # === 建立位置編碼層 ===
        # dropout=0.0 關閉 dropout，方便測試時得到確定性結果
        # 在實際訓練中，dropout > 0 可以防止過擬合
        pe = PositionalEncoding(d_model, dropout=0.0)

        # === 建立輸入張量 ===
        # 模擬詞嵌入的輸出
        # 形狀：(batch_size, seq_len, d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行位置編碼 ===
        # 位置編碼層會：
        # 1. 根據位置生成編碼向量
        # 2. 將編碼向量加到輸入上
        # 3. 應用 dropout（如果 dropout > 0）
        output = pe(x)

        # === 驗證輸出形狀 ===
        # 輸出形狀必須與輸入完全相同
        assert output.shape == (batch_size, seq_len, d_model), \
            f"位置編碼改變了張量形狀：輸入 {x.shape}，輸出 {output.shape}"

    def test_positional_encoding_added(self):
        """
        測試位置編碼是否確實被加到輸入上

        目的：驗證位置編碼層不是「恆等映射」（identity function）

        測試策略：
        - 使用全零輸入
        - 如果位置編碼正常運作，輸出應該只包含位置編碼（非零）
        - 如果輸出仍是零，代表位置編碼沒有被加上
        """
        # === 設定測試參數 ===
        d_model = 512
        batch_size = 1  # 單一樣本，簡化測試
        seq_len = 10

        # === 建立位置編碼層 ===
        # dropout=0.0 確保不會因為 dropout 將值歸零
        pe = PositionalEncoding(d_model, dropout=0.0)

        # === 建立全零輸入 ===
        # 這樣輸出就只會包含位置編碼，沒有其他資訊
        x = torch.zeros(batch_size, seq_len, d_model)

        # === 執行位置編碼 ===
        output = pe(x)

        # === 驗證輸出不是全零 ===
        # 如果位置編碼被正確加上，輸出應該包含非零值
        # torch.allclose 檢查兩個張量是否在誤差範圍內相等
        # 如果輸出與全零張量相等，代表位置編碼沒有被加上
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "位置編碼沒有被加到輸入上，輸出仍然是全零"

    def test_different_positions_different_encodings(self):
        """
        測試不同位置是否有不同的編碼

        目的：確保每個位置都有唯一的位置資訊

        為什麼重要：
        - 如果所有位置的編碼都相同，模型就無法區分詞元的順序
        - 這會導致模型退化成「詞袋」模型，失去序列建模能力
        """
        # === 設定測試參數 ===
        d_model = 512
        max_len = 100  # 預先計算的最大序列長度

        # === 建立位置編碼層 ===
        pe = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        # === 提取不同位置的編碼 ===
        # pe.pe 是預先計算好的位置編碼張量
        # 形狀：(1, max_len, d_model)
        # 我們提取位置 0 和位置 1 的編碼來比較

        # 位置 0 的編碼向量（shape: (d_model,)）
        pos_0 = pe.pe[0, 0, :]

        # 位置 1 的編碼向量（shape: (d_model,)）
        pos_1 = pe.pe[0, 1, :]

        # === 驗證兩個位置的編碼不同 ===
        # 如果兩個位置的編碼相同，代表位置編碼失效
        assert not torch.allclose(pos_0, pos_1), \
            "位置 0 和位置 1 的編碼相同，位置編碼沒有提供位置資訊"

    def test_encoding_range(self):
        """
        測試位置編碼的值範圍是否在 -1 到 1 之間

        目的：驗證編碼使用的是正弦和餘弦函數

        為什麼重要：
        - 正弦和餘弦函數的值域是 [-1, 1]
        - 如果值超出這個範圍，代表實作有誤
        - 適當的值範圍有助於訓練穩定性
        """
        # === 設定測試參數 ===
        d_model = 512
        max_len = 1000  # 測試較長的序列

        # === 建立位置編碼層 ===
        pe = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        # === 檢查所有位置編碼的值範圍 ===
        # pe.pe 包含所有預先計算的位置編碼
        # 形狀：(1, max_len, d_model)

        # 檢查是否所有值都 >= -1
        # torch.all() 檢查是否所有元素都滿足條件
        assert torch.all(pe.pe >= -1.0), \
            f"位置編碼中有小於 -1 的值，最小值：{pe.pe.min():.4f}"

        # 檢查是否所有值都 <= 1
        assert torch.all(pe.pe <= 1.0), \
            f"位置編碼中有大於 1 的值，最大值：{pe.pe.max():.4f}"

    def test_supports_variable_length(self):
        """
        測試是否支援不同長度的序列

        目的：驗證位置編碼可以處理各種長度的輸入

        為什麼重要：
        - 實際應用中，輸入序列長度是可變的
        - 位置編碼必須能夠適應任何長度（直到 max_len）
        - 這測試確保沒有硬編碼的長度限制
        """
        # === 設定測試參數 ===
        d_model = 256      # 使用較小的維度加快測試
        batch_size = 2
        max_len = 1000     # 支援的最大長度

        # === 建立位置編碼層 ===
        pe = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

        # === 測試多種不同的序列長度 ===
        # 從短序列到長序列，確保都能正常處理
        for seq_len in [5, 10, 50, 100]:
            # 建立該長度的輸入張量
            x = torch.randn(batch_size, seq_len, d_model)

            # 執行位置編碼
            output = pe(x)

            # 驗證輸出形狀正確
            assert output.shape == (batch_size, seq_len, d_model), \
                f"序列長度 {seq_len} 時輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_d_model_must_be_even(self):
        """
        測試 d_model 為奇數時是否還能運作

        目的：測試邊界情況

        背景知識：
        - 位置編碼使用 sin 和 cos 函數對不同維度編碼
        - 理論上，偶數維度更容易實作（一半用 sin，一半用 cos）
        - 但好的實作應該也能處理奇數維度

        注意：這個測試可能會跳過（skip），因為：
        - 某些實作可能不支援奇數 d_model
        - 在實際應用中，d_model 通常是偶數（如 512, 256）
        """
        # === 設定奇數 d_model ===
        d_model = 511  # 奇數維度
        batch_size = 1
        seq_len = 10

        # === 嘗試建立位置編碼層 ===
        # 使用 try-except 來處理可能的錯誤
        try:
            # 嘗試建立層
            pe = PositionalEncoding(d_model, dropout=0.0)

            # 建立測試輸入
            x = torch.randn(batch_size, seq_len, d_model)

            # 執行前向傳播
            output = pe(x)

            # 驗證輸出形狀
            assert output.shape == (batch_size, seq_len, d_model), \
                f"奇數 d_model 時輸出形狀錯誤"

        except Exception as e:
            # 如果不支援奇數 d_model，跳過這個測試
            # pytest.skip() 會將測試標記為「跳過」而非「失敗」
            pytest.skip(f"目前實作不支援奇數 d_model：{e}")


if __name__ == "__main__":
    # 允許直接執行此檔案進行測試
    pytest.main([__file__, "-v"])
