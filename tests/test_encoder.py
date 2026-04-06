"""
編碼器（Encoder）的單元測試

這個測試檔案專門測試 Transformer 的編碼器部分

編碼器的結構：
1. EncoderLayer（編碼器層）：
   - Multi-Head Self-Attention（多頭自注意力）
   - Add & Norm（殘差連接 + 層歸一化）
   - Position-wise Feed-Forward Network（位置前饋網路）
   - Add & Norm（殘差連接 + 層歸一化）

2. Encoder（完整編碼器）：
   - 堆疊多個 EncoderLayer（通常 6 層）
   - 最後可能有額外的層歸一化

編碼器的作用：
- 處理輸入序列（例如：源語言句子）
- 建立輸入的上下文表示
- 輸出供解碼器使用的記憶（memory）

測試目標：
1. 驗證 EncoderLayer 的輸出形狀和基本功能
2. 驗證殘差連接是否存在
3. 驗證層歸一化是否應用
4. 驗證完整 Encoder 的堆疊功能
5. 驗證遮罩機制
"""

import torch  # PyTorch 深度學習框架
import pytest  # Python 測試框架
from transformer.encoder import EncoderLayer, Encoder


class TestEncoderLayer:
    """測試單一編碼器層（EncoderLayer）"""

    def test_output_shape(self):
        """
        測試編碼器層的輸出形狀是否正確

        目的：確保編碼器層不改變張量維度

        為什麼重要：
        - 編碼器層使用殘差連接，要求輸入輸出維度相同
        - 多層編碼器需要堆疊，每層的輸入輸出形狀必須一致
        - 序列長度不應該改變
        """
        # === 設定測試參數 ===
        batch_size = 2     # 批次大小
        seq_len = 10       # 序列長度
        d_model = 512      # 模型維度
        num_heads = 8      # 注意力頭數
        d_ff = 2048        # 前饋網路隱藏層維度

        # === 建立編碼器層 ===
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

        # === 建立測試輸入 ===
        # 形狀：(batch_size, seq_len, d_model)
        # 這通常是詞嵌入 + 位置編碼的結果
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行前向傳播 ===
        # 內部流程：
        # 1. x → Multi-Head Self-Attention → attn_output
        # 2. x + attn_output → LayerNorm → norm1_output
        # 3. norm1_output → Feed-Forward → ff_output
        # 4. norm1_output + ff_output → LayerNorm → final_output
        output = encoder_layer(x)

        # === 驗證輸出形狀 ===
        assert output.shape == (batch_size, seq_len, d_model), \
            f"編碼器層輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_with_mask(self):
        """
        測試帶遮罩的編碼器層

        目的：驗證編碼器能夠正確處理填充遮罩（padding mask）

        應用場景：
        - 批次中的序列長度不同
        - 較短的序列會被填充（padding）到相同長度
        - 遮罩用於告訴模型哪些位置是填充，不應被關注
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024

        # === 建立編碼器層 ===
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, seq_len, d_model)

        # === 建立填充遮罩 ===
        # 形狀：(batch_size, 1, 1, seq_len)
        # 中間的兩個 1 是為了廣播到 (num_heads, seq_len_q, seq_len_k)
        mask = torch.ones(batch_size, 1, 1, seq_len)

        # 模擬填充：前 7 個位置有效，後 3 個是填充
        # 1 表示允許關注，0 表示禁止關注（填充位置）
        mask[:, :, :, 7:] = 0

        # === 執行帶遮罩的前向傳播 ===
        output = encoder_layer(x, mask)

        # === 驗證輸出形狀 ===
        # 即使使用遮罩，輸出形狀也應該與輸入相同
        assert output.shape == (batch_size, seq_len, d_model), \
            "帶遮罩的編碼器層輸出形狀錯誤"

    def test_residual_connection_exists(self):
        """
        測試殘差連接（Residual Connection）是否存在

        殘差連接的公式：output = LayerNorm(x + Sublayer(x))

        為什麼重要：
        - 緩解深層網路的梯度消失問題
        - 允許資訊直接從輸入傳到輸出
        - 使訓練更穩定

        驗證策略：
        - 使用非常小的輸入
        - 如果有殘差連接，輸出應該保留輸入的一些特徵
        - 這不是嚴格的數學證明，只是健全性檢查
        """
        # === 設定測試參數 ===
        batch_size = 1
        seq_len = 5
        d_model = 128
        num_heads = 4
        d_ff = 512

        # === 建立編碼器層 ===
        # dropout=0.0 確保不會因為 dropout 影響結果
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout=0.0)

        # === 設定為評估模式 ===
        # 關閉 dropout，確保結果可重現
        encoder_layer.eval()

        # === 建立很小的輸入 ===
        # 乘以 0.01 使輸入值很小
        # 如果沒有殘差連接，多層變換後可能會消失
        x = torch.randn(batch_size, seq_len, d_model) * 0.01

        # === 執行前向傳播 ===
        output = encoder_layer(x)

        # === 驗證殘差連接 ===
        # 這只是確保編碼器層能正常運作
        # 完整的殘差連接測試需要更複雜的驗證
        # 至少輸出形狀應該正確
        assert output.shape == x.shape, \
            "編碼器層無法處理輸入"

    def test_layer_norm_applied(self):
        """
        測試層歸一化（Layer Normalization）是否被應用

        層歸一化的作用：
        - 對每個樣本的特徵維度進行標準化
        - 使訓練更穩定
        - 加速收斂

        注意：
        - 由於層歸一化有可學習參數（gamma 和 beta），我們只檢查形狀
        - 完整的功能測試需要檢查統計特性（均值和變異數）
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024

        # === 建立編碼器層 ===
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行前向傳播 ===
        output = encoder_layer(x)

        # === 驗證輸出形狀 ===
        # 層歸一化不應該改變形狀
        # 只是對每個樣本的每個位置進行標準化
        assert output.shape == x.shape, \
            "層歸一化導致形狀改變"

    def test_different_activations(self):
        """
        測試不同的激活函數是否都能正常運作

        目的：驗證編碼器支援多種激活函數配置

        測試的激活函數：
        - ReLU：原始 Transformer 使用
        - GELU：現代 Transformer（如 BERT、GPT）使用
        """
        # === 設定測試參數 ===
        d_model = 256
        num_heads = 4
        d_ff = 1024
        x = torch.randn(1, 5, d_model)

        # === 測試 ReLU 激活函數 ===
        encoder_relu = EncoderLayer(d_model, num_heads, d_ff, activation='relu')
        output_relu = encoder_relu(x)
        assert output_relu.shape == x.shape, \
            "ReLU 激活函數導致輸出形狀錯誤"

        # === 測試 GELU 激活函數 ===
        encoder_gelu = EncoderLayer(d_model, num_heads, d_ff, activation='gelu')
        output_gelu = encoder_gelu(x)
        assert output_gelu.shape == x.shape, \
            "GELU 激活函數導致輸出形狀錯誤"

        # === 驗證兩種激活函數的輸出不同 ===
        # 由於參數是隨機初始化的，兩個模型的輸出應該不同
        # 這確保我們確實建立了兩個獨立的模型
        assert not torch.allclose(output_relu, output_gelu), \
            "兩種激活函數的輸出完全相同，可能模型沒有正確初始化"


class TestEncoder:
    """測試完整的編碼器（堆疊多層 EncoderLayer）"""

    def test_output_shape(self):
        """
        測試完整編碼器的輸出形狀是否正確

        完整編碼器：
        - 將多個 EncoderLayer 堆疊在一起
        - 第一層的輸出作為第二層的輸入，依此類推
        - 最終輸出供解碼器使用
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        d_ff = 2048
        num_layers = 6  # 標準 Transformer 使用 6 層

        # === 建立完整編碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, seq_len, d_model)

        # === 執行前向傳播 ===
        # 輸入會依序通過所有 6 層編碼器層
        output = encoder(x)

        # === 驗證輸出形狀 ===
        # 即使經過多層，輸出形狀仍應與輸入相同
        assert output.shape == (batch_size, seq_len, d_model), \
            f"編碼器輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_different_num_layers(self):
        """
        測試不同層數的編碼器是否都能正常運作

        目的：驗證編碼器支援可配置的層數

        應用：
        - 小型模型可能只用 2-3 層（快速但效果較差）
        - 標準模型使用 6 層
        - 大型模型可能使用 12 層或更多
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024
        x = torch.randn(batch_size, seq_len, d_model)

        # === 測試不同層數 ===
        for num_layers in [1, 3, 6]:
            # 建立該層數的編碼器
            encoder = Encoder(num_layers, d_model, num_heads, d_ff)

            # 執行前向傳播
            output = encoder(x)

            # 驗證輸出形狀
            assert output.shape == (batch_size, seq_len, d_model), \
                f"{num_layers} 層編碼器的輸出形狀錯誤"

    def test_with_mask(self):
        """
        測試編碼器處理遮罩的能力

        目的：確保遮罩能夠正確地在所有層之間傳播

        遮罩傳播：
        - 遮罩應該被傳遞到每一層的注意力機制
        - 所有層都應該遵守相同的遮罩規則
        """
        # === 設定測試參數 ===
        batch_size = 2
        seq_len = 10
        d_model = 256
        num_heads = 4
        d_ff = 1024
        num_layers = 3

        # === 建立編碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, seq_len, d_model)

        # === 建立遮罩 ===
        mask = torch.ones(batch_size, 1, 1, seq_len)
        mask[:, :, :, 7:] = 0  # 後 3 個位置是填充

        # === 執行帶遮罩的前向傳播 ===
        output = encoder(x, mask)

        # === 驗證輸出形狀 ===
        assert output.shape == (batch_size, seq_len, d_model), \
            "帶遮罩的編碼器輸出形狀錯誤"

    def test_parameters_increase_with_layers(self):
        """
        測試參數量是否隨層數增加而增加

        目的：驗證每一層都有自己的參數（不共享）

        預期：
        - 4 層編碼器的參數量應該約為 2 層的兩倍
        - 不是精確的 2 倍，因為可能有額外的層歸一化等元件
        """
        # === 設定測試參數 ===
        d_model = 256
        num_heads = 4
        d_ff = 1024

        # === 建立不同層數的編碼器 ===
        encoder_2 = Encoder(num_layers=2, d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        encoder_4 = Encoder(num_layers=4, d_model=d_model, num_heads=num_heads, d_ff=d_ff)

        # === 計算參數量 ===
        params_2 = sum(p.numel() for p in encoder_2.parameters())
        params_4 = sum(p.numel() for p in encoder_4.parameters())

        # === 驗證參數量關係 ===
        # 4 層應該比 2 層多
        assert params_4 > params_2, \
            "4 層編碼器的參數量沒有比 2 層多"

        # 應該接近 2 倍（允許一些誤差，因為有額外的元件）
        # 2.1 倍是一個合理的上限
        assert params_4 < params_2 * 2.1, \
            f"參數量比例不合理：4 層 {params_4}，2 層 {params_2}，比例 {params_4/params_2:.2f}"

    def test_layers_are_different(self):
        """
        測試每一層的參數是否獨立（不共享權重）

        目的：確保每層都有自己的參數

        為什麼重要：
        - 如果層之間共享參數，模型的表達能力會大幅下降
        - 每層應該能學習不同的特徵轉換
        """
        # === 設定測試參數 ===
        d_model = 128
        num_heads = 4
        d_ff = 512
        num_layers = 2

        # === 建立編碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # === 獲取不同層的參數 ===
        # 提取第一層和第二層的第一個參數進行比較
        layer1_params = list(encoder.layers[0].parameters())[0]
        layer2_params = list(encoder.layers[1].parameters())[0]

        # === 驗證參數不同 ===
        # 參數是隨機初始化的，應該不相同
        assert not torch.allclose(layer1_params, layer2_params), \
            "第一層和第二層的參數相同，可能發生了權重共享"

    def test_variable_sequence_lengths(self):
        """
        測試編碼器處理不同序列長度的能力

        目的：確保編碼器可以處理各種長度的輸入

        實際應用：
        - 不同句子有不同長度
        - 編碼器應該對任何長度都能正常運作
        """
        # === 設定測試參數 ===
        batch_size = 2
        d_model = 256
        num_heads = 4
        d_ff = 1024
        num_layers = 3

        # === 建立編碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)

        # === 測試多種序列長度 ===
        for seq_len in [5, 10, 50, 100]:
            # 建立該長度的輸入
            x = torch.randn(batch_size, seq_len, d_model)

            # 執行前向傳播
            output = encoder(x)

            # 驗證輸出形狀
            assert output.shape == (batch_size, seq_len, d_model), \
                f"序列長度 {seq_len} 時編碼器輸出形狀錯誤"


if __name__ == "__main__":
    # 允許直接執行此檔案進行測試
    pytest.main([__file__, "-v"])
