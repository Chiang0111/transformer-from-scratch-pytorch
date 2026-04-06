"""
解碼器（Decoder）的單元測試

這個測試檔案專門測試 Transformer 的解碼器部分

解碼器的結構：
1. DecoderLayer（解碼器層）：
   - Masked Multi-Head Self-Attention（遮罩多頭自注意力）
   - Add & Norm（殘差連接 + 層歸一化）
   - Multi-Head Cross-Attention（多頭交叉注意力，關注編碼器輸出）
   - Add & Norm（殘差連接 + 層歸一化）
   - Position-wise Feed-Forward Network（位置前饋網路）
   - Add & Norm（殘差連接 + 層歸一化）

2. Decoder（完整解碼器）：
   - 堆疊多個 DecoderLayer（通常 6 層）
   - 最後可能有額外的層歸一化

解碼器的特殊之處：
- 使用因果遮罩（causal mask）防止看到未來的詞元
- 有交叉注意力層，用於關注編碼器的輸出
- 用於自回歸生成（autoregressive generation）

測試目標：
1. 測試因果遮罩的生成和應用
2. 測試解碼器層的三個子層
3. 測試交叉注意力機制
4. 測試完整解碼器的堆疊
5. 測試編碼器-解碼器集成
"""

import pytest  # Python 測試框架
import torch  # PyTorch 深度學習框架
from transformer.decoder import DecoderLayer, Decoder, create_causal_mask


class TestCausalMask:
    """
    測試因果遮罩（Causal Mask）的建立

    因果遮罩的目的：
    - 在解碼器的自注意力中使用
    - 防止位置 i 關注位置 j（當 j > i 時）
    - 確保生成過程是自回歸的（只能看過去，不能看未來）

    遮罩模式（下三角矩陣）：
    位置  0  1  2  3
    0    [1  0  0  0]  位置 0 只能看自己
    1    [1  1  0  0]  位置 1 可以看 0 和 1
    2    [1  1  1  0]  位置 2 可以看 0、1、2
    3    [1  1  1  1]  位置 3 可以看所有位置
    """

    def test_mask_shape(self):
        """
        測試因果遮罩的形狀是否正確

        預期形狀：(1, 1, size, size)
        - 第一個 1：批次維度（可廣播到任意批次大小）
        - 第二個 1：注意力頭維度（可廣播到所有頭）
        - 後兩個 size：(查詢長度, 鍵長度)
        """
        # === 設定測試參數 ===
        size = 5  # 序列長度

        # === 建立因果遮罩 ===
        mask = create_causal_mask(size)

        # === 驗證遮罩形狀 ===
        assert mask.shape == (1, 1, size, size), \
            f"因果遮罩形狀錯誤：期望 (1, 1, {size}, {size})，得到 {mask.shape}"

    def test_mask_is_lower_triangular(self):
        """
        測試因果遮罩是否為下三角矩陣

        下三角矩陣的特性：
        - 對角線及其下方的元素為 1（允許關注）
        - 對角線上方的元素為 0（禁止關注未來）
        """
        # === 設定測試參數 ===
        size = 4

        # === 建立因果遮罩 ===
        mask = create_causal_mask(size)

        # === 移除批次和頭維度 ===
        # squeeze() 移除大小為 1 的維度
        # (1, 1, 4, 4) → (4, 4)
        mask_2d = mask.squeeze()

        # === 驗證下三角性質 ===
        for i in range(size):
            for j in range(size):
                if j > i:
                    # 對角線上方（未來位置）應該被遮罩（0）
                    assert mask_2d[i, j] == 0, \
                        f"位置 ({i}, {j}) 應該被遮罩，但值為 {mask_2d[i, j]}"
                else:
                    # 對角線及其下方（當前和過去位置）應該可見（1）
                    assert mask_2d[i, j] == 1, \
                        f"位置 ({i}, {j}) 應該可見，但值為 {mask_2d[i, j]}"

    def test_mask_pattern(self):
        """
        測試具體的遮罩模式是否正確

        使用 3x3 的小遮罩來精確驗證模式
        """
        # === 建立因果遮罩 ===
        mask = create_causal_mask(3)

        # === 定義預期的模式 ===
        # 下三角矩陣：
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        expected = torch.tensor([
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]]
        ])

        # === 驗證模式 ===
        # squeeze(0) 移除批次維度，保留頭維度
        # 將整數張量轉換為浮點數進行比較
        assert torch.equal(mask.squeeze(0), expected.float()), \
            "因果遮罩的模式不正確"

    def test_different_sizes(self):
        """
        測試不同大小的因果遮罩是否都能正確生成

        目的：確保函數對各種序列長度都有效
        """
        # === 測試多種大小 ===
        for size in [1, 5, 10, 20]:
            # 建立遮罩
            mask = create_causal_mask(size)

            # 驗證形狀
            assert mask.shape == (1, 1, size, size), \
                f"大小 {size} 的因果遮罩形狀錯誤"

            # 移除批次和頭維度
            mask_2d = mask.squeeze(0).squeeze(0)

            # 驗證對角線元素為 1（可以看自己）
            for i in range(size):
                assert mask_2d[i, i] == 1, \
                    f"對角線位置 ({i}, {i}) 應該為 1"

                # 驗證對角線右上方第一個元素為 0（不能看未來）
                if i < size - 1:
                    assert mask_2d[i, i + 1] == 0, \
                        f"位置 ({i}, {i+1}) 應該為 0（未來位置）"


class TestDecoderLayer:
    """測試單一解碼器層（DecoderLayer）"""

    @pytest.fixture
    def decoder_params(self):
        """
        提供解碼器層的通用參數

        Pytest fixture：
        - 可重用的測試設定
        - 自動注入到測試函數中
        - 避免重複程式碼
        """
        return {
            'd_model': 512,    # 模型維度
            'num_heads': 8,    # 注意力頭數
            'd_ff': 2048,      # 前饋網路隱藏層維度
            'dropout': 0.1     # Dropout 率
        }

    @pytest.fixture
    def decoder_layer(self, decoder_params):
        """
        建立解碼器層實例

        使用 decoder_params fixture 的參數
        """
        return DecoderLayer(**decoder_params)

    def test_output_shape(self, decoder_layer):
        """
        測試解碼器層的輸出形狀是否正確

        注意：
        - 輸出長度與目標序列（x）相同
        - 不是與源序列（encoder_output）相同
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10       # 源序列長度（編碼器輸出）
        tgt_len = 8        # 目標序列長度（解碼器輸入）
        d_model = 512

        # === 建立測試輸入 ===
        # x: 目標序列（例如：目標語言的部分翻譯）
        x = torch.randn(batch_size, tgt_len, d_model)

        # encoder_output: 編碼器的輸出（源序列的表示）
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # === 建立遮罩 ===
        # tgt_mask: 因果遮罩，防止看到未來詞元
        tgt_mask = create_causal_mask(tgt_len)

        # src_mask: 源序列遮罩，處理填充
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        # === 執行前向傳播 ===
        # 內部流程：
        # 1. x → Masked Self-Attention (with tgt_mask) → attn1_output
        # 2. x + attn1_output → LayerNorm → norm1_output
        # 3. norm1_output → Cross-Attention (attend to encoder_output) → attn2_output
        # 4. norm1_output + attn2_output → LayerNorm → norm2_output
        # 5. norm2_output → Feed-Forward → ff_output
        # 6. norm2_output + ff_output → LayerNorm → final_output
        output = decoder_layer(x, encoder_output, tgt_mask, src_mask)

        # === 驗證輸出形狀 ===
        # 輸出長度應該與目標序列相同
        assert output.shape == (batch_size, tgt_len, d_model), \
            f"解碼器層輸出形狀錯誤：期望 {(batch_size, tgt_len, d_model)}，得到 {output.shape}"

    def test_different_source_target_lengths(self, decoder_layer):
        """
        測試源序列和目標序列長度不同的情況

        這是 Transformer 的重要特性：
        - 翻譯時，源語言和目標語言句子長度通常不同
        - 解碼器必須能處理任意長度組合
        """
        # === 設定測試參數 ===
        batch_size = 3
        src_len = 15   # 源序列較長
        tgt_len = 10   # 目標序列較短
        d_model = 512

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # === 建立遮罩 ===
        tgt_mask = create_causal_mask(tgt_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        # === 執行前向傳播 ===
        output = decoder_layer(x, encoder_output, tgt_mask, src_mask)

        # === 驗證輸出長度 ===
        # 輸出長度應該與目標序列相同，而非源序列
        assert output.shape == (batch_size, tgt_len, d_model), \
            f"輸出長度應該是 {tgt_len}（目標長度），而非 {src_len}（源長度）"

    def test_with_masks(self, decoder_layer):
        """
        測試解碼器層處理各種遮罩組合的能力

        測試的組合：
        1. 只有目標遮罩（因果遮罩）
        2. 只有源遮罩（填充遮罩）
        3. 兩種遮罩都有
        4. 兩種遮罩都沒有
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 6
        tgt_len = 5
        d_model = 512

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # === 測試組合 1：只有因果遮罩 ===
        tgt_mask = create_causal_mask(tgt_len)
        output1 = decoder_layer(x, encoder_output, tgt_mask, None)
        assert output1.shape == (batch_size, tgt_len, d_model), \
            "只有目標遮罩時輸出形狀錯誤"

        # === 測試組合 2：只有源遮罩 ===
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        output2 = decoder_layer(x, encoder_output, None, src_mask)
        assert output2.shape == (batch_size, tgt_len, d_model), \
            "只有源遮罩時輸出形狀錯誤"

        # === 測試組合 3：兩種遮罩都有 ===
        output3 = decoder_layer(x, encoder_output, tgt_mask, src_mask)
        assert output3.shape == (batch_size, tgt_len, d_model), \
            "兩種遮罩都有時輸出形狀錯誤"

        # === 測試組合 4：沒有遮罩 ===
        output4 = decoder_layer(x, encoder_output, None, None)
        assert output4.shape == (batch_size, tgt_len, d_model), \
            "沒有遮罩時輸出形狀錯誤"

    def test_parameters_exist(self, decoder_layer):
        """
        測試解碼器層是否包含所有必要的參數

        解碼器層應該有：
        - 自注意力（self-attention）的 W_q, W_k, W_v, W_o
        - 交叉注意力（cross-attention）的 W_q, W_k, W_v, W_o
        - 前饋網路的兩個線性層
        - 三個層歸一化層
        """
        # === 獲取所有參數的名稱 ===
        params = dict(decoder_layer.named_parameters())

        # === 驗證自注意力參數 ===
        assert 'self_attention.W_q.weight' in params, "缺少自注意力的查詢權重"
        assert 'self_attention.W_k.weight' in params, "缺少自注意力的鍵權重"
        assert 'self_attention.W_v.weight' in params, "缺少自注意力的值權重"
        assert 'self_attention.W_o.weight' in params, "缺少自注意力的輸出權重"

        # === 驗證交叉注意力參數 ===
        assert 'cross_attention.W_q.weight' in params, "缺少交叉注意力的查詢權重"
        assert 'cross_attention.W_k.weight' in params, "缺少交叉注意力的鍵權重"
        assert 'cross_attention.W_v.weight' in params, "缺少交叉注意力的值權重"
        assert 'cross_attention.W_o.weight' in params, "缺少交叉注意力的輸出權重"

        # === 驗證前饋網路參數 ===
        assert 'feed_forward.linear1.weight' in params, "缺少前饋網路第一層權重"
        assert 'feed_forward.linear2.weight' in params, "缺少前饋網路第二層權重"

        # === 驗證層歸一化參數 ===
        # 解碼器層有 3 個層歸一化（每個子層之後一個）
        assert 'norm1.weight' in params, "缺少第一個層歸一化"
        assert 'norm2.weight' in params, "缺少第二個層歸一化"
        assert 'norm3.weight' in params, "缺少第三個層歸一化"

    def test_training_vs_eval_mode(self, decoder_layer):
        """
        測試訓練模式和評估模式下的不同行為

        主要差異：
        - 訓練模式：dropout 啟用，結果有隨機性
        - 評估模式：dropout 停用，結果確定性
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        tgt_mask = create_causal_mask(tgt_len)

        # === 測試訓練模式 ===
        decoder_layer.train()

        # 設定隨機種子（理論上應該讓結果可重現）
        torch.manual_seed(42)
        output_train1 = decoder_layer(x, encoder_output, tgt_mask, None)

        # 重設隨機種子
        torch.manual_seed(42)
        output_train2 = decoder_layer(x, encoder_output, tgt_mask, None)

        # 注意：由於 dropout 的隨機性，即使設定相同種子，
        # 結果也可能不同（這是 dropout 的預期行為）

        # === 測試評估模式 ===
        decoder_layer.eval()

        # 設定隨機種子
        torch.manual_seed(42)
        output_eval1 = decoder_layer(x, encoder_output, tgt_mask, None)

        # 重設隨機種子
        torch.manual_seed(42)
        output_eval2 = decoder_layer(x, encoder_output, tgt_mask, None)

        # 評估模式下，相同輸入和種子應該產生相同輸出
        assert torch.allclose(output_eval1, output_eval2), \
            "評估模式下輸出不一致，dropout 可能沒有被正確關閉"

    def test_gradient_flow(self, decoder_layer):
        """
        測試梯度是否能夠正確地流過解碼器層

        為什麼重要：
        - 確保反向傳播能夠更新所有參數
        - 驗證沒有梯度消失或梯度爆炸
        - 確保梯度能傳到編碼器輸出（用於訓練編碼器）
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        # === 建立需要梯度的輸入 ===
        # requires_grad=True 啟用梯度計算
        x = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
        encoder_output = torch.randn(batch_size, src_len, d_model, requires_grad=True)
        tgt_mask = create_causal_mask(tgt_len)

        # === 前向傳播 ===
        output = decoder_layer(x, encoder_output, tgt_mask, None)

        # === 建立損失並反向傳播 ===
        # 使用 sum() 作為簡單的損失函數
        loss = output.sum()
        loss.backward()

        # === 驗證輸入的梯度存在 ===
        assert x.grad is not None, \
            "目標輸入沒有接收到梯度"
        assert encoder_output.grad is not None, \
            "編碼器輸出沒有接收到梯度（這會阻止編碼器訓練）"

        # === 驗證所有參數都有梯度 ===
        for param in decoder_layer.parameters():
            assert param.grad is not None, \
                "某些參數沒有接收到梯度"


class TestDecoder:
    """測試完整的解碼器（堆疊多層 DecoderLayer）"""

    @pytest.fixture
    def decoder_params(self):
        """提供解碼器的通用參數"""
        return {
            'num_layers': 6,   # 層數
            'd_model': 512,    # 模型維度
            'num_heads': 8,    # 注意力頭數
            'd_ff': 2048,      # 前饋網路維度
            'dropout': 0.1     # Dropout 率
        }

    @pytest.fixture
    def decoder(self, decoder_params):
        """建立解碼器實例"""
        return Decoder(**decoder_params)

    def test_output_shape(self, decoder):
        """
        測試完整解碼器的輸出形狀
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8
        d_model = 512

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        tgt_mask = create_causal_mask(tgt_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)

        # === 執行前向傳播 ===
        # 輸入依序通過所有解碼器層
        output = decoder(x, encoder_output, tgt_mask, src_mask)

        # === 驗證輸出形狀 ===
        assert output.shape == (batch_size, tgt_len, d_model), \
            f"解碼器輸出形狀錯誤"

    def test_num_layers(self):
        """
        測試不同層數的解碼器是否都能正常建立
        """
        for num_layers in [1, 2, 4, 6, 12]:
            # 建立該層數的解碼器
            decoder = Decoder(
                num_layers=num_layers,
                d_model=256,
                num_heads=4,
                d_ff=1024
            )

            # 驗證層數
            assert decoder.num_layers == num_layers, \
                f"解碼器層數不正確"
            assert len(decoder.layers) == num_layers, \
                f"實際建立的層數與設定不符"

    def test_parameter_count(self, decoder):
        """
        測試解碼器的參數量是否合理

        6 層的標準 Transformer 解碼器應該有數千萬個參數
        """
        # === 計算總參數量 ===
        total_params = sum(p.numel() for p in decoder.parameters())

        # === 驗證參數量在合理範圍內 ===
        # 應該有足夠多的參數（超過 100 萬）
        assert total_params > 1_000_000, \
            f"參數量太少：{total_params:,}，可能模型結構有問題"

        # 但不應該過多（小於 1 億，健全性檢查）
        assert total_params < 100_000_000, \
            f"參數量過多：{total_params:,}，可能有重複或錯誤"

    def test_layers_have_independent_parameters(self, decoder):
        """
        測試每層是否有獨立的參數（不共享權重）
        """
        # === 獲取不同層的參數 ===
        layer0_weight = decoder.layers[0].self_attention.W_q.weight
        layer1_weight = decoder.layers[1].self_attention.W_q.weight

        # === 驗證參數不同 ===
        # 應該是不同的物件（不共享記憶體）
        assert layer0_weight is not layer1_weight, \
            "不同層共享了參數物件"

        # 應該有不同的值（隨機初始化）
        assert not torch.equal(layer0_weight, layer1_weight), \
            "不同層的參數值相同，可能發生了權重共享"

    def test_with_different_activations(self):
        """
        測試使用不同激活函數的解碼器
        """
        for activation in ['relu', 'gelu']:
            # 建立解碼器
            decoder = Decoder(
                num_layers=2,
                d_model=256,
                num_heads=4,
                d_ff=1024,
                activation=activation
            )

            # 測試前向傳播
            batch_size = 2
            src_len = 5
            tgt_len = 4
            d_model = 256

            x = torch.randn(batch_size, tgt_len, d_model)
            encoder_output = torch.randn(batch_size, src_len, d_model)
            tgt_mask = create_causal_mask(tgt_len)

            output = decoder(x, encoder_output, tgt_mask, None)

            assert output.shape == (batch_size, tgt_len, d_model), \
                f"{activation} 激活函數導致輸出形狀錯誤"

    def test_encoder_output_unchanged(self, decoder):
        """
        測試編碼器輸出在解碼過程中是否保持不變

        為什麼重要：
        - 編碼器輸出應該是唯讀的
        - 解碼器不應該修改編碼器的輸出
        - 這確保多個解碼步驟使用相同的編碼器表示
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 6
        tgt_len = 5
        d_model = 512

        # === 建立測試輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # 複製編碼器輸出以供比較
        encoder_output_copy = encoder_output.clone()

        tgt_mask = create_causal_mask(tgt_len)

        # === 執行解碼 ===
        decoder.eval()
        with torch.no_grad():
            _ = decoder(x, encoder_output, tgt_mask, None)

        # === 驗證編碼器輸出未被修改 ===
        assert torch.equal(encoder_output, encoder_output_copy), \
            "解碼器修改了編碼器的輸出"

    def test_gradient_flow_through_stack(self, decoder):
        """
        測試梯度是否能流過整個解碼器堆疊

        確保：
        - 所有層都能接收梯度
        - 梯度能傳到輸入和編碼器輸出
        """
        # === 設定測試參數 ===
        batch_size = 2
        src_len = 5
        tgt_len = 4
        d_model = 512

        # === 建立需要梯度的輸入 ===
        x = torch.randn(batch_size, tgt_len, d_model, requires_grad=True)
        encoder_output = torch.randn(batch_size, src_len, d_model, requires_grad=True)
        tgt_mask = create_causal_mask(tgt_len)

        # === 前向傳播 ===
        output = decoder(x, encoder_output, tgt_mask, None)

        # === 反向傳播 ===
        loss = output.sum()
        loss.backward()

        # === 驗證輸入有梯度 ===
        assert x.grad is not None, "目標輸入沒有梯度"
        assert encoder_output.grad is not None, "編碼器輸出沒有梯度"

        # === 驗證所有層都有梯度 ===
        for layer_idx, layer in enumerate(decoder.layers):
            for name, param in layer.named_parameters():
                assert param.grad is not None, \
                    f"第 {layer_idx} 層的參數 {name} 沒有梯度"

    def test_sequential_generation_simulation(self, decoder):
        """
        測試模擬自回歸生成的場景

        自回歸生成：
        - 一次生成一個詞元
        - 每次將新詞元加到序列末尾
        - 重複直到生成結束標記或達到最大長度

        這個測試模擬這個過程
        """
        # === 設定測試參數 ===
        batch_size = 1
        src_len = 6
        d_model = 512

        # === 建立編碼器輸出（固定）===
        encoder_output = torch.randn(batch_size, src_len, d_model)

        # === 設定為評估模式 ===
        decoder.eval()

        with torch.no_grad():
            # 模擬生成過程
            max_len = 5

            # 初始序列：只有開始標記
            current_output = torch.randn(batch_size, 1, d_model)

            # 逐步生成
            for step in range(1, max_len):
                # 獲取當前序列長度
                tgt_len = current_output.shape[1]

                # 建立當前長度的因果遮罩
                tgt_mask = create_causal_mask(tgt_len)

                # 解碼當前序列
                output = decoder(current_output, encoder_output, tgt_mask, None)

                # 驗證輸出形狀
                assert output.shape == (batch_size, tgt_len, d_model), \
                    f"生成步驟 {step} 的輸出形狀錯誤"

                # 模擬添加新詞元
                new_token = torch.randn(batch_size, 1, d_model)
                current_output = torch.cat([current_output, new_token], dim=1)

            # 驗證最終長度
            assert current_output.shape[1] == max_len, \
                "生成的序列長度不正確"


class TestDecoderIntegration:
    """編碼器-解碼器集成測試"""

    def test_encoder_decoder_compatibility(self):
        """
        測試編碼器和解碼器是否能協同工作

        這是端到端測試，驗證整個 Transformer 架構
        """
        from transformer.encoder import Encoder

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8
        d_model = 512
        num_heads = 8
        num_layers = 6
        d_ff = 2048

        # === 建立編碼器和解碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # === 建立輸入 ===
        src = torch.randn(batch_size, src_len, d_model)
        tgt = torch.randn(batch_size, tgt_len, d_model)

        # === 建立遮罩 ===
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        tgt_mask = create_causal_mask(tgt_len)

        # === 編碼 ===
        encoder_output = encoder(src, src_mask)
        assert encoder_output.shape == (batch_size, src_len, d_model), \
            "編碼器輸出形狀錯誤"

        # === 解碼 ===
        decoder_output = decoder(tgt, encoder_output, tgt_mask, src_mask)
        assert decoder_output.shape == (batch_size, tgt_len, d_model), \
            "解碼器輸出形狀錯誤"

    def test_full_transformer_forward_pass(self):
        """
        測試完整的 Transformer 前向傳播

        從源序列到目標序列的完整流程
        """
        from transformer.encoder import Encoder

        # === 設定較小的模型以加快測試 ===
        batch_size = 4
        src_len = 12
        tgt_len = 10
        d_model = 256
        num_heads = 4
        num_layers = 2
        d_ff = 1024

        # === 建立編碼器和解碼器 ===
        encoder = Encoder(num_layers, d_model, num_heads, d_ff)
        decoder = Decoder(num_layers, d_model, num_heads, d_ff)

        # === 設定為評估模式 ===
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            # 建立輸入
            src = torch.randn(batch_size, src_len, d_model)
            tgt = torch.randn(batch_size, tgt_len, d_model)

            # 編碼：source → memory
            memory = encoder(src)

            # 解碼：target + memory → output
            tgt_mask = create_causal_mask(tgt_len)
            output = decoder(tgt, memory, tgt_mask)

            # 驗證輸出
            assert output.shape == (batch_size, tgt_len, d_model), \
                "完整 Transformer 的輸出形狀錯誤"
