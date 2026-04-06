"""
完整 Transformer 模型的測試

這個測試檔案測試完整的 Transformer 模型，包括從詞元嵌入到最終輸出的整個流程

測試涵蓋範圍：
1. TokenEmbedding：詞元嵌入查找和縮放
2. Transformer 初始化：參數計數、架構檢查
3. Encode：源序列編碼功能
4. Decode：目標序列解碼功能
5. Forward pass：完整的訓練流程
6. Generate：自回歸推理生成
7. Integration：端到端整合場景

完整 Transformer 的組成：
- 源詞彙的詞嵌入層
- 目標詞彙的詞嵌入層
- 位置編碼層
- 編碼器堆疊
- 解碼器堆疊
- 輸出投影層（將解碼器輸出投影到詞彙空間）
"""

import pytest  # Python 測試框架
import torch  # PyTorch 深度學習框架
import torch.nn as nn  # PyTorch 神經網路模組

from transformer import (
    Transformer,         # 完整的 Transformer 模型
    TokenEmbedding,      # 詞元嵌入層
    create_transformer   # 工廠函數，用於建立模型
)


class TestTokenEmbedding:
    """
    測試詞元嵌入層（Token Embedding Layer）

    詞元嵌入的作用：
    - 將離散的詞元 ID（整數）映射到連續的向量空間
    - 在 Transformer 中，嵌入會乘以 sqrt(d_model) 進行縮放
    - 縮放使嵌入和位置編碼的量級相近
    """

    def test_embedding_output_shape(self):
        """
        測試嵌入層的輸出形狀是否正確

        輸入：詞元 ID 的整數張量 (batch_size, seq_len)
        輸出：嵌入向量 (batch_size, seq_len, d_model)
        """
        # === 設定測試參數 ===
        vocab_size = 1000  # 詞彙表大小
        d_model = 512      # 模型維度（嵌入維度）
        batch_size = 2
        seq_len = 10

        # === 建立嵌入層 ===
        embedding = TokenEmbedding(vocab_size, d_model)

        # === 建立隨機詞元 ID ===
        # torch.randint 生成範圍在 [0, vocab_size) 的隨機整數
        # 形狀：(batch_size, seq_len)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        # === 獲取嵌入 ===
        output = embedding(tokens)

        # === 驗證輸出形狀 ===
        # 應該是 (batch, seq_len, d_model)
        assert output.shape == (batch_size, seq_len, d_model), \
            f"嵌入輸出形狀錯誤：期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"

    def test_embedding_scaling(self):
        """
        測試嵌入是否被正確縮放

        Transformer 論文中的做法：
        - 嵌入向量乘以 sqrt(d_model)
        - 這樣嵌入和位置編碼的量級相近
        - 有助於訓練穩定性
        """
        # === 設定測試參數 ===
        vocab_size = 100
        d_model = 64

        # === 建立嵌入層 ===
        embedding = TokenEmbedding(vocab_size, d_model)

        # === 建立單一詞元 ===
        tokens = torch.tensor([[5]])  # 詞元 ID = 5

        # === 獲取縮放後的嵌入 ===
        scaled_output = embedding(tokens)

        # === 獲取未縮放的嵌入（直接從 nn.Embedding）===
        # embedding.embedding 是內部的 PyTorch Embedding 層
        unscaled_output = embedding.embedding(tokens)

        # === 計算預期的縮放結果 ===
        expected = unscaled_output * (d_model ** 0.5)  # 乘以 sqrt(d_model)

        # === 驗證縮放是否正確 ===
        torch.testing.assert_close(scaled_output, expected), \
            "嵌入的縮放不正確"

    def test_different_tokens_different_embeddings(self):
        """
        測試不同詞元是否有不同的嵌入

        這是嵌入層的基本要求：
        - 不同的詞元應該有不同的向量表示
        - 如果所有詞元的嵌入都相同，模型無法區分它們
        """
        # === 設定測試參數 ===
        vocab_size = 100
        d_model = 64

        # === 建立嵌入層 ===
        embedding = TokenEmbedding(vocab_size, d_model)

        # === 建立兩個不同的詞元 ===
        token1 = torch.tensor([[5]])
        token2 = torch.tensor([[10]])

        # === 獲取嵌入 ===
        emb1 = embedding(token1)
        emb2 = embedding(token2)

        # === 驗證嵌入不同 ===
        assert not torch.allclose(emb1, emb2), \
            "不同詞元的嵌入相同，嵌入層可能沒有正確初始化"

    def test_same_token_same_embedding(self):
        """
        測試相同詞元是否總是得到相同的嵌入

        嵌入查找是確定性的：
        - 相同的詞元 ID 應該總是映射到相同的向量
        - 無論詞元在序列中的位置如何
        - 位置資訊由位置編碼提供，不是嵌入層
        """
        # === 設定測試參數 ===
        vocab_size = 100
        d_model = 64

        # === 建立嵌入層 ===
        embedding = TokenEmbedding(vocab_size, d_model)

        # === 建立包含重複詞元的序列 ===
        # 詞元 5 出現在位置 0 和位置 2
        tokens = torch.tensor([[5, 10, 5, 20]])

        # === 獲取嵌入 ===
        output = embedding(tokens)

        # === 驗證相同詞元的嵌入相同 ===
        # 位置 0 和位置 2 都是詞元 5，嵌入應該完全相同
        torch.testing.assert_close(output[0, 0], output[0, 2]), \
            "相同詞元在不同位置的嵌入不同"


class TestTransformerInitialization:
    """測試 Transformer 模型的初始化"""

    def test_model_creation(self):
        """
        測試能否使用預設參數建立模型

        這是最基本的健全性檢查：確保模型能夠被正確實例化
        """
        # === 建立 Transformer 模型 ===
        model = Transformer(
            src_vocab_size=1000,      # 源詞彙表大小
            tgt_vocab_size=1000,      # 目標詞彙表大小
            d_model=512,              # 模型維度
            num_heads=8,              # 注意力頭數
            num_encoder_layers=6,     # 編碼器層數
            num_decoder_layers=6      # 解碼器層數
        )

        # === 驗證模型組件存在 ===
        assert isinstance(model, nn.Module), \
            "模型不是有效的 PyTorch 模組"
        assert model.src_embedding is not None, \
            "缺少源語言嵌入層"
        assert model.tgt_embedding is not None, \
            "缺少目標語言嵌入層"
        assert model.encoder is not None, \
            "缺少編碼器"
        assert model.decoder is not None, \
            "缺少解碼器"

    def test_small_model_creation(self):
        """
        測試建立小型模型（適合 CPU 訓練）

        小型模型用途：
        - 快速原型開發
        - CPU 訓練和測試
        - 資源受限環境
        """
        # === 建立小型模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,              # 較小的模型維度
            num_heads=4,              # 較少的注意力頭
            num_encoder_layers=2,     # 較少的編碼器層
            num_decoder_layers=2,     # 較少的解碼器層
            d_ff=512                  # 較小的前饋網路維度
        )

        # === 驗證參數量 ===
        param_count = model.count_parameters()

        # 小型模型應該有較少的參數（少於 1000 萬）
        assert param_count < 10_000_000, \
            f"小型模型的參數量過多：{param_count:,}"

    def test_create_transformer_factory(self):
        """
        測試使用工廠函數建立模型

        工廠函數的優點：
        - 提供更簡潔的 API
        - 設定合理的預設值
        - 確保編碼器和解碼器有相同的層數
        """
        # === 使用工廠函數建立模型 ===
        model = create_transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_layers=4,  # 編碼器和解碼器都會是 4 層
            d_ff=1024
        )

        # === 驗證模型類型 ===
        assert isinstance(model, Transformer), \
            "工廠函數沒有返回 Transformer 實例"

        # === 驗證層數 ===
        assert len(model.encoder.layers) == 4, \
            "編碼器層數不正確"
        assert len(model.decoder.layers) == 4, \
            "解碼器層數不正確"

    def test_parameter_count(self):
        """
        測試參數計數功能

        參數計數用途：
        - 估計記憶體需求
        - 比較不同模型大小
        - 除錯模型結構
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=512
        )

        # === 使用模型的計數方法 ===
        param_count = model.count_parameters()

        # === 驗證參數量合理 ===
        assert param_count > 0, \
            "參數量為零，模型可能沒有參數"
        assert param_count < 100_000_000, \
            f"參數量過大：{param_count:,}，可能有錯誤"

        # === 手動計數並比較 ===
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count == manual_count, \
            f"模型的計數方法不正確：{param_count} vs {manual_count}"


class TestTransformerEncode:
    """測試編碼器功能"""

    def test_encode_shape(self):
        """
        測試編碼功能是否產生正確的輸出形狀

        encode() 方法：
        - 輸入：詞元 ID (batch, src_len)
        - 內部：嵌入 → 位置編碼 → 編碼器堆疊
        - 輸出：編碼器的記憶 (batch, src_len, d_model)
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10

        # === 建立源詞元 ID ===
        # 形狀：(batch_size, src_len)
        src = torch.randint(0, 1000, (batch_size, src_len))

        # === 編碼 ===
        # 內部流程：
        # 1. src → TokenEmbedding → (batch, src_len, d_model)
        # 2. → PositionalEncoding → (batch, src_len, d_model)
        # 3. → Encoder → (batch, src_len, d_model)
        memory = model.encode(src)

        # === 驗證輸出形狀 ===
        assert memory.shape == (batch_size, src_len, 128), \
            f"編碼輸出形狀錯誤：期望 {(batch_size, src_len, 128)}，得到 {memory.shape}"

    def test_encode_with_mask(self):
        """
        測試帶遮罩的編碼

        遮罩用途：
        - 處理批次中不同長度的序列
        - 告訴模型哪些位置是填充（padding）
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10

        # === 建立源詞元 ===
        src = torch.randint(0, 1000, (batch_size, src_len))

        # === 建立遮罩 ===
        # 形狀：(batch_size, 1, 1, src_len)
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        src_mask[:, :, :, 5:] = 0  # 後 5 個位置是填充

        # === 帶遮罩編碼 ===
        memory = model.encode(src, src_mask)

        # === 驗證輸出形狀 ===
        # 即使有遮罩，輸出形狀也應該相同
        assert memory.shape == (batch_size, src_len, 128), \
            "帶遮罩的編碼輸出形狀錯誤"

    def test_encode_deterministic(self):
        """
        測試編碼是否是確定性的

        在評估模式下：
        - 相同輸入應該產生相同輸出
        - 不受 dropout 等隨機因素影響
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定為評估模式 ===
        model.eval()

        # === 建立測試輸入 ===
        src = torch.randint(0, 1000, (2, 10))

        # === 編碼兩次 ===
        with torch.no_grad():
            memory1 = model.encode(src)
            memory2 = model.encode(src)

        # === 驗證結果相同 ===
        torch.testing.assert_close(memory1, memory2), \
            "評估模式下編碼不是確定性的"


class TestTransformerDecode:
    """測試解碼器功能"""

    def test_decode_shape(self):
        """
        測試解碼功能是否產生正確的輸出形狀

        decode() 方法：
        - 輸入：目標詞元 ID (batch, tgt_len) + 編碼器記憶
        - 內部：嵌入 → 位置編碼 → 解碼器堆疊
        - 輸出：解碼器輸出 (batch, tgt_len, d_model)
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # === 建立源序列並編碼 ===
        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)

        # === 建立目標序列 ===
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # === 解碼 ===
        output = model.decode(tgt, memory)

        # === 驗證輸出形狀 ===
        assert output.shape == (batch_size, tgt_len, 128), \
            f"解碼輸出形狀錯誤"

    def test_decode_with_causal_mask(self):
        """
        測試帶因果遮罩的解碼

        因果遮罩：
        - 防止位置 i 看到位置 j（當 j > i）
        - 確保自回歸生成的正確性
        """
        from transformer import create_causal_mask

        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # === 編碼源序列 ===
        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)

        # === 建立目標序列 ===
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # === 建立因果遮罩 ===
        tgt_mask = create_causal_mask(tgt_len)

        # === 帶遮罩解碼 ===
        output = model.decode(tgt, memory, tgt_mask)

        # === 驗證輸出形狀 ===
        assert output.shape == (batch_size, tgt_len, 128), \
            "帶因果遮罩的解碼輸出形狀錯誤"

    def test_decode_different_lengths(self):
        """
        測試源序列和目標序列長度不同的情況

        Transformer 的優勢：
        - 可以處理任意長度組合
        - 源和目標長度完全獨立
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2

        # 測試多種長度組合
        src_len = 15  # 源序列較長
        tgt_len = 5   # 目標序列較短

        # === 編碼和解碼 ===
        src = torch.randint(0, 1000, (batch_size, src_len))
        memory = model.encode(src)

        tgt = torch.randint(0, 1000, (batch_size, tgt_len))
        output = model.decode(tgt, memory)

        # === 驗證輸出長度 ===
        # 輸出長度應該與目標相同，而非源
        assert output.shape == (batch_size, tgt_len, 128), \
            f"輸出長度應該是目標長度 {tgt_len}"


class TestTransformerForward:
    """測試完整的前向傳播"""

    def test_forward_shape(self):
        """
        測試完整前向傳播的輸出形狀

        forward() 方法：
        - 輸入：源詞元 ID + 目標詞元 ID
        - 內部：編碼 → 解碼 → 投影到詞彙空間
        - 輸出：logits (batch, tgt_len, tgt_vocab_size)
        """
        # === 設定測試參數 ===
        src_vocab = 1000
        tgt_vocab = 800   # 可以與源詞彙不同
        d_model = 128

        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=d_model,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # === 建立輸入 ===
        src = torch.randint(0, src_vocab, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

        # === 前向傳播 ===
        logits = model(src, tgt)

        # === 驗證輸出形狀 ===
        # 輸出應該是 (batch, tgt_len, tgt_vocab_size)
        # 每個位置都有對整個目標詞彙表的預測
        assert logits.shape == (batch_size, tgt_len, tgt_vocab), \
            f"前向傳播輸出形狀錯誤：期望 {(batch_size, tgt_len, tgt_vocab)}，得到 {logits.shape}"

    def test_forward_with_masks(self):
        """
        測試帶遮罩的完整前向傳播
        """
        from transformer import create_causal_mask

        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # === 建立輸入 ===
        src = torch.randint(0, 1000, (batch_size, src_len))
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # === 建立遮罩 ===
        src_mask = torch.ones(batch_size, 1, 1, src_len)
        tgt_mask = create_causal_mask(tgt_len)

        # === 帶遮罩的前向傳播 ===
        logits = model(src, tgt, src_mask, tgt_mask)

        # === 驗證輸出形狀 ===
        assert logits.shape == (batch_size, tgt_len, 1000), \
            "帶遮罩的前向傳播輸出形狀錯誤"

    def test_forward_backprop(self):
        """
        測試梯度是否能夠流過整個模型

        這是訓練的基本要求：
        - 前向傳播產生輸出
        - 反向傳播計算梯度
        - 所有可訓練參數都應該有梯度
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10
        tgt_len = 8

        # === 建立輸入 ===
        src = torch.randint(0, 1000, (batch_size, src_len))
        tgt = torch.randint(0, 1000, (batch_size, tgt_len))

        # === 前向傳播 ===
        logits = model(src, tgt)

        # === 建立虛擬損失 ===
        # 使用 sum() 作為簡單的損失函數
        loss = logits.sum()

        # === 反向傳播 ===
        loss.backward()

        # === 驗證關鍵參數有梯度 ===
        assert model.src_embedding.embedding.weight.grad is not None, \
            "源嵌入層沒有梯度"
        assert model.tgt_embedding.embedding.weight.grad is not None, \
            "目標嵌入層沒有梯度"
        assert model.output_projection.weight.grad is not None, \
            "輸出投影層沒有梯度"

    def test_forward_batch_size_one(self):
        """
        測試批次大小為 1 的情況

        單樣本批次是常見的邊界情況：
        - 推理時經常使用
        - 某些廣播操作可能出問題
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 建立單樣本輸入 ===
        src = torch.randint(0, 1000, (1, 10))
        tgt = torch.randint(0, 1000, (1, 8))

        # === 前向傳播 ===
        logits = model(src, tgt)

        # === 驗證輸出形狀 ===
        assert logits.shape == (1, 8, 1000), \
            "批次大小為 1 時輸出形狀錯誤"


class TestTransformerGenerate:
    """測試自回歸生成功能"""

    def test_generate_basic(self):
        """
        測試基本的生成功能

        generate() 方法：
        - 自回歸生成目標序列
        - 一次生成一個詞元
        - 直到生成結束標記或達到最大長度
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10

        # === 建立源序列 ===
        src = torch.randint(0, 1000, (batch_size, src_len))

        # === 生成 ===
        # start_token: 開始標記的 ID
        # end_token: 結束標記的 ID
        # max_len: 最大生成長度
        generated = model.generate(src, max_len=20, start_token=1, end_token=2)

        # === 驗證輸出 ===
        # 形狀：(batch, generated_len)
        assert generated.shape[0] == batch_size, \
            "生成的批次大小不正確"
        assert generated.shape[1] <= 20, \
            "生成長度超過最大長度限制"

        # 第一個詞元應該是開始標記
        assert (generated[:, 0] == 1).all(), \
            "生成序列的第一個詞元不是開始標記"

    def test_generate_with_max_len(self):
        """
        測試生成是否遵守最大長度限制
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 建立源序列 ===
        src = torch.randint(0, 1000, (1, 10))

        # === 使用較小的最大長度生成 ===
        max_len = 5
        generated = model.generate(src, max_len=max_len, start_token=1, end_token=2)

        # === 驗證長度限制 ===
        assert generated.shape[1] <= max_len, \
            f"生成長度 {generated.shape[1]} 超過最大長度 {max_len}"

    def test_generate_deterministic_eval_mode(self):
        """
        測試評估模式下生成是否是確定性的

        在評估模式且沒有隨機性的情況下：
        - 相同輸入應該產生相同輸出
        - 這對於可重現的實驗很重要
        """
        # === 建立無 dropout 的模型 ===
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.0  # 關閉 dropout
        )
        model.eval()

        # === 建立源序列 ===
        src = torch.randint(0, 100, (1, 5))

        # === 生成兩次 ===
        with torch.no_grad():
            gen1 = model.generate(src, max_len=10, start_token=1, end_token=2)
            gen2 = model.generate(src, max_len=10, start_token=1, end_token=2)

        # === 驗證結果相同 ===
        assert torch.equal(gen1, gen2), \
            "評估模式下生成不是確定性的"

    def test_generate_stops_at_end_token(self):
        """
        測試生成是否能在遇到結束標記時提前停止

        注意：
        - 實際是否提前停止取決於模型權重
        - 這個測試只確保程式碼能正常執行
        - 不驗證實際的提前停止行為（那需要訓練好的模型）
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 建立源序列 ===
        src = torch.randint(0, 100, (1, 5))

        # === 生成 ===
        # 只要不崩潰就算通過
        generated = model.generate(src, max_len=20, start_token=1, end_token=2)

        # === 驗證基本屬性 ===
        assert generated.shape[1] <= 20, \
            "生成長度超過限制"


class TestTransformerIntegration:
    """整合測試 - 端到端場景"""

    def test_translation_pipeline(self):
        """
        測試完整的翻譯流程

        模擬真實的翻譯任務：
        1. 訓練時：使用已知的源和目標序列
        2. 推理時：使用源序列生成目標序列
        """
        # === 設定測試參數 ===
        src_vocab = 1000  # 源語言詞彙（例如：英文）
        tgt_vocab = 800   # 目標語言詞彙（例如：法文）
        d_model = 128

        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=src_vocab,
            tgt_vocab_size=tgt_vocab,
            d_model=d_model,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 設定測試參數 ===
        batch_size = 2
        src_len = 10

        # === 建立源序列 ===
        src = torch.randint(0, src_vocab, (batch_size, src_len))

        # === 訓練模式：使用已知目標 ===
        tgt_len = 8
        tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

        # 訓練用的前向傳播
        logits = model(src, tgt)
        assert logits.shape == (batch_size, tgt_len, tgt_vocab), \
            "訓練模式輸出形狀錯誤"

        # === 推理模式：生成目標 ===
        model.eval()
        with torch.no_grad():
            generated = model.generate(src, max_len=15, start_token=1, end_token=2)

        assert generated.shape[0] == batch_size, \
            "生成的批次大小不正確"
        assert generated.shape[1] <= 15, \
            "生成長度超過限制"

    def test_different_vocab_sizes(self):
        """
        測試源和目標使用不同詞彙表大小的情況

        實際應用中常見：
        - 不同語言的詞彙表大小通常不同
        - 例如：英文 5000 詞，中文 3000 詞
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=5000,  # 源語言：5000 詞
            tgt_vocab_size=3000,  # 目標語言：3000 詞
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 建立輸入 ===
        src = torch.randint(0, 5000, (2, 12))
        tgt = torch.randint(0, 3000, (2, 10))

        # === 前向傳播 ===
        logits = model(src, tgt)

        # === 驗證輸出 ===
        # 輸出詞彙空間應該是目標語言的大小
        assert logits.shape == (2, 10, 3000), \
            f"輸出詞彙大小應該是目標語言的 3000，得到 {logits.shape[2]}"

    def test_variable_sequence_lengths(self):
        """
        測試各種序列長度組合

        確保模型能夠處理任意長度：
        - 短-短
        - 長-長
        - 短-長
        - 長-短
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 測試多種長度組合 ===
        test_cases = [
            (5, 3),     # 短源，短目標
            (20, 15),   # 長源，長目標
            (10, 3),    # 長源，短目標
            (3, 10),    # 短源，長目標
        ]

        for src_len, tgt_len in test_cases:
            # 建立輸入
            src = torch.randint(0, 1000, (2, src_len))
            tgt = torch.randint(0, 1000, (2, tgt_len))

            # 前向傳播
            logits = model(src, tgt)

            # 驗證輸出
            assert logits.shape == (2, tgt_len, 1000), \
                f"長度組合 ({src_len}, {tgt_len}) 時輸出形狀錯誤"

    def test_training_step_simulation(self):
        """
        模擬完整的訓練步驟

        包括：
        1. 前向傳播
        2. 損失計算
        3. 反向傳播
        4. 參數更新
        """
        # === 建立模型 ===
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        # === 建立優化器 ===
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # === 建立損失函數 ===
        # ignore_index=0: 忽略填充詞元（ID 為 0）
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # === 建立訓練資料 ===
        # 從 1 開始，避免使用填充 ID (0)
        src = torch.randint(1, 1000, (4, 10))      # batch=4
        tgt_input = torch.randint(1, 1000, (4, 8))
        tgt_output = torch.randint(1, 1000, (4, 8))

        # === 前向傳播 ===
        model.train()
        logits = model(src, tgt_input)  # (4, 8, 1000)

        # === 計算損失 ===
        # CrossEntropyLoss 需要：
        # - 輸入：(batch * seq_len, vocab_size)
        # - 目標：(batch * seq_len)
        logits_flat = logits.reshape(-1, 1000)
        tgt_flat = tgt_output.reshape(-1)
        loss = criterion(logits_flat, tgt_flat)

        # === 反向傳播 ===
        optimizer.zero_grad()  # 清空梯度
        loss.backward()         # 計算梯度
        optimizer.step()        # 更新參數

        # === 驗證訓練步驟成功 ===
        assert loss.item() > 0, \
            "損失應該是正數"

        # 如果能執行到這裡，訓練步驟就成功了
        assert True
