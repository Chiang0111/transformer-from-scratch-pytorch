"""
Transformer 元件的單元測試

這個套件包含了完整的 Transformer 架構測試：
- test_attention.py: 測試注意力機制（Scaled Dot-Product 和 Multi-Head Attention）
- test_positional_encoding.py: 測試正弦位置編碼
- test_feedforward.py: 測試前饋神經網路層
- test_encoder.py: 測試編碼器層和完整編碼器堆疊
- test_decoder.py: 測試解碼器層和完整解碼器堆疊
- test_transformer.py: 測試完整的 Transformer 模型
- test_training.py: 測試訓練流程的整合測試

執行所有測試：pytest tests/ -v
執行特定測試：pytest tests/test_attention.py -v
"""
