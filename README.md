# Transformer from Scratch (PyTorch)

[English](https://github.com/Chiang0111/transformer-from-scratch-pytorch) | [繁體中文](README.md)

一個**生產級**的 PyTorch Transformer 架構實作，基於論文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)。

## 為什麼要做這個專案

這個專案連接了「理解 Transformer 概念」與「專業實作」之間的橋樑。它展示了：

- ✅ **深度理解** - 從零實作，而不只是使用 `transformers` 函式庫
- ✅ **生產實務** - 模組化程式碼、型別提示、單元測試、完整文件
- ✅ **簡潔架構** - 每個元件都是獨立、可測試、可重用的
- ✅ **作品集就緒** - 展示 AI 工程技能，而不只是跟著教學做

**目標讀者：** 自學 ML 並準備進入 AI 工程師職位的實務工作者，需要展現紮實基礎與生產級程式碼能力。

## 與其他 Transformer 教學的差異

| 一般教學 | 這個專案 |
|---------|---------|
| 單一 Jupyter notebook | 模組化 Python 套件 |
| 沒有測試 | 每個元件都有單元測試 |
| 最少文件 | 完整的 docstrings + README |
| 「只要能跑就好」 | 生產級程式碼架構 |
| 一個凌亂的 commit | 有思考的 git 歷史紀錄 |
| 沒有訓練範例 | 端到端訓練流程 |
| 需要 GPU | CPU 友善（小模型）|

**理念：** 如果你無法用簡潔且經過測試的程式碼來解釋它，代表你還不夠理解它。

## 專案狀態

🚧 **進行中** - 遵循 [PLAN.md](PLAN.md) 的開發計畫

- [x] Phase 1 部分完成：注意力機制與位置編碼
- [ ] Phase 1：基礎（注意力機制）
- [ ] Phase 2：架構（Encoder/Decoder）
- [ ] Phase 3：訓練（真實資料集）
- [ ] Phase 4：打磨（文件與範例）

## 環境需求

- Python 3.8+
- PyTorch 2.0+
- 不需要 GPU（使用小模型在 CPU 訓練）

## 快速開始

*（Phase 1 完成後提供）*

```python
from transformer import Transformer

# 初始化模型
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=256,
    num_heads=4,
    num_layers=2
)

# 訓練或推論...
```

## 專案結構

```
transformer-from-scratch-pytorch/
├── transformer/              # 核心實作
│   ├── attention.py         # 縮放點積注意力與多頭注意力
│   ├── encoder.py           # Encoder 層
│   ├── decoder.py           # Decoder 層
│   ├── positional_encoding.py
│   ├── feedforward.py
│   └── model.py             # 完整 Transformer
├── tests/                   # 單元測試
├── examples/                # 使用範例
├── PLAN.md                  # 開發路線圖
└── README.md               # 本檔案
```

## 學習資源

- **論文：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **視覺化：** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **實作參考：** [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## 授權

MIT License - 歡迎用於學習和作品集！
