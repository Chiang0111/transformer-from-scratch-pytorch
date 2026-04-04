# Transformer from Scratch (PyTorch)

[English](https://github.com/Chiang0111/transformer-from-scratch-pytorch) | [繁體中文](README.md)

一個**生產級**的 PyTorch Transformer 架構實作，基於論文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)，並包含**完整的教學註解**，就像一本完整的學習指南。

## 🌟 本專案的獨特之處

與一般實作不同，**每一行程式碼都有詳細註解**，包含：
- 📚 **為什麼需要** - 理解每個元件的動機
- 🔍 **如何運作** - 搭配具體例子的逐步說明
- 💡 **直覺理解** - Attention 用圖書館比喻、位置編碼用時鐘比喻
- 📊 **視覺化圖表** - ASCII 圖表展示資料流與轉換過程
- 🎯 **實際例子** - 用「我愛吃蘋果」追蹤整個架構的運作

**程式碼本身就是教學！** 閱讀原始碼檔案即可深入學習 Transformer。

## 為什麼要做這個專案

這個專案連接了「理解 Transformer 概念」與「專業實作」之間的橋樑。它展示了：

- ✅ **深度理解** - 從零實作，而不只是使用 `transformers` 函式庫
- ✅ **生產實務** - 模組化程式碼、型別提示、單元測試、完整文件
- ✅ **簡潔架構** - 每個元件都是獨立、可測試、可重用的
- ✅ **教學卓越** - 1,500+ 行解釋註解（比程式碼本身還多！）
- ✅ **作品集就緒** - 展示 AI 工程技能，而不只是跟著教學做

**目標讀者：** 自學 ML 並準備進入 AI 工程師職位的實務工作者，需要展現紮實基礎與生產級程式碼能力。

## 與其他 Transformer 教學的差異

| 一般教學 | 這個專案 |
|---------|---------|
| 單一 Jupyter notebook | 模組化 Python 套件 |
| 沒有測試 | 每個元件都有單元測試（35 個測試）|
| 最少文件 | **1,500+ 行教學註解** |
| 簡短的行內註解 | 程式碼內含完整學習指南 |
| 「只要能跑就好」 | 生產級程式碼架構 |
| 一個凌亂的 commit | 有思考的 git 歷史紀錄與詳細 commit |
| 沒有訓練範例 | 端到端訓練流程（即將推出）|
| 需要 GPU | CPU 友善（小模型）|
| 只有程式碼 | 程式碼 + 直覺比喻 + 視覺化圖表 |

**理念：** 如果你無法用簡潔且經過測試、並有完整文件的程式碼來解釋它，代表你還不夠理解它。

## 📖 如何從這個專案學習

**本專案設計成像教科書一樣閱讀！** 從這裡開始：

1. **從 Attention 開始** (`transformer/attention.py`)
   - 用圖書館比喻理解 Q, K, V
   - 學習為什麼縮放很重要（√d_k 的解釋）
   - 了解多頭注意力的運作（8 個專家的比喻）

2. **加入位置資訊** (`transformer/positional_encoding.py`)
   - 理解為什麼 Attention 對順序不敏感
   - 用時鐘比喻學習 sin/cos 編碼
   - 看具體的逐位置範例

3. **處理資訊** (`transformer/feedforward.py`)
   - 理解為什麼 Attention 之後還需要 FFN
   - 學習展開→轉換→壓縮的模式
   - 比較 ReLU vs GELU 激活函數

4. **完整的層** (`transformer/encoder.py`)
   - 看所有元件如何整合
   - 理解殘差連接與層歸一化
   - 追蹤「我愛吃蘋果」通過整個編碼器

**每個檔案都包含：**
- 詳細的「為什麼」解釋
- 逐步的「如何」分解
- 具體的數值範例
- 視覺化 ASCII 圖表
- 常見陷阱與解決方案

## 專案狀態

🚧 **進行中** - 遵循 [PLAN.md](PLAN.md) 的開發計畫

- [x] **Phase 1：基礎** ✅ 完成（35 個測試通過）
  - ✅ Attention 機制（包含完整的 Q/K/V 解釋）
  - ✅ 位置編碼（用時鐘比喻解釋 sin/cos 函數）
  - ✅ 前饋網路（用圖書館比喻說明 FFN 的角色）
  - ✅ Encoder 層（完整架構含殘差連接與正規化）
  - ✅ **1,500+ 行教學註解** - 閱讀程式碼即可學習！
- [ ] Phase 2：架構（Decoder/完整 Transformer）
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
