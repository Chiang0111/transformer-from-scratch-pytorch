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
| 沒有測試 | 每個元件都有單元測試（80 個測試）|
| 最少文件 | **2,500+ 行教學註解** |
| 簡短的行內註解 | 程式碼內含完整學習指南 |
| 「只要能跑就好」 | 生產級程式碼架構 |
| 一個凌亂的 commit | 有思考的 git 歷史紀錄與詳細 commit |
| 沒有訓練範例 | **完整訓練流程** ✅ |
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

4. **建構 Encoder** (`transformer/encoder.py`)
   - 看所有元件如何整合
   - 理解殘差連接與層歸一化
   - 追蹤「我愛吃蘋果」通過整個編碼器

5. **加入 Decoder** (`transformer/decoder.py`)
   - 學習遮罩自注意力（因果遮罩）
   - 理解到編碼器記憶的交叉注意力
   - 看自回歸生成的逐步過程

6. **完整模型** (`transformer/transformer.py`)
   - 整合所有元件
   - 訓練 vs 推論模式
   - 從詞元到 logits 的端到端資料流

**每個檔案都包含：**
- 詳細的「為什麼」解釋
- 逐步的「如何」分解
- 具體的數值範例
- 視覺化 ASCII 圖表
- 常見陷阱與解決方案

## 專案狀態

🚧 **進行中** - 遵循 [PLAN.md](PLAN.md) 的開發計畫

- [x] **Phase 1：基礎** ✅ 完成
  - ✅ Attention 機制（包含完整的 Q/K/V 解釋）
  - ✅ 位置編碼（用時鐘比喻解釋 sin/cos 函數）
  - ✅ 前饋網路（用圖書館比喻說明 FFN 的角色）
  - ✅ Encoder 層（完整架構含殘差連接與正規化）
- [x] **Phase 2：Decoder** ✅ 完成
  - ✅ Decoder 層（遮罩自注意力 + 交叉注意力）
  - ✅ 自回歸生成的因果遮罩
  - ✅ Encoder-Decoder 整合測試
- [x] **Phase 3：完整模型與訓練** ✅ 完成（80 個測試通過）
  - ✅ 帶縮放的詞元嵌入
  - ✅ 完整 Transformer 模型（Encoder + Decoder）
  - ✅ 自回歸生成（推論模式）
  - ✅ **訓練基礎設施** - 資料集、工具、訓練迴圈
  - ✅ **3 個訓練任務** - 複製、反轉、排序（不需要外部資料！）
  - ✅ **完整訓練指南** - 參見 [TRAINING.md](TRAINING.md)
  - ✅ **2,500+ 行教學註解** - 閱讀程式碼即可學習！
- [ ] Phase 4：打磨（文件與範例）

## 環境需求

- Python 3.8+
- PyTorch 2.0+
- 不需要 GPU（使用小模型在 CPU 訓練）

## 快速開始

### 安裝
```bash
pip install torch pytest
```

### 執行測試
```bash
pytest tests/ -v  # 80 個測試應該通過
```

### 訓練模型
```bash
# ⚠️ 重要：小模型需要較高的學習率（--lr-factor 10.0）
# 詳見 TRAINING.md 的說明

# 訓練複製任務（最簡單，CPU 上約 15 分鐘）
python train.py --task copy --epochs 30 --lr-factor 10.0

# 訓練反轉任務（中等難度）
python train.py --task reverse --epochs 40 --lr-factor 10.0

# 訓練排序任務（最困難）
python train.py --task sort --epochs 60 --lr-factor 10.0
```

**📖 完整訓練指南請參見 [TRAINING.md](TRAINING.md)**

### 使用模型
```python
from transformer import create_transformer
import torch

# 創建小型 Transformer（CPU 友善）
model = create_transformer(
    src_vocab_size=10000,  # 英文詞彙表
    tgt_vocab_size=8000,   # 中文詞彙表
    d_model=256,           # 比論文的 512 小
    num_heads=4,           # 較少的頭數適合 CPU
    num_layers=2,          # 較淺以便更快訓練
    d_ff=1024              # 較小的 FFN
)

print(f"模型參數數量：{model.count_parameters():,}")

# 訓練模式：教師強迫
src = torch.randint(0, 10000, (2, 10))  # batch=2, src_len=10
tgt = torch.randint(0, 8000, (2, 8))    # batch=2, tgt_len=8
logits = model(src, tgt)  # (2, 8, 8000)

# 推論模式：自回歸生成
model.eval()
generated = model.generate(src, max_len=20, start_token=1, end_token=2)
print(f"生成結果：{generated.shape}")  # (2, <=20)
```

### 測試已訓練的模型
```bash
# 在測試集上評估
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy

# 互動測試
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive
```

參見 [TRAINING.md](TRAINING.md) 獲取詳細訓練指南。

## 專案結構

```
transformer-from-scratch-pytorch/
├── transformer/              # 核心實作（2,500+ 行註解）
│   ├── attention.py         # 縮放點積注意力與多頭注意力
│   ├── positional_encoding.py  # 正弦位置嵌入
│   ├── feedforward.py       # 位置前饋 FFN
│   ├── encoder.py           # Encoder 層（雙向注意力）
│   ├── decoder.py           # Decoder 層（遮罩 + 交叉注意力）
│   ├── transformer.py       # 完整 Transformer 模型 ⭐
│   └── __init__.py          # 公開 API
├── tests/                   # 完整單元測試（80 個測試）
│   ├── test_attention.py    # 7 個測試
│   ├── test_positional_encoding.py  # 5 個測試
│   ├── test_feedforward.py  # 8 個測試
│   ├── test_encoder.py      # 14 個測試
│   ├── test_decoder.py      # 20 個測試
│   ├── test_transformer.py  # 26 個測試 ⭐
│   └── README.md            # 測試文件
├── datasets.py              # 訓練任務（複製/反轉/排序）⭐
├── utils.py                 # 訓練工具 ⭐
├── train.py                 # 主訓練腳本 ⭐
├── test.py                  # 模型評估 ⭐
├── TRAINING.md              # 完整訓練指南 📚
├── PLAN.md                  # 開發路線圖
└── README.md                # 本檔案
```

## 學習資源

- **論文：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **視覺化：** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **實作參考：** [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## 授權

MIT License - 歡迎用於學習和作品集！
