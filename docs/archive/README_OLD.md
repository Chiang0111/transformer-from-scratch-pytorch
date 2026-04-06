# Transformer from Scratch (PyTorch)

[English](README.md) | [中文](https://github.com/Chiang0111/transformer-from-scratch-pytorch/tree/zh-CN)

一個來自 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 的**生產就緒** PyTorch Transformer 架構實作，包含**全面的教育註解**，作為完整的學習指南。

## 🌟 什麼讓這個專案特別

與典型的實作不同，**每一行程式碼都有廣泛的文件記錄**，包括：
- 📚 **為什麼需要它** - 理解每個元件背後的動機
- 🔍 **它如何運作** - 具體範例的逐步解釋
- 💡 **直觀的類比** - Attention 的圖書館類比、位置編碼的時鐘類比
- 📊 **視覺圖表** - 顯示資料流和轉換的 ASCII 藝術
- 🎯 **真實範例** - "I love eating apples" 追蹤整個架構

**程式碼本身就是教學！** 閱讀原始檔案以深入學習 Transformers。

## 為什麼這個儲存庫存在

這個專案彌合了概念上理解 transformers 和專業實作它們之間的差距。它展示：

- ✅ **深入理解** - 從零開始建構，而非僅使用 `transformers` 函式庫
- ✅ **生產實踐** - 模組化程式碼、型別提示、單元測試、適當的文件
- ✅ **乾淨的架構** - 每個元件都是隔離的、已測試的和可重用的
- ✅ **教育卓越** - 1,500+ 行解釋性註解（比程式碼本身還多！）
- ✅ **作品集就緒** - 展示 AI 工程技能，而非僅遵循教學

**目標受眾：** 為 AI 工程師職位做準備的自學 ML 實踐者，需要展示紮實的基礎和生產編碼技能。

## 這與其他 Transformer 教學有何不同

| 大多數教學 | 這個儲存庫 |
|------------|-----------|
| 單一 Jupyter notebook | 模組化 Python 套件 |
| 無測試 | 每個元件的單元測試（80 個測試） |
| 最少的文件 | **2,500+ 行教育註解** |
| 簡短的行內註解 | 程式碼中的完整學習指南 |
| 「只要能運作」 | 生產就緒的程式碼結構 |
| 一個混亂的提交 | 具有詳細提交的周到 git 歷史 |
| 無訓練範例 | **完整的訓練管道** ✅ |
| 需要 GPU | CPU 友善（小型模型） |
| 僅程式碼 | 程式碼 + 直觀類比 + 視覺圖表 |
| 複製貼上論文超參數 | **針對小型模型調整**（實際有效！） |

**理念：** 如果你無法用乾淨、經過測試且具有全面文件的程式碼解釋它，你就還不夠理解它。

## 專案狀態

🚧 **進行中** - 遵循 [PLAN.md](PLAN.md) 中的開發計劃

- [x] **階段 1：基礎** ✅ 完成
  - ✅ 注意力機制（具有全面的 Q/K/V 解釋）
  - ✅ 位置編碼（sin/cos 函數以時鐘類比解釋）
  - ✅ 前饋網路（FFN 角色以圖書館類比澄清）
  - ✅ Encoder 層（具有殘差和標準化的完整架構）
- [x] **階段 2：Decoder** ✅ 完成
  - ✅ Decoder 層（遮罩自注意力 + 交叉注意力）
  - ✅ 用於自回歸生成的因果遮罩
  - ✅ Encoder-Decoder 整合測試
- [x] **階段 3：完整模型和訓練** ✅ 完成（80 個測試通過）
  - ✅ 帶縮放的詞元嵌入
  - ✅ 完整 Transformer 模型（Encoder + Decoder）
  - ✅ 自回歸生成（推論模式）
  - ✅ **訓練基礎設施** - 資料集、工具、訓練迴圈
  - ✅ **3 個訓練任務** - 複製、反轉、排序（無外部資料！）
  - ✅ **完整訓練指南** - 參見 [TRAINING.md](TRAINING.md)
  - ✅ **2,500+ 行教育註解** - 透過閱讀程式碼學習！
- [ ] 階段 4：打磨（文件和範例）

## 📖 如何從這個儲存庫學習

**這個儲存庫設計成像教科書一樣閱讀！** 從這裡開始：

1. **從 Attention 開始**（`transformer/attention.py`）
   - 使用圖書館類比理解 Q、K、V
   - 學習為何縮放至關重要（√d_k 解釋）
   - 看看多頭注意力如何運作（8 個專家類比）

2. **新增位置資訊**（`transformer/positional_encoding.py`）
   - 理解為何 Attention 是順序不可知的
   - 使用時鐘類比學習 sin/cos 編碼
   - 看看逐位置的具體範例

3. **處理資訊**（`transformer/feedforward.py`）
   - 理解為何 Attention 後需要 FFN
   - 學習擴展→轉換→壓縮模式
   - 比較 ReLU vs GELU 激活

4. **建構 Encoder**（`transformer/encoder.py`）
   - 看看所有元件如何整合
   - 理解殘差連接和層標準化
   - 追蹤 "I love eating apples" 通過整個 encoder

5. **新增 Decoder**（`transformer/decoder.py`）
   - 學習遮罩自注意力（因果遮罩）
   - 理解對 encoder 記憶的交叉注意力
   - 看看逐步的自回歸生成

6. **完整模型**（`transformer/transformer.py`）
   - 所有元件的整合
   - 訓練 vs 推論模式
   - 從詞元到 logits 的端到端資料流

**每個檔案包含：**
- 詳細的「為什麼」解釋
- 逐步的「如何」分解  
- 具體的數值範例
- 視覺 ASCII 圖表
- 常見陷阱和解決方案

## 需求

- Python 3.8+
- PyTorch 2.0+
- 不需要 GPU（針對小型模型的 CPU 訓練優化）

## 快速開始

### 安裝
```bash
pip install torch pytest
```

### 執行測試
```bash
pytest tests/ -v  # 應通過 80 個測試
```

### 訓練模型
```bash
# ✅ 使用固定學習率（不是 Transformer 排程！）

# 訓練複製任務（最簡單，CPU 上約 10 分鐘，95-99% 準確率）
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0

# 訓練反轉任務（中等難度，約 20 分鐘，85-95% 準確率）
python train.py --task reverse --epochs 30 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0

# 訓練排序任務（最難，約 30 分鐘，70-85% 準確率）
python train.py --task sort --epochs 50 --fixed-lr 0.0005 --label-smoothing 0.0 --dropout 0.0
```

**📖 完整訓練指南請參見 [TRAINING.md](TRAINING.md)**  
**🐛 模型無法學習？參見 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### 使用模型
```python
from transformer import create_transformer
import torch

# 建立小型 Transformer（CPU 友善）
model = create_transformer(
    src_vocab_size=10000,  # 英文詞彙表
    tgt_vocab_size=8000,   # 中文詞彙表
    d_model=256,           # 比論文的 512 小
    num_heads=4,           # CPU 的較少頭數
    num_layers=2,          # 更淺以便更快訓練
    d_ff=1024              # 較小的 FFN
)

print(f"模型參數：{model.count_parameters():,}")

# 訓練模式：教師強迫
src = torch.randint(0, 10000, (2, 10))  # batch=2, src_len=10
tgt = torch.randint(0, 8000, (2, 8))    # batch=2, tgt_len=8
logits = model(src, tgt)  # (2, 8, 8000)

# 推論模式：自回歸生成
model.eval()
generated = model.generate(src, max_len=20, start_token=1, end_token=2)
print(f"已生成：{generated.shape}")  # (2, <=20)
```

### 測試訓練好的模型
```bash
# 在測試集上評估
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy

# 互動式測試
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive
```

詳細訓練指南請參見 [TRAINING.md](TRAINING.md)。

## 專案結構

```
transformer-from-scratch-pytorch/
├── transformer/              # 核心實作（2,500+ 行註解）
│   ├── attention.py         # 縮放點積和多頭注意力
│   ├── positional_encoding.py  # 正弦位置嵌入
│   ├── feedforward.py       # 位置式 FFN
│   ├── encoder.py           # Encoder 層（雙向注意力）
│   ├── decoder.py           # Decoder 層（遮罩 + 交叉注意力）
│   ├── transformer.py       # 完整 Transformer 模型 ⭐
│   └── __init__.py          # 公開 API
├── tests/                   # 全面的單元測試（80 個測試）
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
├── PLAN.md                  # 開發藍圖
└── README.md                # 本檔案
```

## 學習資源

- **論文：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **視覺化：** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **實作參考：** [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## 授權

MIT License - 歡迎用於學習和作品集！
