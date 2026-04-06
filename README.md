# Transformer 從零開始 (PyTorch)

[![測試](https://img.shields.io/badge/tests-80%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![授權](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[English](https://github.com/Chiang0111/transformer-from-scratch-pytorch) | [繁體中文](README.md)

基於論文 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) 的**生產級** PyTorch Transformer 架構實作，包含 **2,500+ 行教學註解**，作為完整的學習指南。

---

## 🌟 本專案的獨特之處

與一般實作不同，**每一行程式碼都有詳細的文件說明**，包含：
- 📚 **為什麼需要** - 理解每個元件背後的動機
- 🔍 **如何運作** - 搭配具體範例的逐步解釋
- 💡 **直覺比喻** - 用圖書館比喻 Attention、用時鐘比喻位置編碼
- 📊 **視覺化圖表** - ASCII 圖表展示資料流與轉換過程
- 🎯 **實際範例** - 「我愛吃蘋果」完整追蹤整個架構

**程式碼本身就是教學！** 閱讀原始碼檔案即可深入學習 Transformer。

### ✨ 為什麼要建立這個專案

這個專案連接了「理解 Transformer 概念」與「專業實作」之間的橋樑：

- ✅ **深度理解** - 從零實作，而不只是使用 `transformers` 函式庫
- ✅ **生產實務** - 模組化程式碼、型別提示、單元測試、完整文件
- ✅ **簡潔架構** - 每個元件都是獨立、可測試、可重用的
- ✅ **教學卓越** - 2,500+ 行解釋註解（比程式碼本身還多！）
- ✅ **驗證結果** - 在 3 個任務上驗證（98.6%、83%、96% 準確率）
- ✅ **作品集就緒** - 展示 AI 工程技能，而不只是跟著教學做

**目標讀者：** 自學 ML 並準備進入 AI 工程師職位的實務工作者，需要展現紮實基礎與生產級程式碼能力。

---

## 📁 專案結構

```
transformer-from-scratch-pytorch/
│
├── README.md                       ⭐ 你在這裡
├── LICENSE                         📜 MIT 授權
├── CONTRIBUTING.md                 🤝 貢獻指南
├── requirements.txt                📦 相依套件
│
├── transformer/                    🧠 核心實作（2,500+ 行註解）
│   ├── __init__.py                    公開 API
│   ├── attention.py                   縮放點積注意力與多頭注意力
│   ├── positional_encoding.py         正弦/餘弦位置編碼
│   ├── feedforward.py                 位置前饋網路
│   ├── encoder.py                     Transformer 編碼器層
│   ├── decoder.py                     Transformer 解碼器層（含遮罩）
│   └── transformer.py                 ⭐ 完整 Transformer 模型
│
├── tests/                          ✅ 單元測試（80 個測試，全部通過）
│   ├── test_attention.py              測試注意力機制
│   ├── test_positional_encoding.py    測試位置編碼
│   ├── test_feedforward.py            測試前饋網路層
│   ├── test_encoder.py                測試編碼器
│   ├── test_decoder.py                測試解碼器
│   ├── test_transformer.py            ⭐ 完整模型測試
│   └── test_training.py               訓練流程測試
│
├── scripts/                        🚀 訓練與評估
│   ├── train.py                       主要訓練腳本
│   ├── test.py                        模型評估
│   ├── demo.py                        互動式示範
│   └── benchmark.py                   自動化基準測試
│
├── examples/                       💡 使用範例與除錯
│   ├── basic_usage.py                 ⭐ 從這裡開始！簡單範例
│   ├── debug_data.py                  檢查資料集
│   ├── debug_gradients.py             檢查梯度流
│   ├── test_overfit.py                驗證模型可以過擬合
│   └── ...                            其他除錯工具
│
├── docs/                           📚 文件
│   ├── TRAINING.md                    完整訓練指南
│   ├── TROUBLESHOOTING.md             除錯指南
│   ├── RESULTS.md                     基準測試結果
│   ├── VALIDATION.md                  測試方法論
│   └── PLAN.md                        開發路線圖
│
├── datasets.py                     📊 訓練資料集
│   ├── SequenceCopyDataset            複製序列（最簡單）
│   ├── SequenceReverseDataset         反轉序列（中等）
│   └── SequenceSortDataset            排序序列（困難）
│
├── utils.py                        🛠️ 訓練工具
│   ├── LabelSmoothingLoss             標籤平滑（用於翻譯）
│   ├── TransformerLRScheduler         學習率排程
│   ├── create_masks()                 填充與因果遮罩
│   ├── TrainingMetrics                追蹤準確率/損失
│   └── save/load_checkpoint()         檢查點管理
│
├── benchmarks/                     🏆 訓練好的模型檢查點
│   ├── copy/                          98.6% 準確率（已驗證）
│   ├── reverse/                       83.0% 準確率（已驗證）
│   └── sort/                          96.0% 準確率（已驗證）
│
└── checkpoints/                    💾 你的訓練檢查點
    └── .gitkeep                       （實驗用目錄）
```

### 📂 重點檔案探索

| 檔案 | 功能 | 從這裡開始？ |
|------|------|-------------|
| `examples/basic_usage.py` | 簡單使用範例 | ✅ **是** |
| `transformer/attention.py` | 核心注意力機制 | ✅ **是** |
| `transformer/transformer.py` | 完整模型 | 理解各部分後 |
| `scripts/train.py` | 訓練你自己的模型 | 閱讀文件後 |
| `docs/TRAINING.md` | 完整訓練指南 | 訓練前 |
| `tests/test_transformer.py` | 看所有東西如何運作 | 理解測試 |

---

## 🚀 快速開始

### 安裝

```bash
# 複製專案
git clone https://github.com/Chiang0111/transformer-from-scratch-pytorch.git
cd transformer-from-scratch-pytorch

# 切換到中文分支
git checkout zh-CN

# 安裝相依套件
pip install -r requirements.txt

# 執行測試驗證安裝
pytest tests/ -v
```

**輸出：**
```
========================= 80 passed in 12.34s =========================
```

### 執行基本範例

```bash
python examples/basic_usage.py
```

這會展示如何：
1. 建立 transformer 模型
2. 準備輸入資料
3. 執行前向傳播（訓練模式）
4. 生成序列（推論模式）

### 訓練你的第一個模型

```bash
# 複製任務（最簡單，CPU 上約 10 分鐘）
python scripts/train.py --task copy --epochs 20 \
    --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**預期輸出：**
```
Epoch 20/20: Val Seq Acc: 98.6%
✅ 測試集結果：98.6% 準確率
```

參見 [`docs/TRAINING.md`](docs/TRAINING.md) 取得完整訓練指南。

---

## 📖 學習路徑

### 1. 從範例開始（15 分鐘）

```bash
# 理解基本用法
python examples/basic_usage.py

# 看資料長什麼樣子
python examples/debug_data.py
```

### 2. 閱讀核心程式碼（1-2 小時）

**建議閱讀順序：**

1. **`transformer/attention.py`**（從這裡開始！）
   - 用圖書館比喻理解 Q、K、V
   - 學習為什麼縮放很重要（√d_k 的解釋）
   - 看多頭注意力如何運作

2. **`transformer/positional_encoding.py`**
   - 學習為什麼位置很重要
   - 用時鐘比喻理解 sin/cos 編碼

3. **`transformer/feedforward.py`**
   - 看為什麼注意力之後需要 FFN
   - 理解展開→轉換→壓縮模式

4. **`transformer/encoder.py`**
   - 看所有元件如何整合
   - 追蹤範例「我愛吃蘋果」通過編碼器

5. **`transformer/decoder.py`**
   - 學習遮罩自注意力（因果遮罩）
   - 理解到編碼器記憶的交叉注意力

6. **`transformer/transformer.py`**（⭐ 完整圖像）
   - 所有元件的整合
   - 訓練 vs 推論模式
   - 端到端資料流

### 3. 訓練模型（30-60 分鐘）

```bash
# 複製任務（最簡單）
python scripts/train.py --task copy --epochs 20 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0

# 反轉任務（中等）
python scripts/train.py --task reverse --epochs 30 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0

# 排序任務（最困難）
python scripts/train.py --task sort --epochs 10 --fixed-lr 0.0005 \
    --label-smoothing 0.0 --dropout 0.0
```

### 4. 測試訓練好的模型

```bash
# 在測試集上評估
python scripts/test.py --checkpoint benchmarks/copy/checkpoint_best.pt --task copy

# 互動模式 - 試試你自己的序列！
python scripts/test.py --checkpoint benchmarks/copy/checkpoint_best.pt \
    --task copy --interactive
```

---

## 📊 驗證結果

所有三個任務都**顯著超越**最低目標：

| 任務 | 測試準確率 | 目標 | 訓練時間 | 狀態 |
|------|-----------|------|---------|------|
| **複製** | 98.6% | 95.0% | ~10 分鐘 | ✅ 通過 |
| **反轉** | 83.0% | 80.0% | ~20 分鐘 | ✅ 通過 |
| **排序** | 96.0% | 70.0% | ~25 分鐘 | ✅ 通過 |

**關鍵見解：**
- 所有結果都在**保留測試集**上（訓練時從未見過）
- 固定學習率方法對小模型比 Transformer 排程更有效
- 模型學習演算法推理，而不只是記憶
- 排序任務在 3 個 epoch 內就收斂了（不是 50 個！）

參見 [`docs/RESULTS.md`](docs/RESULTS.md) 取得詳細分析。

---

## 💻 使用範例

```python
from transformer import create_transformer
import torch

# 建立小型 transformer（CPU 友善）
model = create_transformer(
    src_vocab_size=10000,  # 英文詞彙表
    tgt_vocab_size=8000,   # 中文詞彙表
    d_model=256,           # 比論文的 512 小
    num_heads=4,           # 較少的頭數適合 CPU
    num_layers=2,          # 較淺以便更快訓練
    d_ff=1024              # 較小的 FFN
)

print(f"模型參數數量：{model.count_parameters():,}")
# 輸出：模型參數數量：16,234,528

# 訓練模式：教師強迫
src = torch.randint(0, 10000, (2, 10))  # batch=2, src_len=10
tgt = torch.randint(0, 8000, (2, 8))    # batch=2, tgt_len=8
logits = model(src, tgt)                # (2, 8, 8000)

# 推論模式：自回歸生成
model.eval()
generated = model.generate(src, max_len=20, start_token=1, end_token=2)
print(f"生成結果：{generated.shape}")  # (2, <=20)
```

---

## 🧪 測試

```bash
# 執行所有 80 個測試
pytest tests/ -v

# 執行特定測試檔案
pytest tests/test_attention.py -v

# 執行涵蓋率測試
pytest tests/ --cov=transformer --cov-report=html
```

**測試涵蓋率：**
- ✅ 80 個單元測試涵蓋所有元件
- ✅ 端到端流程的整合測試
- ✅ 冒煙測試、品質測試、穩健性測試
- ✅ 所有測試在 CPU 上通過（不需要 GPU）

---

## 📚 文件

| 文件 | 說明 |
|------|------|
| [`docs/TRAINING.md`](docs/TRAINING.md) | 包含所有超參數的完整訓練指南 |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | 含決策樹的除錯指南 |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 詳細的基準測試結果與見解 |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | 測試方法論與 ML 最佳實踐 |
| [`docs/PLAN.md`](docs/PLAN.md) | 開發路線圖與專案狀態 |

---

## 🎯 與其他 Transformer 教學的差異

| 一般教學 | 這個專案 |
|---------|---------|
| 單一 Jupyter notebook | 模組化 Python 套件 |
| 沒有測試 | 80 個單元測試，全部通過 |
| 最少文件 | **2,500+ 行教學註解** |
| 簡短的行內註解 | 程式碼內含完整學習指南 |
| 「只要能跑就好」 | 生產級程式碼架構 |
| 一個凌亂的 commit | 有思考的 git 歷史紀錄與詳細 commit |
| 沒有訓練範例 | **完整訓練流程** |
| 需要 GPU | CPU 友善（小模型）|
| 只有程式碼 | 程式碼 + 比喻 + 視覺化圖表 |
| 照抄論文超參數 | **針對小模型調整**（實際可用！）|
| 沒有驗證 | **在 3 個任務上用測試集驗證** |

**理念：** 如果你無法用簡潔且經過測試、並有完整文件的程式碼來解釋它，代表你還不夠理解它。

---

## 🤝 貢獻

歡迎貢獻！這是一個教學專案，所以清晰和全面的註解比效能優化更重要。

參見 [`CONTRIBUTING.md`](CONTRIBUTING.md) 取得指南。

**貢獻方式：**
- 🐛 回報錯誤或問題
- 📝 改進文件或註解
- ✨ 新增訓練任務
- 🌍 翻譯文件
- 🎨 新增視覺化工具
- 📚 建立教學或範例

---

## 📜 授權

MIT License - 歡迎用於學習和作品集！

參見 [`LICENSE`](LICENSE) 取得詳細資訊。

---

## 🙏 致謝

- 原始論文：Vaswani 等人的 ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- 靈感來源：Jay Alammar 的 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 🔗 資源

### 學習更多
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 視覺化解釋
- 📄 [原始論文](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need"
- 🎓 [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - 另一個很棒的資源

### 相關專案
- [PyTorch Transformers](https://github.com/huggingface/transformers) - 生產級函式庫（Hugging Face）
- [Fairseq](https://github.com/facebookresearch/fairseq) - Facebook 的序列建模工具包

---

**致力於透過完整的文件與簡潔的程式碼，讓每個人都能理解 Transformer。**
