# 儲存庫重組計劃

## 目前的問題
- 根目錄中有 30+ 個檔案（雜亂）
- 14 個 markdown 檔案（許多冗餘）
- 除錯腳本與核心程式碼混合
- 難以找到需要的內容

## 建議的結構

```
transformer-from-scratch-pytorch/
│
├── README.md                    ⭐ 新：包含檔案指南的全面說明
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
│
├── docs/                        📚 所有文件
│   ├── TRAINING.md             (訓練使用者指南)
│   ├── TROUBLESHOOTING.md      (除錯指南)
│   ├── RESULTS.md              (基準測試結果)
│   ├── VALIDATION.md           (測試方法)
│   └── PLAN.md                 (開發藍圖)
│
├── transformer/                 🧠 核心實作
│   ├── __init__.py
│   ├── attention.py
│   ├── positional_encoding.py
│   ├── feedforward.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
│
├── scripts/                     🚀 訓練和測試腳本
│   ├── train.py                (主訓練腳本)
│   ├── test.py                 (模型評估)
│   ├── demo.py                 (互動式示範)
│   └── benchmark.py            (執行所有基準測試)
│
├── examples/                    💡 範例腳本和除錯
│   ├── basic_usage.py          (新：簡單使用範例)
│   ├── debug_data.py
│   ├── debug_gradients.py
│   ├── debug_encoder.py
│   ├── debug_generation.py
│   ├── test_overfit.py
│   └── test_simple_training.py
│
├── tests/                       ✅ 單元測試（80 個測試）
│   ├── test_attention.py
│   ├── test_positional_encoding.py
│   ├── test_feedforward.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_transformer.py
│   └── test_training.py
│
├── datasets.py                  📊 資料集實作
├── utils.py                     🛠️ 訓練工具
│
├── benchmarks/                  🏆 訓練好的模型檢查點
│   ├── copy/
│   ├── reverse/
│   └── sort/
│
└── checkpoints/                 💾 供使用者訓練（空的）
```

## 要進行的變更

### 1. 建立 `docs/` 資料夾
```bash
mkdir -p docs/archive
mv TRAINING.md TROUBLESHOOTING.md VALIDATION.md PLAN.md docs/
mv FINAL_RESULTS.md docs/RESULTS.md
```

### 2. 封存內部文件
```bash
mv BENCHMARK_STATUS.md CHANGES_ML_EVAL.md ISSUE_SUMMARY.md \
   NEXT_STEPS.md PHASE3_SUMMARY.md SESSION_SUMMARY.md \
   STATUS.md TRAINING_ISSUES.md \
   DOCUMENTATION_REVIEW.md OPEN_SOURCE_CHECKLIST.md \
   QUICK_START_OPEN_SOURCE.md \
   docs/archive/
```

### 3. 建立 `scripts/` 資料夾
```bash
mkdir -p scripts
mv train.py test.py demo.py benchmark.py scripts/
```

### 4. 建立 `examples/` 資料夾
```bash
mkdir -p examples
mv debug_*.py test_overfit.py test_simple_training.py debug_lr_schedule.py examples/
```

### 5. 更新匯入路徑（如需要）
移動腳本後，更新以下檔案中的任何相對匯入：
- scripts/train.py
- scripts/test.py
- scripts/demo.py
- scripts/benchmark.py

在每個腳本頂部新增：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 6. 更新 .gitignore
```
# 來自使用者訓練的檢查點
checkpoints/*
!checkpoints/.gitkeep

# 保留基準測試結果
!benchmarks/

# 文件封存（僅內部）
docs/archive/
```

### 7. 建立 .gitkeep 檔案
```bash
touch checkpoints/.gitkeep
```

## 檔案數量減少

**之前：** 根目錄中 30+ 個檔案
**之後：** 根目錄中 6 個檔案（README、LICENSE、CONTRIBUTING、requirements.txt、datasets.py、utils.py）

**減少：** 根目錄中的檔案減少 80%！

## 好處

1. ✅ **清晰導航** - 確切知道在哪裡找到東西
2. ✅ **專業結構** - 符合業界標準
3. ✅ **容易上手** - 新使用者可以快速理解佈局
4. ✅ **可擴展** - 容易新增新功能/文件
5. ✅ **乾淨的根目錄** - 僅可見必要檔案

## 更新的 README 結構

```markdown
# Transformer from Scratch

[徽章]

## 📁 儲存庫結構
(清楚指南說明什麼在哪裡)

## 🚀 快速開始
(3 個命令內安裝並執行)

## 📚 文件
(連結至 docs/ 資料夾)

## 🧠 架構
(簡要概述與連結)

## 📊 結果
(連結至 docs/RESULTS.md)

## 🤝 貢獻
(連結至 CONTRIBUTING.md)
```

---

**準備好執行這個計劃了嗎？**
