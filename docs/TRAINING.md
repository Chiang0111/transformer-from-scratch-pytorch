# 訓練指南

在序列任務上訓練 Transformer 模型的完整指南。

---

## 🚀 快速開始（複製貼上即可用）

### 訓練複製任務（最簡單 - 10 分鐘）
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```
**預期結果：** 95-99% 準確率

### 訓練反轉任務（中等 - 20 分鐘）
```bash
python train.py --task reverse --epochs 30 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```
**預期結果：** 85-95% 準確率

### 訓練排序任務（困難 - 30 分鐘）
```bash
python train.py --task sort --epochs 50 --fixed-lr 0.0005 --label-smoothing 0.0 --dropout 0.0
```
**預期結果：** 70-85% 準確率

---

## ⚠️ 重要：為什麼使用這些參數？

### `--fixed-lr 0.001` - 使用固定學習率

**不要在小型模型上使用 Transformer 學習率排程！**

原始論文的排程（`--lr-factor`、`--warmup-steps`）是為以下情境設計的：
- 大型模型（d_model=512、6 層、約 100M 參數）
- 複雜任務（具有數百萬樣本的翻譯任務）

對於小型模型（d_model=128、2 層、約 1M 參數），該排程會產生**過高 5-40 倍**的學習率，導致訓練完全失敗。

✅ **使用：** `--fixed-lr 0.001`（簡單且有效）  
❌ **不要使用：** `--lr-factor 10.0 --warmup-steps 500`（會失敗）

### `--label-smoothing 0.0` - 停用標籤平滑

標籤平滑會將機率分散到錯誤答案上，這會：
- ✅ 有助於翻譯（多個有效輸出）
- ❌ 有害於演算法任務（唯一正確答案）

對於複製/反轉/排序：永遠使用 `0.0`

### `--dropout 0.0` - 對小型資料集停用 Dropout

只有 10K 訓練樣本時，dropout 可能會干擾學習。
- 小型資料集（< 10K）：`--dropout 0.0`
- 大型資料集（> 50K）：`--dropout 0.1`

**→ 如果模型沒有學習，請參閱 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

---

## 📊 資料分割（正確的機器學習實務）

訓練腳本會自動將資料分割為三個集合：

- **訓練集（80%）：** 用於訓練模型
- **驗證集（10%）：** 用於早停和選擇最佳檢查點
- **測試集（10%）：** **訓練期間從未見過** - 僅用於最終評估

**為什麼重要：** 使用驗證集選擇最佳檢查點意味著模型「間接看到」該資料。測試集提供對未見資料真實效能的無偏估計。這是防止資料洩漏的基本機器學習衛生習慣。

**輸出中您會看到：**
```
Dataset ready:
   Task: copy
   Train samples: 8000
   Val samples: 1000
   Test samples: 1000   ← 保留用於最終評估
```

訓練完成後：
```
>> FINAL TEST SET EVALUATION
Test Set Results:
   Seq Acc:   98.60%   ← 這是真實的效能指標
```

測試準確率是您應該報告的數據 - 它代表真正的泛化效能。

---

## 📋 可用任務

### 1. 複製任務 ⭐☆☆☆☆（最簡單）

學習完全複製輸入序列。

```
輸入：  [5, 7, 3, 9]
輸出：  [5, 7, 3, 9]
```

**指令：**
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**預期結果：**
- Epoch 1：約 66% 詞元準確率，約 28% 序列準確率
- Epoch 3：約 98% 詞元準確率，約 89% 序列準確率
- Epoch 5-7：99%+ 詞元準確率，95-98% 序列準確率
- 最終：98-99% 序列準確率

**使用此任務來：**
- 驗證您的設定有效
- 測試超參數變更
- 除錯問題

---

### 2. 反轉任務 ⭐⭐☆☆☆（中等）

學習反轉輸入序列。

```
輸入：  [5, 7, 3, 9]
輸出：  [9, 3, 7, 5]
```

**指令：**
```bash
python train.py --task reverse --epochs 30 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**預期結果：**
- Epoch 5：約 40% 詞元準確率
- Epoch 15：約 75% 詞元準確率
- Epoch 30：約 90% 詞元準確率，約 75-85% 序列準確率

**使用此任務來：**
- 測試位置理解
- 驗證注意力機制運作
- 挑戰模型超越簡單任務

---

### 3. 排序任務 ⭐⭐⭐☆☆（困難）

學習將數字升序排序。

```
輸入：  [7, 3, 9, 5]
輸出：  [3, 5, 7, 9]
```

**指令：**
```bash
python train.py --task sort --epochs 50 --fixed-lr 0.0005 --label-smoothing 0.0 --dropout 0.0
```

**預期結果：**
- Epoch 10：約 25% 詞元準確率
- Epoch 30：約 55% 詞元準確率
- Epoch 50：約 75% 詞元準確率，約 60-70% 序列準確率

**使用此任務來：**
- 測試演算法推理
- 展示 Transformer 能力
- 評估模型效能基準

**注意：** 排序顯著更困難。使用較低的學習率（`0.0005` 而非 `0.001`）以獲得更好的穩定性。

---

## 📊 理解訓練輸出

### 訓練期間

```
Epoch 5/20
------------------------------------------------------------
  Batch 50/141 | Loss: 0.0619 | Token Acc: 98.19% | Seq Acc: 87.34% | LR: 0.001000
  Batch 100/141 | Loss: 0.0587 | Token Acc: 98.28% | Seq Acc: 88.08% | LR: 0.001000

[>] Epoch 5 Summary:
   Train Loss: 0.0559 | Token Acc: 98.36% | Seq Acc: 88.62%
   Val Loss:   0.0461 | Token Acc: 98.67% | Seq Acc: 90.50%
   Time: 56.2s
```

**指標說明：**

- **Loss**：交叉熵損失（越低越好）
  - 隨機猜測：約 2.996（vocab_size=20 時）
  - 良好模型：< 0.5
  - 優秀模型：< 0.1

- **Token Acc**：正確預測的個別詞元百分比
  - 隨機猜測：約 5%（1/20）
  - 最低可接受：> 50%
  - 良好模型：> 90%

- **Seq Acc**：完全正確預測的完整序列百分比（最嚴格的指標）
  - 這才是真正重要的！
  - 良好模型：> 80%
  - 優秀模型：> 95%

- **LR**：目前學習率
  - 使用 `--fixed-lr 0.001` 時，此值保持在 0.001000

### 生成範例（每 5 個 Epoch）

```
============================================================
>> GENERATION TEST - See What The Model Learned!
============================================================

Example 1: [OK] CORRECT
  Input:    [16, 17, 14, 15]
  Expected: [16, 17, 14, 15]
  Got:      [16, 17, 14, 15]

Example 2: [OK] CORRECT
  Input:    [10, 9, 5]
  Expected: [10, 9, 5]
  Got:      [10, 9, 5]
```

**這是最重要的輸出！** 不要只相信指標——要看實際的預測。

**要注意的：**
- ✅ **全部正確：** 模型學習良好
- ⚠️ **部分正確：** 模型正在學習但需要更多 epoch
- ❌ **全部錯誤（空的 `[]` 或重複詞元）：** 模型沒有學習 → 檢查 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 💾 檢查點

檢查點會自動儲存到 `checkpoints/` 目錄：

```
checkpoints/
├── checkpoint_best.pt        # 最佳驗證準確率（使用這個！）
├── checkpoint_latest.pt      # 最新的 epoch
├── checkpoint_epoch_004.pt   # 每 5 個 epoch 儲存
├── checkpoint_epoch_009.pt
└── config.json               # 訓練設定
```

### 使用檢查點

**恢復訓練：**
```bash
python train.py --resume checkpoints/checkpoint_latest.pt --epochs 40
```

**測試訓練好的模型：**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy
```

**互動式測試：**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive
```

---

## 🎯 訓練配方

### 配方 1：快速測試（5 分鐘）
驗證所有東西都運作：

```bash
python train.py \
  --task copy \
  --epochs 10 \
  --num-samples 5000 \
  --batch-size 64 \
  --fixed-lr 0.001 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

**預期結果：** 約 80-90% 序列準確率  
**用途：** 快速健全性檢查

---

### 配方 2：完整複製任務（10 分鐘）
接近完美的效能：

```bash
python train.py \
  --task copy \
  --epochs 20 \
  --num-samples 10000 \
  --batch-size 64 \
  --fixed-lr 0.001 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

**預期結果：** 約 95-99% 序列準確率  
**用途：** 展示 Transformer 有效

---

### 配方 3：挑戰性反轉任務（20 分鐘）
測試序列理解：

```bash
python train.py \
  --task reverse \
  --epochs 30 \
  --num-samples 10000 \
  --batch-size 64 \
  --fixed-lr 0.001 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

**預期結果：** 約 85-95% 序列準確率  
**用途：** 非簡單任務評估

---

### 配方 4：困難排序任務（30 分鐘）
測試演算法推理：

```bash
python train.py \
  --task sort \
  --epochs 50 \
  --num-samples 10000 \
  --batch-size 64 \
  --fixed-lr 0.0005 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

**預期結果：** 約 70-85% 序列準確率  
**用途：** 推動模型極限

---

## ⚙️ 客製化

### 模型大小

**極小型（快速、測試用）：**
```bash
--d-model 64 --num-layers 2 --num-heads 4 --d-ff 256
```
- 約 200K 參數
- 3-5 分鐘訓練
- 適用於：快速實驗

**小型（預設、推薦）：**
```bash
--d-model 128 --num-layers 2 --num-heads 4 --d-ff 512
```
- 約 1M 參數
- 10-15 分鐘訓練
- 適用於：大多數使用情境

**中型（更好的準確率）：**
```bash
--d-model 256 --num-layers 3 --num-heads 8 --d-ff 1024
```
- 約 10M 參數
- 30-45 分鐘訓練
- 適用於：高準確率需求

**大型（最佳結果）：**
```bash
--d-model 512 --num-layers 4 --num-heads 8 --d-ff 2048
```
- 約 50M 參數
- 60-120 分鐘訓練
- 適用於：最大效能

**注意：** 更大的模型使用更多記憶體且訓練較慢，但在簡單任務上不一定總是能達到更好的結果。

---

### 學習率調整

從 `--fixed-lr 0.001` 開始，如有需要再調整：

```bash
# 模型沒有學習（損失沒有下降）：
--fixed-lr 0.002    # 嘗試較高值

# 模型不穩定（損失跳動）：
--fixed-lr 0.0005   # 嘗試較低值

# 排序任務（較難的任務）：
--fixed-lr 0.0005   # 使用較低學習率
```

**經驗法則：**
- 簡單任務（複製）：`0.001`
- 中等任務（反轉）：`0.001`
- 困難任務（排序）：`0.0005`
- 客製化大型模型：`0.0001 - 0.0005`

---

### 資料集大小

```bash
--num-samples 5000    # 快速訓練，較低準確率
--num-samples 10000   # 預設，良好平衡
--num-samples 50000   # 最佳準確率，較長訓練時間
```

**更多資料 = 更好的泛化，但訓練較慢**

---

### 批次大小

```bash
--batch-size 32   # 較少記憶體，較慢訓練
--batch-size 64   # 預設，良好平衡
--batch-size 128  # 較快訓練，更多記憶體
```

**較大批次 = 更快訓練但需要更多 RAM**

---

## 🔍 測試您的模型

### 基本測試

```bash
# 在測試集上測試
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy

# 輸出：
# Test Results:
#    Loss: 0.0234
#    Token Accuracy: 99.87%
#    Sequence Accuracy: 98.50%
```

### 互動式測試

```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive

# 然後輸入您自己的序列：
# Input sequence: 5 7 3 9
# Output: 5 7 3 9
# [OK] CORRECT!
```

---

## 🐛 疑難排解

**模型卡在隨機猜測（約 13% 準確率）？**
→ 參閱 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 完整診斷指南

**快速修復：**
- ✅ 使用 `--fixed-lr 0.001`（不是 Transformer 排程）
- ✅ 使用 `--label-smoothing 0.0`（不是 0.1）
- ✅ 對小型資料集使用 `--dropout 0.0`

**驗證您的設定有效：**
```bash
python test_overfit.py
# 應該達到損失約 0.0000，100% 準確率
```

---

## 📈 預期效能基準

### 複製任務（20 epochs，fixed-lr 0.001）

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 1     | 0.98       | 0.15     | 95%       | 70%     |
| 3     | 0.06       | 0.05     | 98%       | 91%     |
| 5     | 0.03       | 0.02     | 99%       | 96%     |
| 10    | 0.01       | 0.01     | 99%       | 98%     |

### 反轉任務（30 epochs，fixed-lr 0.001）

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 5     | 1.8        | 1.9      | 45%       | 10%     |
| 15    | 0.6        | 0.7      | 80%       | 55%     |
| 30    | 0.2        | 0.3      | 92%       | 80%     |

### 排序任務（50 epochs，fixed-lr 0.0005）

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 10    | 2.2        | 2.3      | 30%       | 5%      |
| 30    | 1.0        | 1.1      | 60%       | 30%     |
| 50    | 0.5        | 0.6      | 78%       | 65%     |

---

## 💡 成功秘訣

1. **永遠從複製任務開始** - 如果這個失敗，表示有問題
2. **使用固定學習率，不是 Transformer 排程** - 簡單對小型模型更好
3. **停用標籤平滑** - 它會損害演算法任務
4. **先執行過擬合測試** - 證明您的架構有效
5. **監控序列準確率** - 這才是真正重要的指標
6. **檢查生成範例** - 不要只相信數字
7. **對困難任務要有耐心** - 排序需要 50+ epochs
8. **儲存最佳檢查點** - 部署時使用 `checkpoint_best.pt`

---

## 🚀 下一步

成功訓練後：

1. ✅ **互動式測試**：`python test.py --interactive`
2. 📊 **嘗試更困難的任務**：複製 → 反轉 → 排序
3. 🔬 **實驗模型大小**：嘗試更大的模型
4. 📈 **擴大資料規模**：嘗試 `--num-samples 50000`
5. 🌐 **真實世界任務**：適應翻譯、摘要等任務
6. 🎓 **閱讀程式碼**：學習 Transformer 內部運作

---

## 📚 其他資源

- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 完整除錯指南
- **[TRAINING_ISSUES.md](TRAINING_ISSUES.md)** - 學習率排程問題的技術深入探討
- **[README.md](README.md)** - 主要專案文件
- **原始論文：** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## ❓ 常見問題

**問：為什麼不使用論文中的 Transformer 學習率排程？**  
答：它是為大型模型（d_model=512）設計的，對小型模型會產生過高 5-40 倍的學習率，導致訓練完全失敗。

**問：我可以在大型模型上使用 Transformer 排程嗎？**  
答：可以！如果您有 d_model ≥ 512 且訓練 50K+ 步，該排程運作良好。對於小型模型（d_model ≤ 256），使用固定學習率。

**問：為什麼停用標籤平滑？**  
答：標籤平滑在有多個有效答案（翻譯）時有幫助。對於只有一個正確答案的演算法任務（複製/反轉/排序），它會使學習更困難。

**問：訓練應該花多長時間？**  
答：在 CPU 上：複製（10 分鐘）、反轉（20 分鐘）、排序（30 分鐘）。在 GPU 上：快 2-5 倍。

**問：我的模型輸出空序列，求助！**  
答：參閱 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。您可能需要 `--fixed-lr 0.001`。

**問：我應該期待的最低準確率是多少？**  
答：複製：95%+、反轉：85%+、排序：70%+。更低表示有問題。

---

訓練愉快！🚀

如果遇到問題，請先查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)！
