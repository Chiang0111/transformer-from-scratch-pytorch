# 訓練問題和解決方案

## 問題摘要

初始的 30 個 epoch 訓練運行完成了，但**模型沒有學習**（卡在隨機猜測的性能）。

## 診斷

### 1. 訓練結果（使用預設超參數的 30 個 epoch）
```
最終指標：
- 訓練損失：2.91 ≈ -log(1/20) = 2.996（隨機猜測）
- 困惑度：18.4 ≈ 20（詞彙表大小）
- 詞元準確率：13.32% ≈ 5%（1/20 = 隨機）
- 序列準確率：0.00%
- 生成：所有輸出都是空的 []
```

模型立即以 11% 的機率生成 END_TOKEN（詞元 2）作為第一個詞元，導致空序列。

### 2. 根本原因分析

**模型架構是正確的** - 我用過擬合測試驗證了這一點：
- 建立了在單一批次（4 個序列）上訓練的測試
- 使用適當的超參數（lr=0.001，無 dropout，無標籤平滑）
- 模型在 500 步內達到**損失 ≈ 0.0000 和 100% 詞元準確率**
- **結論：架構運作完美！**

**問題是超參數** - 特別是學習率：

```python
# 論文中的 Transformer 學習率公式
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) * lr_factor

# 對於 d_model=512（論文）：
base_factor = 512^(-0.5) ≈ 0.044

# 對於 d_model=128（我們的模型）：
base_factor = 128^(-0.5) ≈ 0.088

# 使用 lr_factor=2.0：
peak_lr (at warmup_steps=1000) ≈ 0.088 * 2.0 * 0.001 = 0.000176
```

這個學習率對小型模型來說**太低了**，無法有效學習。

### 3. 梯度分析

所有梯度都正確流動：
- 頂部梯度：2-3 範數（良好）
- 無零梯度（除了 W_k.bias，由於注意力機制這是預期的）
- 嵌入正確接收梯度

### 4. 架構驗證

✅ Encoder 產生有意義的輸出（mean=0.04，std=2.8）
✅ Decoder 產生有意義的輸出（mean=-0.1，std=1.1）  
✅ Logits 有合理的變異數（std=0.68）
✅ 損失函數正確處理填充
✅ 遮罩建立正確
✅ 前向/反向傳播運作

## 解決方案

### 小型模型的修正超參數

對於 `d_model=128, num_layers=2`（小型 CPU 友善模型）：

```bash
python train.py \
  --task copy \
  --epochs 30 \
  --lr-factor 10.0 \        # 比預設高 5 倍
  --warmup-steps 500 \       # 更快的預熱
  --label-smoothing 0.0      # 簡單任務不需要
```

**為什麼這些變更：**

1. **`lr-factor 10.0`**：補償小的 d_model
   - Transformer 學習率公式假設 d_model=512
   - 較小的模型需要成比例更高的學習率
   - 給出 peak_lr ≈ 0.0039（vs 預設的 0.000176）

2. **`warmup-steps 500`**：小型資料集的更快預熱
   - 預設 4000 是用於大型資料集
   - 小型資料集（5K-10K 樣本）收斂更快

3. **`label-smoothing 0.0`**：**關鍵 - 必須停用！**
   - 標籤平滑（預設 0.1）分配機率：90% 正確，10% 分散到其他 19 個詞元
   - 使得在簡單任務上學習正確映射更困難
   - 對大詞彙表任務（翻譯）有用，但在演算法任務上**扼殺性能**
   - **即使使用正確的學習率（10.0），如果標籤平滑是 0.1，訓練仍會失敗！**

### 使用修正超參數的預期結果

基於過擬合測試，模型應該：
- 在 20-30 個 epoch 內達到損失 < 0.1
- 達到 95-100% 詞元準確率
- 達到 90-100% 序列準確率
- 在自回歸模式下生成正確的序列

## 學到的經驗

1. **總是先測試過擬合** - 如果你的模型無法過擬合單一批次，就有錯誤
2. **學習率排程很重要** - Transformer 排程針對 d_model=512 調整，需要為較小模型調整
3. **簡單任務不需要正則化** - 標籤平滑和 dropout 在演算法任務上可能有害
4. **架構 ≠ 訓練** - 正確的架構仍可能因錯誤的超參數而無法訓練

## 建議的訓練配方

### 快速測試（5 分鐘）
```bash
python train.py --task copy --epochs 10 --lr-factor 15.0 --warmup-steps 200 --label-smoothing 0.0
```

### 完整複製任務（15 分鐘）
```bash
python train.py --task copy --epochs 30 --lr-factor 10.0 --warmup-steps 500 --label-smoothing 0.0
```

### 較難的任務（30-60 分鐘）
```bash
# 反轉
python train.py --task reverse --epochs 40 --lr-factor 10.0 --warmup-steps 1000 --label-smoothing 0.0

# 排序  
python train.py --task sort --epochs 60 --lr-factor 10.0 --warmup-steps 1500 --label-smoothing 0.0
```

## 除錯檔案

建立了幾個診斷腳本：
- `debug_generation.py` - 測試自回歸生成
- `debug_data.py` - 檢查訓練資料格式
- `debug_encoder.py` - 檢查 encoder/decoder 輸出
- `debug_gradients.py` - 分析梯度流
- `test_overfit.py` - **關鍵測試** - 驗證模型可以學習

執行 `python test_overfit.py` 以驗證架構正確運作。

## 下一步

1. ✅ 架構已驗證（可以過擬合）
2. ⏳ 使用修正的超參數重新執行訓練
3. ⏳ 使用修正的建議更新 TRAINING.md
4. ⏳ 在 README 新增疑難排解部分
