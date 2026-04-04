# 開發計畫：從零實作 Transformer

## 專案目標
建構一個生產級的 Transformer 實作，展現：
- 對 Transformer 架構的深度理解
- 撰寫簡潔、模組化、經過測試的程式碼能力
- AI 工程師職位所需的技能

---

## Phase 1：基礎（第 1 週）
**目標：** 理解 Transformer 的核心創新

### 任務
- [x] 建立專案結構（資料夾、requirements.txt）
- [x] 實作縮放點積注意力（Scaled Dot-Product Attention）
  - 簡潔的類別與型別提示
  - 解釋數學原理的 docstrings
  - 正確處理遮罩（masking）
- [x] 實作多頭注意力（Multi-Head Attention）
  - 正確分割多個頭
  - 串接並投影輸出
- [x] 實作位置編碼（Positional Encoding）
  - 正弦波編碼
  - 加到輸入嵌入
- [x] 為所有元件撰寫單元測試
  - 測試張量形狀
  - 測試注意力遮罩
  - 測試位置編碼模式
- [x] 實作位置前饋網路（Position-wise Feedforward Network）
  - 標準 FFN 和門控 FFN
  - ReLU 和 GELU 激活函數
- [x] 實作 Encoder Layer
  - 整合所有元件
  - 層歸一化（Layer Normalization）
  - 殘差連接（Residual Connections）
- [x] 實作完整 Encoder（堆疊多層）

**✅ Phase 1 完成！34 個單元測試全部通過**

**Commits：** 每個元件一個描述清楚的 commit

---

## Phase 2：架構（第 2 週）
**目標：** 組裝完整的 Transformer 模型

### 任務
- [ ] 實作 Encoder 層
  - 多頭注意力
  - Add & Norm
  - 前饋網路
  - Add & Norm
- [ ] 實作 Decoder 層
  - 遮罩自注意力（Masked Self-Attention）
  - 交叉注意力到 Encoder（Cross-Attention）
  - 前饋網路
  - 所有殘差連接
- [ ] 堆疊 Encoder 和 Decoder 層
- [ ] 加入嵌入層和最終線性投影
- [ ] 用假資料測試完整前向傳播
- [ ] 為每個層撰寫測試

**Commits：** 每種層一個 commit

---

## Phase 3：訓練（第 3 週）
**目標：** 讓它在真實資料上實際運作

### 任務
- [ ] 準備輕量資料集
  - 小型翻譯任務（英文→法文）
  - 或數字序列任務（如：反轉、排序）
  - 保持資料集小規模以便 CPU 訓練
- [ ] 實作訓練迴圈
  - 標籤平滑（Label Smoothing）
  - 學習率排程（Learning Rate Scheduling）
  - 梯度裁剪（Gradient Clipping）
- [ ] 加入評估指標
  - 追蹤損失
  - 準確率/BLEU 分數
- [ ] 訓練小模型（CPU 友善）
  - 2 層、256 維度、4 個頭
  - 在 CPU 上約 10-20 分鐘訓練時間
- [ ] 儲存和載入檢查點
- [ ] 建立訓練腳本（train.py）

**Commits：** 資料準備、訓練迴圈、評估分別 commit

---

## Phase 4：打磨（第 4 週）
**目標：** 讓它成為作品集就緒的狀態

### 任務
- [ ] 清理程式碼
  - 移除除錯用的 print
  - 一致的命名
  - 到處加上型別提示
  - 完整的 docstrings
- [ ] 撰寫完整的 README
  - 動機與目標
  - 架構圖
  - 使用範例
  - 如何訓練
  - 結果
- [ ] 建立 Jupyter notebook 教學
  - 視覺化注意力權重
  - 逐步解說
  - 解釋設計決策
- [ ] 加入 examples/
  - 簡單的推論腳本
  - 預訓練的小模型（如果可能）
- [ ] 程式碼審查與重構
  - DRY 原則
  - 移除重複
  - 改善可讀性

**Commits：** 打磨的 commits 要有清楚描述

---

## 成功標準

### 技術面
- ✅ 模型在小資料集上訓練並收斂
- ✅ 所有元件都有單元測試
- ✅ 程式碼是模組化且可重用的
- ✅ 適當的型別提示和文件

### 作品集面
- ✅ README 清楚解釋專案
- ✅ 程式碼展現生產實務
- ✅ Git 歷史顯示有思考的開發過程
- ✅ 能在面試中解釋每一行程式碼

### 學習面
- ✅ 深度理解注意力機制
- ✅ 能解釋為什麼 Transformer 有效
- ✅ 知道常見陷阱與解決方案
- ✅ 練習撰寫生產級 ML 程式碼

---

## GPU 需求

**好消息：這個專案不需要 GPU！**

我們會保持模型和資料集小規模，所以一切都能在 CPU 上執行：
- 小模型：2-4 層、256-512 維度
- 輕量資料集：10k-50k 範例
- 訓練時間：CPU 上 10-30 分鐘

如果之後有 GPU，同樣的程式碼會跑得更快 - 不需要任何修改。

---

## 時間預估
- **積極：** 2 週（每天 2-3 小時）
- **舒適：** 4 週（每天 1 小時）
- **深度學習導向：** 6 週（含深入探討）

**建議：** 在 Phase 1 花足夠時間。理解注意力機制就等於理解了 80% 的 Transformer。
