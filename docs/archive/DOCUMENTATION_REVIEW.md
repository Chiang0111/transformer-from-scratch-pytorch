# 文件審查

## 📚 建議：保留 vs 封存

### ✅ 保留（開源使用者必需）

**面向使用者的文件：**
- `README.md` - 主要入口點 ⭐
- `TRAINING.md` - 完整訓練指南 ⭐
- `TROUBLESHOOTING.md` - 除錯指南 ⭐
- `FINAL_RESULTS.md` - 基準測試結果 ⭐
- `VALIDATION.md` - 測試方法
- `PLAN.md` - 專案藍圖

**總計：6 個必要檔案**

### 🗄️ 封存（內部開發筆記）

**冗餘或內部的：**
- `BENCHMARK_STATUS.md` - 與 FINAL_RESULTS.md 重複
- `CHANGES_ML_EVAL.md` - 內部變更日誌（git 歷史記錄已足夠）
- `ISSUE_SUMMARY.md` - 開發調查筆記
- `NEXT_STEPS.md` - 內部規劃（已涵蓋在 PLAN.md 中）
- `PHASE3_SUMMARY.md` - 與 README 專案狀態重複
- `SESSION_SUMMARY.md` - 內部會議筆記
- `STATUS.md` - 與 README 冗餘
- `TRAINING_ISSUES.md` - 已涵蓋在 TROUBLESHOOTING.md 中

**建議：** 為這些檔案建立 `docs/archive/` 資料夾，或直接移除（如需要可在 git 歷史記錄中找到）。

### 📋 行動項目

1. **在根目錄保留 6 個必要文件**
2. **建立 docs/ 資料夾**以組織文件：
   ```
   docs/
   ├── TRAINING.md
   ├── TROUBLESHOOTING.md
   ├── VALIDATION.md
   └── RESULTS.md (重新命名 FINAL_RESULTS.md)
   ```
3. **移除冗餘檔案**或移至 docs/archive/
4. **更新 README**以清楚連結至 docs/

### 🎯 最小開源結構

為了最大清晰度，僅保留：
```
/
├── README.md (連結至所有文件)
├── LICENSE
├── requirements.txt
├── TRAINING.md (或移至 docs/)
├── transformer/
├── tests/
└── ... (程式碼檔案)
```

所有其他文件放在 `docs/` 資料夾中。
