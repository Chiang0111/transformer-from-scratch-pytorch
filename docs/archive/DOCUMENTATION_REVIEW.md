# Documentation Files Review

## 📚 Recommendation: Keep vs Archive

### ✅ KEEP (Essential for open source users)

**User-facing documentation:**
- `README.md` - Main entry point ⭐
- `TRAINING.md` - Complete training guide ⭐
- `TROUBLESHOOTING.md` - Debug guide ⭐
- `FINAL_RESULTS.md` - Benchmark results ⭐
- `VALIDATION.md` - Testing methodology
- `PLAN.md` - Project roadmap

**Total: 6 essential files**

### 🗄️ ARCHIVE (Internal development notes)

**Redundant or internal:**
- `BENCHMARK_STATUS.md` - Duplicates FINAL_RESULTS.md
- `CHANGES_ML_EVAL.md` - Internal changelog (git history sufficient)
- `ISSUE_SUMMARY.md` - Development investigation notes
- `NEXT_STEPS.md` - Internal planning (covered in PLAN.md)
- `PHASE3_SUMMARY.md` - Duplicates README project status
- `SESSION_SUMMARY.md` - Internal session notes
- `STATUS.md` - Redundant with README
- `TRAINING_ISSUES.md` - Covered in TROUBLESHOOTING.md

**Recommendation:** Create a `docs/archive/` folder for these, or just remove them (they're in git history if needed).

### 📋 Action Items

1. **Keep 6 essential docs** in root
2. **Create docs/ folder** for organized documentation:
   ```
   docs/
   ├── TRAINING.md
   ├── TROUBLESHOOTING.md
   ├── VALIDATION.md
   └── RESULTS.md (rename FINAL_RESULTS.md)
   ```
3. **Remove redundant files** or move to docs/archive/
4. **Update README** to link to docs/ clearly

### 🎯 Minimal Open Source Structure

For maximum clarity, keep only:
```
/
├── README.md (with links to all docs)
├── LICENSE
├── requirements.txt
├── TRAINING.md (or move to docs/)
├── transformer/
├── tests/
└── ... (code files)
```

All other documentation in `docs/` folder.
