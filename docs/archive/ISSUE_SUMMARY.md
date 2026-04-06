# Issue Summary: Training Failure Investigation

**Date:** 2026-04-06  
**Status:** ✅ RESOLVED  
**Root Cause:** Transformer LR schedule incompatible with small models

---

## 📖 What Happened - Complete Story

### The Problem

Training a small Transformer model (d_model=128, 2 layers) on a simple copy task completely failed:
- Ran for 30 epochs
- Loss stuck at ~2.9-3.0 (random guessing)
- Accuracy stayed at ~13% (barely better than 5% random)
- Generated only empty sequences or repeated the same token
- Model learned NOTHING

### Initial Investigation (Day 1)

We ran comprehensive diagnostics:

1. **Overfitting test** ✅ PASSED
   - Trained on just 4 samples
   - Reached loss ~0.0000, 100% accuracy
   - **Conclusion:** Architecture is correct!

2. **Gradient analysis** ✅ PASSED
   - All gradients flowing correctly
   - Norms in healthy range (2-3)
   - No vanishing/exploding gradients

3. **Architecture verification** ✅ PASSED
   - Encoder outputs: mean=0.04, std=2.8 ✓
   - Decoder outputs: mean=-0.1, std=1.1 ✓
   - Logits variance: std=0.68 ✓

**Initial diagnosis (INCORRECT):** We thought learning rate was "too low" because the Transformer formula includes `d_model^(-0.5)`, and smaller d_model → smaller base LR. We tried increasing `lr_factor` from 2.0 to 10.0.

### Breakthrough Discovery (Day 2)

We ran controlled experiments comparing different training approaches:

#### Experiment 1: Overfit test (reference)
```bash
LR: Fixed 0.001
Dropout: 0.0
Label smoothing: 0.0
Result: ✅ SUCCESS - Loss 0.0000, 100% accuracy
```

#### Experiment 2: Full training with Transformer schedule
```bash
LR: Transformer schedule (lr_factor=10.0, warmup=500)
Dropout: 0.0
Label smoothing: 0.0
Result: ❌ FAILED - Stuck at 7-13% accuracy
```

#### Experiment 3: Full training with fixed LR
```bash
LR: Fixed 0.001
Dropout: 0.0
Label smoothing: 0.0
Result: ✅ SUCCESS - 98.6% accuracy in 7 epochs!
```

| Epoch | Experiment 2 (Schedule) | Experiment 3 (Fixed LR) |
|-------|-------------------------|-------------------------|
| 1 | 7% accuracy ❌ | 66% → 95% accuracy ✅ |
| 3 | 8% accuracy ❌ | 98% accuracy ✅ |
| 7 | 8% accuracy ❌ | 99.6% accuracy ✅ |

**The pattern was clear: The Transformer LR schedule was the problem!**

---

## 🔍 Root Cause Analysis

### Why the Transformer Schedule Fails

The Transformer LR schedule from the paper:
```python
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) * lr_factor
```

With our settings (d_model=128, warmup=500, lr_factor=10.0):

| Step | Learning Rate | Problem |
|------|---------------|---------|
| 1 | 0.000079 | Too low initially |
| 100 | 0.007906 | **7.9x too high** |
| 500 (peak) | 0.039528 | **39x too high!** |
| Avg (Epoch 1) | 0.005613 | **5.6x too high** |

**What actually works:** Fixed LR = 0.001

### Why This Happens

**The original Transformer paper designed the schedule for:**
- **Large models:** d_model=512, 6 encoder + 6 decoder layers (~100M parameters)
- **Complex tasks:** Machine translation (English ↔ German)
- **Huge datasets:** WMT dataset with millions of sentence pairs
- **Long training:** 100,000+ optimization steps
- **Scale:** Production-grade neural machine translation

**Our setup:**
- **Small model:** d_model=128, 2 encoder + 2 decoder layers (~1M parameters)
- **Simple task:** Copy sequences (trivial pattern learning)
- **Small dataset:** 10,000 synthetic sequences
- **Short training:** ~4,000 steps (30 epochs)
- **Scale:** Educational demonstration

**The schedule that works for "learn all of English grammar" doesn't work for "copy these numbers."**

### Why High LR Causes Failure

When learning rate is too high:
1. Weights "jump" too far in parameter space
2. Model overshoots the optimal solution
3. Never converges to a stable pattern
4. Like trying to land a plane but over-correcting every maneuver

The model oscillates wildly and never learns anything, appearing stuck at random guessing.

---

## ✅ The Solution

### Use Fixed Learning Rate

```bash
# ✅ CORRECT - Simple and works
python train.py --task copy --epochs 20 \
  --fixed-lr 0.001 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

```bash
# ❌ WRONG - Will fail on small models
python train.py --task copy --epochs 30 \
  --lr-factor 10.0 \
  --warmup-steps 500 \
  --label-smoothing 0.0
```

### Expected Results

With fixed LR=0.001:

```
Epoch 1: 66% → 95% accuracy (learning fast!)
Epoch 3: 98% accuracy (near perfect)
Epoch 5: 99% → 99% accuracy (all examples correct)
Epoch 7: 99.6% → 98.6% accuracy (BEST)

Final: 98.6% sequence accuracy
```

All generation examples correct:
```
Example 1: [OK] CORRECT
  Input:    [16, 17, 14, 15]
  Expected: [16, 17, 14, 15]
  Got:      [16, 17, 14, 15]
```

---

## 📝 Changes Made

### Code Changes

1. **train.py**
   - Added `--fixed-lr` parameter to support fixed learning rates
   - Made scheduler optional (None when using fixed LR)
   - Updated LR reporting to handle both fixed and scheduled LRs

2. **utils.py**
   - Fixed checkpoint saving to handle optional scheduler
   - Changed: `scheduler.step_num if scheduler is not None else 0`

### Documentation Updates

1. **TROUBLESHOOTING.md** (NEW)
   - Complete guide for diagnosing training failures
   - Explains why Transformer schedule fails for small models
   - Visual comparisons and decision trees
   - Clear symptoms → diagnosis → solution flow

2. **TRAINING.md** (REWRITTEN)
   - Removed all incorrect Transformer schedule recommendations
   - Updated to use `--fixed-lr 0.001` everywhere
   - Added expected performance benchmarks
   - Clear explanation of when to use fixed LR vs schedule

3. **TRAINING_ISSUES.md** (UPDATED)
   - Added CRITICAL UPDATE section at top
   - Documented the real root cause
   - Kept technical deep-dive for developers

4. **README.md** (UPDATED)
   - Updated Quick Start commands to use fixed LR
   - Added link to TROUBLESHOOTING.md
   - Added note about hyperparameter tuning for small models

---

## 🎓 Key Lessons Learned

### 1. Paper Hyperparameters ≠ Universal

The original Transformer paper's settings were optimized for:
- Large-scale production systems
- Complex natural language tasks
- Massive datasets and long training

**They don't transfer to small educational models.**

### 2. Simpler is Often Better

For small models on simple tasks:
- Fixed LR (0.001) > Complex LR schedule
- No label smoothing > Label smoothing 0.1
- No dropout > Dropout 0.1

**Sometimes the simplest approach is the best approach.**

### 3. Test Overfitting First

The overfitting test was the key diagnostic:
- Proved architecture was correct
- Used simple fixed LR that worked
- Isolated the hyperparameter issue

**Always verify your model CAN learn before debugging WHY it isn't.**

### 4. Trust Your Experiments

Initial diagnosis said "LR too low, increase lr_factor to 10."
- This was based on theory (formula includes d_model^-0.5)
- Seemed logical
- **Was completely wrong!**

Experiments showed the opposite:
- LR was too HIGH, not too low
- Simple fixed LR worked perfectly
- Theory didn't match reality

**Empirical evidence > Theoretical assumptions**

---

## 📚 For Future Reference

### If You Encounter "Model Stuck at Random Guessing"

**Step 1: Run the overfit test**
```bash
python test_overfit.py
```

- ✅ If it passes (loss < 0.1): Hyperparameter problem
- ❌ If it fails: Architecture/code bug

**Step 2: Use fixed LR**
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**Step 3: Check results after 3 epochs**
- ✅ Accuracy > 90%: Working correctly!
- ❌ Accuracy < 50%: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### When to Use What

**Fixed LR (`--fixed-lr 0.001`):**
- ✅ Model size: d_model ≤ 256
- ✅ Dataset: < 50K samples
- ✅ Task: Simple (copy, reverse, sort)
- ✅ Training: < 10K steps
- **→ Use this for educational/tutorial purposes**

**Transformer Schedule (`--lr-factor`, `--warmup-steps`):**
- ✅ Model size: d_model ≥ 512
- ✅ Dataset: millions of samples
- ✅ Task: Complex (translation, generation)
- ✅ Training: 50K+ steps
- **→ Use this for production-scale models**

---

## 🔗 Documentation Structure

For users encountering this issue:

1. **Start here:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
   - Symptoms → Diagnosis → Solution
   - Quick reference for common issues
   - Decision trees and visual guides

2. **Training guide:** [TRAINING.md](TRAINING.md)
   - Complete training recipes
   - Parameter explanations
   - Performance benchmarks

3. **Technical deep-dive:** [TRAINING_ISSUES.md](TRAINING_ISSUES.md)
   - Detailed root cause analysis
   - Experimental results
   - LR schedule mathematics

4. **Quick start:** [README.md](README.md)
   - Copy-paste commands
   - Links to detailed guides

---

## ✨ Final Status

**Problem:** Model completely failed to learn (stuck at random guessing)  
**Root Cause:** Transformer LR schedule produces learning rates 5-40x too high for small models  
**Solution:** Use fixed learning rate (`--fixed-lr 0.001`)  
**Result:** 98.6% sequence accuracy achieved in just 7 epochs ✅

**All training commands updated and tested.**  
**All documentation updated with correct information.**  
**Troubleshooting guide created for future users.**

---

**This issue is now fully resolved and documented!** 🎉
