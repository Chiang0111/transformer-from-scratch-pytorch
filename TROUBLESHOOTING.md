# Troubleshooting Guide

This guide helps you diagnose and fix common training issues. If your model isn't learning, start here!

---

## 🚨 CRITICAL: Model Stuck at Random Guessing

### Symptoms

You'll know you have this problem if you see:

```
Epoch 10/30
------------------------------------------------------------
  Batch 100/141 | Loss: 2.95 | Token Acc: 13.02% | Seq Acc: 0.00%

Example 1: [X] WRONG
  Input:    [5, 7, 3]
  Expected: [5, 7, 3]
  Got:      []   ← Generates empty sequences or same token repeatedly
```

**Key indicators:**
- ✗ Loss stuck around **2.9-3.0** (≈ -log(1/20) = 2.996 for vocab_size=20)
- ✗ Token accuracy around **5-15%** (random guessing is ~5%)
- ✗ Sequence accuracy **stays at 0%** for many epochs
- ✗ Model generates **empty sequences `[]`** or END token immediately
- ✗ Perplexity near vocabulary size (18-20 for vocab_size=20)
- ✗ Validation accuracy **frozen at exact same value** (e.g., always 13.02%)

### Root Cause

**The Transformer learning rate schedule is incompatible with small models on simple tasks.**

The original Transformer paper designed the LR schedule for:
- **Large models:** d_model=512, 6 layers (~100M parameters)
- **Complex tasks:** Machine translation with millions of samples
- **Long training:** 100,000+ steps

Your model is:
- **Small:** d_model=128, 2 layers (~1M parameters)
- **Simple task:** Copy/reverse/sort with 10K samples
- **Short training:** ~4,000 steps (30 epochs)

The LR schedule produces learning rates **5-40x too high** for small models, causing training instability.

### The Solution ✅

**Use fixed learning rate instead of the Transformer schedule:**

```bash
# ✅ CORRECT - Use fixed LR
python train.py --task copy --epochs 20 \
  --fixed-lr 0.001 \
  --label-smoothing 0.0 \
  --dropout 0.0
```

```bash
# ❌ WRONG - Don't use Transformer schedule for small models
python train.py --task copy --epochs 30 \
  --lr-factor 10.0 \
  --warmup-steps 500 \
  --label-smoothing 0.0
```

### Expected Results with Fixed LR

With `--fixed-lr 0.001`, you should see:

```
Epoch 1/20
------------------------------------------------------------
  Batch 100/141 | Loss: 1.28 | Token Acc: 55.58% | Seq Acc: 14.56%
[>] Epoch 1 Summary:
   Train Loss: 0.98 | Token Acc: 66.20% | Seq Acc: 27.57%
   Val Loss:   0.15 | Token Acc: 95.00% | Seq Acc: 70.30%

Epoch 3/20
------------------------------------------------------------
[>] Epoch 3 Summary:
   Train Loss: 0.06 | Token Acc: 98.36% | Seq Acc: 88.62%
   Val Loss:   0.05 | Token Acc: 98.67% | Seq Acc: 90.50%

Example 1: [OK] CORRECT
  Input:    [5, 7, 3]
  Expected: [5, 7, 3]
  Got:      [5, 7, 3]  ← ✅ Actually works!
```

**Progress timeline:**
- **Epoch 1:** 66% token accuracy, 28% sequence accuracy (learning!)
- **Epoch 3:** 98% token accuracy, 89% sequence accuracy (near perfect!)
- **Epoch 5-7:** 99%+ token accuracy, 95-98% sequence accuracy (converged!)

---

## 🧪 How to Verify Your Setup

If you're not sure whether the problem is your code or hyperparameters, run this test:

```bash
python test_overfit.py
```

This trains on **just 4 samples** to see if the model *can* learn at all.

**If you see:**
```
Step 500: Loss=0.0000, Token Acc=100.0%, Seq Acc=33.3%
SUCCESS: Model can overfit! Training setup is correct.
```

✅ **Your architecture is fine!** The problem is just hyperparameters (use `--fixed-lr 0.001`)

**If the overfit test fails:**
❌ **There's a bug in your code.** The model architecture has a problem.

---

## 📊 Understanding Why the LR Schedule Fails

### What the Transformer LR Schedule Does

The schedule from the original paper:
```python
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) * lr_factor
```

With `d_model=128, warmup=500, lr_factor=10.0`:

| Step | Learning Rate | Problem |
|------|---------------|---------|
| 1 | 0.000079 | Too low initially |
| 100 | 0.007906 | **7.9x too high** |
| 500 (peak) | 0.039528 | **39x too high!** |
| Average (Epoch 1) | 0.005613 | **5.6x too high** |

**What works:** Fixed LR = 0.001

### Visual Comparison

```
Transformer Schedule (FAILS):
LR
│        ╱──────╲              ← Reaches 0.0395 (way too high!)
│       ╱         ╲___
│      ╱              ╲___
│     ╱                   ╲___
│____╱                        ╲___
└────────────────────────────────→ Step
    500 (peak)

Fixed LR (WORKS):
LR
│ ────────────────────────────  ← Stable at 0.001
│
│
│
└────────────────────────────────→ Step
```

When LR is too high:
- Weights "jump around" too much
- Model never converges to a solution
- Like trying to land a plane but over-correcting each time

When LR is just right:
- Weights adjust smoothly
- Model finds the pattern
- Converges quickly and reliably

---

## 🔧 Complete Troubleshooting Decision Tree

```
Is your model stuck at random guessing (~13% accuracy)?
│
├─ YES → Use fixed LR instead of Transformer schedule
│        python train.py --task copy --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
│
└─ NO → Is loss decreasing but slowly?
        │
        ├─ YES → Two possibilities:
        │        1. Disable label smoothing: --label-smoothing 0.0
        │        2. Disable dropout for small datasets: --dropout 0.0
        │
        └─ NO → Is loss unstable (jumps up and down)?
                 │
                 ├─ YES → LR might be too high
                 │        Try --fixed-lr 0.0005 (half of default)
                 │
                 └─ NO → Check your data/masks
                          Run: python debug_data.py
```

---

## 📋 Quick Reference: Common Issues

### Issue: Empty sequences `[]`

**Symptom:** Model generates nothing
**Cause:** Predicts END token immediately (LR too high)
**Fix:**
```bash
python train.py --task copy --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

### Issue: Repeats same token (e.g., `[7, 7, 7, 7, ...]`)

**Symptom:** Model stuck on one token
**Cause:** Learned to predict most common token (LR too high)
**Fix:**
```bash
python train.py --task copy --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

### Issue: Loss not decreasing

**Symptom:** Loss stays around 2.9-3.0
**Cause 1:** Using Transformer LR schedule on small model
**Fix:**
```bash
--fixed-lr 0.001
```

**Cause 2:** Label smoothing interfering
**Fix:**
```bash
--label-smoothing 0.0
```

### Issue: Training works but validation doesn't improve

**Symptom:** Train acc: 90%, Val acc: 20%
**Cause:** Overfitting (model memorizing training data)
**Fix:**
```bash
--dropout 0.1              # Add regularization
--num-samples 50000        # Use more data
```

### Issue: Training too slow

**Symptom:** Each epoch takes 5+ minutes
**Fix:**
```bash
--d-model 64               # Smaller model
--num-layers 2             # Fewer layers
--batch-size 32            # Smaller batches
--num-samples 5000         # Less data
```

---

## 🎯 When to Use What

### Use Fixed LR (--fixed-lr 0.001) when:
- ✅ Model size: d_model ≤ 256
- ✅ Task: Simple (copy, reverse, sort)
- ✅ Dataset: Small to medium (< 50K samples)
- ✅ Training: Short (< 10K steps)
- **This covers most educational/tutorial use cases**

### Use Transformer LR Schedule when:
- ✅ Model size: d_model ≥ 512
- ✅ Task: Complex (translation, summarization)
- ✅ Dataset: Large (millions of samples)
- ✅ Training: Long (100K+ steps)
- **This is for production-scale models**

### Label Smoothing:
- ❌ **OFF** (0.0) for: Copy, reverse, sort (one correct answer)
- ✅ **ON** (0.1) for: Translation, generation (multiple valid outputs)

### Dropout:
- ❌ **OFF** (0.0) for: Small datasets (< 10K samples)
- ✅ **ON** (0.1) for: Large datasets (> 50K samples)

---

## 🔬 Debugging Commands

```bash
# 1. Verify architecture works (should reach loss ~0.0)
python test_overfit.py

# 2. Check data format and masks
python debug_data.py

# 3. Check encoder/decoder outputs
python debug_encoder.py

# 4. Check gradient flow
python debug_gradients.py

# 5. Visualize LR schedule
python debug_lr_schedule.py
```

---

## 💡 Pro Tips

1. **Always start with fixed LR** - It just works for small models
2. **Disable label smoothing for algorithmic tasks** - It makes learning harder
3. **Run overfit test first** - Proves your architecture is correct
4. **Monitor sequence accuracy, not just loss** - It's the real metric that matters
5. **Check generation examples** - Don't just trust numbers, see actual outputs
6. **Start simple, scale up** - Copy task first, then reverse, then sort

---

## 📚 Further Reading

- **TRAINING_ISSUES.md** - Deep technical analysis of the LR schedule problem
- **TRAINING.md** - Complete training guide with recipes
- **Original Transformer Paper** - Vaswani et al., 2017 (for large models)

---

**Still stuck?** Open an issue with:
1. Full command you ran
2. Output of `python test_overfit.py`
3. First 5 epochs of training logs
4. OS and PyTorch version
