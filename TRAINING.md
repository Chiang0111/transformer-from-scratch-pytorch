# Training Guide

Complete guide to training your Transformer model on sequence tasks.

---

## 🚀 Quick Start (Copy-Paste Ready)

### Train on Copy Task (Easiest - 10 minutes)
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```
**Expected:** 95-99% accuracy

### Train on Reverse Task (Medium - 20 minutes)
```bash
python train.py --task reverse --epochs 30 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```
**Expected:** 85-95% accuracy

### Train on Sort Task (Hard - 30 minutes)
```bash
python train.py --task sort --epochs 50 --fixed-lr 0.0005 --label-smoothing 0.0 --dropout 0.0
```
**Expected:** 70-85% accuracy

---

## ⚠️ IMPORTANT: Why These Parameters?

### `--fixed-lr 0.001` - Use Fixed Learning Rate

**DO NOT use the Transformer LR schedule for small models!**

The original paper's schedule (`--lr-factor`, `--warmup-steps`) was designed for:
- Large models (d_model=512, 6 layers, ~100M parameters)
- Complex tasks (translation with millions of samples)

For small models (d_model=128, 2 layers, ~1M parameters), the schedule produces learning rates **5-40x too high**, causing training to completely fail.

✅ **Use:** `--fixed-lr 0.001` (simple and works)  
❌ **Don't use:** `--lr-factor 10.0 --warmup-steps 500` (will fail)

### `--label-smoothing 0.0` - Disable Label Smoothing

Label smoothing distributes probability to wrong answers, which:
- ✅ Helps translation (multiple valid outputs)
- ❌ Hurts algorithmic tasks (one correct answer)

For copy/reverse/sort: always use `0.0`

### `--dropout 0.0` - Disable Dropout for Small Datasets

With only 10K training samples, dropout can interfere with learning.
- Small datasets (< 10K): `--dropout 0.0`
- Large datasets (> 50K): `--dropout 0.1`

**→ See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if your model isn't learning**

---

## 📊 Data Splits (Proper ML Practice)

The training script automatically splits data into three sets:

- **Training (80%):** Used to train the model
- **Validation (10%):** Used for early stopping and selecting best checkpoint
- **Test (10%):** **Never seen during training** - final evaluation only

**Why this matters:** Using the validation set to select the best checkpoint means the model "indirectly sees" that data. The test set provides an unbiased estimate of true performance on unseen data. This is basic ML hygiene that prevents data leakage.

**What you'll see in output:**
```
Dataset ready:
   Task: copy
   Train samples: 8000
   Val samples: 1000
   Test samples: 1000   ← Held-out for final evaluation
```

After training completes:
```
>> FINAL TEST SET EVALUATION
Test Set Results:
   Seq Acc:   98.60%   ← This is the real performance metric
```

The test accuracy is what you should report - it represents true generalization performance.

---

## 📋 Available Tasks

### 1. Copy Task ⭐☆☆☆☆ (Easiest)

Learn to copy input sequences exactly.

```
Input:  [5, 7, 3, 9]
Output: [5, 7, 3, 9]
```

**Command:**
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**Expected Results:**
- Epoch 1: ~66% token accuracy, ~28% sequence accuracy
- Epoch 3: ~98% token accuracy, ~89% sequence accuracy
- Epoch 5-7: 99%+ token accuracy, 95-98% sequence accuracy
- Final: 98-99% sequence accuracy

**Use this task to:**
- Verify your setup works
- Test hyperparameter changes
- Debug issues

---

### 2. Reverse Task ⭐⭐☆☆☆ (Medium)

Learn to reverse input sequences.

```
Input:  [5, 7, 3, 9]
Output: [9, 3, 7, 5]
```

**Command:**
```bash
python train.py --task reverse --epochs 30 --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**Expected Results:**
- Epoch 5: ~40% token accuracy
- Epoch 15: ~75% token accuracy
- Epoch 30: ~90% token accuracy, ~75-85% sequence accuracy

**Use this task to:**
- Test positional understanding
- Verify attention mechanisms work
- Challenge the model beyond trivial tasks

---

### 3. Sort Task ⭐⭐⭐☆☆ (Hard)

Learn to sort numbers in ascending order.

```
Input:  [7, 3, 9, 5]
Output: [3, 5, 7, 9]
```

**Command:**
```bash
python train.py --task sort --epochs 50 --fixed-lr 0.0005 --label-smoothing 0.0 --dropout 0.0
```

**Expected Results:**
- Epoch 10: ~25% token accuracy
- Epoch 30: ~55% token accuracy
- Epoch 50: ~75% token accuracy, ~60-70% sequence accuracy

**Use this task to:**
- Test algorithmic reasoning
- Demonstrate Transformer capabilities
- Benchmark model performance

**Note:** Sort is significantly harder. Use lower LR (`0.0005` instead of `0.001`) for better stability.

---

## 📊 Understanding Training Output

### During Training

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

**Metrics Explained:**

- **Loss**: Cross-entropy loss (lower is better)
  - Random guessing: ~2.996 (for vocab_size=20)
  - Good model: < 0.5
  - Excellent model: < 0.1

- **Token Acc**: Percentage of individual tokens predicted correctly
  - Random guessing: ~5% (1/20)
  - Minimum acceptable: > 50%
  - Good model: > 90%

- **Seq Acc**: Percentage of complete sequences predicted perfectly (strictest metric)
  - This is what really matters!
  - Good model: > 80%
  - Excellent model: > 95%

- **LR**: Current learning rate
  - With `--fixed-lr 0.001`, this stays constant at 0.001000

### Generation Examples (Every 5 Epochs)

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

**This is the most important output!** Don't just trust metrics—see actual predictions.

**What to look for:**
- ✅ **All correct:** Model is learning well
- ⚠️ **Some correct:** Model is learning but needs more epochs
- ❌ **All wrong (empty `[]` or repeated tokens):** Model isn't learning → Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 💾 Checkpoints

Checkpoints are automatically saved to `checkpoints/` directory:

```
checkpoints/
├── checkpoint_best.pt        # Best validation accuracy (use this!)
├── checkpoint_latest.pt      # Most recent epoch
├── checkpoint_epoch_004.pt   # Saved every 5 epochs
├── checkpoint_epoch_009.pt
└── config.json               # Training configuration
```

### Using Checkpoints

**Resume training:**
```bash
python train.py --resume checkpoints/checkpoint_latest.pt --epochs 40
```

**Test trained model:**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy
```

**Interactive testing:**
```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive
```

---

## 🎯 Training Recipes

### Recipe 1: Quick Test (5 minutes)
Verify everything works:

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

**Expected:** ~80-90% sequence accuracy  
**Use for:** Quick sanity check

---

### Recipe 2: Full Copy Task (10 minutes)
Near-perfect performance:

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

**Expected:** ~95-99% sequence accuracy  
**Use for:** Demonstrating Transformer works

---

### Recipe 3: Challenging Reverse (20 minutes)
Test sequence understanding:

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

**Expected:** ~85-95% sequence accuracy  
**Use for:** Non-trivial task evaluation

---

### Recipe 4: Hard Sort Task (30 minutes)
Test algorithmic reasoning:

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

**Expected:** ~70-85% sequence accuracy  
**Use for:** Pushing model limits

---

## ⚙️ Customization

### Model Size

**Tiny (Fast, testing):**
```bash
--d-model 64 --num-layers 2 --num-heads 4 --d-ff 256
```
- ~200K parameters
- Trains in 3-5 minutes
- Good for: Quick experiments

**Small (Default, recommended):**
```bash
--d-model 128 --num-layers 2 --num-heads 4 --d-ff 512
```
- ~1M parameters
- Trains in 10-15 minutes
- Good for: Most use cases

**Medium (Better accuracy):**
```bash
--d-model 256 --num-layers 3 --num-heads 8 --d-ff 1024
```
- ~10M parameters
- Trains in 30-45 minutes
- Good for: High accuracy needs

**Large (Best results):**
```bash
--d-model 512 --num-layers 4 --num-heads 8 --d-ff 2048
```
- ~50M parameters
- Trains in 60-120 minutes
- Good for: Maximum performance

**Note:** Larger models use more memory and train slower, but may not always achieve better results on simple tasks.

---

### Learning Rate Tuning

Start with `--fixed-lr 0.001` and adjust if needed:

```bash
# Model not learning (loss not decreasing):
--fixed-lr 0.002    # Try higher

# Model unstable (loss jumping around):
--fixed-lr 0.0005   # Try lower

# Sort task (harder task):
--fixed-lr 0.0005   # Use lower LR
```

**Rule of thumb:**
- Simple tasks (copy): `0.001`
- Medium tasks (reverse): `0.001`
- Hard tasks (sort): `0.0005`
- Custom large models: `0.0001 - 0.0005`

---

### Dataset Size

```bash
--num-samples 5000    # Quick training, lower accuracy
--num-samples 10000   # Default, good balance
--num-samples 50000   # Best accuracy, longer training
```

**More data = better generalization, but slower training**

---

### Batch Size

```bash
--batch-size 32   # Less memory, slower training
--batch-size 64   # Default, good balance
--batch-size 128  # Faster training, more memory
```

**Larger batches = faster training but need more RAM**

---

## 🔍 Testing Your Model

### Basic Testing

```bash
# Test on test set
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy

# Output:
# Test Results:
#    Loss: 0.0234
#    Token Accuracy: 99.87%
#    Sequence Accuracy: 98.50%
```

### Interactive Testing

```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive

# Then enter your own sequences:
# Input sequence: 5 7 3 9
# Output: 5 7 3 9
# [OK] CORRECT!
```

---

## 🐛 Troubleshooting

**Model stuck at random guessing (~13% accuracy)?**
→ See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Complete diagnosis guide

**Quick fixes:**
- ✅ Use `--fixed-lr 0.001` (not Transformer schedule)
- ✅ Use `--label-smoothing 0.0` (not 0.1)
- ✅ Use `--dropout 0.0` for small datasets

**Verify your setup works:**
```bash
python test_overfit.py
# Should reach loss ~0.0000 with 100% accuracy
```

---

## 📈 Expected Performance Benchmarks

### Copy Task (20 epochs, fixed-lr 0.001)

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 1     | 0.98       | 0.15     | 95%       | 70%     |
| 3     | 0.06       | 0.05     | 98%       | 91%     |
| 5     | 0.03       | 0.02     | 99%       | 96%     |
| 10    | 0.01       | 0.01     | 99%       | 98%     |

### Reverse Task (30 epochs, fixed-lr 0.001)

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 5     | 1.8        | 1.9      | 45%       | 10%     |
| 15    | 0.6        | 0.7      | 80%       | 55%     |
| 30    | 0.2        | 0.3      | 92%       | 80%     |

### Sort Task (50 epochs, fixed-lr 0.0005)

| Epoch | Train Loss | Val Loss | Token Acc | Seq Acc |
|-------|------------|----------|-----------|---------|
| 10    | 2.2        | 2.3      | 30%       | 5%      |
| 30    | 1.0        | 1.1      | 60%       | 30%     |
| 50    | 0.5        | 0.6      | 78%       | 65%     |

---

## 💡 Tips for Success

1. **Always start with copy task** - If this fails, something is wrong
2. **Use fixed LR, not Transformer schedule** - Simple is better for small models
3. **Disable label smoothing** - It hurts algorithmic tasks
4. **Run overfit test first** - Proves your architecture works
5. **Monitor sequence accuracy** - It's the metric that really matters
6. **Check generation examples** - Don't just trust numbers
7. **Be patient with hard tasks** - Sort needs 50+ epochs
8. **Save best checkpoint** - Use `checkpoint_best.pt` for deployment

---

## 🚀 Next Steps

After successful training:

1. ✅ **Test interactively**: `python test.py --interactive`
2. 📊 **Try harder tasks**: Copy → Reverse → Sort
3. 🔬 **Experiment with model size**: Try larger models
4. 📈 **Scale up data**: Try `--num-samples 50000`
5. 🌐 **Real-world tasks**: Adapt for translation, summarization
6. 🎓 **Read the code**: Learn how transformers work internally

---

## 📚 Additional Resources

- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Complete debugging guide
- **[TRAINING_ISSUES.md](TRAINING_ISSUES.md)** - Technical deep-dive on LR schedule issue
- **[README.md](README.md)** - Main project documentation
- **Original Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## ❓ FAQ

**Q: Why not use the Transformer LR schedule from the paper?**  
A: It was designed for large models (d_model=512) and produces learning rates 5-40x too high for small models, causing training to fail completely.

**Q: Can I use the Transformer schedule for large models?**  
A: Yes! If you have d_model ≥ 512 and train for 50K+ steps, the schedule works well. For small models (d_model ≤ 256), use fixed LR.

**Q: Why disable label smoothing?**  
A: Label smoothing helps when there are multiple valid answers (translation). For algorithmic tasks with one correct answer (copy/reverse/sort), it makes learning harder.

**Q: How long should training take?**  
A: On CPU: Copy (10 min), Reverse (20 min), Sort (30 min). On GPU: 2-5x faster.

**Q: My model outputs empty sequences, help!**  
A: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md). You likely need `--fixed-lr 0.001`.

**Q: What's the minimum accuracy I should expect?**  
A: Copy: 95%+, Reverse: 85%+, Sort: 70%+. Lower means something is wrong.

---

Happy training! 🚀

If you encounter issues, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first!
