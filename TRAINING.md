# Training Guide

## 🎯 Quick Start

### Train on Copy Task (Easiest)
```bash
python train.py --task copy --epochs 30 --batch-size 64
```

### Train on Reverse Task (Medium)
```bash
python train.py --task reverse --epochs 40 --batch-size 64
```

### Train on Sort Task (Hardest)
```bash
python train.py --task sort --epochs 60 --batch-size 64
```

---

## 📋 Available Tasks

### 1. **Copy Task** ⭐☆☆☆☆ (Easiest)
Learn to copy input sequences exactly.

```
Input:  [5, 7, 3, 9]
Output: [5, 7, 3, 9]
```

**Expected Results:**
- Should reach ~100% accuracy in 20-30 epochs
- Loss should drop below 0.1
- Good for verifying the model works

### 2. **Reverse Task** ⭐⭐☆☆☆ (Medium)
Learn to reverse input sequences.

```
Input:  [5, 7, 3, 9]
Output: [9, 3, 7, 5]
```

**Expected Results:**
- Should reach ~90%+ accuracy in 30-40 epochs
- Requires attention to positional information
- Tests if model understands sequence order

### 3. **Sort Task** ⭐⭐⭐☆☆ (Hard)
Learn to sort numbers in ascending order.

```
Input:  [7, 3, 9, 5]
Output: [3, 5, 7, 9]
```

**Expected Results:**
- May need 50-60 epochs to converge
- Tests algorithmic reasoning
- Most challenging task

---

## ⚙️ Training Parameters

### Model Size

**Small (Fast, CPU-friendly):**
```bash
python train.py --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256
```
- ~0.4M parameters
- Trains in 5-10 min
- Good for testing

**Medium (Balanced):**
```bash
python train.py --d-model 128 --num-layers 2 --num-heads 4 --d-ff 512
```
- ~1M parameters  
- Trains in 10-20 min
- Recommended default

**Large (Better accuracy):**
```bash
python train.py --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024
```
- ~10M parameters
- Trains in 30-60 min
- Best results

### Learning Rate

The model uses Transformer LR schedule with warmup:

```bash
--warmup-steps 500   # Lower for small datasets
--warmup-steps 2000  # Higher for large datasets
--lr-factor 1.0      # Scale overall learning rate
```

**Tips:**
- If loss doesn't decrease: increase `--lr-factor`
- If training is unstable: decrease `--lr-factor` or increase `--warmup-steps`

### Data Size

```bash
--num-samples 5000   # Small, trains fast
--num-samples 10000  # Medium (default)
--num-samples 50000  # Large, better generalization
```

---

## 📊 Understanding Training Output

### During Training

```
Epoch 5/20
------------------------------------------------------------
  Batch 50/71 | Loss: 3.16 | Token Acc: 9.20% | Seq Acc: 0.00% | LR: 0.005281

[>] Epoch 5 Summary:
   Train Loss: 3.1613 | Token Acc: 9.20% | Seq Acc: 0.00%
   Val Loss:   2.9290 | Token Acc: 13.27% | Seq Acc: 0.00%
   Time: 19.1s
```

**Metrics Explained:**
- **Loss**: Lower is better (should decrease over time)
- **Token Acc**: % of individual tokens predicted correctly
- **Seq Acc**: % of complete sequences predicted correctly (strictest metric)
- **LR**: Current learning rate (increases during warmup, then decreases)

### Generation Test (Every 5 Epochs)

```
Example 1: [OK] CORRECT
  Input:    [5, 7, 3]
  Expected: [5, 7, 3]
  Got:      [5, 7, 3]
```

This shows actual model predictions!

---

## 🔍 Testing Trained Models

### Test on Dataset

```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy
```

Output:
```
Test Results:
   Loss: 0.0234
   Token Accuracy: 99.87%
   Sequence Accuracy: 98.50%
```

### Interactive Mode

```bash
python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy --interactive
```

Then type your own sequences:
```
Input sequence: 5 7 3 9
Output: 5 7 3 9
[OK] CORRECT! (Expected: 5 7 3 9)
```

---

## 💾 Checkpoints

Checkpoints are automatically saved in `checkpoints/` directory:

- `checkpoint_latest.pt` - Last epoch
- `checkpoint_best.pt` - Best validation accuracy
- `checkpoint_epoch_XXX.pt` - Every 5 epochs
- `config.json` - Training configuration

### Resume Training

```bash
python train.py --resume checkpoints/checkpoint_latest.pt
```

---

## 🎯 Recommended Training Recipes

### Recipe 1: Quick Test (5 minutes)
Verify everything works:

```bash
python train.py \
  --task copy \
  --epochs 20 \
  --num-samples 2000 \
  --batch-size 64 \
  --d-model 64 \
  --num-layers 2
```

Expected: ~80-90% sequence accuracy

### Recipe 2: Full Copy Task (15 minutes)
Get near-perfect performance:

```bash
python train.py \
  --task copy \
  --epochs 30 \
  --num-samples 10000 \
  --batch-size 64 \
  --d-model 128 \
  --num-layers 2 \
  --warmup-steps 1000
```

Expected: ~95-100% sequence accuracy

### Recipe 3: Challenging Reverse (30 minutes)
Test sequence understanding:

```bash
python train.py \
  --task reverse \
  --epochs 40 \
  --num-samples 10000 \
  --batch-size 64 \
  --d-model 256 \
  --num-layers 3 \
  --warmup-steps 2000
```

Expected: ~85-95% sequence accuracy

---

## 🐛 Troubleshooting

### Model Not Learning (Loss Not Decreasing)

**Problem**: Loss stays high or increases

**Solutions:**
1. Increase learning rate: `--lr-factor 2.0` or `--lr-factor 5.0`
2. Reduce model size if overfitting: `--d-model 64 --num-layers 2`
3. Increase dataset size: `--num-samples 20000`
4. Train longer: `--epochs 50`

### Model Outputs Same Token Repeatedly

**Problem**: Generates `[8, 8, 8, 8, ...]`

**Solutions:**
1. Train longer - this is normal in early epochs
2. Increase warmup: `--warmup-steps 2000`
3. Check generation test after more epochs (epoch 20+)

### Training Too Slow

**Problem**: Takes too long per epoch

**Solutions:**
1. Reduce batch size: `--batch-size 32`
2. Reduce model size: `--d-model 64 --d-ff 256`
3. Reduce dataset: `--num-samples 5000`
4. Use fewer layers: `--num-layers 2`

### Unstable Training (Loss Jumps Around)

**Problem**: Loss increases suddenly

**Solutions:**
1. Reduce learning rate: `--lr-factor 0.5`
2. Increase warmup: `--warmup-steps 4000`
3. Reduce gradient clipping: `--clip-grad 0.5`

---

## 📈 Expected Performance

### Copy Task
| Epochs | Token Acc | Seq Acc | Loss |
|--------|-----------|---------|------|
| 10     | ~50%      | ~20%    | ~1.5 |
| 20     | ~80%      | ~60%    | ~0.5 |
| 30     | ~95%      | ~90%    | ~0.1 |

### Reverse Task
| Epochs | Token Acc | Seq Acc | Loss |
|--------|-----------|---------|------|
| 20     | ~40%      | ~10%    | ~2.0 |
| 40     | ~75%      | ~50%    | ~0.8 |
| 60     | ~90%      | ~75%    | ~0.3 |

### Sort Task
| Epochs | Token Acc | Seq Acc | Loss |
|--------|-----------|---------|------|
| 30     | ~30%      | ~5%     | ~2.5 |
| 60     | ~60%      | ~30%    | ~1.2 |
| 100    | ~80%      | ~60%    | ~0.5 |

---

## 🔬 Advanced Options

### Custom Hyperparameters

```bash
python train.py \
  --task copy \
  --epochs 30 \
  --batch-size 64 \
  --d-model 256 \
  --num-heads 8 \
  --num-layers 4 \
  --d-ff 1024 \
  --dropout 0.1 \
  --lr-factor 1.5 \
  --warmup-steps 2000 \
  --label-smoothing 0.1 \
  --clip-grad 1.0
```

### Full Option List

```bash
python train.py --help
```

---

## Next Steps

After successful training:

1. ✅ **Verify the model learned**: Check generation examples
2. 📊 **Test on new data**: Use `test.py --interactive`
3. 🎯 **Try harder tasks**: Move from copy → reverse → sort
4. 🔬 **Experiment**: Try different hyperparameters
5. 📈 **Scale up**: Larger models, more data
6. 🌐 **Real tasks**: Adapt for translation, summarization, etc.

---

## 💡 Tips for Success

1. **Start Simple**: Always test on copy task first
2. **Monitor Metrics**: Watch both token and sequence accuracy
3. **Be Patient**: Complex tasks need 40-60 epochs
4. **Check Generation**: Don't just trust metrics, see actual outputs
5. **Save Best Model**: Use `checkpoint_best.pt` for deployment
6. **Iterate**: Try different hyperparameters systematically

Happy training! 🚀
