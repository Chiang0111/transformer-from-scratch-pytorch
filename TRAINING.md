# Training Guide

## ⚠️ CRITICAL: Two Required Changes for Small Models

**You MUST use both of these or training will fail:**

1. **Higher learning rate**: `--lr-factor 10.0` (Transformer LR formula designed for d_model=512)
2. **No label smoothing**: `--label-smoothing 0.0` (hurts performance on simple algorithmic tasks)

**Without both fixes, the model will stick at random guessing!**

See [Troubleshooting](#troubleshooting) section for details.

---

## 🎯 Quick Start

### Train on Copy Task (Easiest)
```bash
python train.py --task copy --epochs 30 --lr-factor 10.0 --warmup-steps 500 --label-smoothing 0.0
```

### Train on Reverse Task (Medium)
```bash
python train.py --task reverse --epochs 40 --lr-factor 10.0 --warmup-steps 1000 --label-smoothing 0.0
```

### Train on Sort Task (Hardest)
```bash
python train.py --task sort --epochs 60 --lr-factor 10.0 --warmup-steps 1500 --label-smoothing 0.0
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
python train.py --d-model 64 --num-layers 2 --num-heads 4 --d-ff 256 --lr-factor 15.0
```
- ~0.4M parameters
- Trains in 5-10 min
- Good for testing
- **Needs lr-factor 15.0**

**Medium (Balanced):**
```bash
python train.py --d-model 128 --num-layers 2 --num-heads 4 --d-ff 512 --lr-factor 10.0
```
- ~1M parameters  
- Trains in 10-20 min
- Recommended default
- **Needs lr-factor 10.0**

**Large (Better accuracy):**
```bash
python train.py --d-model 256 --num-layers 4 --num-heads 8 --d-ff 1024 --lr-factor 5.0
```
- ~10M parameters
- Trains in 30-60 min
- Best results
- **Needs lr-factor 5.0**

### Learning Rate

The model uses Transformer LR schedule with warmup:

```bash
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) * lr_factor
```

**CRITICAL: The formula assumes d_model=512 (from the paper).** For smaller models, you MUST scale up the learning rate:

```bash
# For d_model=64:  --lr-factor 15.0
# For d_model=128: --lr-factor 10.0
# For d_model=256: --lr-factor 5.0
# For d_model=512: --lr-factor 2.0 (original paper scale)
```

**Warmup Steps:**
```bash
--warmup-steps 500   # Good for small datasets (5K-10K samples)
--warmup-steps 1000  # Good for medium datasets (10K-50K samples)
--warmup-steps 2000  # Good for large datasets (50K+ samples)
```

**⚠️ Warning:** If you use the default `--lr-factor 2.0` with small models, training will fail (model stuck at random guessing). See troubleshooting section.

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
  --num-layers 2 \
  --lr-factor 15.0 \
  --warmup-steps 500 \
  --label-smoothing 0.0
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
  --lr-factor 10.0 \
  --warmup-steps 500 \
  --label-smoothing 0.0
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
  --lr-factor 5.0 \
  --warmup-steps 1000 \
  --label-smoothing 0.0
```

Expected: ~85-95% sequence accuracy

---

## 🐛 Troubleshooting

### CRITICAL: Model Stuck at Random Guessing ⚠️

**Symptoms:**
- Loss stays around 2.9-3.0 (close to -log(1/20) = 2.996)
- Token accuracy around 5-15% (random guessing)
- Sequence accuracy stays at 0%
- Model generates empty sequences `[]` or END token immediately
- Perplexity near vocab_size (e.g., 18-20 for vocab_size=20)

**Root Cause:**
Learning rate is **WAY TOO LOW** for small models. The Transformer LR formula `lr = d_model^(-0.5) * ...` assumes d_model=512 from the original paper.

**Solution:**
```bash
# For d_model=128 (default small model):
python train.py --task copy --epochs 30 --lr-factor 10.0 --warmup-steps 500

# For d_model=64:
python train.py --task copy --epochs 30 --lr-factor 15.0 --warmup-steps 500

# For d_model=256:
python train.py --task copy --epochs 30 --lr-factor 5.0 --warmup-steps 1000
```

**How to Verify:**
Run the overfitting test to confirm your setup works:
```bash
python test_overfit.py
```
This trains on a single batch. If it reaches loss < 0.1, your architecture is fine and you just need to adjust hyperparameters.

### Model Not Learning (Loss Decreasing Slowly)

**Problem**: Loss decreases but too slowly (still >1.0 after 20 epochs)

**Solutions:**
1. **Disable label smoothing**: `--label-smoothing 0.0` (**CRITICAL** for simple tasks)
2. **Increase learning rate**: `--lr-factor 15.0` (try higher values)
3. **Faster warmup**: `--warmup-steps 500` (for small datasets)
4. **Disable dropout**: `--dropout 0.0` (if dataset is small)

**Note:** Label smoothing (default 0.1) distributes probability mass across all tokens, making it much harder for the model to learn correct mappings on simple algorithmic tasks.

### Model Outputs Empty Sequences

**Problem**: Generates `[]` or just START/END tokens

**Root Cause**: Model predicts END token immediately (learning rate too low)

**Solutions:**
1. **Increase lr-factor to 10.0 or higher** (most common fix)
2. Check generation only after epoch 10+ (early epochs are random)
3. Verify with `python test_overfit.py` that architecture works

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
