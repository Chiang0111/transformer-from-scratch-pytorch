# Training Issues & Solutions

## Issue Summary

The initial 30-epoch training run completed but **the model did not learn** (stuck at random guessing performance).

## Diagnosis

### 1. Training Results (30 epochs with default hyperparameters)
```
Final Metrics:
- Train Loss: 2.91 ≈ -log(1/20) = 2.996 (random guessing)
- Perplexity: 18.4 ≈ 20 (vocabulary size)
- Token Accuracy: 13.32% ≈ 5% (1/20 = random)
- Sequence Accuracy: 0.00%
- Generation: All outputs are empty []
```

The model generated END_TOKEN (token 2) immediately with 11% probability as the first token, resulting in empty sequences.

### 2. Root Cause Analysis

**The model architecture is CORRECT** - I verified this with an overfitting test:
- Created a test that trains on a single batch (4 sequences)
- With proper hyperparameters (lr=0.001, no dropout, no label smoothing)
- Model reached **loss ≈ 0.0000 and 100% token accuracy** in 500 steps
- **Conclusion: Architecture works perfectly!**

**The problem is HYPERPARAMETERS** - specifically the learning rate:

```python
# Transformer LR formula from paper
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)) * lr_factor

# For d_model=512 (paper):
base_factor = 512^(-0.5) ≈ 0.044

# For d_model=128 (our model):
base_factor = 128^(-0.5) ≈ 0.088

# With lr_factor=2.0:
peak_lr (at warmup_steps=1000) ≈ 0.088 * 2.0 * 0.001 = 0.000176
```

This learning rate is **WAY TOO LOW** for the small model to learn effectively.

### 3. Gradient Analysis

All gradients are flowing correctly:
- Top gradients: 2-3 norm (good)
- No zero gradients (except W_k.bias which is expected due to attention mechanism)
- Embeddings receiving gradients properly

### 4. Architecture Verification

✅ Encoder produces meaningful outputs (mean=0.04, std=2.8)
✅ Decoder produces meaningful outputs (mean=-0.1, std=1.1)  
✅ Logits have reasonable variance (std=0.68)
✅ Loss function handles padding correctly
✅ Masks are created properly
✅ Forward/backward pass works

## Solution

### Corrected Hyperparameters for Small Models

For `d_model=128, num_layers=2` (small CPU-friendly models):

```bash
python train.py \
  --task copy \
  --epochs 30 \
  --lr-factor 10.0 \        # 5x higher than default
  --warmup-steps 500 \       # Faster warmup
  --label-smoothing 0.0      # Not needed for simple tasks
```

**Why these changes:**

1. **`lr-factor 10.0`**: Compensates for small d_model
   - Transformer LR formula assumes d_model=512
   - Smaller models need proportionally higher LR
   - Gives peak_lr ≈ 0.0039 (vs 0.000176 with default)

2. **`warmup-steps 500`**: Faster warmup for small datasets
   - Default 4000 is for large datasets
   - Small datasets (5K-10K samples) converge faster

3. **`label-smoothing 0.0`**: **CRITICAL - Must be disabled!**
   - Label smoothing (default 0.1) distributes probability: 90% correct, 10% across other 19 tokens
   - Makes it harder to learn correct mappings on simple tasks
   - Useful for large-vocab tasks (translation) but **KILLS performance** on algorithmic tasks
   - **Even with correct LR (10.0), training STILL FAILS if label smoothing is 0.1!**

### Expected Results with Corrected Hyperparameters

Based on overfitting test, the model should:
- Reach loss < 0.1 in 20-30 epochs
- Achieve 95-100% token accuracy
- Achieve 90-100% sequence accuracy
- Generate correct sequences in autoregressive mode

## Lessons Learned

1. **Always test overfitting first** - If your model can't overfit a single batch, there's a bug
2. **LR schedule matters** - The Transformer schedule is tuned for d_model=512, needs adjustment for smaller models
3. **Simple tasks don't need regularization** - Label smoothing and dropout can hurt on algorithmic tasks
4. **Architecture ≠ Training** - A correct architecture can still fail to train with wrong hyperparameters

## Recommended Training Recipes

### Quick Test (5 min)
```bash
python train.py --task copy --epochs 10 --lr-factor 15.0 --warmup-steps 200 --label-smoothing 0.0
```

### Full Copy Task (15 min)
```bash
python train.py --task copy --epochs 30 --lr-factor 10.0 --warmup-steps 500 --label-smoothing 0.0
```

### Harder Tasks (30-60 min)
```bash
# Reverse
python train.py --task reverse --epochs 40 --lr-factor 10.0 --warmup-steps 1000 --label-smoothing 0.0

# Sort  
python train.py --task sort --epochs 60 --lr-factor 10.0 --warmup-steps 1500 --label-smoothing 0.0
```

## Files for Debugging

Created several diagnostic scripts:
- `debug_generation.py` - Test autoregressive generation
- `debug_data.py` - Inspect training data format
- `debug_encoder.py` - Check encoder/decoder outputs
- `debug_gradients.py` - Analyze gradient flow
- `test_overfit.py` - **Critical test** - verify model can learn

Run `python test_overfit.py` to verify the architecture works correctly.

## Next Steps

1. ✅ Architecture verified (can overfit)
2. ⏳ Re-run training with corrected hyperparameters
3. ⏳ Update TRAINING.md with corrected recommendations
4. ⏳ Add troubleshooting section to README
