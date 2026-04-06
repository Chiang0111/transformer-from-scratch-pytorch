# Benchmark Status Report

**Generated:** 2026-04-06

## Summary

Comprehensive validation of all three training tasks using the fixed hyperparameters.

### Results

| Task | Status | Accuracy | Target | Time | Notes |
|------|--------|----------|--------|------|-------|
| **Copy** | ✅ PASS | 98.6% | 95.0% | ~10 min | Excellent performance |
| **Reverse** | ✅ PASS | 83.0% | 80.0% | ~20 min | Above target threshold |
| **Sort** | ✅ PASS | 96.0% | 70.0% | ~25 min (3 epochs) | **Far exceeds target!** |

## Training Configuration

All tasks use the **fixed learning rate approach** that solves the LR schedule issue:

### Copy Task
```bash
python train.py --task copy --epochs 20 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0
```

### Reverse Task  
```bash
python train.py --task reverse --epochs 30 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0
```

### Sort Task
```bash
python train.py --task sort --epochs 50 --fixed-lr 0.0005 \
    --label-smoothing 0.0 --dropout 0.0
```

## Key Findings

### ✅ What Works
- **Fixed learning rate (0.001)** instead of Transformer schedule
- **No label smoothing (0.0)** for algorithmic tasks
- **No dropout (0.0)** for small datasets
- **Proper train/val/test splits (80/10/10)** to prevent data leakage

### ❌ What Doesn't Work
- Transformer LR schedule with small models (produces rates 5-40x too high)
- Label smoothing on exact-match tasks (distributes probability to wrong answers)
- High dropout on small datasets (interferes with learning)

## Checkpoint Details

### Copy Task
- **Location:** `benchmarks/copy/checkpoint_best.pt`
- **Epoch:** Best from 20 epochs
- **Validation Accuracy:** 98.6%
- **Test Accuracy:** 98.6%
- **Size:** 13.2 MB

### Reverse Task
- **Location:** `benchmarks/reverse/checkpoint_best.pt`
- **Epoch:** Best from 30 epochs  
- **Validation Accuracy:** 84.1%
- **Test Accuracy:** 83.0%
- **Size:** 13.2 MB

### Sort Task
- **Location:** `benchmarks/sort/checkpoint_epoch_003.pt` (or `checkpoint_final.pt`)
- **Epoch:** 3 (converged early - no need for full 50 epochs)
- **Validation Accuracy:** 95.9%
- **Test Accuracy:** 96.0%
- **Size:** 13.2 MB

## Key Insights

**Sort task converged much faster than expected!** With proper hyperparameters:
- Reached 96% accuracy in just 3 epochs
- No need to train for full 50 epochs
- Validation accuracy (95.9%) closely matches test accuracy (96.0%) - good generalization

**All three tasks significantly exceed minimum targets**, validating that the fixed-LR approach is highly effective for small transformer models on algorithmic tasks.

## Next Steps

1. ✅ All benchmark tasks complete and validated
2. ✅ Documentation updated with verified results  
3. ⏳ Commit final benchmark results to repository
4. ⏳ Mark Phase 3 as 100% complete
5. ⏳ Consider Phase 4 polish work (optional)

## Testing the Models

```bash
# Test copy model
python test.py --checkpoint benchmarks/copy/checkpoint_best.pt --task copy

# Test reverse model  
python test.py --checkpoint benchmarks/reverse/checkpoint_best.pt --task reverse

# Test sort model
python test.py --checkpoint benchmarks/sort/checkpoint_epoch_003.pt --task sort
# Or use the final checkpoint copy
python test.py --checkpoint benchmarks/sort/checkpoint_final.pt --task sort

# Interactive mode
python test.py --checkpoint benchmarks/copy/checkpoint_best.pt --task copy --interactive
```

## Validation

All results validated using held-out test sets (10% of data, never seen during training).
This ensures reported accuracies represent true generalization performance.
