"""Debug Transformer LR Schedule"""
import torch

class TransformerLRScheduler:
    def __init__(self, d_model, warmup_steps, factor):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def get_lr(self, step):
        step = max(step, 1)
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return lr

# Test different configurations
configs = [
    ("Current (d_model=128, warmup=500, factor=10)", 128, 500, 10.0),
    ("Fixed LR equivalent", None, None, None),  # for comparison
]

print("=" * 80)
print("TRANSFORMER LR SCHEDULE ANALYSIS")
print("=" * 80)

for name, d_model, warmup, factor in configs:
    if d_model is None:
        print(f"\n{name}: lr = 0.001 (constant)")
        continue

    print(f"\n{name}:")
    print(f"  d_model={d_model}, warmup_steps={warmup}, factor={factor}")

    scheduler = TransformerLRScheduler(d_model, warmup, factor)

    # Show LR at key steps
    steps = [1, 10, 50, 100, 250, 500, 1000, 2000, 4000]
    print(f"\n  Step | Learning Rate")
    print(f"  -----|-------------")

    for step in steps:
        lr = scheduler.get_lr(step)
        marker = " <-- warmup peak" if step == warmup else ""
        print(f"  {step:4d} | {lr:.6f}{marker}")

    # Calculate average LR during first epoch (141 batches)
    epoch1_lr = sum(scheduler.get_lr(s) for s in range(1, 142)) / 141
    print(f"\n  Average LR during Epoch 1: {epoch1_lr:.6f}")
    print(f"  Fixed LR (0.001) / Avg Epoch1 LR: {0.001 / epoch1_lr:.2f}x")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("The Transformer schedule gives MUCH HIGHER learning rates than 0.001")
print("For small models on simple tasks, this causes training instability!")
print("\nSOLUTION: Use a fixed LR or reduce warmup_steps dramatically")
