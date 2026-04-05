"""Check if gradients are flowing"""
import torch
import torch.nn as nn
from transformer import create_transformer
from datasets import create_dataloader
from utils import LabelSmoothingLoss, create_padding_mask, create_target_mask

# Create small dataset
train_loader, _, dataset_info = create_dataloader(
    dataset_type='copy',
    batch_size=4,
    num_samples=20,
    vocab_size=20
)

# Create fresh model (untrained)
model = create_transformer(
    src_vocab_size=20,
    tgt_vocab_size=20,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    dropout=0.1
)

# Get one batch
src, tgt_input, tgt_output = next(iter(train_loader))

print("Batch:")
print(f"  src shape: {src.shape}")
print(f"  tgt_input shape: {tgt_input.shape}")
print(f"  tgt_output shape: {tgt_output.shape}")

# Create masks
pad_idx = dataset_info['pad_token']
src_mask = create_padding_mask(src, pad_idx)
tgt_mask = create_target_mask(tgt_input, pad_idx)

print(f"\nMasks:")
print(f"  src_mask shape: {src_mask.shape}")
print(f"  tgt_mask shape: {tgt_mask.shape}")

# Forward pass
model.train()
logits = model(src, tgt_input, src_mask, tgt_mask)

print(f"\nForward pass:")
print(f"  logits shape: {logits.shape}")
print(f"  logits mean: {logits.mean().item():.4f}")
print(f"  logits std: {logits.std().item():.4f}")

# Compute loss
criterion = LabelSmoothingLoss(
    smoothing=0.1,
    pad_idx=pad_idx
)

loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

print(f"\nLoss:")
print(f"  value: {loss.item():.4f}")

# Backward pass
loss.backward()

# Check gradients
print("\nGradient statistics:")

grad_stats = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        grad_stats.append((name, grad_norm, grad_mean, grad_std))

# Sort by grad norm
grad_stats.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 parameters by gradient norm:")
for i, (name, norm, mean, std) in enumerate(grad_stats[:10], 1):
    print(f"  {i}. {name}")
    print(f"      Norm: {norm:.6f}, Mean: {mean:.6f}, Std: {std:.6f}")

print("\nBottom 10 parameters by gradient norm:")
for i, (name, norm, mean, std) in enumerate(grad_stats[-10:], 1):
    print(f"  {i}. {name}")
    print(f"      Norm: {norm:.6f}, Mean: {mean:.6f}, Std: {std:.6f}")

# Check if any gradients are zero
zero_grads = [(name, param.grad.numel()) for name, param in model.named_parameters()
              if param.grad is not None and param.grad.abs().max().item() == 0]

if zero_grads:
    print(f"\n⚠️  WARNING: {len(zero_grads)} parameters have zero gradients!")
    for name, numel in zero_grads:
        print(f"  - {name} ({numel} parameters)")
else:
    print("\n✅ All parameters have non-zero gradients")

# Check embedding gradients specifically
print("\nEmbedding gradients:")
for name, param in model.named_parameters():
    if 'embedding' in name.lower() and param.grad is not None:
        print(f"  {name}:")
        print(f"    Shape: {param.grad.shape}")
        print(f"    Norm: {param.grad.norm().item():.6f}")
        print(f"    Non-zero elements: {(param.grad != 0).sum().item()} / {param.grad.numel()}")
