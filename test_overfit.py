"""Test if model can overfit a single batch"""
import torch
import torch.nn as nn
from transformer import create_transformer
from utils import LabelSmoothingLoss, create_padding_mask, create_target_mask
from datasets import create_dataloader

print("Testing if model can overfit a single batch...")
print("=" * 60)

# Create tiny dataset
train_loader, _, dataset_info = create_dataloader(
    dataset_type='copy',
    batch_size=4,
    num_samples=4,  # Just 4 samples
    vocab_size=20
)

# Get single batch
src, tgt_input, tgt_output = next(iter(train_loader))

print("Single batch:")
print(f"  src:        {src[0].tolist()}")
print(f"  tgt_output: {tgt_output[0].tolist()}")

# Create small model
model = create_transformer(
    src_vocab_size=20,
    tgt_vocab_size=20,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    dropout=0.0  # No dropout for overfitting test!
)

# Create loss and optimizer
criterion = LabelSmoothingLoss(
    smoothing=0.0,  # No label smoothing for overfitting test
    pad_idx=dataset_info['pad_token']
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Try to overfit this single batch
print("\nTraining on single batch (should reach near-zero loss):")
print("-" * 60)

model.train()
for step in range(500):
    # Create masks
    src_mask = create_padding_mask(src, dataset_info['pad_token'])
    tgt_mask = create_target_mask(tgt_input, dataset_info['pad_token'])

    # Forward
    logits = model(src, tgt_input, src_mask, tgt_mask)

    # Loss
    loss = criterion(logits, tgt_output)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print every 50 steps
    if (step + 1) % 50 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

            # Token accuracy
            mask = (tgt_output != dataset_info['pad_token'])
            correct = ((preds == tgt_output) & mask).sum().item()
            total = mask.sum().item()
            token_acc = 100.0 * correct / total if total > 0 else 0

            # Sequence accuracy
            seq_correct = (preds == tgt_output).all(dim=1).sum().item()
            seq_acc = 100.0 * seq_correct / src.size(0)

            print(f"Step {step+1:3d}: Loss={loss.item():6.4f}, "
                  f"Token Acc={token_acc:5.1f}%, Seq Acc={seq_acc:5.1f}%")

# Final test
print("\n" + "=" * 60)
print("Final predictions:")
model.eval()
with torch.no_grad():
    src_mask = create_padding_mask(src, dataset_info['pad_token'])
    tgt_mask = create_target_mask(tgt_input, dataset_info['pad_token'])
    logits = model(src, tgt_input, src_mask, tgt_mask)
    preds = logits.argmax(dim=-1)

    for i in range(src.size(0)):
        src_clean = [x for x in src[i].tolist() if x not in [0, 1, 2]]
        tgt_clean = [x for x in tgt_output[i].tolist() if x not in [0, 1, 2]]
        pred_clean = [x for x in preds[i].tolist() if x not in [0, 1, 2]]

        status = "OK" if pred_clean == tgt_clean else "FAIL"
        print(f"\nExample {i+1}: [{status}]")
        print(f"  Input:    {src_clean}")
        print(f"  Expected: {tgt_clean}")
        print(f"  Got:      {pred_clean}")

print("\n" + "=" * 60)
if loss.item() < 0.1:
    print("SUCCESS: Model can overfit! Training setup is correct.")
else:
    print(f"FAILURE: Model cannot overfit (loss={loss.item():.4f})")
    print("This indicates a BUG in the model or training code.")
