"""Quick test with simple fixed LR like overfit test"""
import torch
import torch.nn as nn
from transformer import create_transformer
from utils import LabelSmoothingLoss, create_padding_mask, create_target_mask, TrainingMetrics
from datasets import create_dataloader

print("Testing with FIXED LR (no Transformer schedule)")
print("=" * 60)

# Create dataloaders
train_loader, val_loader, _, dataset_info = create_dataloader(
    dataset_type='copy',
    batch_size=64,
    num_samples=10000,
    vocab_size=20
)

# Create model
model = create_transformer(
    src_vocab_size=20,
    tgt_vocab_size=20,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    dropout=0.0  # NO DROPOUT like overfit test
)

print(f"Model parameters: {model.count_parameters():,}")
print(f"Dropout: 0.0 (disabled)")
print()

# Create loss and optimizer
criterion = LabelSmoothingLoss(smoothing=0.0, pad_idx=dataset_info['pad_token'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # FIXED LR like overfit test

# Train for 3 epochs
print("Training with fixed LR=0.001 (no warmup, no schedule)")
print("-" * 60)

for epoch in range(3):
    model.train()
    metrics = TrainingMetrics()

    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        # Create masks
        src_mask = create_padding_mask(src, dataset_info['pad_token'])
        tgt_mask = create_target_mask(tgt_input, dataset_info['pad_token'])

        # Forward
        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits, tgt_output)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update metrics
        with torch.no_grad():
            metrics.update(loss.item(), logits, tgt_output, dataset_info['pad_token'])

        # Print every 50 batches
        if (batch_idx + 1) % 50 == 0:
            m = metrics.get_metrics()
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {m['loss']:.4f} | "
                  f"Token Acc: {m['token_accuracy']:.2f}% | "
                  f"Seq Acc: {m['sequence_accuracy']:.2f}%")

    # Epoch summary
    m = metrics.get_metrics()
    print(f"\n[>] Epoch {epoch+1} Summary:")
    print(f"   Train Loss: {m['loss']:.4f} | "
          f"Token Acc: {m['token_accuracy']:.2f}% | "
          f"Seq Acc: {m['sequence_accuracy']:.2f}%\n")

print("=" * 60)
print("If this works, the issue is the Transformer LR schedule!")
print("If this fails, the issue is something else...")
