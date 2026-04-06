"""
Basic Usage Example - Transformer from Scratch

This example shows how to:
1. Create a transformer model
2. Prepare data
3. Make predictions

Perfect starting point for understanding the code!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformer import create_transformer

print("="*60)
print("BASIC TRANSFORMER USAGE EXAMPLE")
print("="*60)

# Step 1: Create a small transformer model
print("\n[1] Creating transformer model...")
model = create_transformer(
    src_vocab_size=1000,      # Source language vocabulary
    tgt_vocab_size=1000,      # Target language vocabulary
    d_model=128,              # Model dimension (smaller = faster)
    num_heads=4,              # Number of attention heads
    num_layers=2,             # Number of encoder/decoder layers
    d_ff=512,                 # Feedforward dimension
    dropout=0.1               # Dropout rate
)

print(f"[+] Model created with {model.count_parameters():,} parameters")
print(f"    Size: ~{model.count_parameters() * 4 / 1024 / 1024:.2f} MB")

# Step 2: Create sample input data
print("\n[2] Creating sample data...")
batch_size = 2
src_len = 10
tgt_len = 8

# Source sequence (e.g., English sentence)
src = torch.randint(0, 1000, (batch_size, src_len))
print(f"[+] Source shape: {src.shape} (batch x sequence length)")

# Target sequence (e.g., Chinese translation)
tgt = torch.randint(0, 1000, (batch_size, tgt_len))
print(f"[+] Target shape: {tgt.shape}")

# Step 3: Forward pass (training mode)
print("\n[3] Running forward pass (training mode)...")
model.train()
logits = model(src, tgt)
print(f"[+] Output shape: {logits.shape} (batch x sequence x vocabulary)")
print(f"    Logits are scores for each word in the vocabulary")

# Step 4: Autoregressive generation (inference mode)
print("\n[4] Running generation (inference mode)...")
model.eval()

with torch.no_grad():
    generated = model.generate(
        src,
        max_len=15,           # Maximum generation length
        start_token=1,        # <START> token
        end_token=2           # <END> token
    )

print(f"[+] Generated shape: {generated.shape}")
print(f"    Model generated sequences autoregressively!")

# Step 5: Inspect generated sequences
print("\n[5] Sample generated sequences:")
for i in range(batch_size):
    print(f"   Sequence {i+1}: {generated[i].tolist()}")

print("\n" + "="*60)
print("Example complete!")
print("="*60)
print("\nNext steps:")
print("  1. Read the code in transformer/ to understand each component")
print("  2. Run: pytest tests/ -v to see all 80 tests")
print("  3. Train on real tasks: python scripts/train.py --task copy --epochs 20")
print("  4. Check docs/TRAINING.md for complete training guide")
print("="*60)
