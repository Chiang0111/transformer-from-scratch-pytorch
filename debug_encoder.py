"""Check if encoder produces meaningful outputs"""
import torch
from transformer import create_transformer
from utils import load_checkpoint
import json

# Load config
with open('checkpoints/config.json') as f:
    config = json.load(f)

model_config = config['model']

# Create model
model = create_transformer(
    src_vocab_size=model_config['vocab_size'],
    tgt_vocab_size=model_config['vocab_size'],
    d_model=model_config['d_model'],
    num_heads=model_config['num_heads'],
    num_layers=model_config['num_layers'],
    d_ff=model_config['d_ff'],
    dropout=model_config['dropout']
)

load_checkpoint('checkpoints/checkpoint_latest.pt', model)
model.eval()

# Test input: [5, 7, 3]
src = torch.tensor([[5, 7, 3]])

print("Input sequence: [5, 7, 3]")
print(f"Shape: {src.shape}")

# Encode
with torch.no_grad():
    memory = model.encode(src, None)

print(f"\nEncoder output (memory) shape: {memory.shape}")
print(f"Memory statistics:")
print(f"  Mean: {memory.mean().item():.4f}")
print(f"  Std: {memory.std().item():.4f}")
print(f"  Min: {memory.min().item():.4f}")
print(f"  Max: {memory.max().item():.4f}")

# Check if memory is all zeros or all same
if torch.allclose(memory, torch.zeros_like(memory), atol=1e-3):
    print("\n⚠️  WARNING: Memory is all zeros!")
elif memory.std().item() < 0.01:
    print(f"\n⚠️  WARNING: Memory has very low variance!")

# Now check what decoder produces when attending to this memory
print("\n" + "="*60)
print("Checking decoder behavior:")

tgt = torch.tensor([[1]])  # Just START token

with torch.no_grad():
    from transformer.decoder import create_causal_mask
    tgt_mask = create_causal_mask(1)

    # Decode
    output = model.decode(tgt, memory, tgt_mask, None)

    print(f"\nDecoder output shape: {output.shape}")
    print(f"Decoder output statistics:")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")

    # Project to logits
    logits = model.output_projection(output[:, -1, :])

    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits statistics:")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Std: {logits.std().item():.4f}")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")

    # Check if logits are degenerate
    if logits.std().item() < 0.1:
        print("\n⚠️  WARNING: Logits have very low variance - model is uncertain!")

    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(10, dim=-1)

    print(f"\nTop 10 predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        token_name = {0: 'PAD', 1: 'START', 2: 'END'}.get(idx.item(), str(idx.item()))
        print(f"  {i+1}. Token {token_name}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
