"""Quick debug script to see what model generates"""
import torch
from transformer import create_transformer
from utils import load_checkpoint
import json
from pathlib import Path

# Load config
config_path = Path('checkpoints/config.json')
with open(config_path) as f:
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

# Load checkpoint
load_checkpoint('checkpoints/checkpoint_latest.pt', model)
model.eval()

# Test input: [5, 7, 3]
src = torch.tensor([[5, 7, 3]])

print("Testing generation with input: [5, 7, 3]")
print("Expected output (copy task): [5, 7, 3]")
print()

# Generate
with torch.no_grad():
    generated = model.generate(
        src,
        max_len=20,
        start_token=1,
        end_token=2
    )

print(f"Raw generated sequence: {generated.squeeze(0).tolist()}")

# Check what tokens the model predicts at each step
print("\nDetailed generation process:")
with torch.no_grad():
    memory = model.encode(src, None)
    tgt = torch.full((1, 1), 1, dtype=torch.long)  # Start with [1]

    for step in range(10):
        print(f"\nStep {step + 1}:")
        print(f"  Current sequence: {tgt.squeeze(0).tolist()}")

        # Get logits for next token
        from transformer.decoder import create_causal_mask
        tgt_len = tgt.size(1)
        tgt_mask = create_causal_mask(tgt_len)
        output = model.decode(tgt, memory, tgt_mask, None)
        logits = model.output_projection(output[:, -1, :])

        # Show top 5 predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5, dim=-1)

        print(f"  Top 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            print(f"    {i+1}. Token {idx.item()}: {prob.item():.4f}")

        # Get next token
        next_token = logits.argmax(dim=-1, keepdim=True)
        print(f"  Chosen: {next_token.item()}")

        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == 2:  # END_TOKEN
            print("  -> END_TOKEN reached, stopping")
            break

print(f"\nFinal sequence: {tgt.squeeze(0).tolist()}")
print(f"After filtering [0,1,2]: {[x for x in tgt.squeeze(0).tolist() if x not in [0,1,2]]}")
