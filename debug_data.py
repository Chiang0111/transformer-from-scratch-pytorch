"""Check training data format"""
from datasets import create_dataloader

# Create small dataset
train_loader, val_loader, dataset_info = create_dataloader(
    dataset_type='copy',
    batch_size=2,
    num_samples=5,
    vocab_size=20
)

print("Dataset info:")
for key, val in dataset_info.items():
    print(f"  {key}: {val}")

print("\nFirst batch:")
for src, tgt_input, tgt_output in train_loader:
    print(f"\nBatch shapes:")
    print(f"  src: {src.shape}")
    print(f"  tgt_input: {tgt_input.shape}")
    print(f"  tgt_output: {tgt_output.shape}")

    print(f"\nExample 1:")
    print(f"  src:        {src[0].tolist()}")
    print(f"  tgt_input:  {tgt_input[0].tolist()}")
    print(f"  tgt_output: {tgt_output[0].tolist()}")

    print(f"\nExample 2:")
    print(f"  src:        {src[1].tolist()}")
    print(f"  tgt_input:  {tgt_input[1].tolist()}")
    print(f"  tgt_output: {tgt_output[1].tolist()}")

    # Check token distribution in tgt_output
    import torch
    all_tokens = tgt_output.flatten()
    non_pad = all_tokens[all_tokens != 0]

    print(f"\nTarget output token distribution (excluding padding):")
    for token in range(20):
        count = (non_pad == token).sum().item()
        if count > 0:
            print(f"  Token {token}: {count} times")

    break  # Just first batch
