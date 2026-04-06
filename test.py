"""
Test trained Transformer model

Usage:
    python test.py --checkpoint checkpoints/checkpoint_best.pt --task copy
"""

import argparse
import torch
from transformer import create_transformer
from datasets import create_dataloader
from utils import load_checkpoint, create_padding_mask, create_target_mask, TrainingMetrics
import json
from pathlib import Path


@torch.no_grad()
def test_model(model, test_loader, device, pad_idx):
    """Test model on dataset"""
    model.eval()
    metrics = TrainingMetrics()

    for src, tgt_input, tgt_output in test_loader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        src_mask = create_padding_mask(src, pad_idx).to(device)
        tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Dummy loss (just for metrics)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=pad_idx
        )

        metrics.update(loss.item(), logits, tgt_output, pad_idx)

    return metrics.get_metrics()


@torch.no_grad()
def interactive_test(model, vocab_size, device, start_token, end_token, task):
    """Interactive testing - generate outputs for user inputs"""
    model.eval()

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print(f"Task: {task}")
    print("Enter sequences as space-separated numbers (3-19)")
    print("Example: 5 7 3 9")
    print("Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Input sequence: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            # Parse input
            tokens = [int(x) for x in user_input.split()]

            # Validate tokens
            if not all(3 <= t < vocab_size for t in tokens):
                print(f"[X] Error: All tokens must be between 3 and {vocab_size-1}")
                continue

            # Create tensor
            src = torch.tensor(tokens).unsqueeze(0).to(device)

            # Generate
            generated = model.generate(
                src,
                max_len=20,
                start_token=start_token,
                end_token=end_token
            )

            # Clean output
            generated_clean = [x for x in generated.squeeze(0).cpu().tolist()
                             if x not in [0, start_token, end_token]]

            # Show result
            print(f"> Output: {' '.join(map(str, generated_clean))}")

            # Show expected for this task
            if task == 'copy':
                expected = tokens
            elif task == 'reverse':
                expected = tokens[::-1]
            elif task == 'sort':
                expected = sorted(tokens)
            else:
                expected = None

            if expected:
                is_correct = generated_clean == expected
                status = "[OK] CORRECT!" if is_correct else "[X] Wrong"
                print(f"{status} (Expected: {' '.join(map(str, expected))})")

            print()

        except ValueError:
            print("[X] Invalid input. Use space-separated numbers.\n")
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Test Transformer Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='Task type')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--vocab-size', type=int, default=20,
                        help='Vocabulary size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode')

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[!] CUDA not available, using CPU")
        device = 'cpu'

    print("\n" + "="*60)
    print("TRANSFORMER MODEL TESTING")
    print("="*60 + "\n")

    # Load config if available
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = checkpoint_dir / 'config.json'

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("[OK] Loaded config from training")
        model_config = config['model']
    else:
        print("[!] No config found, using defaults")
        model_config = {
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 512,
            'dropout': 0.1,
            'vocab_size': args.vocab_size
        }

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

    model = model.to(device)

    # Load checkpoint
    checkpoint_info = load_checkpoint(args.checkpoint, model)

    print(f"Model parameters: {model.count_parameters():,}")
    print()

    # Test on dataset
    print("Testing on dataset...")
    # Create fresh test data (not from training/validation)
    _, _, test_loader, dataset_info = create_dataloader(
        dataset_type=args.task,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        vocab_size=model_config['vocab_size'],
        train_split=0.8,
        val_split=0.1
    )

    metrics = test_model(model, test_loader, device, dataset_info['pad_token'])

    print(f"[OK] Test Results:")
    print(f"   Loss: {metrics['loss']:.4f}")
    print(f"   Token Accuracy: {metrics['token_accuracy']:.2f}%")
    print(f"   Sequence Accuracy: {metrics['sequence_accuracy']:.2f}%")
    print(f"   Perplexity: {metrics['perplexity']:.4f}")

    # Interactive mode
    if args.interactive:
        interactive_test(
            model,
            model_config['vocab_size'],
            device,
            dataset_info['start_token'],
            dataset_info['end_token'],
            args.task
        )


if __name__ == '__main__':
    main()
