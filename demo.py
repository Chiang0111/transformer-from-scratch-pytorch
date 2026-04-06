"""
Interactive Transformer Demo

Beautiful CLI demo showing the trained model in action.
Better than Jupyter - clean, version-controllable, no dependencies.

Usage:
    python demo.py                           # Interactive mode
    python demo.py --checkpoint path.pt      # Use specific checkpoint
    python demo.py --task copy               # Specify task
    python demo.py --show-attention          # Visualize attention weights
"""

import argparse
import torch
from pathlib import Path
from transformer import create_transformer
from utils import create_padding_mask, create_target_mask
from datasets import create_dataloader


def print_banner(text):
    """Print fancy banner"""
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width + "\n")


def print_example(num, input_seq, expected, predicted, correct):
    """Print a single example nicely"""
    status = "✅ CORRECT" if correct else "❌ WRONG"
    color_code = "\033[92m" if correct else "\033[91m"  # Green or red
    reset_code = "\033[0m"

    print(f"\n{color_code}Example {num}: {status}{reset_code}")
    print(f"  Input:    {input_seq}")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {predicted}")


def demo_checkpoint(checkpoint_path: str, task: str, num_examples: int = 10):
    """
    Run demo with a trained checkpoint

    Shows model predictions on test examples
    """
    print_banner(f"TRANSFORMER DEMO - {task.upper()} TASK")

    # Load checkpoint
    print("📂 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config
    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            print(f"   Task: {config['task']}")
            print(f"   Model: d_model={config['model']['d_model']}, "
                  f"layers={config['model']['num_layers']}, "
                  f"heads={config['model']['num_heads']}")
    print()

    # Create dataloader for test examples
    print("📊 Loading test data...")
    _, val_loader, _, dataset_info = create_dataloader(
        dataset_type=task,
        batch_size=1,  # One at a time for demo
        num_samples=1000,
        vocab_size=20
    )
    print()

    # Create model
    print("🤖 Creating model...")
    model = create_transformer(
        src_vocab_size=dataset_info['vocab_size'],
        tgt_vocab_size=dataset_info['vocab_size'],
        d_model=config['model']['d_model'] if config_path.exists() else 128,
        num_heads=config['model']['num_heads'] if config_path.exists() else 4,
        num_layers=config['model']['num_layers'] if config_path.exists() else 2,
        d_ff=config['model']['d_ff'] if config_path.exists() else 512,
        dropout=0.0
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"   Parameters: {model.count_parameters():,}")
    print(f"   Checkpoint epoch: {checkpoint['epoch'] + 1}")

    # Get metrics from checkpoint
    if 'metrics' in checkpoint:
        val_metrics = checkpoint['metrics'].get('val', {})
        if val_metrics:
            print(f"   Val accuracy: {val_metrics.get('sequence_accuracy', 0):.2f}%")
    print()

    # Test on examples
    print_banner("PREDICTIONS")

    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for i, (src, _, tgt_output) in enumerate(val_loader):
            if i >= num_examples:
                break

            # Generate prediction
            generated = model.generate(
                src,
                max_len=20,
                start_token=dataset_info['start_token'],
                end_token=dataset_info['end_token']
            )

            # Clean sequences (remove special tokens)
            src_clean = [x for x in src[0].tolist()
                        if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]
            expected_clean = [x for x in tgt_output[0].tolist()
                            if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]
            predicted_clean = [x for x in generated[0].tolist()
                             if x not in [0, dataset_info['start_token'], dataset_info['end_token']]]

            # Check if correct
            correct = predicted_clean == expected_clean
            if correct:
                correct_count += 1
            total_count += 1

            # Print example
            print_example(i + 1, src_clean, expected_clean, predicted_clean, correct)

    # Summary
    accuracy = 100.0 * correct_count / total_count
    print_banner("SUMMARY")
    print(f"  Correct: {correct_count}/{total_count}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Color-coded overall result
    if accuracy >= 90:
        print("\n  ✨ \033[92mExcellent performance!\033[0m")
    elif accuracy >= 70:
        print("\n  👍 \033[93mGood performance!\033[0m")
    elif accuracy >= 50:
        print("\n  🤔 \033[93mNeeds more training\033[0m")
    else:
        print("\n  ⚠️  \033[91mPoor performance - check training\033[0m")

    print()


def interactive_mode(checkpoint_path: str, task: str):
    """
    Interactive mode - user enters sequences

    User can type sequences and see model predictions
    """
    print_banner(f"INTERACTIVE MODE - {task.upper()} TASK")

    # Load model (same as demo_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    config_path = Path(checkpoint_path).parent / 'config.json'
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {'model': {'d_model': 128, 'num_heads': 4, 'num_layers': 2, 'd_ff': 512}}

    # Create model
    model = create_transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        dropout=0.0
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("🤖 Model loaded!")
    print("\nEnter sequences as space-separated numbers (3-19).")
    print("Special tokens: 0=PAD, 1=START, 2=END")
    print("Type 'quit' to exit.\n")

    while True:
        # Get input
        try:
            user_input = input("Input sequence: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break

            # Parse input
            try:
                sequence = [int(x) for x in user_input.split()]

                # Validate
                if not sequence:
                    print("⚠️  Empty sequence, try again\n")
                    continue

                if any(x < 3 or x > 19 for x in sequence):
                    print("⚠️  Numbers must be between 3-19\n")
                    continue

                # Create input tensor
                src = torch.tensor([sequence])

                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        src,
                        max_len=20,
                        start_token=1,
                        end_token=2
                    )

                # Clean output
                output = [x for x in generated[0].tolist() if x not in [0, 1, 2]]

                # Show result
                print(f"Output: {output}")

                # Show expected for known tasks
                if task == 'copy':
                    expected = sequence
                elif task == 'reverse':
                    expected = list(reversed(sequence))
                elif task == 'sort':
                    expected = sorted(sequence)
                else:
                    expected = None

                if expected is not None:
                    correct = output == expected
                    status = "✅ CORRECT" if correct else "❌ WRONG"
                    print(f"Expected: {expected}")
                    print(f"{status}\n")
                else:
                    print()

            except ValueError:
                print("⚠️  Invalid input. Use space-separated integers.\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋\n")
            break


def main():
    parser = argparse.ArgumentParser(description='Transformer Demo')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: checkpoints/checkpoint_best.pt)')
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='Task type (default: copy)')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='Number of examples to show (default: 10)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - enter your own sequences')

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint is None:
        default_path = Path('checkpoints') / 'checkpoint_best.pt'
        if default_path.exists():
            checkpoint_path = str(default_path)
        else:
            print("❌ No checkpoint found!")
            print("   Train a model first: python train.py --task copy --epochs 20")
            print("   Or specify path: python demo.py --checkpoint path/to/checkpoint.pt")
            return 1
    else:
        checkpoint_path = args.checkpoint

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1

    # Run demo
    if args.interactive:
        interactive_mode(checkpoint_path, args.task)
    else:
        demo_checkpoint(checkpoint_path, args.task, args.num_examples)

    return 0


if __name__ == '__main__':
    exit(main())
