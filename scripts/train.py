"""
Train Transformer on Sequence Tasks

Usage:
    # Copy task (easiest)
    python scripts/train.py --task copy --epochs 20

    # Reverse task (medium)
    python scripts/train.py --task reverse --epochs 30

    # Sort task (hardest)
    python scripts/train.py --task sort --epochs 50

    # Custom configuration
    python scripts/train.py --task copy --epochs 20 --batch-size 64 --d-model 256
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import torch.nn as nn
import time

from transformer import create_transformer
from datasets import create_dataloader
from utils import (
    LabelSmoothingLoss,
    TransformerLRScheduler,
    create_padding_mask,
    create_target_mask,
    TrainingMetrics,
    save_checkpoint,
    load_checkpoint,
    save_training_config
)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device: str,
    pad_idx: int,
    clip_grad: float = 1.0
) -> dict:
    """
    Train for one epoch

    Args:
        model: Transformer model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on ('cpu' or 'cuda')
        pad_idx: Padding token index
        clip_grad: Gradient clipping value

    Returns:
        Dictionary with training metrics
    """
    model.train()
    metrics = TrainingMetrics()

    for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
        # Move to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        # Create masks
        src_mask = create_padding_mask(src, pad_idx).to(device)
        tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(logits, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # Update weights
        optimizer.step()

        # Update learning rate (if using schedule)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        with torch.no_grad():
            metrics.update(loss.item(), logits, tgt_output, pad_idx)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_metrics = metrics.get_metrics()
            if scheduler is not None:
                lr = scheduler.get_last_lr()
            else:
                lr = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {current_metrics['loss']:.4f} | "
                  f"Token Acc: {current_metrics['token_accuracy']:.2f}% | "
                  f"Seq Acc: {current_metrics['sequence_accuracy']:.2f}% | "
                  f"LR: {lr:.6f}")

    return metrics.get_metrics()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
    pad_idx: int
) -> dict:
    """
    Evaluate model on validation set

    Args:
        model: Transformer model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        pad_idx: Padding token index

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    metrics = TrainingMetrics()

    for src, tgt_input, tgt_output in val_loader:
        # Move to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        # Create masks
        src_mask = create_padding_mask(src, pad_idx).to(device)
        tgt_mask = create_target_mask(tgt_input, pad_idx).to(device)

        # Forward pass
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # Compute loss
        loss = criterion(logits, tgt_output)

        # Update metrics
        metrics.update(loss.item(), logits, tgt_output, pad_idx)

    return metrics.get_metrics()


@torch.no_grad()
def test_generation(
    model: nn.Module,
    test_samples: list,
    device: str,
    start_token: int,
    end_token: int,
    max_len: int = 20
):
    """
    Test autoregressive generation on a few examples

    【What This Does】
    Shows actual model predictions vs expected outputs.
    This is helpful to see if the model is really learning!

    Args:
        model: Transformer model
        test_samples: List of (src, expected_output) tuples
        device: Device
        start_token: Start token ID
        end_token: End token ID
        max_len: Maximum generation length
    """
    model.eval()

    print("\n" + "="*60)
    print(">> GENERATION TEST - See What The Model Learned!")
    print("="*60)

    for i, (src, expected) in enumerate(test_samples, 1):
        # Add batch dimension
        src_batch = src.unsqueeze(0).to(device)

        # Generate
        generated = model.generate(
            src_batch,
            max_len=max_len,
            start_token=start_token,
            end_token=end_token
        )

        # Remove batch dimension and convert to list
        generated = generated.squeeze(0).cpu().tolist()

        # Remove special tokens for display
        src_clean = [x for x in src.tolist() if x not in [0, start_token, end_token]]
        expected_clean = [x for x in expected.tolist() if x not in [0, start_token, end_token]]
        generated_clean = [x for x in generated if x not in [0, start_token, end_token]]

        # Check if correct
        is_correct = generated_clean == expected_clean
        status = "[OK] CORRECT" if is_correct else "[X] WRONG"

        print(f"\nExample {i}: {status}")
        print(f"  Input:    {src_clean}")
        print(f"  Expected: {expected_clean}")
        print(f"  Got:      {generated_clean}")

    print("="*60 + "\n")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Transformer on Sequence Tasks')

    # Task settings
    parser.add_argument('--task', type=str, default='copy',
                        choices=['copy', 'reverse', 'sort'],
                        help='Task to train on')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--vocab-size', type=int, default=20,
                        help='Vocabulary size')

    # Model settings
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training settings
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--fixed-lr', type=float, default=None,
                        help='Use fixed learning rate instead of Transformer schedule (e.g., 0.001)')
    parser.add_argument('--lr-factor', type=float, default=2.0,
                        help='Learning rate factor (only for Transformer schedule)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Number of warmup steps (only for Transformer schedule)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--clip-grad', type=float, default=1.0,
                        help='Gradient clipping value')

    # Other settings
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to train on')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'

    print("\n" + "="*60)
    print(">> TRANSFORMER TRAINING")
    print("="*60)
    print(f"Task: {args.task.upper()}")
    print(f"Device: {device.upper()}")
    print(f"Model: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print("="*60 + "\n")

    # Create dataloaders
    print("[*] Loading data...")
    train_loader, val_loader, test_loader, dataset_info = create_dataloader(
        dataset_type=args.task,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        vocab_size=args.vocab_size
    )

    print(f"[+] Dataset ready:")
    print(f"   Task: {dataset_info['task']}")
    print(f"   Train samples: {dataset_info['train_samples']}")
    print(f"   Val samples: {dataset_info['val_samples']}")
    print(f"   Test samples: {dataset_info['test_samples']}")
    print(f"   Vocab size: {dataset_info['vocab_size']}")
    print()

    # Create model
    print("[*] Building model...")
    model = create_transformer(
        src_vocab_size=dataset_info['vocab_size'],
        tgt_vocab_size=dataset_info['vocab_size'],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    )

    model = model.to(device)

    param_count = model.count_parameters()
    print(f"[+] Model created:")
    print(f"   Parameters: {param_count:,}")
    print(f"   Size: ~{param_count * 4 / 1024 / 1024:.2f} MB")
    print()

    # Create optimizer and scheduler
    if args.fixed_lr is not None:
        # Use fixed learning rate (better for small models/simple tasks)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fixed_lr, betas=(0.9, 0.999), eps=1e-9)
        scheduler = None  # No scheduler needed for fixed LR
        print(f"[*] Using FIXED learning rate: {args.fixed_lr}")
    else:
        # Use Transformer LR schedule (original paper)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = TransformerLRScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            factor=args.lr_factor
        )
        print(f"[*] Using Transformer LR schedule: factor={args.lr_factor}, warmup={args.warmup_steps}")

    # Create loss function
    criterion = LabelSmoothingLoss(
        smoothing=args.label_smoothing,
        pad_idx=dataset_info['pad_token']
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_info = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler
        )
        start_epoch = checkpoint_info['epoch'] + 1
        print(f"[*] Resuming from epoch {start_epoch}")
        print()

    # Save training config
    config = {
        'task': args.task,
        'model': {
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'vocab_size': args.vocab_size
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr_factor': args.lr_factor,
            'warmup_steps': args.warmup_steps,
            'label_smoothing': args.label_smoothing,
            'clip_grad': args.clip_grad
        }
    }
    save_training_config(config, args.checkpoint_dir)

    # Prepare test samples for generation
    test_samples = []
    dataset_obj = train_loader.dataset.dataset
    for i in range(5):  # Get 5 samples
        src, _, tgt_out = dataset_obj[i]
        test_samples.append((src, tgt_out))

    # Training loop
    print("[*] Starting training...")
    print("="*60 + "\n")

    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, dataset_info['pad_token'], args.clip_grad
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, dataset_info['pad_token']
        )

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\n[>] Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | "
              f"Token Acc: {train_metrics['token_accuracy']:.2f}% | "
              f"Seq Acc: {train_metrics['sequence_accuracy']:.2f}%")
        print(f"   Val Loss:   {val_metrics['loss']:.4f} | "
              f"Token Acc: {val_metrics['token_accuracy']:.2f}% | "
              f"Seq Acc: {val_metrics['sequence_accuracy']:.2f}%")
        print(f"   Time: {epoch_time:.1f}s")

        # Save checkpoint every 5 epochs or if best model
        if (epoch + 1) % 5 == 0 or val_metrics['sequence_accuracy'] > best_val_acc:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                args.checkpoint_dir
            )

            if val_metrics['sequence_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['sequence_accuracy']
                # Save as best model
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'train': train_metrics, 'val': val_metrics},
                    args.checkpoint_dir,
                    filename='checkpoint_best.pt'
                )
                print(f"   [!] New best model! Val Seq Acc: {best_val_acc:.2f}%")

        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_generation(
                model, test_samples, device,
                dataset_info['start_token'],
                dataset_info['end_token']
            )

        print()

    # Final test set evaluation (held-out data, never seen during training)
    print("\n" + "="*60)
    print(">> TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation sequence accuracy: {best_val_acc:.2f}%\n")

    # Evaluate on test set (most important metric - true generalization)
    print("="*60)
    print(">> FINAL TEST SET EVALUATION")
    print("="*60)
    print("Evaluating on held-out test set (never seen during training)...\n")

    test_metrics = evaluate(
        model, test_loader, criterion, device, dataset_info['pad_token']
    )

    print(f"[+] Test Set Results:")
    print(f"   Loss:      {test_metrics['loss']:.4f}")
    print(f"   Token Acc: {test_metrics['token_accuracy']:.2f}%")
    print(f"   Seq Acc:   {test_metrics['sequence_accuracy']:.2f}%")
    print(f"   Perplexity: {test_metrics['perplexity']:.2f}")
    print("="*60 + "\n")

    test_generation(
        model, test_samples, device,
        dataset_info['start_token'],
        dataset_info['end_token']
    )

    print("[+] Model checkpoints saved in:", args.checkpoint_dir)
    print("\nTo test the model:")
    print(f"  python test.py --checkpoint {args.checkpoint_dir}/checkpoint_best.pt --task {args.task}")


if __name__ == '__main__':
    main()
