"""
Training utilities for Transformer models

Includes:
- Learning rate schedulers
- Label smoothing loss
- Masking utilities
- Training metrics
- Checkpoint management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import json
from pathlib import Path


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss

    【What Is Label Smoothing?】
    Instead of using hard targets (one-hot vectors),
    we "smooth" the labels by distributing a small amount of
    probability mass to all other classes.

    【Why Use It?】
    - Prevents overconfidence (model being too sure)
    - Improves generalization
    - Reduces overfitting
    - Standard practice in Transformer training

    【How It Works】
    Hard label:     [0, 0, 1, 0, 0]  (100% on correct class)
    Smoothed label: [0.02, 0.02, 0.92, 0.02, 0.02]  (smoothing=0.1)

    Example:
        loss_fn = LabelSmoothingLoss(smoothing=0.1, pad_idx=0)
        loss = loss_fn(logits, targets)
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        """
        Initialize Label Smoothing Loss

        Args:
            smoothing: Smoothing factor (0.0 = no smoothing, 0.1 is common)
            pad_idx: Padding token index (ignore in loss calculation)
        """
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss

        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Ground truth labels [batch_size, seq_len]

        Returns:
            Scalar loss value
        """
        vocab_size = logits.size(-1)

        # Reshape for loss calculation
        # logits: [batch * seq_len, vocab_size]
        # targets: [batch * seq_len]
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        # Start with uniform distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 1))

        # Put most probability on correct class
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Zero out padding positions
        smooth_targets = smooth_targets.masked_fill(
            targets.unsqueeze(1) == self.pad_idx, 0.0
        )

        # KL divergence loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)

        # Don't count padding in loss
        mask = (targets != self.pad_idx).float()
        loss = (loss * mask).sum() / mask.sum()

        return loss


class TransformerLRScheduler:
    """
    Transformer Learning Rate Scheduler (Warmup + Decay)

    【What Is This?】
    The learning rate schedule from the original Transformer paper.
    It increases LR linearly during warmup, then decreases proportionally
    to the inverse square root of the step number.

    【Why Use It?】
    - Prevents unstable training at the start
    - Allows model to explore at beginning (low LR)
    - Then make larger updates (higher LR)
    - Finally fine-tune (decreasing LR)

    【Formula】
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    【Visualization】
        LR
         │    ╱─────╲___
         │   ╱           ╲___
         │  ╱                ╲___
         │ ╱                     ╲___
         └─────────────────────────────→ Step
           warmup    peak      decay

    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

        for step in range(num_steps):
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        """
        Initialize Transformer LR Scheduler

        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension (from Transformer config)
            warmup_steps: Number of warmup steps (4000 in original paper)
            factor: Scaling factor (1.0 by default)
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        """Update learning rate"""
        self.step_num += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Calculate learning rate for current step"""
        step = max(self.step_num, 1)  # Avoid division by zero

        # Original Transformer formula
        lr = self.factor * (
            self.d_model ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

        return lr

    def get_last_lr(self) -> float:
        """Get current learning rate"""
        return self._get_lr()


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask for sequences

    【What Is This?】
    When we batch sequences of different lengths, we pad them with
    a special <pad> token. We don't want the model to attend to
    padding positions.

    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index

    Returns:
        Mask tensor [batch_size, 1, 1, seq_len]
        1 = attend to this position
        0 = ignore this position (padding)

    Example:
        seq = [[5, 8, 3, 0, 0],     # 0 = padding
               [7, 2, 9, 1, 4]]

        mask = [[1, 1, 1, 0, 0],    # Attend to first 3 positions
                [1, 1, 1, 1, 1]]     # Attend to all 5 positions
    """
    # Create mask: 1 where not padding, 0 where padding
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create combined padding + causal mask for target sequences

    【What Is This?】
    For decoder, we need TWO types of masking:
    1. Padding mask (don't attend to <pad>)
    2. Causal mask (don't attend to future positions)

    This function combines both.

    Args:
        tgt: Target sequence [batch_size, tgt_len]
        pad_idx: Padding token index

    Returns:
        Combined mask [batch_size, 1, tgt_len, tgt_len]

    Example:
        tgt = [[1, 5, 8, 0],     # 0 = padding
               [1, 7, 2, 9]]

        mask = [[[1, 0, 0, 0],   # Position 0: can only see itself
                 [1, 1, 0, 0],   # Position 1: can see 0-1
                 [1, 1, 1, 0],   # Position 2: can see 0-2
                 [0, 0, 0, 0]],  # Position 3: padding (all masked)

                [[1, 0, 0, 0],
                 [1, 1, 0, 0],
                 [1, 1, 1, 0],
                 [1, 1, 1, 1]]] # No padding in second sequence
    """
    batch_size, tgt_len = tgt.shape

    # 1. Causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    # 2. Padding mask
    padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]

    # 3. Combine: element-wise AND
    # Position is valid if BOTH:
    # - Not in future (causal)
    # - Not padding
    mask = causal_mask * padding_mask

    return mask.float()


class TrainingMetrics:
    """
    Track and compute training metrics

    Metrics:
    - Loss (training and validation)
    - Token accuracy
    - Sequence accuracy
    - Perplexity
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_sequences = 0
        self.correct_sequences = 0
        self.num_batches = 0

    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_idx: int = 0
    ):
        """
        Update metrics with batch results

        Args:
            loss: Batch loss value
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Ground truth [batch_size, seq_len]
            pad_idx: Padding token to ignore
        """
        self.total_loss += loss
        self.num_batches += 1

        # Get predictions
        preds = logits.argmax(dim=-1)  # [batch_size, seq_len]

        # Create mask for non-padding positions
        mask = (targets != pad_idx)

        # Token-level accuracy
        correct = (preds == targets) & mask
        self.correct_tokens += correct.sum().item()
        self.total_tokens += mask.sum().item()

        # Sequence-level accuracy (all tokens in sequence must be correct)
        # For each sequence, check if all non-padding tokens match
        seq_correct = ((preds == targets) | ~mask).all(dim=1)
        self.correct_sequences += seq_correct.sum().item()
        self.total_sequences += targets.size(0)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics

        Returns:
            Dictionary with metric names and values
        """
        avg_loss = self.total_loss / max(self.num_batches, 1)

        token_acc = 100.0 * self.correct_tokens / max(self.total_tokens, 1)
        seq_acc = 100.0 * self.correct_sequences / max(self.total_sequences, 1)

        # Perplexity = exp(loss)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'loss': avg_loss,
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'perplexity': perplexity
        }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TransformerLRScheduler,
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str = 'checkpoints',
    filename: str = None
):
    """
    Save training checkpoint

    Args:
        model: Transformer model
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch number
        metrics: Training metrics dictionary
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename (auto-generated if None)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    if filename is None:
        filename = f'checkpoint_epoch_{epoch:03d}.pt'

    checkpoint_path = checkpoint_dir / filename

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_step': scheduler.step_num,
        'metrics': metrics
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"[+] Checkpoint saved: {checkpoint_path}")

    # Also save as "latest"
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[TransformerLRScheduler] = None
) -> Dict:
    """
    Load training checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Transformer model (will be updated in-place)
        optimizer: Optimizer (optional, will be updated if provided)
        scheduler: LR scheduler (optional, will be updated if provided)

    Returns:
        Dictionary with checkpoint info (epoch, metrics)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.step_num = checkpoint['scheduler_step']

    print(f"[+] Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Metrics: {checkpoint['metrics']}")

    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics']
    }


def save_training_config(config: Dict, save_dir: str = 'checkpoints'):
    """
    Save training configuration to JSON

    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    config_path = save_dir / 'config.json'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"[+] Config saved: {config_path}")
