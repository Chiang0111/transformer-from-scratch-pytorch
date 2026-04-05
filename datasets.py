"""
Dataset utilities for training Transformer models

This module provides simple datasets for testing and training:
1. Sequence Copy - Learn to copy input sequences
2. Sequence Reverse - Learn to reverse sequences
3. Sequence Sort - Learn to sort numbers

These are ideal for:
- Quick training on CPU (5-10 minutes)
- Verifying the model works end-to-end
- Testing without external data dependencies
"""

import torch
from torch.utils.data import Dataset
import random
from typing import Tuple, List


class SequenceCopyDataset(Dataset):
    """
    Copy Sequence Task - Learn to copy input to output

    【What Is This Task?】
    The simplest possible sequence-to-sequence task.
    Given an input sequence, the model must learn to copy it exactly.

    Example:
        Input:  [3, 7, 2, 9, 1]
        Output: [3, 7, 2, 9, 1]

    【Why Is This Useful?】
    - Tests if the model can learn basic sequence-to-sequence mapping
    - Should converge very quickly (high accuracy in a few epochs)
    - If this doesn't work, something is fundamentally broken
    - No external data needed - generate on the fly

    【Task Difficulty】
    ⭐☆☆☆☆ Very Easy - should reach ~100% accuracy quickly

    【Special Tokens】
    - 0: <pad> - padding token
    - 1: <start> - start of sequence token
    - 2: <end> - end of sequence token
    - 3+: actual data tokens
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_seq_len: int = 3,
        max_seq_len: int = 10,
        vocab_size: int = 20,
        seed: int = 42
    ):
        """
        Initialize Copy Dataset

        Args:
            num_samples: Number of sequences to generate
            min_seq_len: Minimum sequence length
            max_seq_len: Maximum sequence length
            vocab_size: Size of vocabulary (number of unique tokens)
                Must be >= 3 (to account for <pad>, <start>, <end>)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.num_samples = num_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        # Generate all sequences upfront for consistency
        random.seed(seed)
        torch.manual_seed(seed)

        self.sequences = []
        for _ in range(num_samples):
            # Random length for this sequence
            seq_len = random.randint(min_seq_len, max_seq_len)

            # Generate random sequence (tokens from 3 to vocab_size-1)
            # Avoid 0, 1, 2 which are special tokens
            sequence = torch.randint(3, vocab_size, (seq_len,))

            self.sequences.append(sequence)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example

        Returns:
            src: Source sequence (input)
            tgt_input: Target input (with <start>, for teacher forcing)
            tgt_output: Target output (with <end>, for loss calculation)

        Example:
            Original sequence: [5, 8, 3]

            src:        [5, 8, 3]           # Input to encoder
            tgt_input:  [<start>, 5, 8, 3]  # Input to decoder (shifted right)
            tgt_output: [5, 8, 3, <end>]    # Expected output (for loss)
        """
        sequence = self.sequences[idx]

        # Source: just the sequence
        src = sequence.clone()

        # Target input: <start> + sequence (for teacher forcing)
        tgt_input = torch.cat([
            torch.tensor([self.START_TOKEN]),
            sequence
        ])

        # Target output: sequence + <end> (for loss calculation)
        tgt_output = torch.cat([
            sequence,
            torch.tensor([self.END_TOKEN])
        ])

        return src, tgt_input, tgt_output


class SequenceReverseDataset(Dataset):
    """
    Reverse Sequence Task - Learn to reverse input sequences

    【What Is This Task?】
    Given an input sequence, output it in reverse order.

    Example:
        Input:  [3, 7, 2, 9, 1]
        Output: [1, 9, 2, 7, 3]

    【Why Is This Useful?】
    - Tests if model can learn non-trivial transformations
    - Requires attention to positional information
    - Still simple enough to train quickly
    - Classic seq2seq benchmark task

    【Task Difficulty】
    ⭐⭐☆☆☆ Easy - should converge in 10-20 epochs
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_seq_len: int = 3,
        max_seq_len: int = 10,
        vocab_size: int = 20,
        seed: int = 42
    ):
        """Initialize Reverse Dataset (same args as CopyDataset)"""
        super().__init__()

        self.num_samples = num_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        # Generate sequences
        random.seed(seed)
        torch.manual_seed(seed)

        self.sequences = []
        for _ in range(num_samples):
            seq_len = random.randint(min_seq_len, max_seq_len)
            sequence = torch.randint(3, vocab_size, (seq_len,))
            self.sequences.append(sequence)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example

        Returns:
            src: Original sequence
            tgt_input: <start> + reversed sequence
            tgt_output: reversed sequence + <end>
        """
        sequence = self.sequences[idx]

        # Source: original sequence
        src = sequence.clone()

        # Reverse the sequence
        reversed_seq = torch.flip(sequence, dims=[0])

        # Target input: <start> + reversed
        tgt_input = torch.cat([
            torch.tensor([self.START_TOKEN]),
            reversed_seq
        ])

        # Target output: reversed + <end>
        tgt_output = torch.cat([
            reversed_seq,
            torch.tensor([self.END_TOKEN])
        ])

        return src, tgt_input, tgt_output


class SequenceSortDataset(Dataset):
    """
    Sort Sequence Task - Learn to sort numbers in ascending order

    【What Is This Task?】
    Given a sequence of numbers, output them sorted in ascending order.

    Example:
        Input:  [7, 3, 9, 1, 5]
        Output: [1, 3, 5, 7, 9]

    【Why Is This Useful?】
    - More challenging than copy/reverse
    - Tests if model can learn algorithmic reasoning
    - Requires understanding magnitude relationships

    【Task Difficulty】
    ⭐⭐⭐☆☆ Medium - may need more epochs to converge
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_seq_len: int = 3,
        max_seq_len: int = 10,
        vocab_size: int = 20,
        seed: int = 42
    ):
        """Initialize Sort Dataset (same args as CopyDataset)"""
        super().__init__()

        self.num_samples = num_samples
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2

        # Generate sequences
        random.seed(seed)
        torch.manual_seed(seed)

        self.sequences = []
        for _ in range(num_samples):
            seq_len = random.randint(min_seq_len, max_seq_len)
            sequence = torch.randint(3, vocab_size, (seq_len,))
            self.sequences.append(sequence)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example

        Returns:
            src: Original unsorted sequence
            tgt_input: <start> + sorted sequence
            tgt_output: sorted sequence + <end>
        """
        sequence = self.sequences[idx]

        # Source: original sequence
        src = sequence.clone()

        # Sort the sequence
        sorted_seq, _ = torch.sort(sequence)

        # Target input: <start> + sorted
        tgt_input = torch.cat([
            torch.tensor([self.START_TOKEN]),
            sorted_seq
        ])

        # Target output: sorted + <end>
        tgt_output = torch.cat([
            sorted_seq,
            torch.tensor([self.END_TOKEN])
        ])

        return src, tgt_input, tgt_output


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader - handles padding

    【What Does This Do?】
    When creating batches, sequences may have different lengths.
    We need to pad them to the same length so they can be batched together.

    Args:
        batch: List of (src, tgt_input, tgt_output) tuples

    Returns:
        src_batch: Padded source sequences [batch_size, max_src_len]
        tgt_input_batch: Padded target inputs [batch_size, max_tgt_len]
        tgt_output_batch: Padded target outputs [batch_size, max_tgt_len]

    Example:
        Input batch:
            src1: [5, 8, 3]        (len=3)
            src2: [7, 2, 9, 1, 4]  (len=5)

        Output:
            src_batch: [[5, 8, 3, 0, 0],      # padded to len=5
                        [7, 2, 9, 1, 4]]
    """
    # Separate src, tgt_input, tgt_output
    src_batch = [item[0] for item in batch]
    tgt_input_batch = [item[1] for item in batch]
    tgt_output_batch = [item[2] for item in batch]

    # Pad sequences to same length (padding value = 0)
    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )
    tgt_input_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_input_batch, batch_first=True, padding_value=0
    )
    tgt_output_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_output_batch, batch_first=True, padding_value=0
    )

    return src_batch, tgt_input_batch, tgt_output_batch


def create_dataloader(
    dataset_type: str = 'copy',
    batch_size: int = 32,
    num_samples: int = 10000,
    vocab_size: int = 20,
    train_split: float = 0.9
) -> Tuple:
    """
    Create train and validation dataloaders

    Args:
        dataset_type: 'copy', 'reverse', or 'sort'
        batch_size: Batch size for training
        num_samples: Total number of samples to generate
        vocab_size: Vocabulary size
        train_split: Fraction of data for training (rest for validation)

    Returns:
        train_loader, val_loader, dataset_info

    Example:
        train_loader, val_loader, info = create_dataloader('copy', batch_size=32)

        for src, tgt_input, tgt_output in train_loader:
            # src.shape: (32, max_src_len)
            # tgt_input.shape: (32, max_tgt_len)
            # tgt_output.shape: (32, max_tgt_len)
            ...
    """
    # Select dataset
    if dataset_type == 'copy':
        dataset = SequenceCopyDataset(num_samples=num_samples, vocab_size=vocab_size)
    elif dataset_type == 'reverse':
        dataset = SequenceReverseDataset(num_samples=num_samples, vocab_size=vocab_size)
    elif dataset_type == 'sort':
        dataset = SequenceSortDataset(num_samples=num_samples, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Split into train/val
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for Windows compatibility
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Dataset info for model creation
    dataset_info = {
        'vocab_size': vocab_size,
        'pad_token': dataset.PAD_TOKEN,
        'start_token': dataset.START_TOKEN,
        'end_token': dataset.END_TOKEN,
        'task': dataset_type,
        'train_samples': train_size,
        'val_samples': val_size
    }

    return train_loader, val_loader, dataset_info
