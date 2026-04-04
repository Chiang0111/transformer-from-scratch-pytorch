# Transformer from Scratch (PyTorch)

[English](README.md) | [中文](https://github.com/Chiang0111/transformer-from-scratch-pytorch/tree/zh-CN)

A **production-ready** PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## Why This Repository Exists

This project bridges the gap between understanding transformers conceptually and implementing them professionally. It demonstrates:

- ✅ **Deep understanding** - Built from scratch, not just using `transformers` library
- ✅ **Production practices** - Modular code, type hints, unit tests, proper documentation
- ✅ **Clean architecture** - Each component is isolated, tested, and reusable
- ✅ **Portfolio-ready** - Shows AI engineering skills, not just tutorial following

**Target audience:** Self-taught ML practitioners preparing for AI Engineer roles who need to demonstrate strong fundamentals and production coding skills.

## How This Differs from Other Transformer Tutorials

| Most Tutorials | This Repository |
|----------------|-----------------|
| Single Jupyter notebook | Modular Python package |
| No tests | Unit tests for every component |
| Minimal documentation | Comprehensive docstrings + README |
| "Just make it work" | Production-ready code structure |
| One messy commit | Thoughtful git history |
| No training example | End-to-end training pipeline |
| GPU required | CPU-friendly (small model) |

**Philosophy:** If you can't explain it in clean, tested code, you don't understand it well enough.

## Project Status

🚧 **Work in Progress** - Following the development plan in [PLAN.md](PLAN.md)

- [x] **Phase 1: Foundation** ✅ Complete (34 tests passing)
  - Attention mechanisms
  - Positional encoding
  - Feedforward networks
  - Encoder layers
- [ ] Phase 2: Architecture (Decoder/Full Transformer)
- [ ] Phase 3: Training (Real dataset)
- [ ] Phase 4: Polish (Documentation & examples)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- No GPU required (optimized for CPU training with small models)

## Quick Start

*(Coming soon after Phase 1)*

```python
from transformer import Transformer

# Initialize model
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=256,
    num_heads=4,
    num_layers=2
)

# Train or inference...
```

## Project Structure

```
transformer-from-scratch-pytorch/
├── transformer/              # Core implementation
│   ├── attention.py         # Scaled dot-product & multi-head attention
│   ├── encoder.py           # Encoder layers
│   ├── decoder.py           # Decoder layers
│   ├── positional_encoding.py
│   ├── feedforward.py
│   └── model.py             # Full transformer
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── PLAN.md                  # Development roadmap
└── README.md               # This file
```

## Learning Resources

- **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Visualization:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Implementation reference:** [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## License

MIT License - feel free to use for learning and portfolios!
