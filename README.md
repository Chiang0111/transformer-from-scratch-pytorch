# Transformer from Scratch (PyTorch)

[English](README.md) | [中文](https://github.com/Chiang0111/transformer-from-scratch-pytorch/tree/zh-CN)

A **production-ready** PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), with **comprehensive educational comments** that serve as a complete learning guide.

## 🌟 What Makes This Special

Unlike typical implementations, **every line of code is extensively documented** with:
- 📚 **Why it's needed** - Understand the motivation behind each component
- 🔍 **How it works** - Step-by-step explanations with concrete examples
- 💡 **Intuitive analogies** - Library analogy for Attention, clock analogy for positional encoding
- 📊 **Visual diagrams** - ASCII art showing data flow and transformations
- 🎯 **Real examples** - "I love eating apples" traced through entire architecture

**The code itself is the tutorial!** Read the source files to learn Transformers deeply.

## Why This Repository Exists

This project bridges the gap between understanding transformers conceptually and implementing them professionally. It demonstrates:

- ✅ **Deep understanding** - Built from scratch, not just using `transformers` library
- ✅ **Production practices** - Modular code, type hints, unit tests, proper documentation
- ✅ **Clean architecture** - Each component is isolated, tested, and reusable
- ✅ **Educational excellence** - 1,500+ lines of explanatory comments (more than the code itself!)
- ✅ **Portfolio-ready** - Shows AI engineering skills, not just tutorial following

**Target audience:** Self-taught ML practitioners preparing for AI Engineer roles who need to demonstrate strong fundamentals and production coding skills.

## How This Differs from Other Transformer Tutorials

| Most Tutorials | This Repository |
|----------------|-----------------|
| Single Jupyter notebook | Modular Python package |
| No tests | Unit tests for every component (35 tests) |
| Minimal documentation | **1,500+ lines of educational comments** |
| Brief inline comments | Complete learning guide in the code |
| "Just make it work" | Production-ready code structure |
| One messy commit | Thoughtful git history with detailed commits |
| No training example | End-to-end training pipeline (coming) |
| GPU required | CPU-friendly (small model) |
| Code only | Code + intuitive analogies + visual diagrams |

**Philosophy:** If you can't explain it in clean, tested code with comprehensive documentation, you don't understand it well enough.

## Project Status

🚧 **Work in Progress** - Following the development plan in [PLAN.md](PLAN.md)

- [x] **Phase 1: Foundation** ✅ Complete (35 tests passing)
  - ✅ Attention mechanisms (with comprehensive Q/K/V explanations)
  - ✅ Positional encoding (sin/cos function explained with clock analogy)
  - ✅ Feedforward networks (FFN role clarified with library analogy)
  - ✅ Encoder layers (complete architecture with residual & normalization)
  - ✅ **1,500+ lines of educational comments** - Learn by reading the code!
- [ ] Phase 2: Architecture (Decoder/Full Transformer)
- [ ] Phase 3: Training (Real dataset)
- [ ] Phase 4: Polish (Documentation & examples)

## 📖 How to Learn from This Repository

**This repository is designed to be read like a textbook!** Start here:

1. **Start with Attention** (`transformer/attention.py`)
   - Understand Q, K, V with library analogy
   - Learn why scaling is crucial (√d_k explained)
   - See how multi-head attention works (8 experts analogy)

2. **Add Position Info** (`transformer/positional_encoding.py`)
   - Understand why Attention is order-agnostic
   - Learn sin/cos encoding with clock analogy
   - See concrete examples position by position

3. **Process Information** (`transformer/feedforward.py`)
   - Understand why FFN is needed after Attention
   - Learn the expand→transform→compress pattern
   - Compare ReLU vs GELU activations

4. **Complete the Layer** (`transformer/encoder.py`)
   - See how all components integrate
   - Understand residual connections & layer normalization
   - Follow "I love eating apples" through the entire encoder

**Every file contains:**
- Detailed "Why" explanations
- Step-by-step "How" breakdowns  
- Concrete numerical examples
- Visual ASCII diagrams
- Common pitfalls and solutions

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
