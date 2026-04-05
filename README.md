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
| No tests | Unit tests for every component (80 tests) |
| Minimal documentation | **2,500+ lines of educational comments** |
| Brief inline comments | Complete learning guide in the code |
| "Just make it work" | Production-ready code structure |
| One messy commit | Thoughtful git history with detailed commits |
| No training example | End-to-end training pipeline (coming) |
| GPU required | CPU-friendly (small model) |
| Code only | Code + intuitive analogies + visual diagrams |

**Philosophy:** If you can't explain it in clean, tested code with comprehensive documentation, you don't understand it well enough.

## Project Status

🚧 **Work in Progress** - Following the development plan in [PLAN.md](PLAN.md)

- [x] **Phase 1: Foundation** ✅ Complete
  - ✅ Attention mechanisms (with comprehensive Q/K/V explanations)
  - ✅ Positional encoding (sin/cos function explained with clock analogy)
  - ✅ Feedforward networks (FFN role clarified with library analogy)
  - ✅ Encoder layers (complete architecture with residual & normalization)
- [x] **Phase 2: Decoder** ✅ Complete
  - ✅ Decoder layers (masked self-attention + cross-attention)
  - ✅ Causal masking for autoregressive generation
  - ✅ Encoder-Decoder integration tests
- [x] **Phase 3: Complete Model** ✅ Architecture Complete (80 tests passing)
  - ✅ Token embeddings with scaling
  - ✅ Full Transformer model (Encoder + Decoder)
  - ✅ Autoregressive generation (inference mode)
  - ✅ **2,500+ lines of educational comments** - Learn by reading the code!
  - ⏳ Training loop & dataset (in progress)
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

4. **Build the Encoder** (`transformer/encoder.py`)
   - See how all components integrate
   - Understand residual connections & layer normalization
   - Follow "I love eating apples" through the entire encoder

5. **Add the Decoder** (`transformer/decoder.py`)
   - Learn masked self-attention (causal masking)
   - Understand cross-attention to encoder memory
   - See autoregressive generation step-by-step

6. **Complete Model** (`transformer/transformer.py`)
   - Integration of all components
   - Training vs inference modes
   - End-to-end data flow from tokens to logits

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

```bash
# Install dependencies
pip install torch pytest

# Run tests to verify everything works
pytest tests/ -v

# Use the model
python
```

```python
from transformer import create_transformer
import torch

# Create a small Transformer (CPU-friendly)
model = create_transformer(
    src_vocab_size=10000,  # English vocabulary
    tgt_vocab_size=8000,   # Chinese vocabulary
    d_model=256,           # Smaller than paper's 512
    num_heads=4,           # Fewer heads for CPU
    num_layers=2,          # Shallower for faster training
    d_ff=1024              # Smaller FFN
)

print(f"Model parameters: {model.count_parameters():,}")

# Training mode: teacher forcing
src = torch.randint(0, 10000, (2, 10))  # batch=2, src_len=10
tgt = torch.randint(0, 8000, (2, 8))    # batch=2, tgt_len=8
logits = model(src, tgt)  # (2, 8, 8000)

# Inference mode: autoregressive generation
model.eval()
generated = model.generate(src, max_len=20, start_token=1, end_token=2)
print(f"Generated: {generated.shape}")  # (2, <=20)
```

## Project Structure

```
transformer-from-scratch-pytorch/
├── transformer/              # Core implementation (2,500+ lines of comments)
│   ├── attention.py         # Scaled dot-product & multi-head attention
│   ├── positional_encoding.py  # Sinusoidal position embeddings
│   ├── feedforward.py       # Position-wise FFN
│   ├── encoder.py           # Encoder layers (bidirectional attention)
│   ├── decoder.py           # Decoder layers (masked + cross attention)
│   ├── transformer.py       # Complete Transformer model ⭐
│   └── __init__.py          # Public API
├── tests/                   # Comprehensive unit tests (80 tests)
│   ├── test_attention.py    # 7 tests
│   ├── test_positional_encoding.py  # 5 tests
│   ├── test_feedforward.py  # 8 tests
│   ├── test_encoder.py      # 14 tests
│   ├── test_decoder.py      # 20 tests
│   ├── test_transformer.py  # 26 tests ⭐
│   └── README.md            # Test documentation
├── PLAN.md                  # Development roadmap
└── README.md                # This file
```

## Learning Resources

- **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Visualization:** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Implementation reference:** [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

## License

MIT License - feel free to use for learning and portfolios!
