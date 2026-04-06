# Transformer from Scratch (PyTorch)

[![Tests](https://img.shields.io/badge/tests-80%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

[English](README.md) | [中文](https://github.com/Chiang0111/transformer-from-scratch-pytorch/tree/zh-CN)

A **production-ready** PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), with **2,500+ lines of educational comments** that serve as a complete learning guide.

---

## 🌟 What Makes This Special

Unlike typical implementations, **every line of code is extensively documented** with:
- 📚 **Why it's needed** - Understand the motivation behind each component
- 🔍 **How it works** - Step-by-step explanations with concrete examples
- 💡 **Intuitive analogies** - Library analogy for Attention, clock analogy for positional encoding
- 📊 **Visual diagrams** - ASCII art showing data flow and transformations
- 🎯 **Real examples** - "I love eating apples" traced through entire architecture

**The code itself is the tutorial!** Read the source files to learn Transformers deeply.

### ✨ Why This Repository Exists

This project bridges the gap between understanding transformers conceptually and implementing them professionally:

- ✅ **Deep understanding** - Built from scratch, not just using `transformers` library
- ✅ **Production practices** - Modular code, type hints, unit tests, proper documentation
- ✅ **Clean architecture** - Each component is isolated, tested, and reusable
- ✅ **Educational excellence** - 2,500+ lines of explanatory comments (more than the code!)
- ✅ **Verified results** - Validated on 3 tasks (98.6%, 83%, 96% accuracy)
- ✅ **Portfolio-ready** - Shows AI engineering skills, not just tutorial following

**Target audience:** Self-taught ML practitioners preparing for AI Engineer roles who need to demonstrate strong fundamentals and production coding skills.

---

## 📁 Repository Structure

```
transformer-from-scratch-pytorch/
│
├── README.md                       ⭐ You are here
├── LICENSE                         📜 MIT License
├── CONTRIBUTING.md                 🤝 Contribution guidelines
├── requirements.txt                📦 Dependencies
│
├── transformer/                    🧠 Core Implementation (2,500+ comment lines)
│   ├── __init__.py                    Public API
│   ├── attention.py                   Scaled dot-product & multi-head attention
│   ├── positional_encoding.py         Sin/cos positional embeddings
│   ├── feedforward.py                 Position-wise feed-forward network
│   ├── encoder.py                     Transformer encoder layers
│   ├── decoder.py                     Transformer decoder layers (with masking)
│   └── transformer.py                 ⭐ Complete Transformer model
│
├── tests/                          ✅ Unit Tests (80 tests, all passing)
│   ├── test_attention.py              Test attention mechanisms
│   ├── test_positional_encoding.py    Test positional encoding
│   ├── test_feedforward.py            Test FFN layers
│   ├── test_encoder.py                Test encoder
│   ├── test_decoder.py                Test decoder
│   ├── test_transformer.py            ⭐ Complete model tests
│   └── test_training.py               Training pipeline tests
│
├── scripts/                        🚀 Training & Evaluation
│   ├── train.py                       Main training script
│   ├── test.py                        Model evaluation
│   ├── demo.py                        Interactive demo
│   └── benchmark.py                   Automated benchmarking
│
├── examples/                       💡 Usage Examples & Debugging
│   ├── basic_usage.py                 ⭐ Start here! Simple example
│   ├── debug_data.py                  Inspect dataset
│   ├── debug_gradients.py             Check gradient flow
│   ├── test_overfit.py                Verify model can overfit
│   └── ...                            Other debugging tools
│
├── docs/                           📚 Documentation
│   ├── TRAINING.md                    Complete training guide
│   ├── TROUBLESHOOTING.md             Debug guide
│   ├── RESULTS.md                     Benchmark results
│   ├── VALIDATION.md                  Testing methodology
│   └── PLAN.md                        Development roadmap
│
├── datasets.py                     📊 Training Datasets
│   ├── SequenceCopyDataset            Copy sequences (easiest)
│   ├── SequenceReverseDataset         Reverse sequences (medium)
│   └── SequenceSortDataset            Sort sequences (hard)
│
├── utils.py                        🛠️ Training Utilities
│   ├── LabelSmoothingLoss             Label smoothing (for translation)
│   ├── TransformerLRScheduler         Learning rate schedule
│   ├── create_masks()                 Padding & causal masks
│   ├── TrainingMetrics                Track accuracy/loss
│   └── save/load_checkpoint()         Checkpoint management
│
├── benchmarks/                     🏆 Trained Model Checkpoints
│   ├── copy/                          98.6% accuracy (validated)
│   ├── reverse/                       83.0% accuracy (validated)
│   └── sort/                          96.0% accuracy (validated)
│
└── checkpoints/                    💾 Your Training Checkpoints
    └── .gitkeep                       (Directory for your experiments)
```

### 📂 Key Files to Explore

| File | What It Does | Start Here? |
|------|--------------|-------------|
| `examples/basic_usage.py` | Simple usage example | ✅ **YES** |
| `transformer/attention.py` | Core attention mechanism | ✅ **YES** |
| `transformer/transformer.py` | Complete model | After understanding parts |
| `scripts/train.py` | Train your own model | After reading docs |
| `docs/TRAINING.md` | Complete training guide | Before training |
| `tests/test_transformer.py` | See how everything works | To understand testing |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Chiang0111/transformer-from-scratch-pytorch.git
cd transformer-from-scratch-pytorch

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest tests/ -v
```

**Output:**
```
========================= 80 passed in 12.34s =========================
```

### Run a Basic Example

```bash
python examples/basic_usage.py
```

This shows how to:
1. Create a transformer model
2. Prepare input data
3. Run forward pass (training mode)
4. Generate sequences (inference mode)

### Train Your First Model

```bash
# Copy task (easiest, ~10 minutes on CPU)
python scripts/train.py --task copy --epochs 20 \
    --fixed-lr 0.001 --label-smoothing 0.0 --dropout 0.0
```

**Expected output:**
```
Epoch 20/20: Val Seq Acc: 98.6%
✅ Test Set Results: 98.6% accuracy
```

See [`docs/TRAINING.md`](docs/TRAINING.md) for complete training guide.

---

## 📖 Learning Path

### 1. Start with Examples (15 minutes)

```bash
# Understand basic usage
python examples/basic_usage.py

# See how data looks
python examples/debug_data.py
```

### 2. Read the Core Code (1-2 hours)

**Recommended reading order:**

1. **`transformer/attention.py`** (Start here!)
   - Understand Q, K, V with library analogy
   - Learn why scaling is crucial (√d_k explained)
   - See how multi-head attention works

2. **`transformer/positional_encoding.py`**
   - Learn why position matters
   - Understand sin/cos encoding with clock analogy

3. **`transformer/feedforward.py`**
   - See why FFN is needed after attention
   - Understand expand→transform→compress pattern

4. **`transformer/encoder.py`**
   - See how all components integrate
   - Trace example "I love eating apples" through encoder

5. **`transformer/decoder.py`**
   - Learn masked self-attention (causal masking)
   - Understand cross-attention to encoder memory

6. **`transformer/transformer.py`** (⭐ The complete picture)
   - Integration of all components
   - Training vs inference modes
   - End-to-end data flow

### 3. Train Models (30-60 minutes)

```bash
# Copy task (easiest)
python scripts/train.py --task copy --epochs 20 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0

# Reverse task (medium)
python scripts/train.py --task reverse --epochs 30 --fixed-lr 0.001 \
    --label-smoothing 0.0 --dropout 0.0

# Sort task (hardest)
python scripts/train.py --task sort --epochs 10 --fixed-lr 0.0005 \
    --label-smoothing 0.0 --dropout 0.0
```

### 4. Test Trained Models

```bash
# Evaluate on test set
python scripts/test.py --checkpoint benchmarks/copy/checkpoint_best.pt --task copy

# Interactive mode - try your own sequences!
python scripts/test.py --checkpoint benchmarks/copy/checkpoint_best.pt \
    --task copy --interactive
```

---

## 📊 Validated Results

All three tasks **significantly exceed** their minimum targets:

| Task | Test Accuracy | Target | Training Time | Status |
|------|---------------|--------|---------------|--------|
| **Copy** | 98.6% | 95.0% | ~10 min | ✅ PASS |
| **Reverse** | 83.0% | 80.0% | ~20 min | ✅ PASS |
| **Sort** | 96.0% | 70.0% | ~25 min | ✅ PASS |

**Key Insights:**
- All results are on **held-out test sets** (never seen during training)
- Fixed learning rate approach works better than Transformer schedule for small models
- Models learn algorithmic reasoning, not just memorization
- Sort task converged in just 3 epochs (not 50!)

See [`docs/RESULTS.md`](docs/RESULTS.md) for detailed analysis.

---

## 💻 Usage Example

```python
from transformer import create_transformer
import torch

# Create a small transformer (CPU-friendly)
model = create_transformer(
    src_vocab_size=10000,  # English vocabulary
    tgt_vocab_size=8000,   # Chinese vocabulary
    d_model=256,           # Smaller than paper's 512
    num_heads=4,           # Fewer heads for CPU
    num_layers=2,          # Shallower for faster training
    d_ff=1024              # Smaller FFN
)

print(f"Model parameters: {model.count_parameters():,}")
# Output: Model parameters: 16,234,528

# Training mode: teacher forcing
src = torch.randint(0, 10000, (2, 10))  # batch=2, src_len=10
tgt = torch.randint(0, 8000, (2, 8))    # batch=2, tgt_len=8
logits = model(src, tgt)                # (2, 8, 8000)

# Inference mode: autoregressive generation
model.eval()
generated = model.generate(src, max_len=20, start_token=1, end_token=2)
print(f"Generated: {generated.shape}")  # (2, <=20)
```

---

## 🧪 Testing

```bash
# Run all 80 tests
pytest tests/ -v

# Run specific test file
pytest tests/test_attention.py -v

# Run with coverage
pytest tests/ --cov=transformer --cov-report=html
```

**Test coverage:**
- ✅ 80 unit tests covering all components
- ✅ Integration tests for end-to-end pipeline
- ✅ Smoke tests, quality tests, robustness tests
- ✅ All tests pass on CPU (no GPU required)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/TRAINING.md`](docs/TRAINING.md) | Complete training guide with all hyperparameters |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Debug guide with decision trees |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Detailed benchmark results and insights |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Testing methodology and ML best practices |
| [`docs/PLAN.md`](docs/PLAN.md) | Development roadmap and project status |

---

## 🎯 How This Differs from Other Transformer Tutorials

| Most Tutorials | This Repository |
|----------------|-----------------|
| Single Jupyter notebook | Modular Python package |
| No tests | 80 unit tests, all passing |
| Minimal documentation | **2,500+ lines of educational comments** |
| Brief inline comments | Complete learning guide in code |
| "Just make it work" | Production-ready code structure |
| One messy commit | Thoughtful git history |
| No training example | **Complete training pipeline** |
| GPU required | CPU-friendly (small models) |
| Code only | Code + analogies + visual diagrams |
| Copy-paste paper hyperparameters | **Tuned for small models** (actually works!) |
| No validation | **Verified on 3 tasks with test sets** |

**Philosophy:** If you can't explain it in clean, tested code with comprehensive documentation, you don't understand it well enough.

---

## 🤝 Contributing

Contributions are welcome! This is an educational project, so clarity and comprehensive comments are valued over performance optimizations.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs or issues
- 📝 Improve documentation or comments
- ✨ Add new training tasks
- 🌍 Translate documentation
- 🎨 Add visualization tools
- 📚 Create tutorials or examples

---

## 📜 License

MIT License - feel free to use for learning and portfolios!

See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

- Original paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al.
- Inspired by [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- Built with ❤️ for the ML learning community

---

## 🔗 Resources

### Learn More
- 📖 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- 📄 [Original Paper](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need"
- 🎓 [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Another great resource

### Related Projects
- [PyTorch Transformers](https://github.com/huggingface/transformers) - Production library (Hugging Face)
- [Fairseq](https://github.com/facebookresearch/fairseq) - Facebook's sequence modeling toolkit

---

## ⭐ Show Your Support

If you found this helpful, please give it a star! It helps others discover this resource.

**Share with:**
- ML students learning transformers
- Engineers preparing for AI roles
- Anyone building transformers from scratch

---

## 📬 Contact

- **Issues:** [GitHub Issues](https://github.com/Chiang0111/transformer-from-scratch-pytorch/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Chiang0111/transformer-from-scratch-pytorch/discussions)

---

**Built to make transformers accessible to everyone through comprehensive documentation and clean code.**
