# Contributing to Transformer from Scratch

Thank you for your interest in contributing! This is an educational project aimed at helping people understand transformers deeply.

## 🎯 Project Goals

- **Educational first**: Code clarity and comprehensive comments over performance
- **Production practices**: Show how to write professional, maintainable ML code
- **Complete implementation**: Every component built from scratch, no black boxes

## 🤝 How to Contribute

### Reporting Issues

Found a bug or have a suggestion?

1. Check if the issue already exists
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (Python version, OS)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comprehensive comments (this is an educational project!)
   - Include docstrings with examples
   - Add unit tests for new features

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```
   All 80+ tests must pass!

5. **Commit your changes**
   ```bash
   git commit -m "feat: add feature X with comprehensive comments"
   ```
   Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Comments and Documentation

This is an **educational** project - comments are critical!

**Good example:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    【What Is This?】
    The core attention mechanism from "Attention Is All You Need".
    Computes how much each position should attend to every other position.
    
    【Why Scaling?】
    Without scaling by √d_k, the dot products can become very large,
    pushing the softmax into regions with tiny gradients.
    
    Args:
        Q: Query matrix [batch, heads, seq_len, d_k]
        K: Key matrix [batch, heads, seq_len, d_k]
        V: Value matrix [batch, heads, seq_len, d_v]
        mask: Optional mask [batch, 1, seq_len, seq_len]
    
    Returns:
        Attention output [batch, heads, seq_len, d_v]
    
    Example:
        >>> Q = torch.randn(2, 8, 10, 64)  # batch=2, heads=8, len=10, d_k=64
        >>> K = torch.randn(2, 8, 10, 64)
        >>> V = torch.randn(2, 8, 10, 64)
        >>> output = scaled_dot_product_attention(Q, K, V)
        >>> output.shape
        torch.Size([2, 8, 10, 64])
    """
    # Implementation...
```

**Not this:**
```python
def attention(q, k, v):
    # compute attention
    return output
```

### Testing

- Add tests for all new functionality
- Tests should be clear and well-documented
- Include edge cases
- Use descriptive test names

**Good test name:**
```python
def test_attention_mask_prevents_future_positions():
    """Causal mask should block attention to future tokens"""
```

### Type Hints

Use type hints for better code clarity:
```python
def create_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512
) -> Transformer:
```

## 🎓 Contribution Ideas

### Beginner-Friendly

- Fix typos in documentation
- Improve code comments for clarity
- Add more examples to docstrings
- Translate documentation to other languages

### Intermediate

- Add visualization tools (attention heatmaps)
- Create Jupyter notebook tutorials
- Add more training tasks (e.g., addition, multiplication)
- Improve error messages

### Advanced

- Add beam search for generation
- Implement additional architectures (encoder-only, decoder-only)
- Add pre-training support
- Performance optimizations (while keeping clarity!)

## 🚫 What We're NOT Looking For

- **Black-box implementations**: We want every line understandable
- **Performance hacks**: Clarity > speed (this is educational)
- **Removing comments**: More comments = better!
- **External dependencies**: Keep it minimal (just PyTorch, NumPy, pytest)

## 📚 Resources

- [Original Paper](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need"
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Project README](README.md) - Start here to understand the codebase

## 💬 Questions?

- Open an issue for discussion
- Check existing issues and documentation first
- Be respectful and constructive

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make transformers more accessible to everyone! 🚀**
