# Embedding Layer: Understanding the Fundamentals

## Overview

This document summarizes key concepts about embedding layers in neural networks, particularly in the context of Large Language Models (LLMs).

## What is an Embedding Layer?

An **embedding layer** is a learnable lookup table that maps discrete tokens (like words or subwords) to continuous vector representations. In PyTorch, `nn.Embedding(vocab_size, embedding_dim)` creates a matrix of shape `(vocab_size, embedding_dim)` where each row represents a token's vector.

### Key Properties
- **Input**: Integer token IDs (e.g., `[5, 142, 7]`)
- **Output**: Dense vectors (e.g., `[[0.1, -0.5, 0.3, ...], ...]`)
- **Operation**: Simple row lookup from the embedding matrix
- **Linearity**: The operation is inherently linear

## One-Hot Encoding and Embeddings

### The Mathematical Relationship

An embedding lookup is mathematically equivalent to:
```
embedding(token_id) = one_hot(token_id) @ embedding_matrix
```

**Example:**
```python
# Token ID 2 in vocab of size 5
one_hot = [0, 0, 1, 0, 0]

# Embedding matrix (5 x 3)
embedding_matrix = [
    [0.1, 0.2, 0.3],  # token 0
    [0.4, 0.5, 0.6],  # token 1
    [0.7, 0.8, 0.9],  # token 2  ← selected
    [1.0, 1.1, 1.2],  # token 3
    [1.3, 1.4, 1.5]   # token 4
]

# Result: [0.7, 0.8, 0.9]
```

### Why We Don't Actually Use One-Hot Encoding

While conceptually equivalent, we don't use one-hot encoding in practice because:

1. **Memory efficiency**: One-hot vectors are sparse (mostly zeros)
   - Vocab size of 50,000 → 50,000-dimensional vector with only one 1
   - Direct indexing uses far less memory

2. **Computational efficiency**: Matrix multiplication with one-hot is wasteful
   - Direct lookup: O(1) operation
   - Matrix multiplication: O(vocab_size × embedding_dim) operations

3. **Implementation**: Modern frameworks optimize direct indexing

## Are Linear Layers Required for Neural Networks?

**No!** Neural networks can and do use non-linear layers. In fact, **non-linearity is essential** for neural networks to learn complex patterns.

### Common Non-Linear Components

1. **Activation Functions**
   - ReLU: `f(x) = max(0, x)`
   - GELU: `f(x) = x * Φ(x)` (used in GPT)
   - Sigmoid, Tanh, etc.

2. **Non-Linear Layers**
   - Convolutions with non-linear activations
   - Self-attention mechanisms (contain softmax)
   - Layer normalization (contains division/square root)

3. **Why Non-Linearity Matters**
   - Without it, stacking layers is pointless: `f(g(x)) = Wx + b` (still linear!)
   - Non-linearity enables learning complex decision boundaries
   - Universal approximation theorem requires non-linearity

## Can Embeddings Be Non-Linear?

**Technically yes, but practically no.** Here's why:

### Theoretical Possibility

You *could* apply non-linear transformations:
```python
# Non-linear embedding (not standard)
embedding = self.embedding_table[token_id]
non_linear_embedding = torch.tanh(embedding)
```

### Why Embeddings Stay Linear in LLMs

1. **Lack of Context at This Stage**
   - Embeddings are just the initial representation
   - No information about surrounding tokens yet
   - Non-linearity without context doesn't help

2. **Non-Linearity Comes Later**
   - Self-attention layers provide context-aware mixing
   - Feed-forward networks add non-linearity
   - The entire transformer stack is highly non-linear

3. **Flexibility in Learning**
   - Linear embeddings can represent any initial distribution
   - The network learns the optimal representation during training
   - Subsequent layers apply sophisticated non-linear transformations

4. **Mathematical Elegance**
   - Linear embeddings preserve vector space properties
   - Easy to add (token + position embeddings)
   - Residual connections work naturally

5. **Empirical Success**
   - All successful LLMs (GPT, BERT, LLaMA, etc.) use linear embeddings
   - No evidence that non-linear embeddings improve performance
   - The transformer architecture's non-linearity is sufficient

## The GPT Architecture Flow

```
Input Token IDs
      ↓
[Linear] Token Embedding Lookup
      ↓
[Linear] Position Embedding Lookup
      ↓
[Linear] Add them together
      ↓
[Non-Linear] Transformer Blocks:
   - Multi-head self-attention (softmax = non-linear)
   - Layer normalization (non-linear)
   - Feed-forward network with GELU (non-linear)
   - Residual connections
      ↓
[Linear] Final projection to vocab
      ↓
[Non-Linear] Softmax for probabilities
      ↓
Output Token Probabilities
```

## Summary

- **Embeddings are linear**: They're just lookup tables
- **One-hot encoding**: Conceptually equivalent but inefficient in practice
- **Neural networks need non-linearity**: But not at the embedding stage
- **LLMs use linear embeddings**: Because context-aware non-linearity happens in transformer layers
- **The network as a whole is highly non-linear**: Despite linear embeddings

## Code Example

See [gpt_embedding_stem.py](gpt_embedding_stem.py) for our implementation of GPT-style embeddings.

```python
class GPTEmbeddingStem(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_length: int):
        super().__init__()
        # Both are simple linear lookup tables
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Linear operations only
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        return token_embeddings + position_embeddings  # Linear addition
```

## Further Reading

- **Attention Is All You Need** (Vaswani et al., 2017) - Original Transformer paper
- **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) - GPT-2
- **Universal Approximation Theorem** - Why non-linearity matters in neural networks
