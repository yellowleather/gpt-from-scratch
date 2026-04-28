#!/usr/bin/env python3

"""main.py

Main driver for LLM training pipeline.

Usage examples:
    # With uv (recommended, no global install)
    uv sync && uv run python main.py

This script requires tiktoken (see pyproject.toml).
"""

import sys
from urllib.error import HTTPError, URLError

from tokenizer.tokenizer_factory import get_tokenizer
from training_data_provider import get_provider
from data_loader import create_dataset, create_dataloader
from embedding_stemmer import get_embedding_stem
from transformer_block import get_transformer_block


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

CONFIG = {
    # Data source
    "url": (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    ),
    "cache_path": "tokenizer/input/the_verdict.txt",
    "timeout": 30,             # seconds for HTTP requests

    # Dataset / dataloader
    "max_length": 128,         # sliding-window sequence length
    "stride": 64,              # sliding-window step size
    "batch_size": 4,

    # Model architecture
    "vocab_size": 50257,       # GPT-2 BPE vocabulary
    "emb_dim": 256,            # token + position embedding dimension
    "n_heads": 2,              # attention heads (emb_dim must be divisible)
    "n_layers": 2,             # number of stacked transformer blocks
    "drop_rate": 0.0,          # dropout probability
    "qkv_bias": False,         # bias in Q/K/V projections
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Create a training data provider
    try:
        training_data_provider = get_provider(
            provider_type="simple_text",
            url=CONFIG["url"],
            cache_path=CONFIG["cache_path"],
            timeout=CONFIG["timeout"],
            use_cache=True,
        )
    except ValueError as e:
        print(f"Failed to create provider: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch the training text (will use cache if available, otherwise download)
    try:
        raw_text = training_data_provider.get_text()
        print(f"Training data loaded ({len(raw_text)} characters)")
        print(f"Cache location: {CONFIG['cache_path']}")
    except HTTPError as e:
        print(f"HTTP error {e.code} when fetching {CONFIG['url']}: {e.reason}", file=sys.stderr)
        sys.exit(2)
    except URLError as e:
        print(f"URL error when fetching {CONFIG['url']}: {e.reason}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error when fetching training data: {e}", file=sys.stderr)
        sys.exit(4)

    # Obtain tokenizer from factory
    try:
        tokenizer = get_tokenizer()
    except Exception as e:
        print(f"Failed to obtain tokenizer from factory: {e}", file=sys.stderr)
        sys.exit(8)

    try:
        tokens = tokenizer.encode(raw_text)
        print(f"Tokenized into {len(tokens)} tokens; first 20 tokens: {tokens[:20]}")
    except Exception as e:
        print(f"Tokenization failed: {e}", file=sys.stderr)
        sys.exit(9)

    print(f"Tokenizer '{tokenizer.name}' is ready to use.")

    # Create dataset and dataloader
    try:
        dataset = create_dataset(
            dataset_type="gpt_v1",
            txt=raw_text,
            tokenizer=tokenizer,
            max_length=CONFIG["max_length"],
            stride=CONFIG["stride"],
        )
        print(f"Dataset created with {len(dataset)} samples")
        print(f"  max_length: {CONFIG['max_length']}, stride: {CONFIG['stride']}")
    except Exception as e:
        print(f"Failed to create dataset: {e}", file=sys.stderr)
        sys.exit(10)

    try:
        dataloader = create_dataloader(
            dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        print(f"DataLoader created with batch_size={CONFIG['batch_size']}")
    except Exception as e:
        print(f"Failed to create dataloader: {e}", file=sys.stderr)
        sys.exit(11)

    # Show examples from the first batch
    num_examples_to_print = 3
    print(f"\nShowing first batch (up to {num_examples_to_print} examples):")
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print(f"\nInputs shape:  {inputs.shape}")
    print(f"Targets shape: {targets.shape}")

    # Build the embedding stem that fuses token and positional embeddings
    embedding_stem = get_embedding_stem(
        stem_type="gpt",
        vocab_size=CONFIG["vocab_size"],
        embedding_dim=CONFIG["emb_dim"],
        context_length=CONFIG["max_length"],
    )
    embeddings = embedding_stem(inputs)

    print(f"\nEmbedding output shape: {embeddings.shape}")
    print("  Example token embedding slice:", embeddings[0, 0, :5].tolist())

    # Stack n_layers transformer blocks sequentially.
    # Each block contains attention + feed-forward + layer norms + residuals,
    # so the output shape (batch, seq_len, emb_dim) is preserved throughout.
    trf_blocks = [
        get_transformer_block(
            emb_dim=CONFIG["emb_dim"],
            context_length=CONFIG["max_length"],
            n_heads=CONFIG["n_heads"],
            drop_rate=CONFIG["drop_rate"],
            qkv_bias=CONFIG["qkv_bias"],
        )
        for _ in range(CONFIG["n_layers"])
    ]
    x = embeddings
    for block in trf_blocks:
        x = block(x)

    print(f"\nTransformer stack output shape: {x.shape}")
    print(f"  n_layers: {CONFIG['n_layers']}, n_heads: {CONFIG['n_heads']}, head_dim: {CONFIG['emb_dim'] // CONFIG['n_heads']}")

    # Show a few examples from the batch
    num_examples = min(num_examples_to_print, inputs.shape[0])
    for i in range(num_examples):
        input_seq = inputs[i].tolist()
        target_seq = targets[i].tolist()
        example_embeddings = embeddings[i]
        example_out = x[i]
        print(f"\nExample {i + 1}:")
        print(f"  Input tokens:  {input_seq[:10]}..." if len(input_seq) > 10 else f"  Input tokens:  {input_seq}")
        print(f"  Target tokens: {target_seq[:10]}..." if len(target_seq) > 10 else f"  Target tokens: {target_seq}")
        print(f"  Input text:  \"{tokenizer.decode(input_seq[:20])}...\"")
        print(f"  Target text: \"{tokenizer.decode(target_seq[:20])}...\"")
        print(
            "  Embedding first token (dim 0-5):",
            example_embeddings[0, :5].tolist(),
        )
        print(
            "  Embedding mean/std:",
            f"{example_embeddings.mean().item():.4f}/{example_embeddings.std().item():.4f}",
        )
        print(
            "  Transformer out first token (dim 0-5):",
            example_out[0, :5].tolist(),
        )
        print(
            "  Transformer out mean/std:",
            f"{example_out.mean().item():.4f}/{example_out.std().item():.4f}",
        )


if __name__ == "__main__":
    main()
