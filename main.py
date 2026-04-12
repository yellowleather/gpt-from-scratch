#!/usr/bin/env python3

"""main.py

Main driver for LLM training pipeline.

Currently supports downloading text from a URL and tokenizing it.

Usage examples:
    # With uv (recommended, no global install)
    uv sync && uv run python main.py

This script requires tiktoken (see pyproject.toml).
"""

import argparse
import sys
from urllib.error import HTTPError, URLError

from tokenizer.tokenizer_factory import get_tokenizer
from training_data_provider import get_provider
from data_loader import create_dataset, create_dataloader
from embedding_stemmer import get_embedding_stem
from attention import get_attention


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download text from URL and write to output file")
    p.add_argument(
        "--url",
        "-u",
        default=(
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        ),
        help=(
            "Source URL to download text from. If omitted, downloads a small sample file "
            "from rasbt/LLMs-from-scratch (used as a reasonable default for testing)."
        ),
    )
    p.add_argument(
        "--out",
        "-o",
        default="tokenizer/input/the_verdict.txt",
        help=(
            "Destination file path to write the text. By default this is "
            "`tokenizer/input/the_verdict.txt` relative to the project root."
        ),
    )
    p.add_argument("--timeout", type=int, default=30, help="Timeout in seconds for HTTP requests (default: 30)")
    p.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for dataset (default: 256)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Stride for sliding window in dataset (default: 128)",
    )
    p.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=4,
        help="Batch size for data loader (default: 4)",
    )
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Hidden size for the token/position embedding layer (default: 256)",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads for MultiHeadAttentionWrapper (default: 2)",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability for attention weights (default: 0.0)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Create a training data provider
    try:
        training_data_provider = get_provider(
            provider_type="simple_text",
            url=args.url,
            cache_path=args.out,
            timeout=args.timeout,
            use_cache=True,
        )
    except ValueError as e:
        print(f"Failed to create provider: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch the training text (will use cache if available, otherwise download)
    try:
        raw_text = training_data_provider.get_text()
        if args.out:
            print(f"Training data loaded ({len(raw_text)} characters)")
            print(f"Cache location: {args.out}")
        else:
            print(f"Downloaded {len(raw_text)} characters from {args.url}")
    except HTTPError as e:
        print(f"HTTP error {e.code} when fetching {args.url}: {e.reason}", file=sys.stderr)
        sys.exit(2)
    except URLError as e:
        print(f"URL error when fetching {args.url}: {e.reason}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error when fetching training data: {e}", file=sys.stderr)
        sys.exit(4)

    # Obtain tokenizer from factory. Import happens at module top; here we
    # only handle runtime failures from the factory itself.
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

    # Tokenization succeeded; we can use the tokenizer.
    print(f"Tokenizer '{tokenizer.name}' is ready to use.")

    # Create dataset and dataloader
    try:
        dataset = create_dataset(
            dataset_type="gpt_v1",
            txt=raw_text,
            tokenizer=tokenizer,
            max_length=args.max_length,
            stride=args.stride,
        )
        print(f"Dataset created with {len(dataset)} samples")
        print(f"  max_length: {args.max_length}")
        print(f"  stride: {args.stride}")
    except Exception as e:
        print(f"Failed to create dataset: {e}", file=sys.stderr)
        sys.exit(10)

    try:
        dataloader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        print(f"DataLoader created with batch_size={args.batch_size}")
    except Exception as e:
        print(f"Failed to create dataloader: {e}", file=sys.stderr)
        sys.exit(11)

    # Show examples from the first batch
    num_examples_to_print = 3  # Just for display purposes
    print(f"\nShowing first batch (up to {num_examples_to_print} examples):")
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print(f"\nInputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")

    # Build the embedding stem that combines token and positional embeddings
    vocab_size = getattr(tokenizer, "n_vocab", None)
    if vocab_size is None and hasattr(tokenizer, "encoder"):
        vocab_size = len(tokenizer.encoder)
    if vocab_size is None:
        raise AttributeError("Tokenizer does not expose a vocabulary size attribute")

    embedding_dim = args.embedding_dim  # size for the embedding layer
    embedding_stem = get_embedding_stem(
        stem_type="gpt",
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        context_length=args.max_length,
    )

    embeddings = embedding_stem(inputs)

    print(f"\nEmbedding output shape: {embeddings.shape}")
    print("  Example token embedding slice:", embeddings[0, 0, :5].tolist())

    # Pass embeddings through multi-head attention
    attention = get_attention(
        attention_type="multi_head_wrapper",
        d_in=embedding_dim,
        d_out=embedding_dim,
        context_length=args.max_length,
        dropout=args.dropout,
        num_heads=args.num_heads,
    )
    context_vecs = attention(embeddings)
    print(f"\nAttention output shape: {context_vecs.shape}")
    print(f"  num_heads: {args.num_heads}, head_dim: {embedding_dim // args.num_heads}")

    # Show a few examples from the batch
    num_examples = min(num_examples_to_print, inputs.shape[0])
    for i in range(num_examples):
        input_seq = inputs[i].tolist()
        target_seq = targets[i].tolist()
        example_embeddings = embeddings[i]
        example_context = context_vecs[i]
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
            "  Attention first token (dim 0-5):",
            example_context[0, :5].tolist(),
        )
        print(
            "  Attention mean/std:",
            f"{example_context.mean().item():.4f}/{example_context.std().item():.4f}",
        )


        


if __name__ == "__main__":
    main()
