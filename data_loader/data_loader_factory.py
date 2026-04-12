"""data_loader/data_loader_factory.py

Factory for creating data loaders and datasets for LLM training.

This module provides factory functions to create PyTorch datasets and data loaders.
Currently supports GPTDatasetV1, but the factory pattern allows for easy addition
of other dataset types (e.g., GPTDatasetV2 with different chunking strategies,
instruction-tuning datasets, etc.) in the future.
"""

from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from data_loader.gpt_dataset_v1 import GPTDatasetV1


def create_dataset(
    dataset_type: str = "gpt_v1",
    txt: str | None = None,
    tokenizer: Any | None = None,
    max_length: int = 256,
    stride: int = 128,
    **kwargs: Any,
) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    """Factory function to create a dataset.

    Parameters:
    - dataset_type: Type of dataset to create (default: "gpt_v1")
    - txt: Raw text for the dataset (required)
    - tokenizer: Tokenizer instance with encode() method (required)
    - max_length: Maximum sequence length (default: 256)
    - stride: Sliding window stride (default: 128)
    - **kwargs: Additional dataset-specific parameters

    Returns:
    - A PyTorch Dataset instance

    Raises:
    - ValueError: If dataset_type is unknown or required params are missing
    """
    if dataset_type == "gpt_v1":
        if txt is None:
            raise ValueError("txt parameter is required for gpt_v1 dataset")
        if tokenizer is None:
            raise ValueError("tokenizer parameter is required for gpt_v1 dataset")

        return GPTDatasetV1(
            txt=txt,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
        )
    else:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Supported types: gpt_v1"
        )


def create_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader:
    """Factory function to create a PyTorch DataLoader.

    Parameters:
    - dataset: PyTorch Dataset instance
    - batch_size: Number of samples per batch (default: 4)
    - shuffle: Whether to shuffle data (default: True)
    - drop_last: Whether to drop the last incomplete batch (default: True)
    - num_workers: Number of worker processes for data loading (default: 0)
    - **kwargs: Additional DataLoader parameters

    Returns:
    - A PyTorch DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        **kwargs,
    )


__all__ = ["create_dataset", "create_dataloader"]
