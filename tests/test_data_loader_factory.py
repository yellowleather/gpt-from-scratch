"""tests/test_data_loader_factory.py"""

import pytest
import torch
from torch.utils.data import DataLoader

from data_loader.data_loader_factory import create_dataset, create_dataloader
from data_loader.gpt_dataset_v1 import GPTDatasetV1


class TestCreateDataset:
    def test_gpt_v1_returns_dataset_instance(self, mock_tokenizer, sample_text):
        ds = create_dataset(dataset_type="gpt_v1", txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=2)
        assert isinstance(ds, GPTDatasetV1)

    def test_passes_max_length_and_stride(self, mock_tokenizer, sample_text):
        ds = create_dataset(
            dataset_type="gpt_v1",
            txt=sample_text,
            tokenizer=mock_tokenizer,
            max_length=4,
            stride=2,
        )
        inp, _ = ds[0]
        assert inp.shape == (4,)

    def test_missing_txt_raises_value_error(self, mock_tokenizer):
        with pytest.raises(ValueError, match="txt parameter is required"):
            create_dataset(dataset_type="gpt_v1", tokenizer=mock_tokenizer)

    def test_missing_tokenizer_raises_value_error(self, sample_text):
        with pytest.raises(ValueError, match="tokenizer parameter is required"):
            create_dataset(dataset_type="gpt_v1", txt=sample_text)

    def test_unknown_type_raises_value_error(self, mock_tokenizer, sample_text):
        with pytest.raises(ValueError, match="Unknown dataset_type"):
            create_dataset(dataset_type="nonexistent", txt=sample_text, tokenizer=mock_tokenizer)


class TestCreateDataloader:
    def _make_dataset(self, mock_tokenizer, sample_text):
        return create_dataset(
            dataset_type="gpt_v1",
            txt=sample_text,
            tokenizer=mock_tokenizer,
            max_length=4,
            stride=2,
        )

    def test_returns_dataloader(self, mock_tokenizer, sample_text):
        ds = self._make_dataset(mock_tokenizer, sample_text)
        dl = create_dataloader(ds, batch_size=2, shuffle=False, drop_last=False)
        assert isinstance(dl, DataLoader)

    def test_batch_size_is_respected(self, mock_tokenizer, sample_text):
        ds = self._make_dataset(mock_tokenizer, sample_text)
        dl = create_dataloader(ds, batch_size=3, shuffle=False, drop_last=False)
        inputs, targets = next(iter(dl))
        assert inputs.shape[0] == 3

    def test_drop_last_removes_incomplete_batch(self, mock_tokenizer, sample_text):
        ds = self._make_dataset(mock_tokenizer, sample_text)
        # 11 samples, batch_size=4 → 2 full batches + 1 leftover; drop_last drops the leftover
        dl_drop = create_dataloader(ds, batch_size=4, shuffle=False, drop_last=True)
        dl_keep = create_dataloader(ds, batch_size=4, shuffle=False, drop_last=False)
        assert len(list(dl_drop)) < len(list(dl_keep))

    def test_yields_input_target_pairs(self, mock_tokenizer, sample_text):
        ds = self._make_dataset(mock_tokenizer, sample_text)
        dl = create_dataloader(ds, batch_size=2, shuffle=False, drop_last=False)
        inputs, targets = next(iter(dl))
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert inputs.shape == targets.shape
