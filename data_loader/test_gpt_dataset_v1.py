"""tests/test_gpt_dataset_v1.py"""

import torch
import pytest
from data_loader.gpt_dataset_v1 import GPTDatasetV1


class TestLength:
    def test_length_matches_sliding_window_count(self, mock_tokenizer, sample_text):
        # 26 tokens, max_length=4, stride=2 → range(0, 22, 2) → 11 windows
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=2)
        expected = len(range(0, len(sample_text) - 4, 2))
        assert len(ds) == expected

    def test_stride_equal_to_max_length_no_overlap(self, mock_tokenizer, sample_text):
        # Non-overlapping windows
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=4)
        expected = len(range(0, len(sample_text) - 4, 4))
        assert len(ds) == expected

    def test_stride_one_maximum_overlap(self, mock_tokenizer, sample_text):
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=1)
        expected = len(range(0, len(sample_text) - 4, 1))
        assert len(ds) == expected


class TestGetItem:
    def test_returns_tuple_of_tensors(self, mock_tokenizer, sample_text):
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=2)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)

    def test_input_and_target_have_correct_shape(self, mock_tokenizer, sample_text):
        max_length = 4
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=max_length, stride=2)
        inp, tgt = ds[0]
        assert inp.shape == (max_length,)
        assert tgt.shape == (max_length,)

    def test_target_is_input_shifted_by_one(self, mock_tokenizer, sample_text):
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=2)
        inp, tgt = ds[0]
        # With MockTokenizer, token[i] == i, so input[0:4]=[0,1,2,3], target[0:4]=[1,2,3,4]
        assert inp.tolist() == [0, 1, 2, 3]
        assert tgt.tolist() == [1, 2, 3, 4]

    def test_stride_advances_window_correctly(self, mock_tokenizer, sample_text):
        stride = 3
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=stride)
        inp0, _ = ds[0]
        inp1, _ = ds[1]
        # Window 1 should start stride positions after window 0
        assert inp1[0].item() - inp0[0].item() == stride

    def test_last_index_is_accessible(self, mock_tokenizer, sample_text):
        ds = GPTDatasetV1(txt=sample_text, tokenizer=mock_tokenizer, max_length=4, stride=2)
        last_idx = len(ds) - 1
        inp, tgt = ds[last_idx]
        assert inp.shape == (4,)
        assert tgt.shape == (4,)


class TestValidation:
    def test_text_shorter_than_max_length_raises_value_error(self, mock_tokenizer):
        with pytest.raises(ValueError, match="max_length"):
            GPTDatasetV1(txt="abc", tokenizer=mock_tokenizer, max_length=10, stride=1)

    def test_text_equal_to_max_length_raises_value_error(self, mock_tokenizer):
        # 4 chars → 4 tokens, max_length=4: needs > max_length tokens
        with pytest.raises(ValueError, match="max_length"):
            GPTDatasetV1(txt="abcd", tokenizer=mock_tokenizer, max_length=4, stride=1)
