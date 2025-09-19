import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from model.io.save_processed_chunks_to_db import save_processed_chunk_to_db
from utils.result_type import ResultType


@pytest.fixture
def mock_saver():
    return MagicMock()


@pytest.fixture
def sample_chunk_df():
    return pd.DataFrame({
        "source_id": ["src1", "src2"],
        "val": ["a", "b"]
    })


def make_result(result_type, response="", chunk=None, remaining_tokens=42):
    return SimpleNamespace(
        result_type=result_type,
        response=response,
        chunk=chunk,
        remaining_tokens=remaining_tokens
    )


def test_skips_if_not_success(mock_saver, sample_chunk_df):
    result = make_result(ResultType.UNEXPECTED_ERROR, "ignored", sample_chunk_df)
    save_processed_chunk_to_db(
        result, "chunk1", "prompt", "v1", mock_saver
    )
    mock_saver.save.assert_not_called()


def test_raises_if_chunk_missing(mock_saver):
    result = make_result(ResultType.SUCCESS, "resp", chunk=None)
    with pytest.raises(ValueError, match="Missing chunk"):
        save_processed_chunk_to_db(result, "cid", "pr", "mod", mock_saver)


def test_raises_if_responses_and_rows_mismatch(mock_saver, sample_chunk_df):
    # Only 1 line but 2 rows
    result = make_result(
        ResultType.SUCCESS,
        "1: foo",
        sample_chunk_df
    )
    with pytest.raises(ValueError, match="Mismatch"):
        save_processed_chunk_to_db(result, "cid", "pr", "mod", mock_saver)


def test_success_saves_expected_rows(mock_saver, sample_chunk_df):
    # Two lines matching two rows
    resp = "1: foo\n2: bar"
    result = make_result(ResultType.SUCCESS, resp, sample_chunk_df, remaining_tokens=99)
    save_processed_chunk_to_db(
        result, "chunkX", "the prompt", "model-v", mock_saver
    )
    mock_saver.save.assert_called_once()
    saved_rows = mock_saver.save.call_args[0][0]
    assert len(saved_rows) == 2
    for row in saved_rows:
        assert row["prompt"] == "the prompt"
        assert row["chunk_id"] == "chunkX"
        assert row["model_version"] == "model-v"
        assert row["used_tokens"] == 99
