import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from tenacity import RetryError

from model.core.chunk.gemini_chunk_processor import GeminiChunkProcessor
from utils.result_type import ResultType
from utils.chunk_process_result import ChunkProcessResult

runner_path = "model.core.chunk.gemini_chunk_processor.GeminiResilientRunner"



@pytest.fixture
def mock_client():
    return Mock(name="GeminiClient")


@pytest.fixture
def mock_chunk_manager():
    return Mock(name="ChunkManager")


@pytest.fixture
def sample_df():
    return pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})


def test_init_validation_errors(mock_client, mock_chunk_manager):
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        GeminiChunkProcessor("", mock_client, mock_chunk_manager)

    with pytest.raises(ValueError, match="LLM client is not configured"):
        GeminiChunkProcessor("prompt", None, mock_chunk_manager)

    with pytest.raises(ValueError, match="Chunk manager is not initialized"):
        GeminiChunkProcessor("prompt", mock_client, None)


def test_no_more_chunks_returns_no_more_chunks(mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (None, None)
    processor = GeminiChunkProcessor("prompt", mock_client, mock_chunk_manager)
    result = processor.process_next_chunk()
    assert result.result_type == ResultType.NO_MORE_CHUNKS
    assert result.chunk is None
    assert result.error is None


def test_process_success_decrements_tokens_and_marks_processed(sample_df, mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (sample_df, "chunk123")
    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.return_value = ("response text", 5)
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("prompt", mock_client, mock_chunk_manager)
        starting_tokens = processor.remaining_tokens

        result = processor.process_next_chunk()

        assert result.result_type == ResultType.SUCCESS
        assert result.response == "response text"
        assert result.remaining_tokens == starting_tokens - 5
        mock_chunk_manager.mark_chunk_processed.assert_called_once_with("chunk123")
        mock_chunk_manager.save_state.assert_called_once()


def test_process_fatal_error_returns_chunk(sample_df, mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (sample_df, "chunkX")
    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = ValueError("fatal")
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.FATAL_ERROR
        assert isinstance(result.error, ValueError)
        assert result.chunk.equals(sample_df)


def test_process_retryable_error_returns_chunk(sample_df, mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (sample_df, "chunkX")
    retry_exc = Exception("retryable")
    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = RetryError(last_attempt=MagicMock(exception=lambda: retry_exc))
        runner_instance.fatal_errors = (ValueError,)  # Not matching retry_exc type

        processor = GeminiChunkProcessor("prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.RETRYABLE_ERROR
        assert str(result.error) == "retryable"
        assert result.chunk.equals(sample_df)


def test_process_unexpected_error_returns_chunk(sample_df, mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (sample_df, "chunkY")
    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = RuntimeError("something else")
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.UNEXPECTED_ERROR
        assert isinstance(result.error, RuntimeError)
        assert result.chunk.equals(sample_df)
