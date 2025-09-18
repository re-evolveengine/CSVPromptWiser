import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pandas as pd
from tenacity import RetryError

from model.core.chunk.gemini_chunk_processor import GeminiChunkProcessor
from model.core.llms.gemini_client import GeminiClient
from model.core.chunk.chunk_manager import ChunkManager
from utils.result_type import ResultType


@pytest.fixture
def mock_client():
    return Mock(spec=GeminiClient)


@pytest.fixture
def mock_chunk_manager():
    return Mock(spec=ChunkManager)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


def test_initialization_validation(mock_client, mock_chunk_manager):
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        GeminiChunkProcessor("", mock_client, mock_chunk_manager)

    with pytest.raises(ValueError, match="LLM client is not configured"):
        GeminiChunkProcessor("test prompt", None, mock_chunk_manager)

    with pytest.raises(ValueError, match="Chunk manager is not initialized"):
        GeminiChunkProcessor("test prompt", mock_client, None)


from unittest.mock import patch, PropertyMock

def test_process_one_chunk_success(mock_client, mock_chunk_manager, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunk_1")
    expected_response = "Test response"
    expected_tokens = 100

    # Patch it where the processor actually uses it
    with patch("model.core.chunk.gemini_chunk_processor.GeminiResilientRunner") as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.return_value = (expected_response, expected_tokens)
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        initial_tokens = processor.remaining_tokens

        # Patch remaining_total_tokens property from ModelPreference
        with patch.object(
            type(processor.prefs), "remaining_total_tokens", new_callable=PropertyMock
        ) as mock_tokens:
            mock_tokens.return_value = initial_tokens
            result = processor.process_next_chunk()

    assert result.result_type == ResultType.SUCCESS
    assert result.response == expected_response
    assert result.chunk.equals(sample_dataframe)
    assert result.remaining_tokens == initial_tokens - expected_tokens
    mock_chunk_manager.mark_chunk_processed.assert_called_once_with("chunk_1")



def test_process_one_chunk_no_more_chunks(mock_client, mock_chunk_manager):
    mock_chunk_manager.get_next_chunk.return_value = (None, None)
    processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
    result = processor.process_next_chunk()

    assert result.result_type == ResultType.NO_MORE_CHUNKS
    assert result.chunk is None
    assert result.error is None


def test_process_one_chunk_fatal_error(mock_client, mock_chunk_manager, sample_dataframe):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunk_1")
    test_exception = ValueError("Fatal error")

    # Patch where the processor looks up GeminiResilientRunner
    with patch("model.core.chunk.gemini_chunk_processor.GeminiResilientRunner") as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = test_exception
        runner_instance.fatal_errors = (ValueError,)  # Must match exception type

        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

    # Assertions
    assert result.result_type == ResultType.FATAL_ERROR
    assert str(result.error) == "Fatal error"
    assert result.chunk.equals(sample_dataframe)
    assert result.chunk_id is None  # matches current implementation for fatal errors



def test_process_one_chunk_retry_error(mock_client, mock_chunk_manager, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunk_1")
    retry_exc = RetryError(last_attempt=MagicMock(exception=lambda: Exception("Retryable error")))

    # Patch the runner in the namespace where GeminiChunkProcessor uses it
    with patch("model.core.chunk.gemini_chunk_processor.GeminiResilientRunner") as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = retry_exc
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

    assert result.result_type == ResultType.RETRYABLE_ERROR
    assert str(result.error) == "Retryable error"
    assert result.chunk.equals(sample_dataframe)
    assert result.chunk_id is None  # matches current implementation



def test_process_one_chunk_unexpected_error(mock_client, mock_chunk_manager, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunk_42")

    with patch("model.core.chunk.gemini_chunk_processor.GeminiResilientRunner") as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = Exception("Unexpected error")
        runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

    assert result.result_type == ResultType.UNEXPECTED_ERROR
    assert isinstance(result.error, Exception)
    assert str(result.error) == "Unexpected error"
    assert result.chunk.equals(sample_dataframe)
    assert result.chunk_id is None  # processor does not return chunk_id in unexpected error



def test_fatal_error_result_contains_none_chunk_id(mock_client, mock_chunk_manager, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "test_chunk_id")

    with patch('model.core.chunk.gemini_chunk_processor.GeminiResilientRunner') as mock_runner:
        mock_runner_instance = mock_runner.return_value
        mock_runner_instance.run.side_effect = ValueError("Fatal error")
        mock_runner_instance.fatal_errors = (ValueError,)

        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

    assert result.result_type == ResultType.FATAL_ERROR
    assert isinstance(result.error, ValueError)
    assert "Fatal error" in str(result.error)
    assert result.chunk.equals(sample_dataframe)
    assert result.chunk_id is None

