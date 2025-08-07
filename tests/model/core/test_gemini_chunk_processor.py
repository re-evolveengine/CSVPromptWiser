import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import pandas as pd
from tenacity import RetryError

from streamlit_dir.gemini_chunk_processor import GeminiChunkProcessor
from model.core.llms.gemini_client import GeminiClient
from model.core.chunk.chunk_manager import ChunkManager
from model.utils.chunk_process_result import ChunkProcessResult
from model.utils.result_type import ResultType


@pytest.fixture
def mock_client():
    return Mock(spec=GeminiClient)


@pytest.fixture
def mock_chunk_manager():
    manager = Mock(spec=ChunkManager)
    return manager


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})


def test_initialization_validation(mock_client, mock_chunk_manager):
    # Test empty prompt
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        GeminiChunkProcessor("", mock_client, mock_chunk_manager)

    # Test missing client
    with pytest.raises(ValueError, match="LLM client is not configured"):
        GeminiChunkProcessor("test prompt", None, mock_chunk_manager)

    # Test missing chunk manager
    with pytest.raises(ValueError, match="Chunk manager is not initialized"):
        GeminiChunkProcessor("test prompt", mock_client, None)


def test_process_one_chunk_success(mock_client, mock_chunk_manager, sample_dataframe):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = sample_dataframe
    expected_response = "Test response"
    expected_tokens = 100

    with patch('streamlit_dir.gemini_chunk_processor.GeminiResilientRunner') as mock_runner:
        # Configure the mock runner
        mock_runner_instance = mock_runner.return_value
        mock_runner_instance.run.return_value = (expected_response, expected_tokens)
        mock_runner_instance.fatal_errors = (ValueError,)

        # Test
        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        
        # Store the initial tokens after processor initialization
        initial_tokens = processor.remaining_tokens
        
        # Mock the prefs to have the same initial token count
        with patch.object(processor.prefs, 'get_remaining_total_tokens', return_value=initial_tokens):
            result = processor.process_next_chunk()

        # Assertions
        assert result.result_type == ResultType.SUCCESS
        assert result.response == expected_response
        assert result.chunk.equals(sample_dataframe)
        
        # Verify tokens were subtracted correctly
        assert result.remaining_tokens == initial_tokens - expected_tokens
        mock_chunk_manager.mark_chunk_processed.assert_called_once()


def test_process_one_chunk_no_more_chunks(mock_client, mock_chunk_manager):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = None

    # Test
    processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
    result = processor.process_next_chunk()

    # Assertions
    assert result.result_type == ResultType.NO_MORE_CHUNKS
    mock_chunk_manager.mark_chunk_processed.assert_not_called()


def test_process_one_chunk_fatal_error(mock_client, mock_chunk_manager, sample_dataframe):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = sample_dataframe
    test_exception = ValueError("Fatal error")

    with patch('streamlit_dir.gemini_chunk_processor.GeminiResilientRunner') as mock_runner:
        # Configure the mock runner
        mock_runner_instance = mock_runner.return_value
        mock_runner_instance.run.side_effect = test_exception
        mock_runner_instance.fatal_errors = (ValueError,)  # This must match the exception type

        # Test
        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        # Assertions
        assert result.result_type == ResultType.FATAL_ERROR
        assert str(result.error) == "Fatal error"
        assert result.chunk.equals(sample_dataframe)


def test_process_one_chunk_retry_error(mock_client, mock_chunk_manager, sample_dataframe):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = sample_dataframe
    test_exception = Exception("Retryable error")

    with patch('streamlit_dir.gemini_chunk_processor.GeminiResilientRunner') as mock_runner:
        # Configure the mock runner
        mock_runner_instance = mock_runner.return_value
        mock_runner_instance.run.side_effect = RetryError(last_attempt=MagicMock(exception=lambda: test_exception))
        mock_runner_instance.fatal_errors = (ValueError,)  # This must be different from the raised exception

        # Test
        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        # Assertions
        assert result.result_type == ResultType.RETRYABLE_ERROR
        assert str(result.error) == "Retryable error"
        assert result.chunk.equals(sample_dataframe)


def test_process_one_chunk_unexpected_error(mock_client, mock_chunk_manager, sample_dataframe):
    # Setup
    mock_chunk_manager.get_next_chunk.return_value = sample_dataframe
    test_exception = KeyError("Unexpected error")

    with patch('streamlit_dir.gemini_chunk_processor.GeminiResilientRunner') as mock_runner:
        # Configure the mock runner
        mock_runner_instance = mock_runner.return_value
        mock_runner_instance.run.side_effect = test_exception
        mock_runner_instance.fatal_errors = (ValueError,)  # This must be different from the raised exception

        # Test
        processor = GeminiChunkProcessor("test prompt", mock_client, mock_chunk_manager)
        result = processor.process_next_chunk()

        # Assertions
        assert result.result_type == ResultType.UNEXPECTED_ERROR
        assert str(result.error) == "'Unexpected error'"
        assert result.chunk.equals(sample_dataframe)