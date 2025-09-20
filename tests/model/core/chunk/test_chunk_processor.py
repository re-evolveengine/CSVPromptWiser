import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from tenacity import RetryError

from model.core.chunk.chunk_processor import ChunkProcessor
from model.core.llms.base_llm_client import BaseLLMClient
from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.io.model_prefs import ModelPreference
from utils.result_type import ResultType

runner_path = "model.core.chunk.chunk_processor.GeminiResilientRunner"


@pytest.fixture
def mock_client():
    return Mock(spec=GeminiClient)


@pytest.fixture
def mock_chunk_manager():
    return Mock(spec=ChunkManager)


@pytest.fixture
def mock_model_preference():
    mock_pref = Mock(spec=ModelPreference)
    mock_pref.remaining_total_tokens = 10000  # Default value for tests
    return mock_pref


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})


def test_initialization_validation(mock_client, mock_chunk_manager, mock_model_preference):
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        ChunkProcessor("", mock_client, mock_chunk_manager, mock_model_preference)

    with pytest.raises(ValueError, match="LLM client is not configured"):
        ChunkProcessor("test prompt", None, mock_chunk_manager, mock_model_preference)

    with pytest.raises(ValueError, match="Chunk manager is not initialized"):
        ChunkProcessor("test prompt", mock_client, None, mock_model_preference)


def test_no_more_chunks(mock_client, mock_chunk_manager, mock_model_preference):
    mock_chunk_manager.get_next_chunk.return_value = (None, None)
    processor = ChunkProcessor("test prompt", mock_client, mock_chunk_manager, mock_model_preference)
    result = processor.process_next_chunk()
    assert result.result_type == ResultType.NO_MORE_CHUNKS
    assert result.chunk is None
    assert result.error is None


def test_fatal_error(mock_client, mock_chunk_manager, mock_model_preference, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunkX")

    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = ValueError("fatal")
        runner_instance.fatal_errors = (ValueError,)

        processor = ChunkProcessor("prompt", mock_client, mock_chunk_manager, mock_model_preference)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.FATAL_ERROR
        assert isinstance(result.error, ValueError)
        assert result.chunk.equals(sample_dataframe)


def test_retryable_error(mock_client, mock_chunk_manager, mock_model_preference, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunkX")
    retry_exc = Exception("retryable error")

    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = RetryError(last_attempt=MagicMock(exception=lambda: retry_exc))
        runner_instance.fatal_errors = (ValueError,)

        processor = ChunkProcessor("prompt", mock_client, mock_chunk_manager, mock_model_preference)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.RETRYABLE_ERROR
        assert result.chunk.equals(sample_dataframe)
        assert str(result.error) == "retryable error"


def test_unexpected_error(mock_client, mock_chunk_manager, mock_model_preference, sample_dataframe):
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunkY")

    with patch(runner_path) as mock_runner_cls:
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.side_effect = RuntimeError("weird")
        runner_instance.fatal_errors = (ValueError,)

        processor = ChunkProcessor("prompt", mock_client, mock_chunk_manager, mock_model_preference)
        result = processor.process_next_chunk()

        assert result.result_type == ResultType.UNEXPECTED_ERROR
        assert isinstance(result.error, RuntimeError)
        assert result.chunk.equals(sample_dataframe)
