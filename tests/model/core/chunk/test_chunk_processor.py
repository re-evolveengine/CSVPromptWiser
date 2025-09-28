import pytest
import pandas as pd
import types
import streamlit as st
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from tenacity import RetryError

# Mock Streamlit secrets
st.secrets = types.SimpleNamespace()
st.secrets.is_local = True

from model.core.chunk.chunk_processor import ChunkProcessor
from model.core.llms.base_llm_client import BaseLLMClient
from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.io.model_prefs import ModelPreference
from utils.result_type import ResultType
from utils.exceptions import TokenBudgetExceededError

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


def test_token_budget_exceeded(mock_client, mock_chunk_manager, mock_model_preference, sample_dataframe):
    # Set up the test with a chunk that would exceed the token budget
    mock_chunk_manager.get_next_chunk.return_value = (sample_dataframe, "chunkZ")
    
    with patch(runner_path) as mock_runner_cls:
        # Mock the runner to return token usage that would exceed the budget
        runner_instance = mock_runner_cls.return_value
        runner_instance.run.return_value = ("result", 150)  # 150 tokens used
        runner_instance.fatal_errors = (ValueError,)  # Mock the fatal_errors tuple
        
        # Create processor with limited token budget
        processor = ChunkProcessor("prompt", mock_client, mock_chunk_manager, mock_model_preference)
        processor.remaining_tokens = 100  # Set a low token budget
        
        # Process the chunk - should return TOKENS_BUDGET_EXCEEDED result
        result = processor.process_next_chunk()
        
        # Verify the result
        assert result.result_type == ResultType.TOKENS_BUDGET_EXCEEDED
        assert result.chunk.equals(sample_dataframe)
        assert isinstance(result.error, TokenBudgetExceededError)
        assert result.error.used_tokens == 150
        assert result.error.remaining_tokens == 100
