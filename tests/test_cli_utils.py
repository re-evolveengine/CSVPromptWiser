import os
import builtins
from unittest.mock import patch, MagicMock, mock_open
import pytest
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions
from tenacity import RetryError
import pandas as pd

# Import functions to test
from cli.cli_utils import (
    load_api_key,
    get_model_selection,
    ask_int_input,
    run_gemini_chunk_processor,
    handle_model_selection
)
from model.io.model_prefs import ModelPreference
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner

# Fixtures
@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables."""
    monkeypatch.setenv("KEY", "test_api_key")
    return "test_api_key"

@pytest.fixture
def mock_model_preference():
    """Fixture to mock ModelPreference."""
    with patch('cli.cli_utils.ModelPreference') as mock:
        instance = mock.return_value
        instance.get_selected_model_name.return_value = None
        instance.get_model_list.return_value = []
        yield instance

@pytest.fixture
def mock_gemini_provider():
    """Fixture to mock GeminiModelProvider."""
    with patch('cli.cli_utils.GeminiModelProvider') as mock:
        instance = mock.return_value
        instance.get_usable_model_names.return_value = ["gemini-pro", "gemini-1.5-pro"]
        yield instance

@pytest.fixture
def mock_chunk_manager():
    """Fixture to mock ChunkManager."""
    manager = MagicMock()
    manager.process_chunks.side_effect = lambda fn: [fn(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}))]
    return manager

# Test load_api_key
def test_load_api_key_from_env(monkeypatch, mock_env):
    """Test loading API key from environment variable."""
    with patch('dotenv.load_dotenv'), \
         patch('os.getenv', return_value=mock_env), \
         patch('builtins.print') as mock_print:
        result = load_api_key()
        assert result == "test_api_key"
        mock_print.assert_called_with("✅ API key loaded from .env.")

def test_load_api_key_from_input(monkeypatch):
    """Test loading API key from user input when not in env."""
    with patch('dotenv.load_dotenv'), \
         patch('os.getenv', return_value=None), \
         patch('builtins.input', return_value="user_input_key"), \
         patch('dotenv.set_key'), \
         patch('dotenv.find_dotenv', return_value=".env"), \
         patch('builtins.print') as mock_print:
        result = load_api_key()
        assert result == "user_input_key"
        mock_print.assert_called_with("✅ API key saved to .env.")

# Test get_model_selection
def test_get_model_selection_valid_input():
    """Test model selection with valid input."""
    models = ["model1", "model2", "model3"]
    with patch('builtins.input', return_value="2"):
        result = get_model_selection(models)
        assert result == "model2"

def test_get_model_selection_invalid_input_then_valid(monkeypatch):
    """Test model selection with invalid then valid input."""
    models = ["model1", "model2", "model3"]
    input_values = ["5", "0", "abc", "1"]
    with patch('builtins.input', side_effect=input_values), \
         patch('builtins.print') as mock_print:
        result = get_model_selection(models)
        assert result == "model1"
        assert mock_print.call_count == 3  # 3 invalid inputs

# Test ask_int_input
def test_ask_int_input_valid():
    """Test integer input with valid value."""
    with patch('builtins.input', return_value="42"):
        result = ask_int_input("Enter a number: ")
        assert result == 42

def test_ask_int_input_invalid_then_valid():
    """Test integer input with invalid then valid value."""
    with patch('builtins.input', side_effect=["not a number", "42"]), \
         patch('builtins.print') as mock_print:
        result = ask_int_input("Enter a number: ")
        assert result == 42
        mock_print.assert_called_with("Please enter a valid integer.")

# Test run_gemini_chunk_processor
def test_run_gemini_chunk_processor_success(mock_chunk_manager):
    """Test successful chunk processing."""
    with patch('cli.cli_utils.GeminiClient') as mock_client, \
         patch('cli.cli_utils.GeminiResilientRunner') as mock_runner:
        # Setup mocks
        mock_runner.return_value.fatal_errors = (ValueError,)
        mock_runner.return_value.run.return_value = "test response"

        # Call function
        results, success = run_gemini_chunk_processor(
            prompt="test prompt",
            model_name="test-model",
            api_key="test-key",
            chunk_manager=mock_chunk_manager
        )

        # Assertions
        assert success is True
        assert len(results) == 1
        assert results[0]["response"] == "test response"
        mock_chunk_manager.process_chunks.assert_called_once()

def test_run_gemini_chunk_processor_user_error(mock_chunk_manager):
    """Test chunk processing with user error."""
    with patch('cli.cli_utils.GeminiClient'), \
         patch('cli.cli_utils.GeminiResilientRunner') as mock_runner, \
         patch('builtins.print') as mock_print:
        # Setup mocks to raise user error
        mock_runner.return_value.fatal_errors = (ValueError,)
        mock_runner.return_value.run.side_effect = ValueError("Invalid API key")

        # Call function
        results, success = run_gemini_chunk_processor(
            prompt="test prompt",
            model_name="test-model",
            api_key="test-key",
            chunk_manager=mock_chunk_manager
        )

        # Assertions
        assert success is False
        assert len(results) == 0
        mock_print.assert_any_call("[User Error] Skipping chunk due to user error: Invalid API key")

# Test handle_model_selection
def test_handle_model_selection_with_saved_model(mock_model_preference, mock_gemini_provider):
    """Test model selection with a previously saved model."""
    # Setup
    mock_model_preference.get_selected_model_name.return_value = "saved-model"
    
    with patch('builtins.input', return_value='y'), \
         patch('builtins.print') as mock_print:
        result = handle_model_selection("test-api-key")
        
        # Assertions
        assert result == "saved-model"
        # Add \n to match the actual print output
        mock_print.assert_any_call("\n✅ Using saved model: saved-model")

def test_handle_model_selection_with_saved_models(mock_model_preference, mock_gemini_provider):
    """Test model selection with a list of saved models."""
    # Setup
    mock_model_preference.get_selected_model_name.return_value = None
    mock_model_preference.get_model_list.return_value = ["model1", "model2"]
    
    with patch('builtins.input', side_effect=['y', '1']), \
         patch('builtins.print') as mock_print:
        result = handle_model_selection("test-api-key")
        
        # Assertions
        assert result == "model1"
        # Add \n to match the actual print output
        mock_print.assert_any_call("\n✅ Using saved models.")

def test_handle_model_selection_fetch_new_models(mock_model_preference, monkeypatch):
    """Test model selection by fetching new models."""
    # Setup
    mock_model_preference.get_selected_model_name.return_value = None
    mock_model_preference.get_model_list.return_value = []
    
    # Create a mock for the provider instance
    mock_provider = MagicMock()
    mock_provider.get_usable_model_names.return_value = ["gemini-pro", "gemini-1.5-pro"]
    
    # Patch the GeminiModelProvider to return our mock
    with patch('cli.cli_utils.GeminiModelProvider', return_value=mock_provider) as mock_provider_cls, \
         patch('builtins.input', return_value='1'), \
         patch('builtins.print'):
        
        # Call the function
        result = handle_model_selection("test-api-key")
        
        # Assertions
        assert result in ["gemini-pro", "gemini-1.5-pro"]
        mock_provider_cls.assert_called_once_with("test-api-key")
        mock_provider.get_usable_model_names.assert_called_once()
        mock_model_preference.save_selected_model_name.assert_called_once()
        mock_model_preference.save_model_list.assert_called_once_with(["gemini-pro", "gemini-1.5-pro"])
