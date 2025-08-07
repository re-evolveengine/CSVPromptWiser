import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from model.core.llms.gemini_client import GeminiClient


@pytest.fixture
def mock_gemini_client():
    """Fixture to create a GeminiClient with a mock API key."""
    return GeminiClient(
        model="gemini-pro",
        api_key="test-api-key",
        generation_config={"temperature": 0.2}
    )


def test_gemini_client_initialization(mock_gemini_client):
    """Test that GeminiClient initializes with correct attributes."""
    assert mock_gemini_client.model == "gemini-pro"
    assert mock_gemini_client.api_key == "test-api-key"
    assert mock_gemini_client.generation_config["temperature"] == 0.2


@patch('google.generativeai.configure')
@patch('google.generativeai.GenerativeModel')
def test_init_llm_success(mock_genai_model, mock_configure, mock_gemini_client):
    """Test successful initialization of the Gemini model."""
    # Setup mock
    mock_model = MagicMock()
    mock_genai_model.return_value = mock_model

    # Call the method
    result = mock_gemini_client._init_llm()

    # Assertions
    mock_configure.assert_called_once_with(api_key="test-api-key")
    mock_genai_model.assert_called_once_with(
        model_name="gemini-pro",
        generation_config=mock_gemini_client.generation_config
    )
    assert result == mock_model


@patch('google.generativeai.configure')
def test_init_llm_failure(mock_configure, mock_gemini_client):
    """Test initialization failure of the Gemini model."""
    # Setup mock to raise an exception
    mock_configure.side_effect = Exception("API Error")

    # Test that the exception is properly raised
    with pytest.raises(RuntimeError, match="Failed to initialize Gemini client: API Error"):
        mock_gemini_client._init_llm()


def test_format_input(mock_gemini_client):
    """Test the _format_input method with a sample DataFrame."""
    # Setup test data
    prompt = "Test prompt"
    df = pd.DataFrame({
        'col1': ['value1', 'value2'],
        'col2': [1, 2]
    })

    # Call the method
    result = mock_gemini_client._format_input(prompt, df)

    # Assert the output format
    expected_output = (
        "Test prompt\n"
        "\n"
        "Row 1:\n"
        "- col1: value1\n"
        "- col2: 1\n"
        "\n"
        "Row 2:\n"
        "- col1: value2\n"
        "- col2: 2"
    )
    assert result == expected_output


@patch('model.core.llms.gemini_client.logger')
@patch.object(GeminiClient, '_format_input')
def test_call_success(mock_format_input, mock_logger, mock_gemini_client):
    """Test successful execution of the call method."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_gemini_client.llm = mock_llm

    # Mock the count_tokens response
    mock_count_response = MagicMock()
    mock_count_response.total_tokens = 10
    mock_llm.count_tokens.return_value = mock_count_response

    # Mock the generate_content response
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_llm.generate_content.return_value = mock_response

    # Mock format_input
    mock_format_input.return_value = "Formatted input"

    # Call the method
    prompt = "Test prompt"
    df = pd.DataFrame({'col1': ['value1']})
    result_text, result_tokens = mock_gemini_client.call(prompt, df)

    # Assertions
    mock_format_input.assert_called_once_with(prompt, df)
    assert result_text == "Test response"
    assert result_tokens == 20  # input_tokens + output_tokens (10 + 10)
    mock_logger.info.assert_called_once()


@patch('model.core.llms.gemini_client.logger')
def test_call_failure(mock_logger, mock_gemini_client):
    """Test error handling in the call method."""
    # Setup mocks
    mock_llm = MagicMock()
    mock_gemini_client.llm = mock_llm

    # Make count_tokens raise an exception
    mock_llm.count_tokens.side_effect = Exception("Test error")

    # Test that the exception is properly handled
    with pytest.raises(RuntimeError, match="Gemini call failed: Test error"):
        mock_gemini_client.call("test", pd.DataFrame())

    # Verify error was logged
    mock_logger.error.assert_called_once_with("Gemini call failed: Test error")