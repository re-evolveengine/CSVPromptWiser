# tests/model/core/llms/test_gemini_client.py
import builtins
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import model.core.llms.gemini_client as gemini_client_module
from model.core.llms.gemini_client import GeminiClient


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "col1": [1, 2],
        "col2": ["a", "b"]
    })


def test_init_llm_success(monkeypatch):
    # Arrange
    mock_model = MagicMock()
    monkeypatch.setattr(gemini_client_module.genai, "configure", MagicMock())
    monkeypatch.setattr(gemini_client_module.genai, "GenerativeModel", MagicMock(return_value=mock_model))

    # Act
    client = GeminiClient(model="gemini-model", api_key="fake-key")

    # Assert
    gemini_client_module.genai.configure.assert_called_once_with(api_key="fake-key")
    gemini_client_module.genai.GenerativeModel.assert_called_once_with(
        model_name="gemini-model",
        generation_config=client.generation_config
    )
    assert client.llm is mock_model


def test_init_llm_failure(monkeypatch):
    monkeypatch.setattr(gemini_client_module.genai, "configure", MagicMock(side_effect=Exception("boom")))
    with pytest.raises(RuntimeError) as exc:
        GeminiClient(model="gemini-model", api_key="fake-key")
    assert "Failed to initialize Gemini client" in str(exc.value)


def test_call_success(monkeypatch, sample_df):
    # Arrange - patch init to skip real API calls
    mock_llm = MagicMock()
    client = GeminiClient.__new__(GeminiClient)  # bypass __init__
    client.model = "gemini-model"
    client.api_key = "fake-key"
    client.generation_config = {}
    client.llm = mock_llm

    # Prepare mock token counting
    mock_llm.count_tokens.side_effect = [
        MagicMock(total_tokens=10),  # input tokens
        MagicMock(total_tokens=5)    # output tokens
    ]
    mock_llm.generate_content.return_value = MagicMock(text="the response")

    # Act
    result_text, total_tokens = client.call("prompt here", sample_df)

    # Assert
    assert result_text == "the response"
    assert total_tokens == 15
    assert mock_llm.count_tokens.call_count == 2
    mock_llm.generate_content.assert_called_once()


def test_call_success_with_empty_text(monkeypatch, sample_df):
    # Arrange
    mock_llm = MagicMock()
    client = GeminiClient.__new__(GeminiClient)
    client.model = "gemini-model"
    client.api_key = "fake-key"
    client.generation_config = {}
    client.llm = mock_llm

    mock_llm.count_tokens.side_effect = [
        MagicMock(total_tokens=4),
        MagicMock(total_tokens=6)
    ]
    mock_llm.generate_content.return_value = MagicMock(text=None)

    text, tokens = client.call("prompt here", sample_df)

    assert text == ""
    assert tokens == 10



def test_call_failure(sample_df):
    # Arrange
    mock_llm = MagicMock()
    mock_llm.count_tokens.return_value.total_tokens = 4
    mock_llm.generate_content.side_effect = Exception("API down")

    client = GeminiClient.__new__(GeminiClient)
    client.model = "gemini-model"
    client.api_key = "fake-key"
    client.generation_config = {}
    client.llm = mock_llm

    # Act + Assert
    # The exception should propagate directly, not be wrapped
    with pytest.raises(Exception) as exc_info:
        client.call("prompt here", sample_df)
    assert "API down" in str(exc_info.value)
