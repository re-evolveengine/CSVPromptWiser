# tests/model/core/llms/test_gemini_model_provider.py
import pytest
import types
import streamlit as st
from unittest.mock import patch, MagicMock

# Mock Streamlit secrets
st.secrets = types.SimpleNamespace()
st.secrets.is_local = True

import model.core.llms.gemini_model_provider as gmp_module
from model.core.llms.gemini_model_provider import GeminiModelProvider


@pytest.fixture
def provider(monkeypatch):
    # Patch genai.configure to avoid contacting the real API
    monkeypatch.setattr(gmp_module.genai, "configure", MagicMock())
    return GeminiModelProvider(api_key="fake-key")


def test_init_configures_api_key(monkeypatch):
    mock_configure = MagicMock()
    monkeypatch.setattr(gmp_module.genai, "configure", mock_configure)

    provider = GeminiModelProvider(api_key="secret-123")

    mock_configure.assert_called_once_with(api_key="secret-123")
    assert provider.api_key == "secret-123"


def test_test_model_success(monkeypatch, provider):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text.strip.return_value = "Hello world"
    # Simulate response.text being a normal string
    mock_model.generate_content.return_value.text = "  hello  "

    with patch.object(gmp_module.genai, "GenerativeModel", return_value=mock_model) as mock_ctor:
        assert provider._test_model("model-name") is True
        mock_ctor.assert_called_once_with("model-name")
        mock_model.generate_content.assert_called_once_with("Hello")


def test_test_model_failure(monkeypatch, provider):
    with patch.object(gmp_module.genai, "GenerativeModel", side_effect=Exception("bad")):
        assert provider._test_model("model-name") is False


def test_get_usable_model_names_filters_and_returns(monkeypatch, provider):
    # Prepare fake models
    model1 = MagicMock()
    model1.name = "projects/123/model-alpha"
    model1.supported_generation_methods = ["generateContent"]

    model2 = MagicMock()
    model2.name = "model-beta"
    model2.supported_generation_methods = ["generateContent"]

    model3 = MagicMock()
    model3.name = "model-no-content"
    model3.supported_generation_methods = ["somethingElse"]

    monkeypatch.setattr(gmp_module.genai, "list_models", MagicMock(return_value=[model1, model2, model3]))

    # Patch _test_model to return True for model1 and False for model2
    monkeypatch.setattr(provider, "_test_model", lambda name: name != "model-beta")

    result = provider.get_usable_model_names()

    # model1 passes, name split on '/'
    # model2 fails _test_model
    # model3 skipped because no generateContent support
    assert result == ["123"]

    gmp_module.genai.list_models.assert_called_once()


def test_get_usable_model_names_handles_short_names(monkeypatch, provider):
    model_short = MagicMock()
    model_short.name = "shortname"
    model_short.supported_generation_methods = ["generateContent"]

    monkeypatch.setattr(gmp_module.genai, "list_models", MagicMock(return_value=[model_short]))
    monkeypatch.setattr(provider, "_test_model", lambda name: True)

    result = provider.get_usable_model_names()

    assert result == ["shortname"]
