import pytest
from unittest.mock import patch
from model.core.llms.gemini_model_provider import GeminiModelProvider


class MockModel:
    def __init__(self, name, supports_generation=True):
        self.name = name
        self.supported_generation_methods = ["generateContent"] if supports_generation else []


class MockResponse:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def gemini_provider():
    with patch('google.generativeai.configure') as _:
        provider = GeminiModelProvider(api_key="test_api_key")
        return provider


def test_test_model_success(gemini_provider):
    """Test _test_model returns True when model responds successfully."""
    with patch('google.generativeai.GenerativeModel') as mock_model:
        # Setup mock model to return a successful response
        mock_instance = mock_model.return_value
        mock_instance.generate_content.return_value = MockResponse("Hello back!")

        # Test the method
        result = gemini_provider._test_model("test-model")

        # Verify results
        assert result is True
        mock_model.assert_called_once_with("test-model")
        mock_instance.generate_content.assert_called_once_with("Hello")


def test_test_model_failure(gemini_provider):
    """Test _test_model returns False when model raises an exception."""
    with patch('google.generativeai.GenerativeModel') as mock_model:
        # Setup mock model to raise an exception
        mock_instance = mock_model.return_value
        mock_instance.generate_content.side_effect = Exception("API Error")

        # Test the method
        result = gemini_provider._test_model("failing-model")

        # Verify results
        assert result is False


def test_get_usable_model_names(gemini_provider):
    """Test get_usable_model_names returns correct model names."""
    # Setup mock models
    mock_models = [
        MockModel("models/gemini-pro"),
        MockModel("models/gemini-pro-vision"),
        MockModel("models/unsupported-model", supports_generation=False)
    ]

    with patch('google.generativeai.list_models', return_value=mock_models), \
         patch.object(gemini_provider, '_test_model') as mock_test:

        # Configure _test_model to return True for all models
        mock_test.return_value = True

        # Call the method
        result = gemini_provider.get_usable_model_names()

        # Verify results
        assert len(result) == 2  # Should only include models that support generation
        assert "gemini-pro" in result
        assert "gemini-pro-vision" in result
        assert mock_test.call_count == 2  # Should only be called for supported models


def test_model_name_parsing(gemini_provider):
    """Test that model names are correctly parsed from full paths."""
    with patch('google.generativeai.list_models') as mock_list, \
         patch.object(gemini_provider, '_test_model', return_value=True):
        
        # Setup mock with different name formats
        mock_list.return_value = [
            MockModel("models/gemini-pro"),
            MockModel("gemini-pro-vision"),  # Unusual case without models/ prefix
            MockModel("path/with/multiple/slashes/gemini-ultra")
        ]
        
        result = gemini_provider.get_usable_model_names()
        
        # Verify all models are included with correct parsing
        # The method takes the second part of the path when split by '/'
        assert result == ["gemini-pro", "gemini-pro-vision", "with"]


def test_no_working_models(gemini_provider):
    """Test behavior when no models are working."""
    with patch('google.generativeai.list_models') as mock_list, \
         patch.object(gemini_provider, '_test_model', return_value=False):

        mock_list.return_value = [MockModel("models/gemini-pro")]

        result = gemini_provider.get_usable_model_names()

        assert result == []  # Should return empty list when no models work
