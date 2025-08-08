import pytest
from unittest import mock

from model.io.prompt_pref import PromptPreference


class TestPromptPreference:
    @pytest.fixture
    def mock_temp_file(self, tmp_path):
        # Create a temporary file path
        temp_file = tmp_path / "test_prompt_prefs.json"
        return temp_file

    def test_save_and_load_prompt(self, mock_temp_file, monkeypatch):
        # Create an instance of PromptPreference
        pp = PromptPreference()
        
        # Mock the _load_all and _save_all methods
        with mock.patch.object(pp, '_load_all', return_value={}) as mock_load, \
             mock.patch.object(pp, '_save_all') as mock_save:
            
            test_prompt = "Test prompt content"
            
            # Save the prompt
            pp.save_prompt(test_prompt)
            
            # Verify _save_all was called with the correct data
            mock_save.assert_called_once_with({"prompt": test_prompt})
            
            # Set up the mock to return the saved data
            mock_load.return_value = {"prompt": test_prompt}
            
            # Load the prompt and verify
            loaded_prompt = pp.load_prompt()
            assert loaded_prompt == test_prompt

    def test_save_and_load_example_response(self, mock_temp_file):
        # Create an instance of PromptPreference
        pp = PromptPreference()
        
        # Mock the _load_all and _save_all methods
        with mock.patch.object(pp, '_load_all', return_value={}) as mock_load, \
             mock.patch.object(pp, '_save_all') as mock_save:
            
            test_response = "Test example response"
            
            # Save the example response
            pp.save_example_response(test_response)
            
            # Verify _save_all was called with the correct data
            mock_save.assert_called_once_with({"example_response": test_response})
            
            # Set up the mock to return the saved data
            mock_load.return_value = {"example_response": test_response}
            
            # Load the response and verify
            loaded_response = pp.load_example_response()
            assert loaded_response == test_response

    def test_load_nonexistent_file(self, tmp_path):
        # Create an instance of PromptPreference
        pp = PromptPreference()
        
        # Mock _load_all to return empty dict (simulating non-existent file)
        with mock.patch.object(pp, '_load_all', return_value={}):
            # Should return empty strings for non-existent file
            assert pp.load_prompt() == ""
            assert pp.load_example_response() == ""

    def test_save_multiple_fields(self, mock_temp_file):
        # Create an instance of PromptPreference
        pp = PromptPreference()
        
        # Mock the _load_all and _save_all methods
        with mock.patch.object(pp, '_load_all', return_value={}) as mock_load, \
             mock.patch.object(pp, '_save_all') as mock_save:
            
            test_prompt = "Test prompt"
            test_response = "Test response"
            
            # Save both fields
            pp.save_prompt(test_prompt)
            pp.save_example_response(test_response)
            
            # Verify the final state
            expected_data = {
                "prompt": test_prompt,
                "example_response": test_response
            }
            mock_save.assert_called_with(expected_data)
            
            # Set up mock to return the final state
            mock_load.return_value = expected_data
            
            # Verify both fields can be loaded
            assert pp.load_prompt() == test_prompt
            assert pp.load_example_response() == test_response

    def test_directory_creation(self, tmp_path):
        # Create a path with a non-existent parent directory
        test_file = tmp_path / "nonexistent" / "prefs.json"
        
        # Create an instance of PromptPreference with the test file path
        with mock.patch('streamlit_dir.prompt_pref.PROMPT_PREF_PATH', test_file):
            pp = PromptPreference()
            
            # Mock _load_all and _save_all to avoid actual file operations
            with mock.patch.object(pp, '_load_all', return_value={}), \
                 mock.patch.object(pp, '_save_all'):
                
                # This should not raise an exception
                pp.save_prompt("test")
                
                # Verify the parent directory was created
                assert test_file.parent.exists()
