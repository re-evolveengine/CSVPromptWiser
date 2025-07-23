import pytest
import sys
from unittest.mock import patch, MagicMock
import pandas as pd

from model.utils.data_prompt_arger import DataPromptArger
from model.utils.dataset_loader import DatasetLoader

class TestDataPromptArg:
    def test_get_prompt(self):
        # Test with a sample prompt and dataset
        test_args = ["script_name.py", "--prompt", "test prompt", "--dataset", "test.csv"]
        
        with patch.object(sys, 'argv', test_args):
            dpa = DataPromptArger()
            assert dpa.get_prompt() == "test prompt"

    def test_get_dataset_returns_dataframe(self):
        # Test dataset loading with a mock
        test_args = ["script_name.py", "--prompt", "test", "--dataset", "test.csv"]
        
        with patch.object(sys, 'argv', test_args):
            dpa = DataPromptArger()
            
            # Mock the DatasetLoader to return test data
            with patch('model.utils.data_prompt_arger.DatasetLoader') as mock_loader:
                # Create a sample DataFrame
                test_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
                mock_loader.return_value.load.return_value = test_df
                
                # Test get_dataset returns a DataFrame
                df = dpa.get_dataset()
                assert isinstance(df, pd.DataFrame)
                assert df.shape == (2, 2)
                
                # Test print methods don't raise errors
                dpa.print_df_head()
                dpa.print_df_shape()
    
    def test_missing_prompt_raises_error(self):
        # Test that missing required --prompt raises SystemExit
        with patch.object(sys, 'argv', ["script_name.py"]):
            with pytest.raises(SystemExit):
                DataPromptArger()

    def test_dataset_loading_error_handling(self):
        # Test error handling when dataset loading fails
        test_args = ["script_name.py", "--prompt", "test", "--dataset", "test.csv"]
        
        with patch.object(sys, 'argv', test_args):
            dpa = DataPromptArger()
            
            # Mock DatasetLoader to raise an error
            with patch('model.utils.data_prompt_arger.DatasetLoader') as mock_loader:
                # Configure the mock to raise an error when load is called
                mock_loader.return_value.load.side_effect = FileNotFoundError("File not found")
                
                with pytest.raises(FileNotFoundError):
                    dpa.get_dataset()
                    
    def test_get_dataset_without_filename_raises_error(self):
        # Test that get_dataset() raises ValueError when no filename is provided
        test_args = ["script_name.py", "--prompt", "test"]
        
        with patch.object(sys, 'argv', test_args):
            dpa = DataPromptArger()
            
            with patch('model.utils.data_prompt_arger.DatasetLoader') as mock_loader:
                # Mock the load method to verify it's not called
                mock_loader.return_value.load.side_effect = ValueError("This should not be called")
                
                with pytest.raises(ValueError, match="Please provide a file name"):
                    # This should raise ValueError before even trying to call load()
                    dpa.get_dataset()
