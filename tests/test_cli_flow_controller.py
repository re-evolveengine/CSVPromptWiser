import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import pandas as pd

from cli.cli_flow_controller import CLIFlowController
from model.utils.constants import TEMP_DIR_CLI, DATA_DIR_CLI, RESULTS_DIR_CLI


class TestCLIFlowController(unittest.TestCase):
    def setUp(self):
        self.controller = CLIFlowController()
        self.sample_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

    @patch('cli.cli_flow_controller.load_api_key')
    @patch('cli.cli_flow_controller.handle_model_selection')
    def test_step_0_choose_model(self, mock_handle_model, mock_load_key):
        # Setup
        mock_load_key.return_value = 'test_api_key'
        mock_handle_model.return_value = 'gemini-pro'
        
        # Execute
        self.controller.step_0_choose_model()
        
        # Assert
        mock_load_key.assert_called_once()
        mock_handle_model.assert_called_once_with('test_api_key')
        self.assertEqual(self.controller.api_key, 'test_api_key')
        self.assertEqual(self.controller.model_name, 'gemini-pro')

    @patch('cli.cli_flow_controller.ChunkJSONInspector')
    def test_step_1_check_existing_chunk_file_no_file(self, mock_inspector):
        # Setup
        mock_instance = MagicMock()
        mock_instance.find_valid_chunk_file.return_value = None
        mock_inspector.return_value = mock_instance
        
        # Execute
        self.controller.step_1_check_existing_chunk_file()
        
        # Assert
        self.assertIsNone(self.controller.chunk_file)
        self.assertFalse(self.controller.resume)
