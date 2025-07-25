import os
import json
import tempfile
import shutil
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open

from model.utils.gemini_result_saver import GeminiResultSaver
from model.utils.constants import JSON_CHUNK_VERSION


class TestGeminiResultSaver:
    """Test cases for GeminiResultSaver class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        df1 = pd.DataFrame({"text": ["sample text 1", "sample text 2"], "id": [1, 2]})
        df2 = pd.DataFrame({"text": ["sample text 3"], "id": [3]})

        return [
            {
                "chunk": df1,
                "prompt": "Test prompt 1",
                "response": "Test response 1"
            },
            {
                "chunk": df2,
                "prompt": "Test prompt 2",
                "response": "Test response 2"
            }
        ]

    @pytest.fixture
    def temp_dir(self):
        """Create and clean up a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_save_results_valid_input(self, sample_results, temp_dir):
        """Test saving results with valid input."""
        output_path = os.path.join(temp_dir, "output.json")
        metadata = {"test_metadata": "test_value"}

        # Call the method under test
        GeminiResultSaver.save_results_to_json(
            results=sample_results,
            file_path=output_path,
            metadata=metadata
        )

        # Verify the file was created
        assert os.path.exists(output_path)

        # Verify the content
        with open(output_path, 'r') as f:
            data = json.load(f)

        # Check metadata and version
        assert data["version"] == JSON_CHUNK_VERSION
        assert data["metadata"] == metadata

        # Check summary
        assert data["summary"]["total_chunks"] == 2
        assert len(data["summary"]["processed_ids"]) == 2

        # Check chunks
        assert len(data["chunks"]) == 2
        assert len(data["chunks"][0]["data"]) == 2  # First chunk has 2 rows
        assert len(data["chunks"][1]["data"]) == 1  # Second chunk has 1 row
        assert data["chunks"][0]["prompt"] == "Test prompt 1"
        assert data["chunks"][1]["response"] == "Test response 2"

    def test_save_results_empty_input(self):
        """Test that empty results raise a ValueError."""
        with pytest.raises(ValueError, match="No results to save"):
            GeminiResultSaver.save_results_to_json([], "dummy_path.json")

    def test_save_results_file_creation(self, sample_results, temp_dir):
        """Test that files are created with proper permissions and structure."""
        output_path = os.path.join(temp_dir, "nested", "output.json")

        # Call the method under test
        GeminiResultSaver.save_results_to_json(sample_results, output_path)

        # Verify the file was created in the nested directory
        assert os.path.exists(output_path)

        # Verify the parent directory was created with proper permissions
        parent_dir = os.path.dirname(output_path)
        assert os.path.isdir(parent_dir)
        assert os.access(parent_dir, os.W_OK)

    def test_save_results_error_handling(self, sample_results, temp_dir):
        """Test error handling during file writing."""
        output_path = os.path.join(temp_dir, "output.json")
        
        # Create a read-only file to cause a write failure
        with open(output_path, 'w') as f:
            f.write("dummy content")
        os.chmod(output_path, 0o444)

        # Test that PermissionError is raised when write fails
        with pytest.raises((OSError, PermissionError)):  # Accept either exception type
            GeminiResultSaver.save_results_to_json(sample_results, output_path)

        # Clean up permissions for test cleanup
        os.chmod(output_path, 0o644)

    def test_save_results_temporary_file_cleanup(self, sample_results, temp_dir):
        """Test that temporary files are cleaned up on error."""
        output_path = os.path.join(temp_dir, "output.json")
        temp_path = f"{output_path}.tmp"

        # Create the output file as a directory to cause an error
        os.makedirs(output_path, exist_ok=True)

        # Test that the temporary file is cleaned up on error
        with pytest.raises(OSError):
            GeminiResultSaver.save_results_to_json(sample_results, output_path)

        # Verify the temporary file was cleaned up
        assert not os.path.exists(temp_path)

    def test_save_results_metadata_handling(self, sample_results, temp_dir):
        """Test that metadata is properly included in the output."""
        output_path = os.path.join(temp_dir, "output.json")
        metadata = {
            "model": "test-model",
            "timestamp": "2023-01-01",
            "config": {"key": "value"}
        }

        # Call the method under test
        GeminiResultSaver.save_results_to_json(
            results=sample_results,
            file_path=output_path,
            metadata=metadata
        )

        # Verify the metadata in the output
        with open(output_path, 'r') as f:
            data = json.load(f)

        assert data["metadata"] == metadata
