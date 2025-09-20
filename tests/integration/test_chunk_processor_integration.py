import pytest
import pandas as pd
import tempfile
import os
import shelve
import dbm
import io
from contextlib import contextmanager
from google.api_core import exceptions as api_exceptions
from pathlib import Path

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.chunk.chunk_processor import ChunkProcessor
from model.io.model_prefs import ModelPreference
from utils.result_type import ResultType


@contextmanager
def temp_shelf():
    """Context manager for a temporary in-memory shelf database."""
    # Create a temporary file for the shelf
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_db:
        db_path = temp_db.name
    
    try:
        # Create and initialize the shelf
        with shelve.open(db_path, flag='n') as shelf:
            # Initialize with default values if needed
            shelf['remaining_total_tokens'] = 10000
            shelf['total_tokens'] = 10000
            shelf['chunk_size'] = 1000
        
        # Yield the path to the shelf file
        yield db_path
    finally:
        # Clean up the temporary file
        try:
            # Remove all possible shelf files
            for ext in ('', '.bak', '.dat', '.dir'):
                file_path = f"{db_path}{ext}"
                if os.path.exists(file_path):
                    os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary shelf file: {e}")

chunk_file_path = Path(r"C:\Users\Alchemist\PycharmProjects\PromptPilot\tests\integration\chunks.json")


class DummyModel:
    """Simulates Gemini API responses for testing ChunkProcessor."""
    def __init__(self, responses=None, errors=None):
        self.responses = responses or []
        self.errors = errors or []
        self.call_count = 0

    def count_tokens(self, contents):
        class Result:
            total_tokens = 1
        return Result()

    def generate_content(self, formatted_input):
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return type("Resp", (), {"text": self.responses.pop(0)})
        raise RuntimeError("No more responses configured")


def test_json_path():
    from pathlib import Path

    # Optional safety check
    assert chunk_file_path.exists(), f"chunks.json not found at {chunk_file_path}"

@pytest.mark.parametrize("scenario", [
    "success",
    "retryable",
    "fatal",
    "unexpected",
])
def test_chunk_processor_end_to_end(monkeypatch, scenario):
    # Use the real chunks.json from the repo so ChunkManager version check passes
    cm = ChunkManager(json_path=str(chunk_file_path))

    # Configure DummyModel behavior
    if scenario == "success":
        model = DummyModel(responses=["OK-1"])
    elif scenario == "retryable":
        model = DummyModel(errors=[api_exceptions.DeadlineExceeded("try again")],
                           responses=["OK-after-retry"])
    elif scenario == "fatal":
        model = DummyModel(errors=[api_exceptions.PermissionDenied("stop right there")])
    elif scenario == "unexpected":
        class WeirdError(RuntimeError):
            pass
        model = DummyModel(errors=[WeirdError("boom!")])
    else:
        raise ValueError(f"Unknown scenario {scenario}")

    # Avoid calling real API
    monkeypatch.setattr(
        "model.core.llms.gemini_client.genai.configure",
        lambda **kwargs: None
    )
    monkeypatch.setattr(
        "model.core.llms.gemini_client.genai.GenerativeModel",
        lambda **kwargs: model
    )

    client = GeminiClient("gemini-test", api_key="fake-key")
    
    # Create a temporary shelf database for ModelPreference
    with temp_shelf() as db_path:
        # Initialize ModelPreference with the temporary database
        model_preference = ModelPreference(db_path=db_path)
        # The shelf is already initialized with default values
    
        processor = ChunkProcessor(
            prompt="prompt", 
            client=client, 
            chunk_manager=cm,
            model_preference=model_preference
        )

        result = processor.process_next_chunk()

        if scenario == "success":
            assert result.result_type == ResultType.SUCCESS
            assert "OK" in result.response
        elif scenario == "retryable":
            assert result.result_type == ResultType.SUCCESS
            assert "OK" in result.response
        elif scenario == "fatal":
            assert result.result_type == ResultType.FATAL_ERROR
            assert "stop right there" in str(result.error)
            assert isinstance(result.error, api_exceptions.PermissionDenied)
        elif scenario == "unexpected":
            assert result.result_type == ResultType.UNEXPECTED_ERROR
            assert "boom!" in str(result.error)
