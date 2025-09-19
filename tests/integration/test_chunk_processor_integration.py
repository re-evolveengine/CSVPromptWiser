import pytest
import pandas as pd
from google.api_core import exceptions as api_exceptions
from pathlib import Path

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.chunk.chunk_processor import ChunkProcessor
from utils.result_type import ResultType

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
    processor = ChunkProcessor(prompt="prompt", client=client, chunk_manager=cm)

    result = processor.process_next_chunk()

    if scenario == "success":
        assert result.result_type == ResultType.SUCCESS
        assert "OK" in result.response
    elif scenario == "retryable":
        assert result.result_type == ResultType.SUCCESS
        assert "OK" in result.response
    elif scenario == "fatal":
        assert result.result_type == ResultType.FATAL_ERROR
        assert isinstance(result.error, api_exceptions.PermissionDenied)
    elif scenario == "unexpected":
        assert result.result_type == ResultType.UNEXPECTED_ERROR
