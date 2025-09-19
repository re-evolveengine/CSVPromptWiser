import types
import pandas as pd
import pytest
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

import google.generativeai as genai

from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner


class DummyModel:
    """Fake google.generativeai.GenerativeModel for testing."""

    def __init__(self, responses=None, errors=None):
        self._responses = responses or []
        self._errors = errors or []
        self.call_count = 0

    def count_tokens(self, contents):
        # Always return a fake token count
        return types.SimpleNamespace(total_tokens=len(contents))

    def generate_content(self, formatted_input):
        # Simulate sequence of errors/successes
        if self.call_count < len(self._errors):
            err = self._errors[self.call_count]
            self.call_count += 1
            raise err
        resp_text = self._responses[self.call_count] if self.call_count < len(self._responses) else "ok"
        self.call_count += 1
        return types.SimpleNamespace(text=resp_text)


@pytest.fixture
def fake_model(monkeypatch):
    """Patch genai.GenerativeModel to return a DummyModel instance on init."""
    model_instance = DummyModel()
    monkeypatch.setattr(genai, "GenerativeModel", lambda **kwargs: model_instance)
    return model_instance


@pytest.fixture
def runner(fake_model, monkeypatch):
    # Patch genai.configure so it doesn't hit network
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)
    client = GeminiClient(model="gemini-test", api_key="fake-key")
    return GeminiResilientRunner(client=client)


def test_run_success(fake_model, runner):
    fake_model._responses = ["hello world"]
    df = pd.DataFrame([{"col": "val"}])
    resp, tokens = runner.run("prompt text", df)
    assert resp == "hello world"
    # Token count is sum of input length (formatted_input string) and output length
    assert isinstance(tokens, int)
    assert tokens > 0
    assert fake_model.call_count == 1


def test_run_retryable_error_then_success(monkeypatch):
    # First call raises a retryable error, second call succeeds
    model_instance = DummyModel(
        responses=["final result"],
        errors=[api_exceptions.DeadlineExceeded("timeout!")]
    )
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda **kwargs: model_instance)
    client = GeminiClient("gemini-test", api_key="fake-key")
    runner = GeminiResilientRunner(client)
    df = pd.DataFrame([{"x": 1}])
    resp, tokens = runner.run("prompt", df)
    assert resp == "ok"
    assert model_instance.call_count == 2


def test_run_fatal_error(monkeypatch):
    model_instance = DummyModel(
        errors=[api_exceptions.PermissionDenied("no access")]
    )
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda **kwargs: model_instance)
    client = GeminiClient("gemini-test", api_key="fake-key")
    runner = GeminiResilientRunner(client)
    df = pd.DataFrame([{"x": 2}])
    with pytest.raises(api_exceptions.PermissionDenied):
        runner.run("prompt", df)
    assert model_instance.call_count == 1


def test_run_unexpected_error(monkeypatch):
    class WeirdError(RuntimeError):
        pass

    model_instance = DummyModel(errors=[WeirdError("boom!")])
    monkeypatch.setattr(genai, "configure", lambda **kwargs: None)
    monkeypatch.setattr(genai, "GenerativeModel", lambda **kwargs: model_instance)
    client = GeminiClient("gemini-test", api_key="fake-key")
    runner = GeminiResilientRunner(client)
    df = pd.DataFrame([{"x": 3}])
    with pytest.raises(WeirdError):
        runner.run("prompt", df)
    assert model_instance.call_count == 1
