import pytest
from unittest.mock import MagicMock
from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from model.core.runners.gemini_resilient_runner import GeminiResilientRunner


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def runner(mock_client):
    return GeminiResilientRunner(client=mock_client, max_attempts=2, wait_seconds=0)


def test_run_success(runner, mock_client):
    mock_client.call.return_value = "Success"
    result = runner.run("Prompt here")
    assert result == "Success"
    mock_client.call.assert_called_once()


def test_run_retryable_error_then_success(runner, mock_client):
    # First call raises retryable error, second call succeeds
    mock_client.call.side_effect = [
        api_exceptions.DeadlineExceeded("Temporary issue"),
        "Recovered"
    ]
    result = runner.run("Prompt")
    assert result == "Recovered"
    assert mock_client.call.call_count == 2


def test_run_user_error_fails_fast(runner, mock_client):
    mock_client.call.side_effect = api_exceptions.PermissionDenied("Access denied")
    with pytest.raises(api_exceptions.PermissionDenied):
        runner.run("Prompt")
    mock_client.call.assert_called_once()


def test_run_unknown_error_fails(runner, mock_client):
    mock_client.call.side_effect = ValueError("Unexpected")
    with pytest.raises(ValueError):
        runner.run("Prompt")
    mock_client.call.assert_called_once()


def test_should_retry_and_should_fail_fast_types(runner):
    assert runner._should_retry(api_exceptions.Aborted("retry")) is True
    assert runner._should_fail_fast(api_exceptions.Unauthenticated("fail fast")) is True
    assert runner._should_retry(ValueError("nope")) is False
    assert runner._should_fail_fast(ValueError("nope")) is False
