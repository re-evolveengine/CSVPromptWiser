# tests/model/core/runners/test_resilient_llm_runner.py
import pytest
from unittest.mock import MagicMock, call

from model.core.llms.resilient_llm_runner import ResilientLLMRunner


class MyRetryableError(Exception):
    pass


class MyFatalError(Exception):
    pass


class DummyRunner(ResilientLLMRunner):
    @property
    def retryable_errors(self):
        return (MyRetryableError,)

    @property
    def fatal_errors(self):
        return (MyFatalError,)


@pytest.fixture
def dummy_client():
    return MagicMock()


@pytest.fixture
def runner(dummy_client):
    # Using small max_attempts for fast tests
    return DummyRunner(dummy_client, max_attempts=2)


def test_should_retry_and_fail_fast_checks(runner):
    assert runner._should_retry(MyRetryableError())
    assert not runner._should_retry(Exception())
    assert runner._should_fail_fast(MyFatalError())
    assert not runner._should_fail_fast(Exception())


def test_run_returns_value_on_success(runner, dummy_client):
    dummy_client.call.return_value = "ok", 42
    result = runner.run("prompt", None)
    assert result == ("ok", 42)
    dummy_client.call.assert_called_once_with("prompt", None)


def test_run_retries_on_retryable_error(runner, dummy_client):
    # Fail first time, succeed second time
    dummy_client.call.side_effect = [
        MyRetryableError("temporary"),
        ("done", 1),
    ]
    result = runner.run("prompt", None)
    assert result == ("done", 1)
    assert dummy_client.call.call_count == 2


def test_run_raises_immediately_on_fatal_error(runner, dummy_client):
    dummy_client.call.side_effect = MyFatalError("bad request")
    with pytest.raises(MyFatalError):
        runner.run("prompt")
    # Only one call because fatal stops retries
    assert dummy_client.call.call_count == 1


def test_run_raises_immediately_on_unexpected_error(runner, dummy_client):
    dummy_client.call.side_effect = ValueError("weird failure")
    with pytest.raises(ValueError):
        runner.run("prompt")
    # Unexpected errors also stop retries
    assert dummy_client.call.call_count == 1
