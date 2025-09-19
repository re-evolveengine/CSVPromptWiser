# tests/model/core/runners/test_gemini_resilient_runner.py
import types
import builtins
from unittest.mock import MagicMock
import pytest

from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from model.core.llms.gemini_resilient_runner import GeminiResilientRunner


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    return MagicMock()


@pytest.fixture
def runner(mock_client):
    """Fixture that provides a GeminiResilientRunner instance with a mock client."""
    return GeminiResilientRunner(client=mock_client)


def test_retryable_errors_contains_expected_exceptions(runner):
    retryable = runner.retryable_errors
    # All expected retryable exceptions should be in the tuple
    expected = (
        api_exceptions.DeadlineExceeded,
        api_exceptions.ServiceUnavailable,
        api_exceptions.InternalServerError,
        api_exceptions.Aborted,
        ConnectionError,
        TimeoutError,
    )
    # Compare as sets to ignore ordering
    assert set(retryable) == set(expected)
    assert isinstance(retryable, tuple)
    # Ensure they are all classes (not instances)
    assert all(isinstance(exc, type) for exc in retryable)


def test_fatal_errors_contains_expected_exceptions(runner):
    fatal = runner.fatal_errors
    expected = (
        api_exceptions.ResourceExhausted,
        api_exceptions.PermissionDenied,
        api_exceptions.Unauthenticated,
        api_exceptions.InvalidArgument,
        auth_exceptions.DefaultCredentialsError,
    )
    assert set(fatal) == set(expected)
    assert isinstance(fatal, tuple)
    assert all(isinstance(exc, type) for exc in fatal)


def test_properties_are_read_only(runner):
    # The properties should not allow reassignment
    with pytest.raises(AttributeError):
        runner.retryable_errors = ()
    with pytest.raises(AttributeError):
        runner.fatal_errors = ()
