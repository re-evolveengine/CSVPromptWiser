import unittest
from unittest.mock import Mock
import time

from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from model.core.error_handler.gemini_error_handler import GeminiErrorHandler


class TestGeminiErrorHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.handler = GeminiErrorHandler(client=self.mock_client, max_retries=2, retry_delay=0.1)

    def test_retryable_errors_property(self):
        """Test that retryable_errors contains the expected error types."""
        expected_errors = (
            api_exceptions.DeadlineExceeded,
            api_exceptions.ServiceUnavailable,
            api_exceptions.InternalServerError,
            api_exceptions.Aborted,
            ConnectionError,
            TimeoutError,
        )
        self.assertEqual(self.handler.retryable_errors, expected_errors)

    def test_user_errors_property(self):
        """Test that user_errors contains the expected error types."""
        expected_errors = (
            api_exceptions.PermissionDenied,
            api_exceptions.Unauthenticated,
            api_exceptions.InvalidArgument,
            api_exceptions.ResourceExhausted,
            auth_exceptions.DefaultCredentialsError,
        )
        self.assertEqual(self.handler.user_errors, expected_errors)

    def test_run_with_retryable_error(self):
        """Test that run() retries on retryable errors."""
        # Mock the client to raise a retryable error twice, then succeed
        self.mock_client.call.side_effect = [
            api_exceptions.DeadlineExceeded("Timeout"),
            "Success"
        ]

        result = self.handler.run("test prompt")
        self.assertEqual(result, "Success")
        self.assertEqual(self.mock_client.call.call_count, 2)

    def test_run_with_user_error(self):
        """Test that run() raises user errors immediately."""
        # Mock the client to raise a user error
        self.mock_client.call.side_effect = api_exceptions.PermissionDenied("Access denied")

        with self.assertRaises(api_exceptions.PermissionDenied):
            self.handler.run("test prompt")
        self.mock_client.call.assert_called_once()

    def test_run_exceeds_max_retries(self):
        """Test that run() raises RuntimeError when max retries are exceeded."""
        # Mock the client to always raise a retryable error
        self.mock_client.call.side_effect = api_exceptions.DeadlineExceeded("Timeout")
        
        with self.assertRaises(RuntimeError) as context:
            self.handler.run("test prompt")
        self.assertIn("Max retries exceeded", str(context.exception))
        # With max_retries=2, we expect 2 total calls (1 initial + 1 retry)
        self.assertEqual(self.mock_client.call.call_count, 2)

    def test_run_with_unexpected_error(self):
        """Test that unexpected errors are re-raised."""
        self.mock_client.call.side_effect = ValueError("Unexpected error")

        with self.assertRaises(ValueError):
            self.handler.run("test prompt")
        self.mock_client.call.assert_called_once()


if __name__ == '__main__':
    unittest.main()
