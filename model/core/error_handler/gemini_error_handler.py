# model/core/error_handlers/gemini_error_handler.py

from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from model.core.error_handler.llm_error_handler import LLMErrorHandler


class GeminiErrorHandler(LLMErrorHandler):
    @property
    def retryable_errors(self):
        return (
            api_exceptions.DeadlineExceeded,
            api_exceptions.ServiceUnavailable,
            api_exceptions.InternalServerError,
            api_exceptions.Aborted,
            ConnectionError,
            TimeoutError,
        )

    @property
    def user_errors(self):
        return (
            api_exceptions.PermissionDenied,
            api_exceptions.Unauthenticated,
            api_exceptions.InvalidArgument,
            api_exceptions.ResourceExhausted,
            auth_exceptions.DefaultCredentialsError,
        )
