# model/core/runners/gemini_resilient_runner.py

from google.api_core import exceptions as api_exceptions
from google.auth import exceptions as auth_exceptions

from model.core.llms.resilient_llm_runner import ResilientLLMRunner


class GeminiResilientRunner(ResilientLLMRunner):
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
    def fatal_errors(self):
        return (
            api_exceptions.ResourceExhausted,
            api_exceptions.PermissionDenied,
            api_exceptions.Unauthenticated,
            api_exceptions.InvalidArgument,
            auth_exceptions.DefaultCredentialsError,
        )
