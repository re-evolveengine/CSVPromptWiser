# model/core/runners/resilient_llm_runner.py

from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


class ResilientLLMRunner(ABC):
    def __init__(self, client, max_attempts=3, wait_seconds=12):
        self.client = client
        self.max_attempts = max_attempts
        self.wait_seconds = wait_seconds

    @property
    @abstractmethod
    def retryable_errors(self):
        """Define which exceptions can be retried."""
        pass

    @property
    @abstractmethod
    def user_errors(self):
        """Define which exceptions must be shown to user immediately."""
        pass

    def _should_retry(self, exception):
        return isinstance(exception, self.retryable_errors)

    def _should_fail_fast(self, exception):
        return isinstance(exception, self.user_errors)

    def run(self, prompt, df=None):
        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_fixed(self.wait_seconds),
            retry=retry_if_exception_type(self.retryable_errors),
            reraise=True
        )
        def _call():
            try:
                return self.client.call(prompt, df)
            except self.user_errors as e:
                print(f"[User Error] {type(e).__name__}: {e}")
                raise
            except self.retryable_errors as e:
                print(f"[Retrying] {type(e).__name__}: {e}")
                raise
            except Exception as e:
                print(f"[Unknown Error] {type(e).__name__}: {e}")
                raise

        return _call()
