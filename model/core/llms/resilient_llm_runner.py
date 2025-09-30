# model/core/runners/resilient_llm_runner.py

from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential


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
    def fatal_errors(self):
        """Define which exceptions must be shown to user immediately."""
        pass

    def _should_retry(self, exception):
        return isinstance(exception, self.retryable_errors)

    def _should_fail_fast(self, exception):
        return isinstance(exception, self.fatal_errors)

    def run(self, prompt, df=None):
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(self.max_attempts),
            retry=retry_if_exception_type(self.retryable_errors),
            reraise=True
        )
        def _call():
            try:
                return self.client.call(prompt, df)
            except self.fatal_errors:
                raise
            except self.retryable_errors:
                raise
            except Exception:
                raise

        return _call()
