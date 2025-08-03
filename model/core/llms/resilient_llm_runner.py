# model/core/runners/resilient_llm_runner.py

from abc import ABC, abstractmethod
import logging

from responses import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type, wait_exponential, before_sleep_log


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

    logger = logging.getLogger(__name__)

    def run(self, prompt, df=None):
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(self.max_attempts),
            retry=retry_if_exception_type(self.retryable_errors),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        def _call():
            try:
                return self.client.call(prompt, df)
            except self.user_errors as e:
                # Don't retry on user errors — re-raise immediately
                print(f"[User Error] {type(e).__name__}: {e}")
                raise
            except self.retryable_errors as e:
                # Will be retried by tenacity
                print(f"[Retryable Error] {type(e).__name__}: {e}")
                raise
            except Exception as e:
                # Unknown exceptions — raise to avoid silent failures
                print(f"[Unexpected Error] {type(e).__name__}: {e}")
                raise

        return _call()
