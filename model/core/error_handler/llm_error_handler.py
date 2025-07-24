# model/core/error_handlers/base_error_handler.py

from abc import ABC, abstractmethod
import time
import traceback


class LLMErrorHandler(ABC):
    def __init__(self, client, max_retries=3, retry_delay=2):
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    @abstractmethod
    def retryable_errors(self):
        """Errors for which retries should be attempted."""
        pass

    @property
    @abstractmethod
    def user_errors(self):
        """Errors that should be surfaced to the user immediately."""
        pass

    def run(self, prompt, df=None):
        attempt = 0
        while attempt < self.max_retries:
            try:
                return self.client.call(prompt, df)
            except self.retryable_errors as e:
                attempt += 1
                print(f"[Retryable Error] {type(e).__name__}: {e} (Attempt {attempt})")
                time.sleep(self.retry_delay)
            except self.user_errors as e:
                print(f"[User Error] {type(e).__name__}: {e}")
                raise
            except Exception as e:
                print(f"[Unexpected Error] {type(e).__name__}: {e}")
                traceback.print_exc()
                raise
        raise RuntimeError("Max retries exceeded.")
