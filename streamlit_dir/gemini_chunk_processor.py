from typing import List, Optional, Callable, Tuple
import pandas as pd
from tenacity import RetryError

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.model_prefs import ModelPreference
from model.core.llms.gemini_error_handler import GeminiErrorHandler


class GeminiChunkProcessor:
    def __init__(
        self,
        prompt: str,
        client: GeminiClient,
        chunk_manager: ChunkManager,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ):
        self.prompt = prompt
        self.client = client
        self.chunk_manager = chunk_manager
        self.progress_callback = progress_callback

        self.runner = GeminiResilientRunner(client=client)
        self.prefs = ModelPreference()
        self.error_handler = GeminiErrorHandler()

        self.results: List[dict] = []
        self.errors: List[str] = []
        self.remaining_tokens = self.prefs.get_remaining_total_tokens()
        self.count = 0

    def process_one_chunk(self) -> Tuple[List[dict], List[str]]:
        if not self._validate():
            return self.results, self.errors

        df = self.chunk_manager.get_next_chunk()
        if df is None:
            return self.results, self.errors  # No chunks left

        try:
            response, used_tokens = self.runner.run(self.prompt, df)

            self.remaining_tokens -= used_tokens
            self.prefs.save_remaining_total_tokens(self.remaining_tokens)

            result = {
                "chunk": df,
                "prompt": self.prompt,
                "response": response,
                "remaining_tokens": self.remaining_tokens
            }
            self.results.append(result)
            self.chunk_manager.mark_chunk_processed()

        except self.error_handler.user_errors as ue:
            if self._is_fatal_user_error(ue):
                raise ue
            self.errors.append(f"[User Error] Skipped chunk: {ue}")

        except RetryError as re:
            last_exc = re.last_attempt.exception()
            self.errors.append(f"[Retryable Error] Skipped chunk after retries: {last_exc}")

        except Exception as e:
            raise RuntimeError(f"[Unexpected Error] Gemini processing failed: {e}") from e

        finally:
            self.count += 1
            if self.progress_callback:
                self.progress_callback(self.count, 1, self.remaining_tokens)

        return self.results, self.errors

    def _validate(self) -> bool:
        if not self.client:
            self.errors.append("[Fatal Error] Gemini client is not set.")
            return False
        if not self.prompt:
            self.errors.append("[Fatal Error] Prompt is missing.")
            return False
        if not self.chunk_manager:
            self.errors.append("[Fatal Error] Chunk manager is not set.")
            return False
        return True

    def _is_fatal_user_error(self, error: Exception) -> bool:
        from google.api_core import exceptions as api_exceptions
        return isinstance(error, (api_exceptions.PermissionDenied, api_exceptions.Unauthenticated))
