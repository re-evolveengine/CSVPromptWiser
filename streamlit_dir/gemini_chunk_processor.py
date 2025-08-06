from typing import List, Tuple, Optional, Callable
import pandas as pd
from tenacity import RetryError
from enum import Enum, auto

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.model_prefs import ModelPreference
from model.utils.chunk_process_result import ChunkProcessResult
from model.utils.result_type import ResultType


class GeminiChunkProcessor:
    def __init__(
        self,
        prompt: str,
        client: GeminiClient,
        chunk_manager: ChunkManager,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ):
        self.prompt = prompt
        self.client = client
        self.chunk_manager = chunk_manager
        self.progress_callback = progress_callback

        self.runner = GeminiResilientRunner(client=self.client)
        self.prefs = ModelPreference()
        self.count = 0
        self.remaining_tokens = self.prefs.get_remaining_total_tokens()

        self._validate_inputs()

    def _validate_inputs(self):
        if not self.prompt:
            raise ValueError("Prompt must not be empty.")
        if not self.client:
            raise ValueError("LLM client is not configured.")
        if not self.chunk_manager:
            raise ValueError("Chunk manager is not initialized.")

    def process_one_chunk(self) -> ChunkProcessResult:
        """Processes a single chunk and returns a typed result."""
        df = self.chunk_manager.get_next_chunk()
        if df is None:
            return ChunkProcessResult(ResultType.NO_MORE_CHUNKS)

        try:
            response, used_tokens = self.runner.run(self.prompt, df)
            self.remaining_tokens -= used_tokens
            self.prefs.save_remaining_total_tokens(self.remaining_tokens)

            self.chunk_manager.mark_chunk_processed()
            self.count += 1

            if self.progress_callback:
                total = self.chunk_manager.total_chunks
                self.progress_callback(self.count, total, self.remaining_tokens)

            return ChunkProcessResult(
                result_type=ResultType.SUCCESS,
                response=response,
                chunk=df,
                remaining_tokens=self.remaining_tokens
            )

        except self.runner.fatal_errors as ue:
            return ChunkProcessResult(
                result_type=ResultType.FATAL_ERROR,
                chunk=df,
                error=ue
            )
        except RetryError as re:
            last_exc = re.last_attempt.exception()
            return ChunkProcessResult(
                result_type=ResultType.RETRYABLE_ERROR,
                chunk=df,
                error=last_exc
            )
        except Exception as e:
            return ChunkProcessResult(
                result_type=ResultType.UNEXPECTED_ERROR,
                chunk=df,
                error=e
            )
