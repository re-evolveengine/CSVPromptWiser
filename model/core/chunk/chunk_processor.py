import logging

from tenacity import RetryError

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.base_llm_client import BaseLLMClient
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner
from model.io.model_prefs import ModelPreference
from utils.exceptions import TokenBudgetExceededError
from utils.providers import get_model_prefs
from utils.chunk_process_result import ChunkProcessResult
from utils.result_type import ResultType

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ChunkProcessor:
    def __init__(
        self,
        prompt: str,
        client: BaseLLMClient,
        chunk_manager: ChunkManager,
        model_preference: ModelPreference,
    ):
        self.prompt = prompt
        self.client = client
        self.chunk_manager = chunk_manager

        self._validate_inputs()

        if isinstance(client, GeminiClient):
            self.runner = GeminiResilientRunner(client=self.client)
        else:
            raise ValueError("Unsupported LLM client type")

        self.prefs = model_preference
        # TODO: create a dictionary for different llms to get each remaining tokens
        self.remaining_tokens = self.prefs.remaining_total_tokens


    def _validate_inputs(self):
        if not self.prompt:
            raise ValueError("Prompt must not be empty.")
        if not self.client:
            raise ValueError("LLM client is not configured.")
        if not self.chunk_manager:
            raise ValueError("Chunk manager is not initialized.")

    def process_next_chunk(self) -> ChunkProcessResult:

        """Processes a single chunk and returns a typed result."""
        chunk_data = self.chunk_manager.get_next_chunk()
        if not chunk_data or chunk_data[0] is None:
            return ChunkProcessResult(ResultType.NO_MORE_CHUNKS)

        df, chunk_id = chunk_data

        try:

            response, used_tokens = self.runner.run(self.prompt, df)

            if self.remaining_tokens - used_tokens <= 0:
                raise TokenBudgetExceededError(used_tokens, self.remaining_tokens)

            # rewrite unit and integration tests

            self.remaining_tokens -= used_tokens
            self.prefs.remaining_total_tokens = self.remaining_tokens

            self.chunk_manager.mark_chunk_processed(chunk_id)
            self.chunk_manager.save_state()

            return ChunkProcessResult(
                result_type=ResultType.SUCCESS,
                response=response,
                chunk=df,
                remaining_tokens=self.remaining_tokens,
                chunk_id=chunk_id
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

        except TokenBudgetExceededError as ve:
            return ChunkProcessResult(
                result_type=ResultType.TOKENS_BUDGET_EXCEEDED,
                chunk=df,
                error=ve
            )

        except Exception as e:
            return ChunkProcessResult(
                result_type=ResultType.UNEXPECTED_ERROR,
                chunk=df,
                error=e
            )
