import pandas as pd
from tenacity import RetryError
from typing import List, Tuple, Optional, Callable

from model.core.chunk.chunk_manager import ChunkManager
from model.core.llms.gemini_client import GeminiClient
from model.core.llms.gemini_resilient_runner import GeminiResilientRunner


def run_gemini_chunk_processor_ui(
    prompt: str,
    client: GeminiClient,
    chunk_manager: ChunkManager,
    max_chunks: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[List[dict], List[str]]:
    """
    Applies the given prompt to up to `max_chunks` data chunks using a pre-initialized Gemini client.

    Args:
        prompt (str): The user prompt.
        client (BaseLLMClient): An already configured GeminiClient instance.
        chunk_manager (ChunkManager): Manages access to the chunked DataFrame.
        max_chunks (int, optional): Maximum number of chunks to process.
        progress_callback (callable, optional): Function to call with (i, total) after each chunk is processed.

    Returns:
        results: List of dicts, each with keys {"chunk", "prompt", "response"}.
        errors: List of any chunk-level error messages encountered.
    """
    runner = GeminiResilientRunner(client=client)
    results: List[dict] = []
    errors: List[str] = []

    if client is None or prompt is None or chunk_manager is None:
        errors.append("[Fatal Error] Client, prompt, or chunk manager is None.")
        return results, errors

    total = chunk_manager.remaining_chunks if max_chunks is None else min(max_chunks, chunk_manager.remaining_chunks)
    count = 0

    def process_fn(df: pd.DataFrame):
        nonlocal count
        try:
            response = runner.run(prompt, df)
            results.append({
                "chunk": df,
                "prompt": prompt,
                "response": response
            })
        except runner.user_errors as ue:
            errors.append(f"[User Error] Skipped chunk: {ue}")
        except RetryError as re:
            last_exc = re.last_attempt.exception()
            errors.append(f"[Retryable Error] Skipped chunk after retries: {last_exc}")
        except Exception as e:
            errors.append(f"[Unexpected Error] Skipped chunk: {e}")
        finally:
            count += 1
            if progress_callback:
                progress_callback(count, total)

    try:
        chunk_manager.process_chunks(
            func=process_fn,
            show_progress=False,
            max_chunks=max_chunks
        )
    except Exception as e:
        errors.append(f"[Fatal Error] Processing aborted: {e}")

    return results, errors
