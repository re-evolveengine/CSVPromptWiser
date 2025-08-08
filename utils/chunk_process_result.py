from typing import Optional
import pandas as pd
from utils.result_type import ResultType


class ChunkProcessResult:
    def __init__(
            self,
            result_type: ResultType,
            response: Optional[str] = None,
            chunk: Optional[pd.DataFrame] = None,
            error: Optional[Exception] = None,
            remaining_tokens: Optional[int] = None,
            chunk_id: Optional[str] = None
    ):
        self.result_type = result_type
        self.response = response
        self.chunk = chunk
        self.error = error
        self.remaining_tokens = remaining_tokens
        self.chunk_id = chunk_id
