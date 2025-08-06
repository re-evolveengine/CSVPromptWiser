from typing import Optional
import pandas as pd
from model.utils.result_type import ResultType


class ChunkProcessResult:
    def __init__(
        self,
        result_type: ResultType,
        response: Optional[str] = None,
        chunk: Optional[pd.DataFrame] = None,
        error: Optional[Exception] = None,
        remaining_tokens: Optional[int] = None
    ):
        self.result_type = result_type
        self.response = response
        self.chunk = chunk
        self.error = error
        self.remaining_tokens = remaining_tokens

