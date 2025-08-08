from abc import ABC, abstractmethod
from typing import Any, Tuple
import pandas as pd

from utils import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P
)


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    """

    def __init__(self, model: str, api_key: str, generation_config: dict = None):
        self.model = model
        self.api_key = api_key
        self.generation_config = generation_config or {
            "temperature": DEFAULT_TEMPERATURE,
            "top_k": DEFAULT_TOP_K,
            "top_p": DEFAULT_TOP_P
        }
        self.model_name = model  # ✅ Add this to make it accessible externally
        self.llm = self._init_llm()

    @abstractmethod
    def _init_llm(self) -> Any:
        """Initialize and return a configured LLM client."""
        pass

    @abstractmethod
    def call(self, prompt: str, df: pd.DataFrame) -> Tuple[str, int]:
        """
        Format input and call the LLM.
        Returns a tuple of (response_text, token_count).
        """
        pass

    def _format_input(self, prompt: str, df: pd.DataFrame) -> str:
        """
        Combines prompt and DataFrame into a structured string.
        """
        output = [prompt.strip(), ""]

        df = df.fillna("")
        for idx, row in df.iterrows():
            lines = [f"Row {idx + 1}:"]
            for col in df.columns:
                lines.append(f"- {col}: {row[col]}")
            output.append("\n".join(lines))

        return "\n\n".join(output)
