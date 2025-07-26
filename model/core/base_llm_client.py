# core/clients/base_llm_client.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

from model.utils.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P
)


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM clients.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.generation_config = {
            "temperature": DEFAULT_TEMPERATURE,
            "top_k": DEFAULT_TOP_K,
            "top_p": DEFAULT_TOP_P
        }
        self.llm = self._init_llm()

    @abstractmethod
    def _init_llm(self) -> Any:
        """Initialize and return a configured LLM client."""
        pass

    def call(self, prompt: str, df: pd.DataFrame) -> str:
        """
        Format input and call the LLM.
        """
        formatted_input = self._format_input(prompt, df)
        return self.llm.invoke(formatted_input)

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
