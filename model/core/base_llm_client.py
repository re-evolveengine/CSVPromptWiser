# core/clients/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseLLMClient(ABC):
    """
    Abstract base class for all LangChain-based LLM clients.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.llm = self._init_llm()

    @abstractmethod
    def _init_llm(self) -> Any:
        """Initialize and return a LangChain-compatible LLM object."""
        pass

    def call(self, prompt: str, df: pd.DataFrame) -> str:
        """
        Format input and call the LLM.
        """
        formatted_input = self._format_input(prompt, df)
        return self.llm.invoke(formatted_input)

    def _format_input(self, prompt: str, df: pd.DataFrame) -> str:
        """
        General-purpose formatter: combines prompt and a DataFrame in a clean bullet-list format.

        Args:
            prompt (str): The instruction or question for the LLM.
            df (pd.DataFrame): Any structured data from ChunkManager.

        Returns:
            str: A readable string suitable for LLM input.
        """
        output = [prompt.strip(), ""]

        df = df.fillna("")
        for idx, row in df.iterrows():
            lines = [f"Row {idx + 1}:"]
            for col in df.columns:
                lines.append(f"- {col}: {row[col]}")
            output.append("\n".join(lines))

        return "\n\n".join(output)

