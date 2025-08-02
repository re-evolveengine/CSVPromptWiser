# core/clients/base_llm_client.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import tiktoken

from model.utils.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P, MODEL_TOKEN_LIMITS
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
        self.llm = self._init_llm()

    def find_optimal_row_number(self, prompt: str, row_df: pd.DataFrame, example_response: str, usage_ratio: float = 0.8) -> int:
        """
        Calculate the optimal number of rows that can be processed within the model's token limit.
        
        Args:
            prompt: The prompt template to be used for processing
            row_df: A DataFrame containing example row(s) to calculate token usage
            example_response: Example response from the model for a single row
            usage_ratio: Fraction of the model's total token limit to use (default: 0.8)
            
        Returns:
            int: Maximum number of rows that can be processed within the token limit
        """
        encoding = tiktoken.encoding_for_model(self.model)
        max_tokens = MODEL_TOKEN_LIMITS.get(self.model, 8192)
        usable_tokens = int(max_tokens * usage_ratio)
        prompt_tokens = len(encoding.encode(prompt.strip()))

        # Prepare a single row block (formatted)
        def format_row(row):
            lines = ["Row 1:"]
            for col in row.index:
                lines.append(f"- {col}: {row[col]}")
            return "\n".join(lines)

        example_row_text = format_row(row_df.iloc[0])
        row_input_tokens = len(encoding.encode(example_row_text))
        row_output_tokens = len(encoding.encode(example_response))
        tokens_per_row = row_input_tokens + row_output_tokens

        available_for_rows = usable_tokens - prompt_tokens

        return max(0, available_for_rows // tokens_per_row)

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
