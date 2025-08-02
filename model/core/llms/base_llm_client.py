# core/clients/base_llm_client.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import tiktoken

from model.utils.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P, 
    SAFE_PROMPT_LIMITS
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
        Uses SAFE_PROMPT_LIMITS for more conservative chunking.
        
        Args:
            prompt: The prompt template to be used for processing
            row_df: A DataFrame containing example row(s) to calculate token usage
            example_response: Example response from the model for a single row
            usage_ratio: Fraction of the safe token limit to use (default: 0.8)
            
        Returns:
            int: Maximum number of rows that can be processed within the safe token limit
        """
        try:
            # Get the appropriate safe token limit for this model
            max_tokens = None
            model_lower = self.model.lower()
            
            # Find the first matching model family in SAFE_PROMPT_LIMITS
            for model_prefix, limit in SAFE_PROMPT_LIMITS.items():
                if model_prefix in model_lower:
                    max_tokens = limit
                    break
            
            # Fall back to default if no specific match found
            if max_tokens is None:
                max_tokens = SAFE_PROMPT_LIMITS.get('default', 32000)
            
            # Calculate usable tokens
            usable_tokens = int(max_tokens * usage_ratio)
            
            # Get encoding (fall back to cl100k_base if model not found)
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            # Calculate prompt tokens
            prompt_tokens = len(encoding.encode(prompt.strip()))

            # Calculate tokens for a single row
            def format_row(row):
                lines = ["Row 1:"]
                for col in row.index:
                    lines.append(f"- {col}: {row[col]}")
                return "\n".join(lines)

            example_row_text = format_row(row_df.iloc[0])
            row_input_tokens = len(encoding.encode(example_row_text))
            row_output_tokens = len(encoding.encode(example_response))
            tokens_per_row = row_input_tokens + row_output_tokens

            # Ensure we have at least some tokens per row
            if tokens_per_row == 0:
                return 10  # Safe default if we can't calculate

            # Calculate rows that fit, with a reasonable maximum
            available_for_rows = (usable_tokens - prompt_tokens) // tokens_per_row
            return min(max(1, available_for_rows), 100)  # Cap at 100 rows
            
        except Exception as e:
            print(f"Warning: Could not calculate optimal chunk size: {str(e)}")
            return 10  # Return a safe default number of rows

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
