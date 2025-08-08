import tiktoken
import pandas as pd

from utils import SAFE_PROMPT_LIMITS


class PromptOptimizer:
    """
    Handles optimization of prompts and chunking based on token limits.
    """

    def __init__(self, model_name: str):
        """
        Initialize the PromptOptimizer with a specific model.

        Args:
            model_name: Name of the model being used (e.g., 'gpt-4', 'gemini-pro')
        """
        self.model_name = model_name.lower()

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
            max_tokens = self._get_safe_token_limit()

            # Calculate usable tokens
            usable_tokens = int(max_tokens * usage_ratio)

            # Get encoding (fall back to cl100k_base if model not found)
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            # Calculate prompt tokens
            prompt_tokens = len(encoding.encode(prompt.strip()))

            # Calculate tokens for a single row
            example_row_text = self._format_row(row_df.iloc[0])
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

    def _get_safe_token_limit(self) -> int:
        """
        Get the safe token limit for the current model.

        Returns:
            int: Safe token limit for the model
        """
        # Find the first matching model family in SAFE_PROMPT_LIMITS
        for model_prefix, limit in SAFE_PROMPT_LIMITS.items():
            if model_prefix in self.model_name:
                return limit

        # Fall back to default if no specific match found
        return SAFE_PROMPT_LIMITS.get('default', 32000)

    @staticmethod
    def _format_row(row) -> str:
        """
        Format a single row into a string representation.

        Args:
            row: A single row from a DataFrame

        Returns:
            str: Formatted string representation of the row
        """
        lines = ["Row 1:"]
        for col in row.index:
            lines.append(f"- {col}: {row[col]}")
        return "\n".join(lines)

    def calculate_used_tokens(self, prompt: str, row_df: pd.DataFrame, example_response: str, num_rows: int) -> int:
        """
        Calculate the total number of tokens used for a given prompt, number of rows, and example response.
        
        Args:
            prompt: The prompt template being used
            row_df: A DataFrame containing example row(s) to calculate token usage
            example_response: Example response from the model for a single row
            num_rows: Number of rows being processed
            
        Returns:
            int: Total number of tokens used
        """
        try:
            # Get encoding (fall back to cl100k_base if model not found)
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            # Calculate prompt tokens
            prompt_tokens = len(encoding.encode(prompt.strip()))

            # Calculate tokens for a single row input and output
            example_row_text = self._format_row(row_df.iloc[0])
            row_input_tokens = len(encoding.encode(example_row_text))
            row_output_tokens = len(encoding.encode(example_response))
            
            # Calculate total tokens: prompt + (input + output) * number of rows
            total_tokens = prompt_tokens + (row_input_tokens + row_output_tokens) * num_rows
            
            return total_tokens
            
        except Exception as e:
            print(f"Warning: Could not calculate token usage: {str(e)}")
            return 0  # Return 0 if calculation fails


    def calculate_max_chunks_with_quota(
        self,
        prompt: str,
        row_df: pd.DataFrame,
        example_response: str,
        rows_per_chunk: int,
        token_quota: int
    ) -> int:
        """
        Calculate how many chunks can fit within a given token quota.

        Args:
            prompt: Prompt template used for chunk processing
            row_df: A DataFrame with example data
            example_response: Sample response used to estimate output tokens
            rows_per_chunk: Number of rows per chunk
            token_quota: Total tokens available (e.g., daily quota)

        Returns:
            int: Maximum number of chunks that can be processed within the quota
        """
        try:
            tokens_per_chunk = self.calculate_used_tokens(
                prompt=prompt,
                row_df=row_df,
                example_response=example_response,
                num_rows=rows_per_chunk
            )

            if tokens_per_chunk == 0:
                return 0  # Avoid division by zero

            max_chunks = token_quota // tokens_per_chunk
            return max_chunks

        except Exception as e:
            print(f"Warning: Could not calculate max chunks: {str(e)}")
            return 0

