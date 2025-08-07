from typing import Any, Tuple
import logging
import pandas as pd
import google.generativeai as genai

from model.core.llms.base_llm_client import BaseLLMClient

# Set up logger
logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    def _init_llm(self) -> Any:
        try:
            genai.configure(api_key=self.api_key)
            return genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def call(self, prompt: str, df: pd.DataFrame) -> Tuple[str, int]:
        """
        Call Gemini LLM and return the response along with total token count (input + output).

        Returns:
            text (str): The generated response text.
            total_tokens (int): Total tokens used (input + output).
        """
        formatted_input = self._format_input(prompt, df)

        try:
            # Count input tokens
            input_tokens = self.llm.count_tokens(contents=formatted_input).total_tokens

            # Generate content
            response = self.llm.generate_content(formatted_input)
            text = response.text or ""

            # Count output tokens
            output_tokens = self.llm.count_tokens(contents=text).total_tokens

            total_tokens = input_tokens + output_tokens

            logger.info(
                f"Gemini token usage — Input: {input_tokens}, "
                f"Output: {output_tokens}, Total: {total_tokens}"
            )

            return text, total_tokens

        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            raise RuntimeError(f"Gemini call failed: {e}")
