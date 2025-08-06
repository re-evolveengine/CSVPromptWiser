from typing import Any, Tuple
import logging
import pandas as pd
import google.generativeai as genai

from model.core.llms.base_llm_client import BaseLLMClient

# Add logger
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
        Call Gemini LLM and return both the response and token count.
        """
        formatted_input = self._format_input(prompt, df)
        try:
            # First get the token count
            count_response = self.llm.count_tokens(contents=formatted_input)
            input_tokens = count_response.total_tokens
            
            # Then generate the response
            response = self.llm.generate_content(formatted_input)
            text = response.text
            
            # Log the token usage
            logger.info(f"Token count for Gemini API call - Input: {input_tokens}")
            
            # Note: The Gemini API doesn't provide output token count in the response
            # For now, we'll return input_tokens as the token count
            # You might want to estimate output tokens or find another way to get this info
            return text, input_tokens
            
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            raise RuntimeError(f"Gemini call failed: {e}")