from typing import Any, Tuple
import pandas as pd
import google.generativeai as genai

from model.core.llms.base_llm_client import BaseLLMClient


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
            response = self.llm.generate_content(formatted_input)
            text = response.text
            # Handle case where usage_metadata is not available
            try:
                token_count = getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
            except AttributeError:
                token_count = 0  # Fallback if usage_metadata is not available
            return text, token_count
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {e}")
