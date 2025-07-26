# core/clients/gemini_client.py

from typing import Any
import pandas as pd
import google.generativeai as genai

from model.core.base_llm_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Gemini LLM client using the official google-generativeai SDK.
    """

    def _init_llm(self) -> Any:
        try:
            genai.configure(api_key=self.api_key)

            return genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def call(self, prompt: str, df: pd.DataFrame) -> str:
        """
        Format input and call Gemini using `generate_content`.
        """
        formatted_input = self._format_input(prompt, df)
        try:
            response = self.llm.generate_content(formatted_input)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {str(e)}")
