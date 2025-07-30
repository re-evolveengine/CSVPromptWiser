from typing import Any
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

    def call(self, prompt: str, df: pd.DataFrame) -> str:
        """
        Override base method to use `generate_content` (Gemini-specific).
        """
        formatted_input = self._format_input(prompt, df)
        try:
            response = self.llm.generate_content(formatted_input)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {str(e)}")

    def reconfigure(self, new_config: dict):
        """
        Optional: Update generation config and reinitialize model.
        """
        self.generation_config = new_config
        self.llm = self._init_llm()
