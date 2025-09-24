# core/clients/gemini_model_provider.py

from typing import List

import google.generativeai as genai


class GeminiModelProvider:
    """
    Fetches usable Gemini models via a dummy prompt check.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def _test_model(self, model_name: str) -> bool:
        """Returns True if model responds to a dummy prompt."""
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello")
            return bool(response.text.strip())
        except Exception:
            return False

    def get_usable_model_names(self) -> List[str]:
        """Returns names of Gemini models that support generateContent and are working."""
        models = genai.list_models()
        working_models = []

        # Iterate models without external progress utilities to avoid extra deps
        for model in models:
            if "generateContent" not in model.supported_generation_methods:
                continue
            if self._test_model(model.name):
                parts = model.name.split("/")
                if len(parts) > 1:
                    working_models.append(parts[1].strip())
                else:
                    working_models.append(model.name.strip())

        return working_models
