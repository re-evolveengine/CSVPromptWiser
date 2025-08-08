import json

from utils.constants import PROMPT_PREF_PATH


class PromptPreference:
    def __init__(self):
        self.file_path = PROMPT_PREF_PATH
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_prompt(self, prompt: str):
        data = self._load_all()
        data["prompt"] = prompt
        self._save_all(data)

    def load_prompt(self) -> str:
        data = self._load_all()
        return data.get("prompt", "")

    def save_example_response(self, example_response: str):
        data = self._load_all()
        data["example_response"] = example_response
        self._save_all(data)

    def load_example_response(self) -> str:
        data = self._load_all()
        return data.get("example_response", "")

    def _load_all(self) -> dict:
        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_all(self, data: dict):
        with self.file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
