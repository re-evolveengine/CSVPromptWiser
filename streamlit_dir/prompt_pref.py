import json
from pathlib import Path

PROMPT_PREF_PATH = Path(".prompt_pref.json")


class PromptPreference:
    def __init__(self):
        self.file_path = PROMPT_PREF_PATH

    def save_prompt(self, prompt: str):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt}, f)

    def load_prompt(self) -> str:
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("prompt", "")
        return ""
