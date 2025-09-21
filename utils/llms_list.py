from enum import Enum

class LLMProvider(Enum):
    GEMINI = "Gemini"
    CHATGPT = "ChatGPT"
    DEEPSEEK = "DeepSeek"
    GROK = "Grok"

    @property
    def is_available(self) -> bool:
        return self in {LLMProvider.GEMINI}

    @property
    def display_name(self) -> str:
        return self.value if self.is_available else f"{self.value} (Coming Soon)"
