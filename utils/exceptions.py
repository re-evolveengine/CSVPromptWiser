# utils/exceptions.py
class TokenBudgetExceededError(Exception):
    """Raised when the token usage would exceed the remaining allowed tokens."""

    def __init__(self, used_tokens: int, remaining_tokens: int):
        message = (
            f"Token usage ({used_tokens}) exceeds remaining tokens ({remaining_tokens})."
        )
        super().__init__(message)
        self.used_tokens = used_tokens
        self.remaining_tokens = remaining_tokens
