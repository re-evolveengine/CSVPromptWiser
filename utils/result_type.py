from enum import Enum, auto


class ResultType(Enum):
    SUCCESS = auto()
    RETRYABLE_ERROR = auto()
    FATAL_ERROR = auto()
    UNEXPECTED_ERROR = auto()
    NO_MORE_CHUNKS = auto()
    TOKENS_EXCEEDED = auto()