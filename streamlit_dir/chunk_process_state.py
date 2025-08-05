from dataclasses import dataclass

@dataclass
class ChunkProcessorState:
    current_chunk: int = 0
    batch_total: int = 1
    remaining_tokens: int = 0
    total_tokens: int = 1  # Default to 1 to avoid division by zero
