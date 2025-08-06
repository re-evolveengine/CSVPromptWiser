from dataclasses import dataclass

@dataclass
class ChunkProcessorState:
    processed_chunk_count: int = 0
    chunk_count: int = 100
    remaining_tokens: int = 0
    total_tokens: int = 1  # Default to 1 to avoid division by zero


@dataclass
class CurrentChunkState:
    processed_chunk_count: int = 0
    total_chunk_count: int = 0

