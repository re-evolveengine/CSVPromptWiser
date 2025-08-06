from dataclasses import dataclass

@dataclass
class ChunkProcessState:
    processed_current_chunk_count: int = 0
    current_chunk_count: int = 5
    processed_total_chunk_count: int = 0
    total_chunk_count: int = 100
    remaining_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CurrentChunkState:
    processed_chunk_count: int = 0
    total_chunk_count: int = 0

