from dataclasses import dataclass, field, asdict
from typing import List, Optional
import uuid
from datetime import datetime


@dataclass
class ChunkingConfig:
    min_tokens: int = 100
    max_tokens: int = 500
    similarity_threshold: float = 0.7
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class Chunk:
    text: str
    sentences: List[str]
    token_count: int
    start_sentence: int
    end_sentence: int
    document_id: str
    section_heading: Optional[str] = None
    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4()}")
    chunk_type: str = "content"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)
