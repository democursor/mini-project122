from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
from datetime import datetime
import json


@dataclass
class Entity:
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    normalized_form: Optional[str] = None


@dataclass
class Keyphrase:
    phrase: str
    score: float
    ngram_length: int = 1


@dataclass
class ConceptExtractionResult:
    chunk_id: str
    document_id: str
    entities: List[Entity]
    keyphrases: List[Keyphrase]
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
