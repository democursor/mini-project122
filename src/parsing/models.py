from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Metadata:
    title: str = "Unknown"
    authors: List[str] = field(default_factory=lambda: ["Unknown"])
    year: Optional[int] = None
    abstract: str = ""


@dataclass
class Section:
    heading: str
    content: str
    start_page: int
    end_page: int


@dataclass
class ParsedDocument:
    document_id: str
    metadata: Metadata
    sections: List[Section]
    full_text: str
    page_count: int
    parsing_date: str = ""
    
    def to_dict(self):
        return asdict(self)
