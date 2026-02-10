from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PaperNode:
    """Represents a paper node in the knowledge graph"""
    id: str
    title: str
    abstract: str
    year: Optional[int]
    page_count: int
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'year': self.year,
            'page_count': self.page_count
        }


@dataclass
class ConceptNode:
    """Represents a concept node in the knowledge graph"""
    id: str
    name: str
    normalized_name: str
    type: str
    frequency: int = 1
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'normalized_name': self.normalized_name,
            'type': self.type,
            'frequency': self.frequency
        }


@dataclass
class MentionsRelationship:
    """Represents a MENTIONS relationship between paper and concept"""
    paper_id: str
    concept_name: str
    frequency: int
    confidence: float
    
    def to_dict(self):
        return {
            'paper_id': self.paper_id,
            'concept_name': self.concept_name,
            'frequency': self.frequency,
            'confidence': self.confidence
        }
