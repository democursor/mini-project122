import logging
import time
from typing import List, Optional, Dict
import re

from .models import Entity, Keyphrase, ConceptExtractionResult

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """
    Enhanced concept extractor with domain-specific models
    Supports multiple domains: scientific, biomedical, computer science
    """
    
    def __init__(self, domain: str = "scientific", use_domain_models: bool = True):
        """
        Initialize extractor with domain-specific configuration
        
        Args:
            domain: Target domain (scientific, biomedical, computer_science, general)
            use_domain_models: Whether to use domain-specific models
        """
        self.domain = domain
        self.use_domain_models = use_domain_models
        self.ner_model = None
        self.kw_model = None
        self.domain_patterns = self._load_domain_patterns()
        
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-specific regex patterns for concept extraction"""
        patterns = {
            "scientific": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:theorem|hypothesis|principle|law|effect)\b',
                r'\b(?:algorithm|method|approach|technique|framework)\b',
                r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b',  # Acronyms
            ],
            "biomedical": [
                r'\b(?:protein|gene|enzyme|receptor|antibody|antigen)\b',
                r'\b[A-Z][A-Z0-9]{2,}\b',  # Gene names
                r'\b(?:disease|syndrome|disorder|condition|pathology)\b',
                r'\b(?:treatment|therapy|drug|medication|intervention)\b',
            ],
            "computer_science": [
                r'\b(?:neural network|deep learning|machine learning|AI|ML|DL)\b',
                r'\b(?:algorithm|data structure|complexity|optimization)\b',
                r'\b(?:CNN|RNN|LSTM|GAN|BERT|GPT|Transformer)\b',
                r'\b(?:training|inference|model|architecture)\b',
            ]
        }
        return patterns.get(self.domain, patterns["scientific"])
    
    def _load_ner_model(self):
        """Load domain-specific NER model with fallback"""
        if self.ner_model is None:
            try:
                import spacy
                
                # Try domain-specific models first
                if self.use_domain_models:
                    model_priority = self._get_model_priority()
                    
                    for model_name in model_priority:
                        try:
                            self.ner_model = spacy.load(model_name)
                            logger.info(f"Loaded domain-specific NER model: {model_name}")
                            return
                        except OSError:
                            logger.debug(f"Model {model_name} not available, trying next")
                            continue
                
                # Fallback to general model
                self.ner_model = spacy.load("en_core_web_sm")
                logger.info("Loaded general NER model (en_core_web_sm)")
                
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                raise
    
    def _get_model_priority(self) -> List[str]:
        """Get prioritized list of models based on domain"""
        model_map = {
            "scientific": ["en_core_sci_md", "en_core_sci_sm", "en_core_web_md"],
            "biomedical": ["en_ner_bc5cdr_md", "en_core_sci_md", "en_core_web_md"],
            "computer_science": ["en_core_sci_md", "en_core_web_md"],
            "general": ["en_core_web_md", "en_core_web_sm"]
        }
        return model_map.get(self.domain, model_map["scientific"])
    
    def _load_keyphrase_model(self):
        """Load domain-specific keyphrase extraction model"""
        if self.kw_model is None:
            try:
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                
                # Select embedding model based on domain
                embedding_model = self._get_embedding_model()
                sentence_model = SentenceTransformer(embedding_model)
                self.kw_model = KeyBERT(model=sentence_model)
                logger.info(f"Loaded KeyBERT with {embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load KeyBERT: {e}")
                raise
    
    def _get_embedding_model(self) -> str:
        """Get domain-specific embedding model"""
        # If use_domain_models is False, always use lightweight general model
        if not self.use_domain_models:
            return "all-MiniLM-L6-v2"
        
        embedding_map = {
            "scientific": "allenai/scibert_scivocab_uncased",
            "biomedical": "dmis-lab/biobert-base-cased-v1.1",
            "computer_science": "allenai/scibert_scivocab_uncased",
            "general": "all-MiniLM-L6-v2"
        }
        
        model = embedding_map.get(self.domain, "all-MiniLM-L6-v2")
        
        # Fallback to general model if domain model not available
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(model)
            return model
        except Exception as e:
            logger.warning(f"Domain model {model} not available, using general model: {e}")
            return "all-MiniLM-L6-v2"
    
    def extract(self, chunk_id: str, document_id: str, text: str) -> ConceptExtractionResult:
        """
        Extract concepts with domain-specific enhancements
        
        Args:
            chunk_id: Unique chunk identifier
            document_id: Document identifier
            text: Text to extract concepts from
            
        Returns:
            ConceptExtractionResult with entities and keyphrases
        """
        start_time = time.time()
        
        try:
            # Extract entities using NER
            entities = self._extract_entities(text)
            
            # Extract keyphrases using KeyBERT
            keyphrases = self._extract_keyphrases(text)
            
            # Extract domain-specific patterns
            pattern_concepts = self._extract_pattern_concepts(text)
            
            # Merge and deduplicate
            all_keyphrases = self._merge_keyphrases(keyphrases, pattern_concepts)
            
            processing_time = time.time() - start_time
            
            result = ConceptExtractionResult(
                chunk_id=chunk_id,
                document_id=document_id,
                entities=entities,
                keyphrases=all_keyphrases,
                processing_time=processing_time
            )
            
            logger.info(
                f"Extracted {len(entities)} entities and {len(all_keyphrases)} keyphrases "
                f"(domain: {self.domain})"
            )
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return ConceptExtractionResult(
                chunk_id=chunk_id,
                document_id=document_id,
                entities=[],
                keyphrases=[],
                processing_time=time.time() - start_time
            )
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities with improved confidence scoring"""
        try:
            self._load_ner_model()
            doc = self.ner_model(text)
            
            entities = []
            seen_texts = set()  # Deduplicate
            
            for ent in doc.ents:
                # Skip if already seen
                normalized = ent.text.lower().strip()
                if normalized in seen_texts or len(normalized) < 2:
                    continue
                
                seen_texts.add(normalized)
                
                # Calculate confidence based on entity properties
                confidence = self._calculate_entity_confidence(ent)
                
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                    normalized_form=normalized
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _calculate_entity_confidence(self, ent) -> float:
        """Calculate confidence score for entity"""
        base_confidence = 0.7
        
        # Boost for longer entities (more specific)
        if len(ent.text.split()) > 1:
            base_confidence += 0.1
        
        # Boost for capitalized entities
        if ent.text[0].isupper():
            base_confidence += 0.05
        
        # Boost for specific entity types
        high_confidence_types = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT'}
        if ent.label_ in high_confidence_types:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _extract_pattern_concepts(self, text: str) -> List[Keyphrase]:
        """Extract concepts using domain-specific regex patterns"""
        concepts = []
        seen_phrases = set()
        
        for pattern in self.domain_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                phrase = match.group().strip()
                normalized = phrase.lower()
                
                if normalized not in seen_phrases and len(phrase) > 3:
                    seen_phrases.add(normalized)
                    concepts.append(Keyphrase(
                        phrase=phrase,
                        score=0.75,  # Pattern-based concepts get fixed score
                        ngram_length=len(phrase.split())
                    ))
        
        return concepts
    
    def _extract_keyphrases(self, text: str, top_k: int = 15) -> List[Keyphrase]:
        """Extract keyphrases with domain-aware settings"""
        try:
            self._load_keyphrase_model()
            
            # Domain-specific parameters
            ngram_range = self._get_ngram_range()
            diversity = self._get_diversity_score()
            
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=ngram_range,
                stop_words='english',
                top_n=top_k,
                use_mmr=True,
                diversity=diversity
            )
            
            keyphrases = []
            for phrase, score in keywords:
                # Filter low-quality keyphrases
                if self._is_valid_keyphrase(phrase, score):
                    kp = Keyphrase(
                        phrase=phrase,
                        score=float(score),
                        ngram_length=len(phrase.split())
                    )
                    keyphrases.append(kp)
            
            return keyphrases
        except Exception as e:
            logger.error(f"Keyphrase extraction failed: {e}")
            return []
    
    def _get_ngram_range(self) -> tuple:
        """Get domain-specific n-gram range"""
        ngram_map = {
            "scientific": (1, 4),  # Longer phrases for scientific terms
            "biomedical": (1, 3),
            "computer_science": (1, 3),
            "general": (1, 3)
        }
        return ngram_map.get(self.domain, (1, 3))
    
    def _get_diversity_score(self) -> float:
        """Get domain-specific diversity score for MMR"""
        diversity_map = {
            "scientific": 0.6,  # Higher diversity for scientific papers
            "biomedical": 0.5,
            "computer_science": 0.5,
            "general": 0.5
        }
        return diversity_map.get(self.domain, 0.5)
    
    def _is_valid_keyphrase(self, phrase: str, score: float) -> bool:
        """Validate keyphrase quality"""
        # Minimum score threshold
        if score < 0.3:
            return False
        
        # Minimum length
        if len(phrase) < 3:
            return False
        
        # Not just numbers
        if phrase.replace('.', '').replace(',', '').isdigit():
            return False
        
        # Not just stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = phrase.lower().split()
        if all(word in stopwords for word in words):
            return False
        
        return True
    
    def _merge_keyphrases(self, keyphrases: List[Keyphrase], 
                         pattern_concepts: List[Keyphrase]) -> List[Keyphrase]:
        """Merge and deduplicate keyphrases from different sources"""
        merged = {}
        
        # Add KeyBERT keyphrases
        for kp in keyphrases:
            normalized = kp.phrase.lower().strip()
            merged[normalized] = kp
        
        # Add pattern-based concepts (don't override higher scores)
        for kp in pattern_concepts:
            normalized = kp.phrase.lower().strip()
            if normalized not in merged or merged[normalized].score < kp.score:
                merged[normalized] = kp
        
        # Sort by score and return top results
        result = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return result[:20]  # Limit to top 20
