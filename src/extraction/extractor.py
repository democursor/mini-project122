import logging
import time
from typing import List

from .models import Entity, Keyphrase, ConceptExtractionResult

logger = logging.getLogger(__name__)


class ConceptExtractor:
    def __init__(self):
        self.ner_model = None
        self.kw_model = None
        
    def _load_ner_model(self):
        if self.ner_model is None:
            try:
                import spacy
                try:
                    self.ner_model = spacy.load("en_core_sci_md")
                    logger.info("Loaded scientific NER model")
                except OSError:
                    logger.warning("Scientific model not found, using general model")
                    self.ner_model = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                raise
    
    def _load_keyphrase_model(self):
        if self.kw_model is None:
            try:
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                
                sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.kw_model = KeyBERT(model=sentence_model)
                logger.info("Loaded KeyBERT model")
            except Exception as e:
                logger.error(f"Failed to load KeyBERT: {e}")
                raise
    
    def extract(self, chunk_id: str, document_id: str, text: str) -> ConceptExtractionResult:
        start_time = time.time()
        
        try:
            entities = self._extract_entities(text)
            keyphrases = self._extract_keyphrases(text)
            
            processing_time = time.time() - start_time
            
            result = ConceptExtractionResult(
                chunk_id=chunk_id,
                document_id=document_id,
                entities=entities,
                keyphrases=keyphrases,
                processing_time=processing_time
            )
            
            logger.info(f"Extracted {len(entities)} entities and {len(keyphrases)} keyphrases")
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ConceptExtractionResult(
                chunk_id=chunk_id,
                document_id=document_id,
                entities=[],
                keyphrases=[],
                processing_time=time.time() - start_time
            )
    
    def _extract_entities(self, text: str) -> List[Entity]:
        try:
            self._load_ner_model()
            doc = self.ner_model(text)
            
            entities = []
            for ent in doc.ents:
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.8,
                    normalized_form=ent.text.lower()
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _extract_keyphrases(self, text: str, top_k: int = 10) -> List[Keyphrase]:
        try:
            self._load_keyphrase_model()
            
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_k,
                use_mmr=True,
                diversity=0.5
            )
            
            keyphrases = []
            for phrase, score in keywords:
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
