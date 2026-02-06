import logging
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class SemanticChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.config.model_name)
                logger.info(f"Loaded embedding model: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def chunk_document(self, document_id: str, text: str, 
                      section_heading: str = None) -> List[Chunk]:
        try:
            self._load_model()
            
            sentences = self._split_sentences(text)
            
            if len(sentences) < 2:
                return [self._create_single_chunk(document_id, text, sentences, section_heading)]
            
            embeddings = self.model.encode(sentences, convert_to_numpy=True)
            boundaries = self._find_boundaries(embeddings)
            chunks = self._create_chunks(document_id, sentences, boundaries, section_heading)
            
            logger.info(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return [self._create_single_chunk(document_id, text, [text], section_heading)]
    
    def _split_sentences(self, text: str) -> List[str]:
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {e}, using simple split")
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def _find_boundaries(self, embeddings: np.ndarray) -> List[int]:
        boundaries = [0]
        
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0][0]
            
            if sim < self.config.similarity_threshold:
                boundaries.append(i + 1)
        
        boundaries.append(len(embeddings))
        return boundaries
    
    def _create_chunks(self, document_id: str, sentences: List[str], 
                      boundaries: List[int], section_heading: str) -> List[Chunk]:
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            token_count = len(chunk_text.split())
            
            if token_count >= self.config.min_tokens or i == len(boundaries) - 2:
                chunk = Chunk(
                    text=chunk_text,
                    sentences=chunk_sentences,
                    token_count=token_count,
                    start_sentence=start_idx,
                    end_sentence=end_idx,
                    document_id=document_id,
                    section_heading=section_heading
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_single_chunk(self, document_id: str, text: str, 
                            sentences: List[str], section_heading: str) -> Chunk:
        return Chunk(
            text=text,
            sentences=sentences,
            token_count=len(text.split()),
            start_sentence=0,
            end_sentence=len(sentences),
            document_id=document_id,
            section_heading=section_heading
        )
