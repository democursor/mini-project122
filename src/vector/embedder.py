"""Embedding generation module"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import logging
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    max_seq_length: int = 512


class EmbeddingGenerator:
    """Generates semantic embeddings for text chunks"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()
    
    def _load_model(self) -> SentenceTransformer:
        """Load and configure sentence transformer model"""
        model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device
        )
        model.max_seq_length = self.config.max_seq_length
        
        self.logger.info(f"Loaded embedding model: {self.config.model_name}")
        return model
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        self.logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()
