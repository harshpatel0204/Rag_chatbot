"""
Embedding Module

This module handles generating embeddings using sentence-transformers.
It uses the all-MiniLM-L6-v2 model which is fast and produces quality embeddings.
"""

from typing import List, Union
import logging
import numpy as np

from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(__file__).rsplit('\\', 2)[0])
from config import EMBEDDING_MODEL

# Set up logging
logger = logging.getLogger(__name__)


class Embedder:
    """
    A class to generate embeddings using sentence-transformers.
    
    Features:
    - Lazy loading of the model to save memory
    - Batch embedding generation
    - Compatible with FAISS vector store
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading the model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedder (model loaded on first use)."""
        self.model_name = EMBEDDING_MODEL
        
    def _load_model(self):
        """Load the embedding model if not already loaded."""
        if Embedder._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            Embedder._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the embedding model, loading it if necessary."""
        self._load_model()
        return Embedder._model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            Numpy array of the embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Integer dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        return self.model.get_sentence_embedding_dimension()


# Global embedder instance for convenience
_embedder = None

def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def embed_text(text: str) -> np.ndarray:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector as numpy array
    """
    return get_embedder().get_embedding(text)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        Numpy array of embeddings
    """
    return get_embedder().get_embeddings(texts)
