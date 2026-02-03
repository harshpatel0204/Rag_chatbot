"""
Vector Store Module

This module handles storing and retrieving document embeddings using FAISS.
It provides local persistence so embeddings don't need to be regenerated.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

import faiss

import sys
sys.path.append(str(__file__).rsplit('\\', 2)[0])
from config import VECTORSTORE_DIR
from embeddings.embedder import get_embedder

# Set up logging
logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store with local persistence.
    
    Features:
    - Add documents with automatic embedding generation
    - Similarity search with score filtering
    - Save/load to disk for persistence
    - Reset functionality
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize the vector store.
        
        Args:
            store_path: Directory to store the index and metadata
        """
        self.store_path = Path(store_path) if store_path else VECTORSTORE_DIR
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.store_path / "faiss.index"
        self.metadata_path = self.store_path / "metadata.pkl"
        
        self.embedder = get_embedder()
        self.index = None
        self.documents = []  # Store document content and metadata
        
        # Try to load existing store
        if self.exists():
            self.load()
    
    def exists(self) -> bool:
        """Check if a saved vector store exists."""
        return self.index_path.exists() and self.metadata_path.exists()
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Extract text content
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedder.get_embeddings(texts)
        
        # Initialize index if needed
        if self.index is None:
            embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dim)
            logger.info(f"Created new FAISS index with dimension {embedding_dim}")
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        
        # Save to disk
        self.save()
    
    def search(self, query: str, k: int = 4) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than we have
        distances, indices = self.index.search(query_embedding, k)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                # Normalize to 0-1 range approximately
                score = 1.0 / (1.0 + distances[0][i])
                results.append((doc, score))
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self) -> None:
        """Save the vector store to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save document metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved vector store to {self.store_path}")
    
    def load(self) -> bool:
        """
        Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.exists():
                logger.info("No existing vector store found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            
            # Load document metadata
            with open(self.metadata_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            logger.info(f"Loaded vector store with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.reset()
            return False
    
    def reset(self) -> None:
        """Reset the vector store, clearing all data."""
        self.index = None
        self.documents = []
        
        # Delete saved files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        
        logger.info("Vector store reset")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the store."""
        return len(self.documents)
    
    def get_sources(self) -> List[str]:
        """Get unique source names (PDF filenames) in the store."""
        sources = set()
        for doc in self.documents:
            if 'metadata' in doc and 'source' in doc['metadata']:
                sources.add(doc['metadata']['source'])
        return list(sources)


# Global store instance
_store = None

def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store() -> None:
    """Reset the global vector store."""
    global _store
    if _store is not None:
        _store.reset()
    _store = None
