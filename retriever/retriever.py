"""
Retriever Module

This module handles retrieving relevant documents from the vector store
based on query similarity.
"""

from typing import List, Dict, Optional, Tuple
import logging

import sys
sys.path.append(str(__file__).rsplit('\\', 2)[0])
from config import TOP_K_DOCUMENTS, SIMILARITY_THRESHOLD
from vectorstore.store import get_vector_store, VectorStore

# Set up logging
logger = logging.getLogger(__name__)


class Retriever:
    """
    Document retriever that searches the vector store.
    
    Features:
    - Similarity-based retrieval
    - Score threshold filtering
    - Context formatting for LLM
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store instance (uses global if not provided)
        """
        self.vector_store = vector_store or get_vector_store()
        self.top_k = TOP_K_DOCUMENTS
        self.threshold = SIMILARITY_THRESHOLD
    
    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of documents to retrieve (uses default if not provided)
            threshold: Minimum similarity score (uses default if not provided)
            
        Returns:
            List of relevant documents with metadata and scores
        """
        k = k or self.top_k
        threshold = threshold or self.threshold
        
        # Search the vector store
        results = self.vector_store.search(query, k=k)
        
        # Filter by threshold
        filtered_results = []
        for doc, score in results:
            if score >= threshold:
                filtered_results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score
                })
        
        logger.info(f"Retrieved {len(filtered_results)} documents above threshold {threshold}")
        return filtered_results
    
    def retrieve_with_context(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve documents and format them as context for LLM.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (formatted_context, documents)
        """
        documents = self.retrieve(query, k=k)
        
        if not documents:
            return "", []
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page', 'N/A')
            content = doc['content']
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source}, Page: {page}\n"
                f"Content:\n{content}\n"
            )
        
        formatted_context = "\n---\n".join(context_parts)
        
        return formatted_context, documents
    
    def has_documents(self) -> bool:
        """Check if the vector store has any documents."""
        return self.vector_store.get_document_count() > 0
    
    def get_sources(self) -> List[str]:
        """Get list of source documents in the store."""
        return self.vector_store.get_sources()


# Global retriever instance
_retriever = None

def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve_context(query: str, k: int = TOP_K_DOCUMENTS) -> Tuple[str, List[Dict]]:
    """
    Convenience function to retrieve context for a query.
    
    Args:
        query: The search query
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (formatted_context, documents)
    """
    return get_retriever().retrieve_with_context(query, k=k)
