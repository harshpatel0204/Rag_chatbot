"""
Utility Functions

This module contains helper functions used across the application.
"""

from typing import List, Dict
import logging
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(__file__).rsplit('\\', 2)[0])
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, VERBOSE_LOGGING, LOG_FILE

# Set up logging
def setup_logging():
    """Configure logging for the application."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
    
    handlers = [logging.StreamHandler()]
    
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def chunk_text(
    text: str, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into chunks using recursive character splitting.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    logger.debug(f"Split text into {len(chunks)} chunks")
    
    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split documents into smaller chunks while preserving metadata.
    
    Args:
        documents: List of document dicts with 'content' and 'metadata'
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        List of chunked documents with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_documents = []
    
    for doc in documents:
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})
        
        # Split the content
        chunks = text_splitter.split_text(content)
        
        # Create new documents for each chunk
        for i, chunk in enumerate(chunks):
            chunked_doc = {
                'content': chunk,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            }
            chunked_documents.append(chunked_doc)
    
    logger.info(f"Chunked {len(documents)} documents into {len(chunked_documents)} chunks")
    return chunked_documents


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: The text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: The original filename
        
    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename
