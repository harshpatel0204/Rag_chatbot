"""
PDF Loading Module

This module handles loading and extracting text from PDF documents.
It supports both single and multiple PDF files.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

from PyPDF2 import PdfReader

# Set up logging
logger = logging.getLogger(__name__)


class PDFLoader:
    """
    A class to load and extract text from PDF documents.
    
    Features:
    - Load single or multiple PDFs
    - Extract text with page-level metadata
    - Handle corrupted or password-protected PDFs gracefully
    """
    
    def __init__(self):
        """Initialize the PDF loader."""
        pass
    
    def load_pdf(self, file_path: str) -> List[Dict]:
        """
        Load a single PDF file and extract text from all pages.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing:
            - 'content': The text content of the page
            - 'metadata': Dictionary with 'source' and 'page' info
        """
        documents = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"PDF file not found: {file_path}")
            return documents
        
        try:
            reader = PdfReader(str(file_path))
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    
                    # Skip empty pages
                    if text and text.strip():
                        documents.append({
                            'content': text.strip(),
                            'metadata': {
                                'source': file_path.name,
                                'page': page_num + 1,
                                'total_pages': len(reader.pages)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {file_path.name}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            
        return documents
    
    def load_pdf_from_bytes(self, file_bytes, filename: str) -> List[Dict]:
        """
        Load a PDF from bytes (useful for Streamlit file uploads).
        
        Args:
            file_bytes: Bytes of the PDF file
            filename: Original filename for metadata
            
        Returns:
            List of dictionaries containing page content and metadata
        """
        documents = []
        
        try:
            from io import BytesIO
            pdf_stream = BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    
                    if text and text.strip():
                        documents.append({
                            'content': text.strip(),
                            'metadata': {
                                'source': filename,
                                'page': page_num + 1,
                                'total_pages': len(reader.pages)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {filename}: {e}")
                    continue
                    
            logger.info(f"Successfully loaded {len(documents)} pages from {filename}")
            
        except Exception as e:
            logger.error(f"Error loading PDF from bytes ({filename}): {e}")
            
        return documents
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Dict]:
        """
        Load multiple PDF files.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            Combined list of all documents from all PDFs
        """
        all_documents = []
        
        for file_path in file_paths:
            documents = self.load_pdf(file_path)
            all_documents.extend(documents)
            
        logger.info(f"Loaded {len(all_documents)} total pages from {len(file_paths)} PDFs")
        return all_documents


# Convenience function for quick loading
def load_pdf(file_path: str) -> List[Dict]:
    """
    Convenience function to load a single PDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of documents with content and metadata
    """
    loader = PDFLoader()
    return loader.load_pdf(file_path)


def load_multiple_pdfs(file_paths: List[str]) -> List[Dict]:
    """
    Convenience function to load multiple PDFs.
    
    Args:
        file_paths: List of paths to PDF files
        
    Returns:
        Combined list of all documents
    """
    loader = PDFLoader()
    return loader.load_multiple_pdfs(file_paths)
