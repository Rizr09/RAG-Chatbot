"""
Document processing module for loading and chunking PDF documents.
"""

import os
from typing import List
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all loaded documents
        """
        all_documents = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return all_documents
        
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return all_documents
        
        for filename in pdf_files:
            file_path = os.path.join(directory_path, filename)
            documents = self.load_pdf(file_path)
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source_file'] = filename
                doc.metadata['source_path'] = file_path
            
            all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {len(pdf_files)} PDF files")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        try:
            chunked_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            return []
    
    def process_documents(self, directory_path: str) -> List[Document]:
        """
        Complete document processing pipeline.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of processed and chunked documents
        """
        # Load documents
        documents = self.load_documents_from_directory(directory_path)
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        return chunked_documents
