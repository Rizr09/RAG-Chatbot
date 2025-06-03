import os
import logging
from typing import List
from document_processor import DocumentProcessor
from vector_store import VectorStore

logger = logging.getLogger(__name__)

def process_and_add_documents(vector_store: VectorStore, documents_dir: str) -> bool:
    """Helper function to process documents and add them to the vector store."""
    logger.info("Vector store is empty. Processing documents...")
    if not os.path.exists(documents_dir):
        logger.error(f"Documents directory '{documents_dir}' not found.")
        return False
    pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{documents_dir}'.")
        return False
        
    doc_processor = DocumentProcessor() # Assuming default chunk_size/overlap is fine here
    processed_docs = doc_processor.process_documents(documents_dir)
    if processed_docs:
        vector_store.add_documents(processed_docs)
        logger.info("Documents processed and added to vector store.")
        return True
    else:
        logger.error("Failed to process documents. RAG system might not function correctly.")
        return False 