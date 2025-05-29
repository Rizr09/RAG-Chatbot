"""
Vector store module for document embeddings and similarity search.
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            api_key: Google Gemini API key
            persist_directory: Directory to persist the vector database
        """
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Initialize Chroma vector store
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma with persistent storage
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="market_outlook_docs"
            )
            logger.info(f"Vector store initialized with persist directory: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        try:
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                return []
            
            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} similar documents for query: {query[:50]}...")
            
            return docs
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of tuples (document, score)
        """
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                return []
            
            # Perform similarity search with scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(docs_with_scores)} similar documents with scores")
            
            return docs_with_scores
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            return []
    
    def get_retriever(self, k: int = 4):
        """
        Get a retriever for the vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever object
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            if not self.vectorstore:
                return 0
            
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            logger.info(f"Collection contains {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0
