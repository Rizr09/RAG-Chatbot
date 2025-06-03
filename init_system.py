"""
Initialization script to set up and test the RAG system.
"""

import os
import sys
from dotenv import load_dotenv
from vector_store import VectorStore
from rag_system import RAGSystem
from utils import process_and_add_documents

def main():
    """Initialize and test the RAG system."""
    print("Initializing Telecommunication, Informatics, Cyber, and Internet Law RAG System...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file")
        sys.exit(1)
    
    print("API key found")
    
    # Check documents directory
    documents_dir = "./documents_retrieval"
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory not found: {documents_dir}")
        sys.exit(1)
    
    pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in {documents_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        print(f"   • {pdf_file}")
    
    try:
        # Initialize components
        print("\nInitializing vector store...")
        vector_store = VectorStore(api_key=api_key)
        
        print("Initializing RAG system...")
        rag_system = RAGSystem(api_key=api_key, vector_store=vector_store)
        
        # Check if documents are already processed
        doc_count = vector_store.get_collection_count()
        
        if doc_count == 0:
            print("\nProcessing documents...")
            success = process_and_add_documents(vector_store, documents_dir)
            
            if not success:
                print("Error: Failed to process documents. Exiting.")
                sys.exit(1)
            print("Documents processed and indexed successfully")
        else:
            print(f"Vector store already contains {doc_count} documents")
          # Test the system with a sample question
        print("\nTesting the system...")
        test_question = "Apa saja peraturan terkait perlindungan data pribadi di Indonesia?"
        print(f"Question: {test_question}")
        
        response = rag_system.chat_with_context(test_question)
        
        if "error" not in response:
            print("\nSystem test successful!")
            print(f"Answer preview: {response['answer'][:200]}...")
            print(f"Sources found: {len(response['source_documents'])}")
        else:
            print(f"System test failed: {response['error']}")
            print("The web interface may still function properly")
        
        print("\nRAG system is ready!")
        print("Run 'streamlit run app.py' to start the web interface")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
