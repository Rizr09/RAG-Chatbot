"""
Initialization script to set up and test the RAG system.
"""

import os
import sys
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem

def main():
    """Initialize and test the RAG system."""
    print("🚀 Initializing Market Outlook RAG System...")
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        sys.exit(1)
    
    print("✅ API key found")
    
    # Check documents directory
    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        print(f"❌ Error: Documents directory not found: {documents_dir}")
        sys.exit(1)
    
    pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {documents_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        print(f"   • {pdf_file}")
    
    try:
        # Initialize components
        print("\n📚 Initializing document processor...")
        doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        print("🔗 Initializing vector store...")
        vector_store = VectorStore(api_key=api_key)
        
        print("🤖 Initializing RAG system...")
        rag_system = RAGSystem(api_key=api_key, vector_store=vector_store)
        
        # Check if documents are already processed
        doc_count = vector_store.get_collection_count()
        
        if doc_count == 0:
            print("\n📖 Processing documents...")
            processed_docs = doc_processor.process_documents(documents_dir)
            
            if processed_docs:
                print(f"📊 Adding {len(processed_docs)} document chunks to vector store...")
                vector_store.add_documents(processed_docs)
                print("✅ Documents processed and indexed successfully")
            else:
                print("❌ Error: Failed to process documents")
                sys.exit(1)
        else:
            print(f"✅ Vector store already contains {doc_count} documents")
          # Test the system with a sample question
        print("\n🧪 Testing the system...")
        test_question = "What are the key market trends mentioned in the documents?"
        print(f"Question: {test_question}")
        
        # Try to reinitialize QA chain if needed
        if not rag_system.qa_chain:
            print("🔄 Reinitializing QA chain...")
            rag_system._reinitialize_qa_chain()
        
        response = rag_system.ask_question(test_question)
        
        if "error" not in response:
            print("\n✅ System test successful!")
            print(f"Answer preview: {response['answer'][:200]}...")
            print(f"Sources found: {len(response['source_documents'])}")
        else:
            print(f"❌ System test failed: {response['error']}")
            # Don't exit, as the web interface might still work
            print("⚠️ The web interface may still function properly")
        
        print("\n🎉 RAG system is ready!")
        print("💡 Run 'streamlit run app.py' to start the web interface")
        
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
