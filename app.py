"""
Streamlit web application for the Market Outlook RAG Chatbot.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
import logging
import re
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI & Finance Research Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for full dark theme
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stApp > div {
        background-color: #0e1117;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #64b5f6;
        background-color: #1e1e1e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #fafafa;
    }
    
    .user-message {
        background-color: #1a237e;
        border-left-color: #3f51b5;
        color: #e8eaf6;
    }
    
    .assistant-message {
        background-color: #263238;
        border-left-color: #66bb6a;
        color: #e8f5e8;
    }
    
    .source-document {
        background-color: #2d2d2d;
        border: 1px solid #404040;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #e0e0e0;
    }
    
    .source-header {
        font-weight: 600;
        color: #64b5f6;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #404040;
        text-align: center;
        color: #fafafa;
    }
    
    .header-container {
        background-color: #1e1e1e;
        padding: 2rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        border-bottom: 3px solid #64b5f6;
        color: #fafafa;
    }
    
    .stTextInput > div > div > input {
        border-radius: 2rem;
        border: 2px solid #404040;
        padding: 0.75rem 1rem;
        background-color: #2d2d2d;
        color: #fafafa;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #64b5f6;
        outline: none;
    }
      .stButton > button {
        border-radius: 2rem;
        border: none;
        background-color: #1976d2;
        color: white;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-1px);
    }
      /* New Conversation button specific styling */
    .new-conversation-button {
        background: linear-gradient(135deg, #37474f, #455a64) !important;
        border: 1px solid #546e7a !important;
        color: #eceff1 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 0.75rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    .new-conversation-button:hover {
        background: linear-gradient(135deg, #455a64, #546e7a) !important;
        border-color: #607d8b !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    
    /* Confirmation buttons styling */
    .confirm-button {
        border-radius: 0.5rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .confirm-yes {
        background-color: #d32f2f !important;
        border-color: #f44336 !important;
        color: white !important;
    }
    
    .confirm-yes:hover {
        background-color: #b71c1c !important;
        transform: translateY(-1px) !important;
    }
    
    .confirm-no {
        background-color: #546e7a !important;
        border-color: #607d8b !important;
        color: white !important;
    }
    
    .confirm-no:hover {
        background-color: #455a64 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    
    .css-1lcbmhc {
        background-color: #1e1e1e;
    }
    
    /* Markdown text styling */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fafafa;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #64b5f6;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: #fafafa;
    }
    
    .streamlit-expanderContent {
        background-color: #1e1e1e;
        color: #fafafa;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > textarea {
        background-color: #2d2d2d;
        color: #fafafa;
        border: 2px solid #404040;
        border-radius: 1rem;
    }
    
    .stChatInput > div > div > textarea:focus {
        border-color: #64b5f6;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #1b5e20;
        color: #c8e6c9;
    }
    
    .stError {
        background-color: #b71c1c;
        color: #ffcdd2;
    }
    
    .stWarning {
        background-color: #e65100;
        color: #ffe0b2;
    }
    
    /* Remove any remaining white backgrounds */
    div[data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    div[data-testid="stToolbar"] {
        background-color: #0e1117;
    }
    
    div[data-testid="stHeader"] {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching."""
    try:
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("Gemini API key not found. Please check your .env file.")
            return None, None, None
          # Initialize components
        with st.spinner("Initializing document processor..."):
            doc_processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        with st.spinner("Initializing vector store..."):
            vector_store = VectorStore(api_key=api_key)
        
        with st.spinner("Initializing RAG system..."):
            rag_system = RAGSystem(api_key=api_key, vector_store=vector_store)
        
        return doc_processor, vector_store, rag_system
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None

@st.cache_resource
def load_documents():
    """Load and process documents with caching."""
    doc_processor, vector_store, rag_system = initialize_system()
    
    if not all([doc_processor, vector_store, rag_system]):
        return None, None, None, 0
    
    try:
        # Check if documents are already loaded
        doc_count = vector_store.get_collection_count()
          # Count actual PDF files for display
        documents_dir = "./documents_retrieval"
        pdf_file_count = 0
        if os.path.exists(documents_dir):
            pdf_files = [f for f in os.listdir(documents_dir) if f.lower().endswith('.pdf')]
            pdf_file_count = len(pdf_files)
        
        if doc_count > 0:
            st.success(f"Found {doc_count} document chunks from {pdf_file_count} PDF files already loaded in vector store")
            return doc_processor, vector_store, rag_system, pdf_file_count
        
        # Load documents if not already loaded
        if not os.path.exists(documents_dir):
            st.error(f"Documents directory not found: {documents_dir}")
            return None, None, None, 0
        
        with st.spinner("Processing documents..."):
            processed_docs = doc_processor.process_documents(documents_dir)
        
        if not processed_docs:
            st.warning("No documents found to process")
            return doc_processor, vector_store, rag_system, 0
        
        with st.spinner("Adding documents to vector store..."):
            vector_store.add_documents(processed_docs)
        
        doc_count = len(processed_docs)
        st.success(f"Successfully processed and indexed {doc_count} document chunks from {pdf_file_count} PDF files")
        
        return doc_processor, vector_store, rag_system, pdf_file_count
        
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None, None, None, 0

def main():
    """Main application function."""    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style="margin:0; color:#64b5f6; text-align:center;">AI & Finance Research Q&A</h1>
        <p style="margin:0.5rem 0 0 0; text-align:center; color:#b0bec5;">
            Ask questions about research papers in Artificial Intelligence and Finance.
        </p>
        <p style="margin:0.8rem 0 0 0; text-align:center; color:#78909c; font-size:0.85rem;">
            Made by /rizr09 | Powered by Gemini
        </p>
    </div>
    """, unsafe_allow_html=True)
      # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "system_initialized" not in st.session_state:
        st.session_state.system_initialized = False
    
    # Sidebar
    with st.sidebar:
        st.markdown("### System Information")
        
        # Load documents and initialize system
        doc_processor, vector_store, rag_system, doc_count = load_documents()
        
        if all([doc_processor, vector_store, rag_system]):
            st.session_state.rag_system = rag_system
            st.session_state.system_initialized = True
            
            # System metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#64b5f6;">{doc_count}</h3>
                    <p style="margin:0; font-size:0.8rem; color:#b0bec5;">PDF Files</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""                <div class="metric-card">
                    <h3 style="margin:0; color:#66bb6a;">Ready</h3>
                    <p style="margin:0; font-size:0.8rem; color:#b0bec5;">Status</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Conversation controls
            st.markdown("### Conversation")
            
            # Only show clear option if there are messages
            if st.session_state.get("messages", []):
                # Clear button with confirmation
                if "confirm_clear" not in st.session_state:
                    st.session_state.confirm_clear = False
                
                if not st.session_state.confirm_clear:
                    if st.button("Start New Conversation", 
                                key="clear_chat", 
                                help="Clear current conversation and start fresh",
                                use_container_width=True):
                        st.session_state.confirm_clear = True
                        st.rerun()
                else:
                    st.markdown("""
                    <div style="background-color: #1e293b; padding: 1rem; border-radius: 0.5rem; border: 1px solid #475569; margin-bottom: 1rem;">
                        <p style="margin: 0; color: #f1f5f9; font-weight: 500; text-align: center;">
                            Clear current conversation?
                        </p>
                        <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.85rem; text-align: center;">
                            This action cannot be undone
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Yes, Clear", 
                                    key="confirm_yes", 
                                    help="Confirm clearing the conversation",
                                    use_container_width=True):
                            st.session_state.messages = []
                            st.session_state.confirm_clear = False
                            st.success("New conversation started")
                            st.rerun()
                    with col2:
                        if st.button("Cancel", 
                                    key="confirm_no",
                                    help="Keep current conversation",
                                    use_container_width=True):
                            st.session_state.confirm_clear = False
                            st.rerun()
            else:                st.markdown("""
                <div style="background-color: #065f46; padding: 0.75rem; border-radius: 0.5rem; border-left: 3px solid #10b981;">
                    <div style='color: #a7f3d0; font-weight: 500; text-align: center;'>Ready for your questions</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("System not properly initialized")
            st.session_state.system_initialized = False    # Main chat interface
    if st.session_state.system_initialized:
        # Show welcome message for first time users
        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                👋 **Welcome to AI & Finance Research Q&A!**

                I'm your AI assistant for exploring research papers. Ask me about:

                •  **Key findings & methodologies in AI papers**  
                •  **Concepts and theories in finance research**  
                •  **Connections between AI and financial applications**  
                •  **Specific details from uploaded documents**

                Feel free to ask anything related to the provided research documents!
                """)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages using proper chat elements
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="🧑‍💼"):
                        st.markdown(message["content"])
                else: # assistant message
                    with st.chat_message("assistant", avatar="🤖"):
                        if message.get("type") == "document_links":
                            # For document links, message["content"] is the user_message from the LLM.
                            st.markdown(message["content"]) # Display the LLM's intro message for the documents
                            
                            if "paths" in message and message["paths"]:
                                # The user_message (message["content"]) should already state the context, 
                                # so no need for an additional header like "Dokumen yang relevan..." unless desired.
                                for doc_path in message["paths"]:
                                    try:
                                        with open(doc_path, "rb") as f:
                                            file_bytes = f.read()
                                        file_name = os.path.basename(doc_path)
                                        st.download_button(
                                            label=f"📥 Unduh {file_name}",
                                            data=file_bytes,
                                            file_name=file_name,
                                            mime="application/octet-stream"
                                        )
                                    except FileNotFoundError:
                                        st.error(f"File tidak ditemukan: {doc_path}. Mungkin file telah dipindahkan atau dihapus.")
                                    except Exception as e:
                                        st.error(f"Gagal memuat file {doc_path} untuk diunduh: {e}")
                                if message["paths"]: # Only show caption if there were paths to begin with
                                    st.caption("Klik tombol di atas untuk mengunduh dokumen.")
                            else: # Case where type is document_links but paths is empty or missing
                                # This might indicate an issue upstream if paths were expected.
                                # The content (user_message from LLM) might still be displayed.
                                if not message.get("paths"): # If paths are explicitly empty/None
                                     st.markdown("Saya berniat mengirimkan dokumen, tetapi tidak ada file spesifik yang ditemukan atau dapat diakses saat ini.")

                        elif "sources" in message and message["sources"]: # Standard Q&A with sources
                            st.markdown(message["content"]) # Display the textual answer
                            with st.expander(f"📄 View Sources ({len(message['sources'])} documents)", expanded=False):
                                for i, source in enumerate(message["sources"]):
                                    st.markdown(f"""
                                    <div class="source-document">
                                        <div class="source-header">📋 Source {i+1}: {source.get('source_file', 'Unknown')} (Page: {source.get('page', 'N/A')})</div>
                                        <div style="margin-top: 0.5rem; line-height: 1.5;">{source.get('content', 'No content available')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else: # Fallback for simple text message from assistant (e.g., errors, or answers without sources)
                            st.markdown(message["content"])
                                
          # Chat input
        if prompt := st.chat_input("💬 Ask about AI/Finance papers or request a document..."):
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with chat_container: # Ensure it's displayed within the same container
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.markdown(prompt)
            
            # Get RAG system response
            rag_system = st.session_state.rag_system
            with st.spinner("Thinking..."):
                # Pass the entire conversation history
                # The RAGSystem's chat_with_context now expects this directly
                response = rag_system.chat_with_context(
                    question=prompt, 
                    conversation_history=st.session_state.messages[:-1] # Pass history *before* current prompt
                )
            
            # Process and display assistant's response
            with chat_container: # Ensure it's displayed within the same container
                with st.chat_message("assistant", avatar="🤖"):
                    if response.get("type") == "documents":
                        # Store a message that reflects the intent to provide documents and the paths
                        assistant_message_content = response.get("user_message", "Berikut dokumen yang relevan:")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": assistant_message_content, # User-facing message from LLM
                            "type": "document_links", # Custom type for Streamlit handling
                            "paths": response.get("document_paths", [])
                        })
                        # Display the user-facing message
                        st.markdown(assistant_message_content)
                        
                        # Offer download buttons
                        doc_paths = response.get("document_paths", [])
                        if doc_paths:
                            for doc_path in doc_paths:
                                try:
                                    # Ensure the path is valid and the file exists
                                    if os.path.exists(doc_path) and os.path.isfile(doc_path):
                                        with open(doc_path, "rb") as f:
                                            file_bytes = f.read()
                                        file_name = os.path.basename(doc_path)
                                        st.download_button(
                                            label=f"📥 Unduh {file_name}",
                                            data=file_bytes,
                                            file_name=file_name,
                                            mime="application/octet-stream",
                                            key=f"download_{file_name}_{time.time()}" # Unique key
                                        )
                                    else:
                                        st.error(f"File tidak ditemukan atau path tidak valid: {doc_path}")
                                except FileNotFoundError:
                                    st.error(f"File tidak ditemukan: {doc_path}. Mungkin file telah dipindahkan atau dihapus.")
                                except Exception as e:
                                    st.error(f"Gagal memuat file {doc_path} untuk diunduh: {e}")
                            if doc_paths:
                                st.caption("Klik tombol di atas untuk mengunduh dokumen.")
                        else:
                            st.markdown("Saya berniat mengirimkan dokumen, tetapi tidak ada file spesifik yang ditemukan atau dapat diakses saat ini.")

                    elif response.get("type") == "answer":
                        answer = response.get("answer", "Maaf, saya tidak bisa merespons saat ini.")
                        sources = response.get("source_documents", [])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer, 
                            "sources": sources
                        })
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"📄 View Sources ({len(sources)} documents)", expanded=False):
                                for i, source in enumerate(sources):
                                    st.markdown(f"""
                                    <div class="source-document">
                                        <div class="source-header">📋 Source {i+1}: {source.get('source_file', 'Unknown')} (Page: {source.get('page', 'N/A')})</div>
                                        <div style="margin-top: 0.5rem; line-height: 1.5;">{source.get('content', 'No content available')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                    elif response.get("type") == "error":
                        error_message = response.get("answer", "Terjadi kesalahan internal.")
                        st.session_state.messages.append({"role": "assistant", "content": error_message, "type": "error"})
                        st.error(error_message)
                        
                    else: # Fallback for any other response type or if type is missing
                        unknown_response_content = str(response) # Display the whole response if its structure is unexpected
                        st.session_state.messages.append({"role": "assistant", "content": unknown_response_content, "type": "unknown"})
                        st.warning(f"Menerima respons yang tidak dikenal: {unknown_response_content}")
            
            # Rerun to update the chat display immediately after processing
            st.rerun()

    else:
        st.warning("RAG system is not initialized. Please check logs or .env configuration.")

if __name__ == "__main__":
    main()
