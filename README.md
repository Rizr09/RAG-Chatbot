# Market Outlook RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot powered by Gemini 2.5 Pro API for answering questions about market outlooks, investment strategies, and economic trends based on your PDF documents.

## Features

- **Intelligent Q&A**: Ask questions about market trends, investment strategies, and economic outlook
- **Document Retrieval**: Automatically finds relevant information from your PDF documents
- **Market Focus**: Specialized for financial and market analysis documents
- **Minimalist UI**: Clean, modern interface built with Streamlit
- **Persistent Storage**: ChromaDB vector database for efficient document retrieval
- **Powered by Gemini 2.5 Pro**: State-of-the-art language model for accurate responses

## Prerequisites

- Python 3.8+
- Google Gemini API key
- PDF documents (market outlook reports, investment strategies, etc.)

## Installation

1. **Clone/Navigate to the project directory**
   ```bash
   cd "d:\Root\Playground\Python\RAG"
   ```

2. **Activate the virtual environment**
   ```bash
   .\myenv\Scripts\Activate.ps1
   ```

3. **Install dependencies** (already installed in your environment)
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root with your Google Gemini API key:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
   - Ensure this file is not committed to version control (add to `.gitignore`)

5. **Add your documents**
   - Place your PDF files in the `documents/` folder
   - Supported documents include market outlook reports, investment strategies, and economic analyses

## Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser to `http://localhost:8501`
   - The application will automatically process your PDF documents on first run

3. **Ask questions**
   - Use the chat interface to ask questions about market outlook
   - Examples:
     - "What are the key market trends for 2025?"
     - "What investment strategies are recommended?"
     - "What are the main economic risks highlighted?"
     - "How do different institutions view equity markets?"

## Project Structure

```
RAG/
├── app.py                 # Main Streamlit application
├── document_processor.py  # PDF processing and chunking
├── vector_store.py       # ChromaDB vector store management
├── rag_system.py         # RAG logic with Gemini integration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API key)
├── documents/            # PDF documents folder
│   ├── Deutsche Bank Annual-Outlook-2025 Perspectives.pdf
│   ├── JPmorgan Global outlook 2025 -Building on strength.pdf
│   ├── Morgan Stanely 2025 BEAT [Bonds, equities, alternative, transition].pdf
│   └── Rothschild 2025 perpectives (English version).pdf
├── chroma_db/           # ChromaDB storage (created automatically)
└── myenv/               # Python virtual environment
```

## How It Works

1. **Document Processing**: PDF documents are loaded and split into chunks for better retrieval
2. **Embedding Generation**: Text chunks are converted to embeddings using Google's embedding model
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
4. **Query Processing**: User questions are embedded and matched against document chunks
5. **Response Generation**: Relevant chunks are sent to Gemini 2.5 Pro for answer generation
6. **Source Attribution**: Answers include references to source documents

## Configuration

### Chunk Settings
- **Chunk Size**: 1000 characters (adjustable in `document_processor.py`)
- **Chunk Overlap**: 200 characters (ensures context continuity)

### Retrieval Settings
- **Default K**: 6 documents retrieved per query
- **Temperature**: 0.1 (for consistent, factual responses)
- **Max Tokens**: 2048 (for comprehensive answers)

## Customization

### Adding New Documents
1. Place new PDF files in the `documents/` folder
2. Restart the application to process new documents
3. Or clear the ChromaDB cache to reprocess all documents

### Modifying the Prompt
Edit the prompt template in `rag_system.py` to customize the assistant's behavior:

```python
template = """Your custom prompt here..."""
```

### Adjusting UI
Modify the CSS styles in `app.py` to change the appearance:

```python
st.markdown("""<style>...</style>""", unsafe_allow_html=True)
```

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Verify your Gemini API key in `.env`
   - Ensure the key has proper permissions

2. **Document Loading Issues**
   - Check PDF file integrity
   - Ensure PDFs contain extractable text (not just images)

3. **Memory Issues**
   - Reduce chunk size or number of retrieved documents
   - Process fewer documents at once

4. **Slow Performance**
   - First run takes longer due to document processing
   - Subsequent runs use cached embeddings

### Reset Vector Database
If you need to reprocess all documents:

```python
# In Python console
from vector_store import VectorStore
vs = VectorStore(api_key="your_key")
vs.delete_collection()
```

## Performance Tips

1. **Optimal Document Size**: Keep PDFs under 50MB for best performance
2. **Question Quality**: Be specific in your questions for better results
3. **Context Window**: The system works best with focused, document-specific queries

## Security Notes

- Keep your API key secure and don't commit it to version control
- The `.env` file should be added to `.gitignore` in production
- Consider using environment variables or secure key management in production

## License

This project is for educational and personal use. Please ensure you have the right to process the PDF documents you're using.

## Support

For issues or questions:
1. Check the Streamlit logs in the console
2. Verify all dependencies are properly installed
3. Ensure your API key is valid and has sufficient quota
