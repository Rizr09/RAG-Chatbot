import logging
import os
import re # Added for Markdown escaping
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode # Added for specifying parse mode
from rag_system import RAGSystem
from vector_store import VectorStore # Assuming VectorStore is needed for RAGSystem initialization
from document_processor import DocumentProcessor # Assuming DocumentProcessor might be needed for full setup
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Helper Function for Markdown V2 Escaping ---
def escape_markdown_v2(text: str) -> str:
    """Escapes text for Telegram MarkdownV2 parsing."""
    # Chars to escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DOCUMENTS_DIR = "./documents_retrieval" # Make sure this path is correct
PERSIST_DIRECTORY = "./chroma_db"

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not found in .env file")
    exit()
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env file")
    exit()

# --- RAG System Initialization ---
try:
    logger.info("Initializing VectorStore...")
    vector_store = VectorStore(api_key=GEMINI_API_KEY, persist_directory=PERSIST_DIRECTORY)
    
    # Check if documents need processing (simplified from init_system.py)
    doc_count = vector_store.get_collection_count()
    if doc_count == 0:
        logger.info("Vector store is empty. Processing documents...")
        if not os.path.exists(DOCUMENTS_DIR):
            logger.error(f"Documents directory '{DOCUMENTS_DIR}' not found. Cannot initialize RAG system.")
            exit()
        pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logger.error(f"No PDF files found in '{DOCUMENTS_DIR}'. Cannot initialize RAG system.")
            exit()
            
        doc_processor = DocumentProcessor()
        processed_docs = doc_processor.process_documents(DOCUMENTS_DIR)
        if processed_docs:
            vector_store.add_documents(processed_docs)
            logger.info("Documents processed and added to vector store.")
        else:
            logger.error("Failed to process documents. RAG system might not function correctly.")
            # Decide if to exit or continue with potentially limited functionality
            # exit() 

    logger.info("Initializing RAGSystem...")
    rag_system = RAGSystem(api_key=GEMINI_API_KEY, vector_store=vector_store)
    if not rag_system.qa_chain: # Ensure the QA chain is ready
        logger.info("QA chain not initialized by default, attempting to initialize.")
        rag_system._reinitialize_qa_chain() # Initialize with default (no specific query yet)
    logger.info("RAGSystem initialized successfully.")

except Exception as e:
    logger.error(f"Error initializing RAG system: {e}", exc_info=True)
    exit()

# --- Chat History Management ---
# Stores conversation history and last active timestamp: {chat_id: {"history": [...], "last_active": timestamp}}
chat_histories = {}

# Helper: timeout dalam detik (30 menit)
CHAT_HISTORY_TIMEOUT = 30 * 60

def _should_reset_history(chat_id):
    """Cek apakah history user perlu direset karena timeout."""
    if chat_id not in chat_histories:
        return False
    last_active = chat_histories[chat_id].get("last_active", 0)
    return (time.time() - last_active) > CHAT_HISTORY_TIMEOUT

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mengirim pesan sambutan dalam bahasa Indonesia saat /start."""
    chat_id = update.effective_chat.id
    # Reset history saat /start
    chat_histories[chat_id] = {"history": [], "last_active": time.time()}
    welcome_message = (
        "<b>Halo! Saya chatbot yang sudah terintegrasi dengan database paper milik @rizr09.</b>\n\n"
        "Saya siap membantu Anda menjawab pertanyaan seputar AI & Finance, atau mengirimkan dokumen riset yang relevan.\n\n"
        "<b>Contoh penggunaan:</b>\n"
        "1. <b>QnA:</b>\n"
        "   <i>apa itu sukuk dan bagaimana mekanismenya?</i>\n"
        "2. <b>Document retrieval:</b>\n"
        "   <i>kirimkan paper tentang pemanfaatan sentimen pasar untuk peramalan harga saham</i>\n\n"
        "Gunakan <b>/reset</b> untuk menghapus memori percakapan (bukan menghapus chat).\n\n"
        "Bot ini <b>bilingual</b> (Indonesia & Inggris). Silakan bertanya dalam kedua bahasa tersebut.\n\n"
        "Selamat mencoba!"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current chat."""
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = {"history": [], "last_active": time.time()}
    await update.message.reply_text("Memori percakapan Anda telah direset. Silakan mulai bertanya kembali!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages and responds using the RAG system."""
    chat_id = update.effective_chat.id
    user_message_text = update.message.text

    # Reset history jika timeout
    if _should_reset_history(chat_id):
        chat_histories[chat_id] = {"history": [], "last_active": time.time()}

    if chat_id not in chat_histories:
        chat_histories[chat_id] = {"history": [], "last_active": time.time()}

    # Update last active
    chat_histories[chat_id]["last_active"] = time.time()

    # Append user message to history
    chat_histories[chat_id]["history"].append({"role": "user", "parts": [{"text": user_message_text}]})

    # Get response from RAG system
    try:
        # The rag_system.chat_with_context expects the current question AND the history
        # The question itself is the last user message in the history.
        response_data = rag_system.chat_with_context(
            question=user_message_text, # Current question
            conversation_history=chat_histories[chat_id]["history"][:-1] # History *before* current question
        )
        
        raw_bot_response_for_history = "Sorry, I encountered an issue processing your request." # Default
        formatted_bot_response_for_telegram = escape_markdown_v2(raw_bot_response_for_history)

        # --- Tambahan: Kirim dokumen jika response tipe documents/provide_document ---
        if response_data.get("type") in ["documents", "provide_document"]:
            document_paths = response_data.get("document_paths") or response_data.get("paths")
            user_message_content = response_data.get("user_message", "Here are the documents I found:")
            await update.message.reply_text(escape_markdown_v2(user_message_content), parse_mode=ParseMode.MARKDOWN_V2)
            if document_paths:
                for doc_path in document_paths:
                    if os.path.exists(doc_path):
                        with open(doc_path, "rb") as f:
                            await update.message.reply_document(f)
                    else:
                        await update.message.reply_text(f"File not found: {doc_path}")
            # Tetap tambahkan ke history agar percakapan konsisten
            chat_histories[chat_id]["history"].append({"role": "model", "parts": [{"text": user_message_content}]})
            return
        elif response_data.get("type") == "error":
            raw_bot_response_for_history = response_data.get("answer", raw_bot_response_for_history)
            formatted_bot_response_for_telegram = escape_markdown_v2(raw_bot_response_for_history)
        elif response_data.get("type") == "provide_document":
             # Handle document provision response
            user_message_content = response_data.get("user_message", "I found some documents for you:")
            search_query_content = response_data.get('search_query_for_docs', 'your query')
            
            raw_bot_response_for_history = f"{user_message_content} Suggested search: {search_query_content}"
            
            escaped_user_message = escape_markdown_v2(user_message_content)
            # Search query content does not need to be escaped when inside a ``` block
            formatted_bot_response_for_telegram = f"{escaped_user_message}\nI suggest searching for documents with keywords like:\n```\n{search_query_content}\n```"
        else: # Assuming this is an answer
            raw_bot_response_for_history = response_data.get("answer", raw_bot_response_for_history)
            
            # Process **bold** tags into *bold* and escape other text
            parts = raw_bot_response_for_history.split('**')
            processed_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 1: # This part was between **
                    processed_parts.append("*" + escape_markdown_v2(part) + "*")
                else:
                    processed_parts.append(escape_markdown_v2(part))
            formatted_bot_response_for_telegram = "".join(processed_parts)

    except Exception as e:
        logger.error(f"Error getting response from RAG system: {e}", exc_info=True)
        raw_bot_response_for_history = "I am having trouble connecting to my knowledge base. Please try again later."
        formatted_bot_response_for_telegram = escape_markdown_v2(raw_bot_response_for_history)

    # Append bot response to history
    chat_histories[chat_id]["history"].append({"role": "model", "parts": [{"text": raw_bot_response_for_history}]})
    
    await update.message.reply_text(formatted_bot_response_for_telegram, parse_mode=ParseMode.MARKDOWN_V2)

def main() -> None:
    """Starts the Telegram bot."""
    logger.info("Starting Telegram bot...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))

    # Message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    logger.info("Bot polling started.")
    application.run_polling()

if __name__ == '__main__':
    main() 