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
from typing import List, Dict

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
    if not isinstance(text, str): # Ensure text is a string
        text = str(text)
    # Chars to escape: _ * [ ] ( ) ~ ` > # + - = | { } . ! \ (added backslash)
    escape_chars = r'_*[]()~`>#+-=|{}.!\\'  # Define characters that need escaping including single backslash
    # Escape each special character by prefixing it with a single backslash
    return re.sub(rf'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- Text Cleaning Helper ---
def clean_response_text(text: str) -> str:
    """Cleans known artifacts like \\1, \\\\1, and the SOH character (ASCII 0x01)."""
    if not isinstance(text, str):
        # Attempt to convert to string, as some inputs might be non-string
        try:
            processed_text = str(text)
        except Exception:
            # If conversion fails, return an empty string or a placeholder
            # to prevent downstream errors with non-string types.
            return "[Error: Non-string data]"
    else:
        processed_text = text

    # Remove literal string "\\1" (double backslash, one)
    processed_text = processed_text.replace('\\\\1', '')
    # Remove literal string "\\1" (single backslash, one)
    processed_text = processed_text.replace('\\1', '')
    # Remove ASCII SOH character (Ctrl-A), which might be displayed as \\1 or similar
    processed_text = processed_text.replace('\\x01', '') # Hex escape for SOH
    processed_text = processed_text.replace(chr(1), '')   # chr(1) is SOH

    # Additional check for the specific issue where "." and "!" might be replaced.
    # This is speculative and for debugging.
    # If we see messages like "reset1 Silakan...", this might indicate the issue.
    # For now, let's assume the above cleaning is sufficient.

    return processed_text

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
            
        doc_processor = DocumentProcessor() # Assuming default chunk_size/overlap is fine here
        processed_docs = doc_processor.process_documents(DOCUMENTS_DIR)
        if processed_docs:
            vector_store.add_documents(processed_docs)
            logger.info("Documents processed and added to vector store.")
        else:
            logger.error("Failed to process documents. RAG system might not function correctly.")
            # Not exiting, to allow bot to run even if doc processing fails initially

    logger.info("Initializing RAGSystem...")
    rag_system = RAGSystem(api_key=GEMINI_API_KEY, vector_store=vector_store)
    # The qa_chain is no longer pre-initialized like this. ConversationalRetrievalChain is built on demand.
    # if not rag_system.qa_chain: 
    #     logger.info("QA chain not initialized by default, attempting to initialize.")
    #     rag_system._reinitialize_qa_chain() 
    logger.info("RAGSystem initialized successfully.")

except Exception as e:
    logger.error(f"Error initializing RAG system: {e}", exc_info=True)
    exit()

# --- Chat History Management ---
# Stores conversation history: {chat_id: {"history": List[Dict[str, str]], "last_active": timestamp}}
# History format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
chat_histories = {}
CHAT_HISTORY_TIMEOUT = 30 * 60 # 30 minutes

def get_chat_history(chat_id: int) -> List[Dict[str, str]]:
    """Retrieves and updates activity timestamp for chat history."""
    if chat_id not in chat_histories or (time.time() - chat_histories[chat_id].get("last_active", 0)) > CHAT_HISTORY_TIMEOUT:
        chat_histories[chat_id] = {"history": [], "last_active": time.time()}
    else:
        chat_histories[chat_id]["last_active"] = time.time()
    return chat_histories[chat_id]["history"]

def add_to_chat_history(chat_id: int, role: str, content: str):
    """Adds a message to the chat history for the given chat_id."""
    history = get_chat_history(chat_id) # Ensures "last_active" is updated
    history.append({"role": role, "content": content})

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and resets chat history."""
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = {"history": [], "last_active": time.time()} # Reset history
    
    welcome_message = (
        "<b>Halo! Saya chatbot yang sudah terintegrasi dengan database paper milik @rizr09.</b>\n\n"
        "Saya dapat menjawab pertanyaan riset AI & keuangan, mengirimkan dokumen riset relevan, dan menghapus memori percakapan (/reset).\n\n"
        "<b>Contoh pertanyaan:</b>\n"
        "1. apa itu LoRA berdasarkan dokumen yang ada?\n"
        "2. kirim dokumennya coba (apabila ingin coreference berdasarkan chat sebelumnya)\n"
        "3. What is the difference between sukuk and bonds?\n\n"
        "Bot dapat menerima pertanyaan dalam Bahasa Indonesia dan Inggris. Silakan mulai bertanya!"
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears the conversation history for the current chat."""
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = {"history": [], "last_active": time.time()}
    cleaned_message = clean_response_text("Memori percakapan Anda telah direset. Silakan mulai bertanya kembali!")
    await update.message.reply_text(escape_markdown_v2(cleaned_message), parse_mode=ParseMode.MARKDOWN_V2)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages and responds using the RAG system."""
    chat_id = update.effective_chat.id
    user_message_text = update.message.text

    current_chat_history = get_chat_history(chat_id) # Gets history and updates last_active
    add_to_chat_history(chat_id, "user", user_message_text)

    await update.message.chat.send_action(action="typing")

    bot_response_content_for_history = "Sorry, I encountered an issue." # Default for history

    try:
        # Pass history *before* current user message
        response_data = rag_system.chat_with_context(
            question=user_message_text,
            conversation_history=current_chat_history # This now includes history up to the message *before* the current one
        )
        
        response_type = response_data.get("type")
        
        if response_type == "documents":
            user_facing_message = response_data.get("user_message", "Here are the documents I found:")
            cleaned_user_message = clean_response_text(user_facing_message)
            await update.message.reply_text(escape_markdown_v2(cleaned_user_message), parse_mode=ParseMode.MARKDOWN_V2)
            
            document_paths = response_data.get("document_paths", [])
            if document_paths:
                for doc_path in document_paths:
                    if os.path.exists(doc_path) and os.path.isfile(doc_path):
                        try:
                            with open(doc_path, "rb") as f:
                                await update.message.reply_document(f, connect_timeout=60, read_timeout=60) # Added timeouts
                        except Exception as e:
                            logger.error(f"Failed to send document {doc_path}: {e}")
                            cleaned_error_doc_msg = clean_response_text(f"Maaf, gagal mengirim dokumen: {os.path.basename(doc_path)}.")
                            await update.message.reply_text(escape_markdown_v2(cleaned_error_doc_msg), parse_mode=ParseMode.MARKDOWN_V2)
                    else:
                        logger.warning(f"Document path not found or not a file: {doc_path}")
                        cleaned_notfound_doc_msg = clean_response_text(f"Maaf, file dokumen tidak ditemukan: {os.path.basename(doc_path)}.")
                        await update.message.reply_text(escape_markdown_v2(cleaned_notfound_doc_msg), parse_mode=ParseMode.MARKDOWN_V2)
            else: # No documents found by RAG system even if intent was to provide
                 cleaned_no_doc_reply = clean_response_text("Saya mencari dokumen yang relevan, namun tidak ada yang ditemukan saat ini.")
                 await update.message.reply_text(escape_markdown_v2(cleaned_no_doc_reply), parse_mode=ParseMode.MARKDOWN_V2)

            bot_response_content_for_history = cleaned_user_message + (f" (Sent {len(document_paths)} documents)" if document_paths else " (No documents sent)")

        elif response_type == "answer":
            answer_text = response_data.get("answer", "Sorry, I encountered an issue processing your request.")
            cleaned_answer = clean_response_text(answer_text)
            await update.message.reply_text(escape_markdown_v2(cleaned_answer), parse_mode=ParseMode.MARKDOWN_V2)
            bot_response_content_for_history = cleaned_answer

        elif response_type == "error":
            error_message = response_data.get("answer", "An internal error occurred.") # "answer" field contains user-friendly error
            cleaned_error_message = clean_response_text(error_message)
            await update.message.reply_text(escape_markdown_v2(cleaned_error_message), parse_mode=ParseMode.MARKDOWN_V2)
            bot_response_content_for_history = cleaned_error_message
        
        else: # Unknown response type
            logger.warning(f"Received unknown response type from RAGSystem: {response_type} - Full response: {response_data}")
            unknown_response_text = "I received an unexpected response. Please try rephrasing."
            # This is a fixed string, no LLM content, so clean_response_text not strictly needed but harmless.
            cleaned_unknown_response = clean_response_text(unknown_response_text)
            await update.message.reply_text(escape_markdown_v2(cleaned_unknown_response), parse_mode=ParseMode.MARKDOWN_V2)
            bot_response_content_for_history = cleaned_unknown_response

    except Exception as e:
        logger.error(f"Error in handle_message: {e}", exc_info=True)
        error_reply = "Maaf, terjadi kendala teknis saat memproses permintaan Anda. Silakan coba lagi nanti."
        # This is a fixed string.
        cleaned_fallback_error = clean_response_text(error_reply)
        await update.message.reply_text(escape_markdown_v2(cleaned_fallback_error), parse_mode=ParseMode.MARKDOWN_V2)
        bot_response_content_for_history = cleaned_fallback_error

    # Add bot's final response (or summary) to history
    add_to_chat_history(chat_id, "assistant", bot_response_content_for_history)

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