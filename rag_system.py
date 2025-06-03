"""
RAG (Retrieval-Augmented Generation) system using Gemini 2.5 Pro.
"""

from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.tracers.schemas import Run
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import logging
from googletrans import Translator
import os
import asyncio
import threading # Added for running async calls in a separate thread
import json # Added for parsing LLM output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to run an async coroutine in a separate thread with its own event loop
def run_async_in_thread(coro):
    result = None
    exception = None

    def target():
        nonlocal result, exception
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
        except Exception as e:
            exception = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join() # Wait for the thread to complete

    if exception:
        # Log the exception or handle it as needed before raising
        logger.error(f"Exception in async thread: {exception}")
        raise exception
    return result

class RAGSystem:
    def __init__(self, api_key: str, vector_store):
        """
        Initialize the RAG system.
        
        Args:
            api_key: Google Gemini API key
            vector_store: Vector store instance
        """
        self.api_key = api_key
        self.vector_store = vector_store
        self.translator = Translator()
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=api_key,
            temperature=0.15,
            max_output_tokens=2048
        )
        
        # Create custom prompt template for combining documents
        self.combine_docs_prompt = self._create_combine_docs_prompt()
        
        # Create prompt for condensing question
        self.condense_question_prompt = self._create_condense_question_prompt()

    def _create_combine_docs_prompt(self) -> PromptTemplate:
        """Create a custom prompt template for AI and Finance research paper Q&A, used by the combine_docs_chain."""
        
        template = """You are an expert AI and Finance Research Analyst. Your primary goal is to assist users by either answering their questions based on the provided `Context` or by providing them with the relevant documents if their query indicates a request for the document itself. Use the `Chat History` to understand the context of the conversation.

**Chat History:**
{chat_history}

**Response Modes:**

1.  **Answering Mode:**
    *   If the user asks a question seeking information, insights, summaries, or specific details *from* the documents, provide a comprehensive, accurate, and insightful textual answer derived *solely* from the provided `Context`.
    *   Do not incorporate any external knowledge or make assumptions not supported by the given context.
    *   Follow the detailed answering instructions below.

2.  **Document Provisioning Mode:**
    *   If you judge that the user's query is primarily a request *for* one or more documents, papers, or files themselves (e.g., "send me the paper on X", "can I get the document about Y?", "find the report on Z and related articles"), then you MUST respond *ONLY* with a single JSON object in the following exact format. Do not add any text before or after this JSON object:

```json
{{
  "intent": "provide_document",
  "search_query_for_docs": "<keywords you determine are best for finding the requested document(s), considering the chat history and current question>",
  "user_message": "<a short, friendly message for the user, e.g., 'Certainly, I found the following document(s) related to your request for X (based on our conversation):'>"
}}
```

            *   The `search_query_for_docs` should be your best assessment of the core subject of the document(s) the user wants, considering the full conversation.

**Detailed Answering Instructions (for Answering Mode):**
*   **Context Reliance:** Your entire response must be derived *solely* from the provided `Context`.
*   **Language Handling:** If the user's question was in Indonesian, provide your answer in Indonesian. Otherwise, answer in the language of the question.
*   **Conciseness & Relevance:** Keep your response concise. Prioritize and extract only the most relevant information that directly addresses the user's question.
*   **Structured Answer:** Include a direct answer, supporting evidence (data, findings, quotes), relevant insights, and source attribution (document name, page number if available).
*   **Information Gap Handling:** If the `Context` is insufficient, state (in the appropriate language): "Saya tidak memiliki cukup informasi..." or "I do not have enough information..."
*   **Tone:** Maintain a professional, analytical, and objective tone.

**IMPORTANT:** Choose ONLY ONE mode per query. If providing documents, ONLY output the JSON. Otherwise, provide a textual answer.

**Context:**
{context}

**Question:**
{question}

**Answer:**
""" # Ensure no stray characters after this final triple quote.
        
        return PromptTemplate(
            template=template,
            input_variables=["chat_history", "context", "question"]
        )

    def _create_condense_question_prompt(self) -> PromptTemplate:
        """Create a prompt template for condensing the current question and chat history into a standalone question."""
        template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
        return PromptTemplate.from_template(template)

    def _get_custom_retriever(self, translated_question_for_retrieval: str = None, original_question: str = None) -> BaseRetriever:
        """
        Creates a custom retriever that combines results from original and translated queries.
        """
        try:
            if translated_question_for_retrieval and original_question:
                base_retriever_orig = self.vector_store.get_retriever(k=5)
                base_retriever_trans = self.vector_store.get_retriever(k=5)

                original_docs = []
                if base_retriever_orig:
                    # Use invoke for newer Langchain versions if get_relevant_documents is deprecated
                    original_docs = base_retriever_orig.invoke(original_question)
                
                translated_docs = []
                if base_retriever_trans:
                    translated_docs = base_retriever_trans.invoke(translated_question_for_retrieval)

                combined_docs_dict = {}
                for doc in original_docs + translated_docs:
                    doc_key = (doc.page_content, doc.metadata.get('source_file'), doc.metadata.get('page'))
                    if doc_key not in combined_docs_dict:
                        combined_docs_dict[doc_key] = doc
                
                unique_combined_docs = list(combined_docs_dict.values())[:6]

                class CustomRetriever(BaseRetriever):
                    documents: List[Document]

                    class Config:
                        arbitrary_types_allowed = True
                        
                    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                        return self.documents
                    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                        return self.documents
                
                if not unique_combined_docs:
                    logger.warning("No documents found for combined query. QA might be uninformative.")
                    class EmptyRetriever(BaseRetriever):
                        class Config:
                            arbitrary_types_allowed = True
                        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                            return []
                        async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                            return []
                    return EmptyRetriever()
                else:
                    return CustomRetriever(documents=unique_combined_docs)
            else: # Fallback to original behavior if no translation
                retriever = self.vector_store.get_retriever(k=6)
                if not retriever:
                    logger.error("Failed to get default retriever from vector store.")
                    raise ValueError("Retriever not available.")
                return retriever
        except Exception as e:
            logger.error(f"Error creating custom retriever: {e}", exc_info=True)
            # Fallback to a simple retriever on error
            try:
                retriever = self.vector_store.get_retriever(k=3) # Reduced k for fallback
                if not retriever:
                    raise ValueError("Fallback retriever also failed.")
                logger.warning("Fell back to a simple retriever due to an error in custom retriever creation.")
                return retriever
            except Exception as fallback_e:
                logger.error(f"Critical error: Could not create any retriever: {fallback_e}", exc_info=True)
                # Return an empty retriever as a last resort
                class EmptyRetriever(BaseRetriever):
                    class Config:
                        arbitrary_types_allowed = True
                    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]: return []
                    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]: return []
                return EmptyRetriever()

    def _create_conversational_qa_chain(self, retriever: BaseRetriever, chat_history_for_memory: List[Tuple[str, str]]):
        """Create the ConversationalRetrievalChain."""
        try:
            # Memory for the conversation
            # We re-create it for each call to ensure it's per-user and reset on new sessions.
            # The `chat_history_massages` from Streamlit will be used to populate this.
            # However, ConversationalRetrievalChain manages its own memory internally if we pass chat_history.
            # Let's simplify and pass the history directly to the chain.

            # Document combining chain
            combine_docs_chain = load_qa_chain(
                llm=self.llm,
                chain_type="stuff", # "stuff" is good for relatively small contexts
                prompt=self.combine_docs_prompt,
                document_variable_name="context" # Ensure this matches the prompt
            )

            # Question generator chain
            question_generator_chain = LLMChain(
                llm=self.llm,
                prompt=self.condense_question_prompt
            )
            
            qa_chain = ConversationalRetrievalChain(
                retriever=retriever,
                combine_docs_chain=combine_docs_chain,
                question_generator=question_generator_chain,
                return_source_documents=True,
                # memory=memory, # We will pass chat_history directly
            )
            logger.info("ConversationalRetrievalChain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating ConversationalRetrievalChain: {str(e)}", exc_info=True)
            return None

    def answer_conversational(self, question: str, chat_history_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Answer a question using conversational context.
        
        Args:
            question: The current question from the user.
            chat_history_messages: A list of dictionaries, where each dict has "role" and "content".
                                  e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            Dictionary containing answer and source documents, or document paths and user message.
        """
        try:
            original_question = question
            translated_question_for_retrieval = None
            question_lang = 'en' # Default to English

            # Language detection and translation for retrieval query (if Indonesian)
            try:
                detected_lang_result = run_async_in_thread(self.translator.detect(question))
                detected_lang = detected_lang_result.lang
                
                if detected_lang.startswith('id'):
                    question_lang = 'id'
                    translation_result = run_async_in_thread(self.translator.translate(question, src='id', dest='en'))
                    translated_question_for_retrieval = translation_result.text
                    logger.info(f"Original (ID): '{question}', Translated (EN) for retrieval: '{translated_question_for_retrieval}'")
            except Exception as e:
                logger.warning(f"Language detection/translation for input query failed: {e}. Proceeding with original question for retrieval if needed.")

            # Prepare retriever
            retriever = self._get_custom_retriever(
                translated_question_for_retrieval=translated_question_for_retrieval,
                original_question=original_question
            )

            # Convert Streamlit chat history to Langchain's expected format (list of tuples for ConversationalRetrievalChain)
            # Or list of BaseMessage objects if using memory more directly
            formatted_chat_history = []
            for msg in chat_history_messages:
                if msg["role"] == "user":
                    formatted_chat_history.append((msg["content"], "")) # User query
                elif msg["role"] == "assistant":
                    # Find the preceding user message to pair with this assistant message
                    if formatted_chat_history and formatted_chat_history[-1][1] == "":
                        last_user_query = formatted_chat_history.pop()[0]
                        formatted_chat_history.append((last_user_query, msg["content"]))
                    else: # Should not happen if history is well-formed user-assistant pairs
                        formatted_chat_history.append(("", msg["content"])) # Or handle as error


            # Create or get the conversational chain
            # The chain needs to be (re)created with the potentially new retriever strategy
            conversational_chain = self._create_conversational_qa_chain(retriever, formatted_chat_history)
            
            if not conversational_chain:
                return {
                    "type": "error",
                    "answer": "The RAG system's conversational chain could not be initialized. Please check the setup.",
                    "source_documents": [],
                    "error": "Conversational chain not available"
                }
            
            # Invoke the chain with the current question and chat history
            # The `question` is the current user utterance.
            # `chat_history` is the history of (human_message, ai_message) tuples.
            llm_response_raw = conversational_chain.invoke({
                "question": original_question, # Pass the original question
                "chat_history": formatted_chat_history
            })
            
            raw_answer_text = llm_response_raw.get("answer", "").strip() # 'answer' is the key from ConversationalRetrievalChain
            source_documents_from_chain = llm_response_raw.get("source_documents", [])

            # Attempt to parse the LLM's response as JSON for document provisioning intent
            try:
                potential_json_str = raw_answer_text
                if raw_answer_text.startswith("```json"):
                    potential_json_str = raw_answer_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif raw_answer_text.startswith("```") and raw_answer_text.endswith("```"):
                    potential_json_str = raw_answer_text[3:-3].strip()
                
                json_start_index = potential_json_str.find('{')
                json_end_index = potential_json_str.rfind('}')

                if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                    extracted_json_str = potential_json_str[json_start_index : json_end_index+1]
                    llm_output_json = json.loads(extracted_json_str)
                    if isinstance(llm_output_json, dict) and llm_output_json.get("intent") == "provide_document":
                        search_query_for_docs = llm_output_json.get("search_query_for_docs", original_question) # Fallback to original question if not specified
                        user_message = llm_output_json.get("user_message", "Here are the documents I found based on our conversation:")
                        
                        logger.info(f"LLM signaled 'provide_document' intent. Search query for docs: '{search_query_for_docs}'")
                        document_paths = self.get_documents_for_query(search_query_for_docs, k=3) # Use the dedicated method
                        
                        return {
                            "type": "documents",
                            "document_paths": document_paths,
                            "user_message": user_message,
                            "query_used_for_retrieval": search_query_for_docs
                        }
                    else:
                        logger.info("Parsed JSON from LLM but not 'provide_document' intent. Proceeding with Answering Mode.")
                        raise json.JSONDecodeError("JSON parsed but not provide_document intent", extracted_json_str, 0)
                else:
                    logger.info(f"No JSON block found in LLM response: '{raw_answer_text[:100]}...'. Proceeding with Answering Mode.")
                    raise json.JSONDecodeError("No JSON block found", raw_answer_text, 0)

            except json.JSONDecodeError:
                logger.info(f"LLM response ('{raw_answer_text[:100]}...') is not the expected provide_document JSON. Proceeding with Answering Mode.")
            # Fall-through to Answering Mode

            answer = raw_answer_text
            
            if question_lang == 'id':
                try:
                    detected_answer_lang_result = run_async_in_thread(self.translator.detect(answer))
                    detected_answer_lang = detected_answer_lang_result.lang
                    if detected_answer_lang and not detected_answer_lang.startswith('id'):
                        logger.info(f"Translating answer from {detected_answer_lang} to ID. Original answer: '{answer[:100]}...'")
                        translated_answer_result = run_async_in_thread(self.translator.translate(answer, src=detected_answer_lang, dest='id'))
                        translated_answer = translated_answer_result.text
                        answer = translated_answer
                        logger.info(f"Translated answer (ID): '{answer[:100]}...'")
                except Exception as e:
                    logger.warning(f"Answer translation to Indonesian failed: {e}. Returning original answer.")
            
            sources = []
            for i, doc in enumerate(source_documents_from_chain):
                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "source_file": doc.metadata.get("source_file", f"Document {i+1}"),
                    "page": doc.metadata.get("page", "Unknown")
                }
                sources.append(source_info)
            
            logger.info(f"Generated answer for question: {original_question[:50]}...")
            
            return {
                "type": "answer",
                "answer": answer,
                "source_documents": sources,
                "question": original_question # Storing original question for context if needed by UI
            }
            
        except Exception as e:
            logger.error(f"Error processing question in answer_conversational: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "answer": f"An error occurred while processing your question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Get relevant documents for a query without generating an answer.
        (This method might be less used now with conversational chain, but kept for direct doc search if needed)
        """
        try:
            # This method directly uses the vector store, not the conversational chain's retriever
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_documents_for_query(self, query: str, k: int = 3) -> List[str]:
        """
        Get relevant document file paths for a query.
        Handles potential translation for Indonesian queries.
        Returns a list of unique source file paths.
        This is used by the 'provide_document' intent.
        """
        try:
            original_query = query
            translated_query_for_retrieval = None
            
            try:
                detected_lang_result = run_async_in_thread(self.translator.detect(query))
                detected_lang = detected_lang_result.lang
                if detected_lang.startswith('id'):
                    translation_result = run_async_in_thread(self.translator.translate(query, src='id', dest='en'))
                    translated_query_for_retrieval = translation_result.text
                    logger.info(f"Original (ID) for doc path retrieval: '{query}', Translated (EN): '{translated_query_for_retrieval}'")
            except Exception as e:
                logger.warning(f"Language detection/translation for doc path retrieval failed: {e}. Proceeding with original query.")

            relevant_docs = []
            # Use the vector_store's similarity search directly here.
            # The k value here is for how many docs to fetch per query (original/translated)
            docs_orig = self.vector_store.similarity_search(original_query, k=k) 
            relevant_docs.extend(docs_orig)
            
            if translated_query_for_retrieval:
                docs_trans = self.vector_store.similarity_search(translated_query_for_retrieval, k=k)
                relevant_docs.extend(docs_trans)
            
            source_file_paths = set()
            for doc in relevant_docs:
                if doc.metadata and 'source_path' in doc.metadata:
                    source_file_paths.add(doc.metadata['source_path'])
                elif doc.metadata and 'source_file' in doc.metadata:
                    source_file_path = doc.metadata['source_file']
                    # Attempt to make it a full path if it's just a filename.
                    # This logic assumes documents are in a specific directory if path isn't absolute.
                    if not os.path.isabs(source_file_path):
                        # Assuming 'documents_retrieval' as the base, adjust if necessary.
                        # This path needs to be consistent with where Streamlit expects to find files for download.
                        resolved_path = os.path.join(".", "documents_retrieval", os.path.basename(source_file_path))
                        if os.path.exists(resolved_path):
                            source_file_paths.add(resolved_path)
                        else:
                            logger.warning(f"Could not resolve relative source_file to an existing path: {source_file_path} (tried {resolved_path})")
                    else:
                        source_file_paths.add(source_file_path)
            
            if not source_file_paths and relevant_docs:
                logger.warning("Found relevant documents but could not extract source paths.")

            logger.info(f"Found {len(source_file_paths)} unique document path(s) for query: {query[:50]}...")
            return list(source_file_paths)
            
        except Exception as e:
            logger.error(f"Error retrieving document paths in get_documents_for_query: {str(e)}", exc_info=True)
            return []
    
    def chat_with_context(self, question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Chat with context from previous conversation.
        This is the main entry point for the Streamlit app.
        
        Args:
            question: Current question
            conversation_history: List of message dicts [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Response dictionary
        """
        if conversation_history is None:
            conversation_history = []
            
        # The conversation_history is now passed to answer_conversational
        # which handles the ConversationalRetrievalChain and its memory aspects.
        return self.answer_conversational(question, chat_history_messages=conversation_history)
