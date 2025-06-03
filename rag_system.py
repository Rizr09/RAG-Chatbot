"""
RAG (Retrieval-Augmented Generation) system using Gemini 2.5 Pro.
"""

from typing import List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.tracers.schemas import Run
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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
            temperature=0.05,
            max_output_tokens=2048
        )
        
        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize retrieval chain
        self.qa_chain = self._create_qa_chain()
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for AI and Finance research paper Q&A."""
        
        template = """You are an expert AI and Finance Research Analyst. Your primary goal is to assist users by either answering their questions based on the provided `Context` or by providing them with the relevant documents if their query indicates a request for the document itself.

**Response Modes:**

1.  **Answering Mode:**
    *   If the user asks a question seeking information, insights, summaries, or specific details *from* the documents, provide a comprehensive, accurate, and insightful textual answer derived *solely* from the provided `Context`.
    *   Do not incorporate any external knowledge or make assumptions not supported by the given context.
    *   Follow the detailed answering instructions below.

2.  **Document Provisioning Mode:**
    *   If you judge that the user's query is primarily a request *for* one or more documents, papers, or files themselves (e.g., "send me the paper on X", "can I get the document about Y?", "find the report on Z and related articles"), then you MUST respond *ONLY* with a single JSON object in the following exact format. Do not add any text before or after this JSON object:

```json
{{  "intent": "provide_document",
  "search_query_for_docs": "<keywords you determine are best for finding the requested document(s)>",
  "user_message": "<a short, friendly message for the user, e.g., 'Certainly, I found the following document(s) related to your request for X:'>"
}}
```

            *   The `search_query_for_docs` should be your best assessment of the core subject of the document(s) the user wants.

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
        
        # The input_variables should ONLY be 'context' and 'question' as used in the template for substitution.
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"] # Strictly 'context' and 'question'
        )
    
    def _create_qa_chain(self, translated_question_for_retrieval: str = None, original_question: str = None):
        """Create the QA retrieval chain."""
        try:
            # Use a combined query for document retrieval if a translated query is provided
            # This helps fetch documents that might match either the original or translated query
            if translated_question_for_retrieval and original_question:
                # Retrieve documents for original and translated queries separately
                # and combine them, removing duplicates.
                # We ask for more documents (k=5 for each) to increase recall before combining.
                # Ensure retriever itself is valid before calling get_relevant_documents
                base_retriever_orig = self.vector_store.get_retriever(k=5)
                base_retriever_trans = self.vector_store.get_retriever(k=5)

                original_docs = []
                if base_retriever_orig:
                    original_docs = base_retriever_orig.get_relevant_documents(original_question)
                
                translated_docs = []
                if base_retriever_trans:
                    translated_docs = base_retriever_trans.get_relevant_documents(translated_question_for_retrieval)

                # Combine and deduplicate documents based on content and source_file to avoid redundant context
                combined_docs_dict = {}
                for doc in original_docs + translated_docs:
                    doc_key = (doc.page_content, doc.metadata.get('source_file'), doc.metadata.get('page'))
                    if doc_key not in combined_docs_dict:
                        combined_docs_dict[doc_key] = doc
                
                unique_combined_docs = list(combined_docs_dict.values())[:6] # Keep up to 6 unique docs

                # Create a custom retriever from these unique combined documents
                # Langchain's retriever interface expects get_relevant_documents and aget_relevant_documents
                class CustomRetriever(BaseRetriever): # Inherit from BaseRetriever
                    documents: List[Document]

                    class Config: # Add Pydantic config for arbitrary types
                        arbitrary_types_allowed = True
                        
                    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                        return self.documents
                    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                        return self.documents
                
                if not unique_combined_docs: # If no documents found, use a retriever that returns nothing
                    logger.warning("No documents found for combined query. QA might be uninformative.")
                    # Fallback to an empty retriever or handle as an error condition.
                    # For now, let's create a retriever that returns an empty list.
                    class EmptyRetriever(BaseRetriever):
                        class Config:
                            arbitrary_types_allowed = True
                        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                            return []
                        async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                            return []
                    retriever = EmptyRetriever()

                else:
                    retriever = CustomRetriever(documents=unique_combined_docs)


            else: # Fallback to original behavior if no translation
                retriever = self.vector_store.get_retriever(k=6)

            if not retriever:
                logger.error("Failed to get retriever from vector store or custom retriever creation failed")
                return None
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "document_variable_name": "context"
                },
                return_source_documents=True            )
            
            logger.info("QA chain created successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            return None
    
    def _reinitialize_qa_chain(self, translated_question_for_retrieval: str = None, original_question: str = None):
        """Reinitialize the QA chain if it failed initially."""
        # Always recreate the chain if we might need a new retriever strategy (e.g., due to translation)
        self.qa_chain = self._create_qa_chain(translated_question_for_retrieval=translated_question_for_retrieval, original_question=original_question)
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources, or get document paths if LLM decides so.
        Handles potential translation for Indonesian questions.
        
        Args:
            question: The question to ask
            
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

            # Reinitialize QA chain, potentially with a combined retriever strategy based on translated query
            self._reinitialize_qa_chain(translated_question_for_retrieval=translated_question_for_retrieval, original_question=original_question)
            
            if not self.qa_chain:
                return {
                    "type": "error", # Added type for clarity
                    "answer": "The RAG system is not properly initialized. Please check the setup.",
                    "source_documents": [],
                    "error": "QA chain not available"
                }
            
            # The 'query' to RetrievalQA should be the one the LLM sees and answers.
            # If original was Indonesian, we use that, as per prompt instructions for language handling.
            llm_query = original_question
            llm_response_raw = self.qa_chain.invoke({"query": llm_query})
            
            raw_answer_text = llm_response_raw.get("result", "").strip()
            source_documents_from_chain = llm_response_raw.get("source_documents", [])

            # Attempt to parse the LLM's response as JSON for document provisioning intent
            try:
                potential_json_str = raw_answer_text
                # If the LLM includes markdown ```json ... ```, try to extract it.
                if raw_answer_text.startswith("```json"):
                    potential_json_str = raw_answer_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif raw_answer_text.startswith("```") and raw_answer_text.endswith("```"):
                    potential_json_str = raw_answer_text[3:-3].strip()
                
                # More general extraction if it's just embedded
                json_start_index = potential_json_str.find('{')
                json_end_index = potential_json_str.rfind('}')

                if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                    extracted_json_str = potential_json_str[json_start_index : json_end_index+1]
                    llm_output_json = json.loads(extracted_json_str)
                    if isinstance(llm_output_json, dict) and llm_output_json.get("intent") == "provide_document":
                        search_query_for_docs = llm_output_json.get("search_query_for_docs", original_question)
                        user_message = llm_output_json.get("user_message", "Here are the documents I found:")
                        
                        logger.info(f"LLM signaled 'provide_document' intent. Search query for docs: '{search_query_for_docs}'")
                        document_paths = self.get_documents_for_query(search_query_for_docs, k=3)
                        
                        return {
                            "type": "documents",
                            "document_paths": document_paths,
                            "user_message": user_message,
                            "query_used_for_retrieval": search_query_for_docs
                        }
                    else:
                        # Parsed JSON but not the expected intent, treat as normal answer (or part of an answer)
                        logger.info("Parsed JSON from LLM but not 'provide_document' intent. Proceeding with Answering Mode.")
                        raise json.JSONDecodeError("JSON parsed but not provide_document intent", extracted_json_str, 0) # Force fallback
                else:
                    # No clear JSON block found
                    logger.info(f"No JSON block found in LLM response: '{raw_answer_text[:100]}...'. Proceeding with Answering Mode.")
                    raise json.JSONDecodeError("No JSON block found", raw_answer_text, 0) # Force fallback

            except json.JSONDecodeError:
                logger.info(f"LLM response ('{raw_answer_text[:100]}...') is not the expected provide_document JSON. Proceeding with Answering Mode.")
            # Fall-through to Answering Mode if JSON parsing/intent matching failed

            # If not provide_document intent, proceed as Answering Mode
            answer = raw_answer_text # This will be the original text if JSON parsing failed
            
            # If the original question was Indonesian and the answer came out in English, translate it back.
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
            
            # Format source information from the chain
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
                "type": "answer", # Added type for clarity
                "answer": answer,
                "source_documents": sources,
                "question": original_question
            }
            
        except Exception as e:
            logger.error(f"Error processing question in ask_question: {str(e)}", exc_info=True) # Added exc_info
            return {
                "type": "error", # Added type for clarity
                "answer": f"An error occurred while processing your question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Get relevant documents for a query without generating an answer.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_documents_for_query(self, query: str, k: int = 3) -> List[str]:
        """
        Get relevant document file paths for a query.
        Handles potential translation for Indonesian queries.
        Returns a list of unique source file paths.
        
        Args:
            query: Search query
            k: Number of documents to retrieve per language query
            
        Returns:
            List of unique source file paths
        """
        try:
            original_query = query
            translated_query_for_retrieval = None
            
            try:
                # Run async detection in a separate thread
                detected_lang_result = run_async_in_thread(self.translator.detect(query))
                detected_lang = detected_lang_result.lang
                if detected_lang.startswith('id'):
                    # Run async translation in a separate thread
                    translation_result = run_async_in_thread(self.translator.translate(query, src='id', dest='en'))
                    translated_query_for_retrieval = translation_result.text
                    logger.info(f"Original (ID) for doc retrieval: '{query}', Translated (EN): '{translated_query_for_retrieval}'")
            except Exception as e:
                logger.warning(f"Language detection/translation for doc retrieval failed: {e}. Proceeding with original query.")

            relevant_docs = []
            # Retrieve for original query
            docs_orig = self.vector_store.similarity_search(original_query, k=k)
            relevant_docs.extend(docs_orig)
            
            # Retrieve for translated query if available
            if translated_query_for_retrieval:
                docs_trans = self.vector_store.similarity_search(translated_query_for_retrieval, k=k)
                relevant_docs.extend(docs_trans)
            
            # Extract unique source file paths
            source_file_paths = set()
            for doc in relevant_docs:
                if doc.metadata and 'source_path' in doc.metadata:
                    source_file_paths.add(doc.metadata['source_path'])
                elif doc.metadata and 'source_file' in doc.metadata: # Fallback to source_file if source_path not present
                    # This assumes source_file is just the filename, may need adjustment if it's a path
                    # For now, we'll log a warning if we have to use this and it's not an absolute path
                    if os.path.isabs(doc.metadata['source_file']):
                        source_file_paths.add(doc.metadata['source_file'])
                    else:
                        # If it's a relative path or just a filename, we need to decide how to make it accessible
                        # For now, let's assume it's relative to a known documents directory if not absolute.
                        # This part might need to be more robust depending on how `source_file` is populated.
                        logger.warning(f"Using non-absolute 'source_file' metadata: {doc.metadata['source_file']}. Its usability depends on context.")
                        # Attempt to construct a path assuming it's a filename in a known directory
                        # For this example, let's assume `documents_retrieval` is the base. This is a simplification.
                        # A more robust solution would ensure `source_path` is always populated correctly.
                        potential_path = os.path.join(".", "documents_retrieval", doc.metadata['source_file']) 
                        if os.path.exists(potential_path):
                           source_file_paths.add(potential_path)
                        else:
                           logger.warning(f"Could not resolve path for source_file: {doc.metadata['source_file']}")

            if not source_file_paths and relevant_docs: # If we have docs but no paths
                logger.warning("Found relevant documents but could not extract source paths.")

            logger.info(f"Found {len(source_file_paths)} unique document(s) for query: {query[:50]}...")
            return list(source_file_paths)
            
        except Exception as e:
            logger.error(f"Error retrieving document paths: {str(e)}")
            return []
    
    def chat_with_context(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Chat with context from previous conversation.
        
        Args:
            question: Current question
            conversation_history: Previous conversation history
            
        Returns:
            Response dictionary
        """
        # For now, treat each question independently
        # This can be enhanced to maintain conversation context
        return self.ask_question(question)
