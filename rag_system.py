"""
RAG (Retrieval-Augmented Generation) system using Gemini 2.5 Pro.
"""

from typing import List, Dict, Any, Tuple
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
import json # Added for parsing LLM output
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=api_key,
            temperature=0.05,
            max_output_tokens=2048
        )
        
        # Create custom prompt template for combining documents
        self.combine_docs_prompt = self._create_combine_docs_prompt()
        
        # Create prompt for condensing question
        self.condense_question_prompt = self._create_condense_question_prompt()

    def _create_combine_docs_prompt(self) -> PromptTemplate:
        """Create a custom prompt template for AI and Finance research paper Q&A, used by the combine_docs_chain."""
        
        template = """Anda adalah seorang ahli multibahasa di bidang Hukum Telekomunikasi, Informatika, Siber, dan Internet di Indonesia. Tujuan utama Anda adalah membantu pengguna dengan menjawab pertanyaan mereka berdasarkan `Konteks` yang diberikan, atau dengan menyediakan dokumen yang relevan jika permintaan mereka mengindikasikan permintaan untuk dokumen itu sendiri.

**PENTING: Deteksi Bahasa dan Respons**
- **Deteksi bahasa dari `Pertanyaan` pengguna.**
- **Jawab dalam bahasa yang SAMA dengan bahasa `Pertanyaan` pengguna.** Misalnya, jika pertanyaan dalam Bahasa Inggris, jawab dalam Bahasa Inggris. Jika dalam Bahasa Indonesia, jawab dalam Bahasa Indonesia.

**Mode Respons:**

1.  **Mode Menjawab:**
    *   Jika pengguna mengajukan pertanyaan untuk mencari informasi, wawasan, ringkasan, atau detail spesifik *dari* dokumen, berikan jawaban tekstual yang komprehensif, akurat, dan mendalam yang berasal *hanya* dari `Konteks` yang disediakan.
    *   Jangan memasukkan pengetahuan eksternal atau membuat asumsi yang tidak didukung oleh konteks yang diberikan.
    *   Ikuti instruksi menjawab terperinci di bawah ini.

2.  **Mode Penyediaan Dokumen:**
    *   Jika Anda menilai bahwa permintaan pengguna adalah permintaan *untuk* satu atau lebih dokumen, makalah, atau file itu sendiri (misalnya, "kirimkan saya peraturan tentang X", "can I get the law about Y?", "temukan kebijakan internet tentang Z dan pasal-pasal terkait"), maka Anda HARUS merespons *HANYA* dengan satu objek JSON dalam format yang sama persis berikut ini. Jangan menambahkan teks apa pun sebelum atau sesudah objek JSON ini:
    *   **PENTING**: Jika pengguna meminta dokumen yang sangat spesifik (misalnya, dengan menyebutkan nomor dan tahun seperti "UU Nomor 3 Tahun 1989"), `search_query_for_docs` harus sama persis dengan nama dokumen tersebut untuk memastikan pencarian yang akurat. Jika pengguna meminta beberapa dokumen spesifik, gabungkan nama-nama tersebut dalam query.
    *   Perkirakan jumlah dokumen yang diminta pengguna. Jika mereka meminta satu dokumen spesifik, setel `document_count` ke 1. Jika mereka meminta dua, setel ke 2, dan seterusnya. Jika permintaan bersifat umum ("kirimkan saya dokumen tentang telekomunikasi"), Anda dapat menyetel `document_count` ke angka yang wajar seperti 3 atau 5.

```json
{{
  "intent": "provide_document",
  "search_query_for_docs": "<kata kunci yang menurut Anda terbaik untuk menemukan dokumen yang diminta, dengan mempertimbangkan pertanyaan saat ini. Terjemahkan kueri ini ke Bahasa Indonesia jika pertanyaan asli dalam bahasa lain.>",
  "user_message": "<pesan singkat dan ramah untuk pengguna dalam bahasa ASLI pengguna, mis., 'Tentu, saya menemukan dokumen berikut...' atau 'Sure, here are the documents...'>",
  "document_count": <jumlah dokumen yang Anda perkirakan diminta pengguna>
}}
```

            *   `search_query_for_docs` harus merupakan penilaian terbaik Anda tentang subjek inti dari dokumen yang diinginkan pengguna, dan HARUS dalam Bahasa Indonesia untuk pencarian dokumen.

**Instruksi Menjawab Terperinci (untuk Mode Menjawab):**
*   **Ketergantungan Konteks:** Seluruh respons Anda harus berasal *hanya* dari `Konteks` yang disediakan.
*   **Penanganan Bahasa:** Selalu berikan jawaban Anda dalam bahasa yang sama dengan `Pertanyaan` pengguna.
*   **Ringkas & Relevan:** Jaga agar respons Anda tetap ringkas. Prioritaskan dan ekstrak hanya informasi yang paling relevan yang secara langsung menjawab pertanyaan pengguna.
*   **Jawaban Terstruktur:** Sertakan jawaban langsung, bukti pendukung (data, temuan, kutipan), wawasan relevan, dan atribusi sumber (nama dokumen, nomor halaman jika tersedia).
*   **Penanganan Kesenjangan Informasi:** Jika `Konteks` tidak mencukupi, nyatakan: "Saya tidak memiliki cukup informasi..." (atau terjemahan yang sesuai).
*   **Nada:** Pertahankan nada yang profesional, analitis, dan objektif.

**PENTING:** Pilih HANYA SATU mode per permintaan. Jika menyediakan dokumen, HANYA keluarkan JSON. Jika tidak, berikan jawaban tekstual.

**Konteks:**
{context}

**Pertanyaan:**
{input}

**Jawaban:**
""" # Ensure no stray characters after this final triple quote.
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )

    def _create_condense_question_prompt(self) -> PromptTemplate:
        """Create a prompt template for condensing the current question and chat history into a standalone question."""
        template = """Berdasarkan percakapan berikut dan pertanyaan lanjutan, ubah pertanyaan lanjutan tersebut menjadi pertanyaan yang dapat berdiri sendiri. Pertahankan bahasa asli dari 'Input Lanjutan'.

Riwayat Obrolan:
{chat_history}

Input Lanjutan: {question}
Pertanyaan mandiri:"""
        return PromptTemplate.from_template(template)

    def _get_retriever_for_query(self, query: str, k: int = 5) -> List[Document]:
        """Helper to get documents for a single query."""
        retriever = self.vector_store.get_retriever(k=k)
        if retriever:
            return retriever.invoke(query)
        return []

    def _combine_and_deduplicate_docs(self, *doc_lists: List[Document]) -> List[Document]:
        """Helper to combine and deduplicate documents from multiple lists."""
        combined_docs_dict = {}
        for doc_list in doc_lists:
            for doc in doc_list:
                doc_key = (doc.page_content, doc.metadata.get('source_file'), doc.metadata.get('page'))
                if doc_key not in combined_docs_dict:
                    combined_docs_dict[doc_key] = doc
        return list(combined_docs_dict.values())

    def _create_custom_retriever_instance(self, documents: List[Document]) -> BaseRetriever:
        """Helper to create either CustomRetriever or EmptyRetriever."""
        class CustomRetriever(BaseRetriever):
            documents: List[Document]

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                return self.documents

            async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                return self.documents

        if not documents:
            logger.warning("No documents found. QA might be uninformative.")
            class EmptyRetriever(BaseRetriever):
                class Config:
                    arbitrary_types_allowed = True
                def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
                    return []
                async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
                    return []
            return EmptyRetriever()
        else:
            return CustomRetriever(documents=documents)

    def _get_custom_retriever(self, translated_question_for_retrieval: str = None, original_question: str = None) -> BaseRetriever:
        """
        Creates a custom retriever that combines results from original and translated queries.
        """
        try:
            if translated_question_for_retrieval and original_question:
                original_docs = self._get_retriever_for_query(original_question, k=5)
                translated_docs = self._get_retriever_for_query(translated_question_for_retrieval, k=5)
                
                unique_combined_docs = self._combine_and_deduplicate_docs(original_docs, translated_docs)[:5]
                return self._create_custom_retriever_instance(unique_combined_docs)
            else: # Fallback to original behavior if no translation
                retriever = self.vector_store.get_retriever(k=5)
                if not retriever:
                    logger.error("Failed to get default retriever from vector store.")
                    raise ValueError("Retriever not available.")
                return retriever
        except Exception as e:
            logger.error(f"Error creating custom retriever: {e}", exc_info=True)
            # Fallback to a simple retriever on error
            try:
                retriever = self.vector_store.get_retriever(k=5) # Reduced k for fallback
                if not retriever:
                    raise ValueError("Fallback retriever also failed.")
                logger.warning("Fell back to a simple retriever due to an error in custom retriever creation.")
                return retriever
            except Exception as fallback_e:
                logger.error(f"Critical error: Could not create any retriever: {fallback_e}", exc_info=True)
                # Return an empty retriever as a last resort
                return self._create_custom_retriever_instance([])

    def _create_rag_chain(self, retriever: BaseRetriever):
        """Create a RAG chain that is aware of conversation history."""
        try:
            # Contextualize question prompt
            contextualize_q_system_prompt = """Berdasarkan riwayat obrolan dan pertanyaan terbaru pengguna, yang mungkin merujuk pada konteks dalam riwayat obrolan, rumuskan pertanyaan mandiri yang dapat dipahami tanpa riwayat obrolan. JANGAN menjawab pertanyaan itu, cukup rumuskan ulang jika perlu dan jika tidak, kembalikan sebagaimana adanya. Pertahankan bahasa asli dari pertanyaan pengguna."""
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                self.llm, retriever, contextualize_q_prompt
            )

            # Answering prompt
            qa_prompt = ChatPromptTemplate.from_template(self.combine_docs_prompt.template)
            
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            logger.info("RAG chain created successfully.")
            return rag_chain
        except Exception as e:
            logger.error(f"Error creating RAG chain: {str(e)}", exc_info=True)
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
            question_for_rag = question # Use the original question directly

            # Prepare a simple retriever.
            retriever = self.vector_store.get_retriever(k=5)
            if not retriever:
                logger.error("Failed to get retriever from vector store.")
                return {
                    "type": "error",
                    "answer": "The RAG system's document retriever could not be initialized.",
                    "source_documents": [],
                    "error": "Retriever not available"
                }

            # Convert Streamlit chat history to Langchain's expected format
            formatted_chat_history = []
            for msg in chat_history_messages:
                if msg["role"] == "user":
                    formatted_chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_chat_history.append(AIMessage(content=msg["content"]))


            # Create or get the conversational chain
            rag_chain = self._create_rag_chain(retriever)
            
            if not rag_chain:
                return {
                    "type": "error",
                    "answer": "The RAG system's conversational chain could not be initialized. Please check the setup.",
                    "source_documents": [],
                    "error": "Conversational chain not available"
                }
            
            # Invoke the chain with the current question and chat history
            llm_response_raw = rag_chain.invoke({
                "input": question_for_rag,
                "chat_history": formatted_chat_history
            })
            
            raw_answer_text = llm_response_raw.get("answer", "").strip() # 'answer' is the key from the chain
            source_documents_from_chain = llm_response_raw.get("context", [])

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
                        search_query_for_docs = llm_output_json.get("search_query_for_docs", question_for_rag) # Fallback to rag question
                        user_message = llm_output_json.get("user_message", "Berikut dokumen yang saya temukan berdasarkan percakapan kita:")
                        document_count = llm_output_json.get("document_count", 1) # Default to 1 if not present
                        
                        logger.info(f"LLM signaled 'provide_document' intent. Search query for docs: '{search_query_for_docs}', count: {document_count}")
                        document_paths = self.get_documents_for_query(search_query_for_docs, k=document_count) # Use the dedicated method
                        
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

            # The answer from the LLM is now in Indonesian. No translation needed.
            answer = raw_answer_text
            
            sources = []
            source_filenames = set() # Use a set to store unique filenames
            for i, doc in enumerate(source_documents_from_chain):
                source_file = doc.metadata.get("source_file")
                if source_file:
                    # Get filename without extension
                    filename_without_ext = os.path.splitext(os.path.basename(source_file))[0]
                    source_filenames.add(filename_without_ext)

                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "source_file": doc.metadata.get("source_file", f"Document {i+1}"),
                    "page": doc.metadata.get("page", "Unknown")
                }
                sources.append(source_info)

            if source_filenames:
                answer += "\n\nsumber:\n" + "\n".join(sorted(list(source_filenames)))
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            return {
                "type": "answer",
                "answer": answer,
                "source_documents": sources,
                "question": question # Storing original question for context if needed by UI
            }
            
        except Exception as e:
            logger.error(f"Error processing question in answer_conversational: {str(e)}", exc_info=True)
            return {
                "type": "error",
                "answer": f"An error occurred while processing your question: {str(e)}",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
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
    
    def get_documents_for_query(self, query: str, k: int = 5) -> List[str]:
        """
        Get relevant document file paths for a query.
        The query is expected to be in Indonesian, as generated by the LLM.
        Returns a list of unique source file paths.
        This is used by the 'provide_document' intent.
        """
        try:
            # The query from the LLM is now in Indonesian, so no translation is needed.
            logger.info(f"Retrieving document paths for Indonesian query: '{query}'")
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
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
