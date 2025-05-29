"""
RAG (Retrieval-Augmented Generation) system using Gemini 2.5 Pro.
"""

from typing import List, Dict, Any
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import logging

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
            temperature=0.25,
            max_output_tokens=2048
        )
        
        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize retrieval chain
        self.qa_chain = self._create_qa_chain()
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for market outlook Q&A."""
        
        template = """
        You are an expert financial analyst and CFA charterholder with extensive experience in crafting market outlooks and investment perspectives. Your primary objective is to provide a comprehensive, accurate, and insightful answer to the user's question by *exclusively* leveraging the information provided in the `Context` from financial reports and market analysis documents. Do not incorporate any external knowledge or make assumptions not supported by the given context.

        **Instructions:**
        1.  **Context Reliance:** Your entire response must be derived *solely* from the provided `Context`.
        2.  **Conciseness & Relevance:** Keep your response concise, aiming for under 1250 words. Prioritize and extract only the most relevant information that directly addresses the user's question.
        3.  **Structured Answer:** Provide a well-structured answer that includes:
            *   **Direct Answer:** A clear, direct, and succinct answer to the user's question.
            *   **Supporting Evidence:** Specific data points, key findings, or direct quotes from the `Context` that substantiate your answer.
            *   **Relevant Market Insights/Trends:** Synthesize broader market insights or economic trends that are explicitly mentioned or strongly implied within the `Context`.
            *   **Source Attribution:** For every piece of information used, clearly cite the name of the document or source file from the `Context`.
        4.  **Information Gap Handling:** If the provided `Context` does not contain sufficient or relevant information to accurately answer the question, you must state: "I do not have enough information in the provided documents to accurately answer this question based on the given context."
        5.  **Tone:** Maintain a professional, analytical, and objective tone throughout your response.

        **Context:**
        {context}

        **Question:**
        {question}

        **Answer:**
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self):
        """Create the QA retrieval chain."""
        try:
            retriever = self.vector_store.get_retriever(k=6)
            if not retriever:
                logger.error("Failed to get retriever from vector store")
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
    
    def _reinitialize_qa_chain(self):
        """Reinitialize the QA chain if it failed initially."""
        if not self.qa_chain:
            self.qa_chain = self._create_qa_chain()
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            # Reinitialize QA chain if needed
            if not self.qa_chain:
                self._reinitialize_qa_chain()
            
            if not self.qa_chain:
                return {
                    "answer": "The RAG system is not properly initialized. Please check the setup.",
                    "source_documents": [],
                    "error": "QA chain not available"
                }
            
            # Get response from the chain
            response = self.qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = response.get("result", "No answer generated")
            source_documents = response.get("source_documents", [])
            
            # Format source information
            sources = []
            for i, doc in enumerate(source_documents):
                source_info = {
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "source_file": doc.metadata.get("source_file", f"Document {i+1}"),
                    "page": doc.metadata.get("page", "Unknown")
                }
                sources.append(source_info)
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            return {
                "answer": answer,
                "source_documents": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
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
