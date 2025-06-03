# benchmark_rag.py

import os
import time
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
from utils import process_and_add_documents

def main():
    # 1) Load env & API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return

    # 2) Init vector store & RAG system
    vector_store = VectorStore(api_key=api_key)
    rag_system = RAGSystem(api_key=api_key, vector_store=vector_store)

    # 3) Jika vector store kosong, proses dokumen
    if vector_store.get_collection_count() == 0:
        print("Vector store kosong → memproses dokumen...")
        success = process_and_add_documents(vector_store, "./documents_retrieval")
        if not success:
            print("Gagal memproses dokumen. Benchmark tidak dapat dilanjutkan.")
            return
        print(f"   • Terindeks {vector_store.get_collection_count()} chunk teks")

    # 4) Sample query untuk benchmark
    sample_query = "Bagaimana undang-undang mengatur penggunaan data pribadi oleh penyelenggara sistem elektronik di Indonesia?"

    # 5) Warm-up
    _ = vector_store.similarity_search_with_score(sample_query, k=6)
    _ = rag_system.answer_conversational(sample_query, [])

    # 6) Benchmarking
    n_runs = 10
    retrieval_times = []
    total_times = []

    for i in range(n_runs):
        t0 = time.time()
        # retrieval saja
        _ = vector_store.similarity_search_with_score(sample_query, k=6)
        t1 = time.time()
        # full RAG QA
        _ = rag_system.answer_conversational(sample_query, [])
        t2 = time.time()

        retrieval_ms = (t1 - t0) * 1000
        total_ms     = (t2 - t0) * 1000

        retrieval_times.append(retrieval_ms)
        total_times.append(total_ms)
        print(f"Run {i+1:2d}: retrieval {retrieval_ms:.0f} ms, end-to-end {total_ms:.0f} ms")

    avg_ret = sum(retrieval_times) / n_runs
    avg_tot = sum(total_times)     / n_runs

    print("\n=== Hasil Rata-Rata (10 runs) ===")
    print(f"Average retrieval (top-6): {avg_ret:.0f} ms")
    print(f"Average end-to-end QA:    {avg_tot:.0f} ms")

if __name__ == "__main__":
    main()