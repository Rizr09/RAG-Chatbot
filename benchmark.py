# benchmark_rag.py

import os
import time
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
from utils import process_and_add_documents

def run_benchmark_for_query(rag_system, vector_store, query, query_name, n_runs=10):
    """Runs a benchmark for a given query and prints the results."""
    print(f"\n--- Benchmarking for: {query_name} ---")
    print(f"Query: \"{query}\"")

    # Warm-up run to avoid cold-start penalties
    _ = vector_store.similarity_search_with_score(query, k=4)
    _ = rag_system.answer_conversational(query, [])

    retrieval_times = []
    total_times = []

    for i in range(n_runs):
        t0 = time.time()
        # Benchmark retrieval only
        _ = vector_store.similarity_search_with_score(query, k=4)
        t1 = time.time()
        # Benchmark full RAG QA
        _ = rag_system.answer_conversational(query, [])
        t2 = time.time()

        retrieval_ms = (t1 - t0) * 1000
        total_ms = (t2 - t0) * 1000

        retrieval_times.append(retrieval_ms)
        total_times.append(total_ms)
        print(f"  Run {i+1:2d}: retrieval {retrieval_ms:.0f} ms, end-to-end {total_ms:.0f} ms")

    avg_ret = sum(retrieval_times) / n_runs
    avg_tot = sum(total_times) / n_runs

    print(f"\n  Average Results ({n_runs} runs) for '{query_name}':")
    print(f"  - Average retrieval (top-10): {avg_ret:.0f} ms")
    print(f"  - Average end-to-end QA:    {avg_tot:.0f} ms")
    return avg_tot

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

    # 3) If vector store is empty, process documents
    if vector_store.get_collection_count() == 0:
        print("Vector store is empty -> processing documents...")
        success = process_and_add_documents(vector_store, "./documents_retrieval")
        if not success:
            print("Failed to process documents. Benchmark cannot continue.")
            return
        print(f"   • Indexed {vector_store.get_collection_count()} text chunks")

    # 4) Sample queries for benchmarking
    query_id = "Apakah sudah ada UU yang membahas perlindungan data pribadi anak?"
    query_en = "Is there a law that discusses the protection of children's personal data?"

    # 5) Run benchmarks
    print("\nStarting RAG System Benchmark...")
    avg_id = run_benchmark_for_query(rag_system, vector_store, query_id, "Indonesian Query (No Translation)")
    avg_en = run_benchmark_for_query(rag_system, vector_store, query_en, "English Query (with Translation)")

    print("\n\n=== Overall Benchmark Summary ===")
    print(f"Indonesian (No Translation) Average: {avg_id:.0f} ms")
    print(f"English (with Translation) Average:  {avg_en:.0f} ms")
    if avg_id > 0:
        overhead = ((avg_en - avg_id) / avg_id) * 100
        print(f"Translation overhead: +{overhead:.1f}%")
    print("=================================")


if __name__ == "__main__":
    main()