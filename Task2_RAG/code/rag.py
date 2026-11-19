"""
Task 2: Retrieval-Augmented Generation (RAG) From Scratch.
Implemented over the World Travel & Tourism Council (WTTC) AI Report (44 pages).
No LangChain, LlamaIndex, or third-party RAG frameworks.
Embedding model: sentence-transformers/all-MiniLM-L6-v2.
Vector search: Pure NumPy cosine similarity (with optional FAISS IndexFlatIP support).
Answer generation: Document-grounded LLM completions via standard OpenAI-compatible client.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Ensure local offline HF cache is prioritized
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chunking import build_and_save_chunks, extract_pages, create_chunks
from src.embeddings import get_embedding_model, build_all_embeddings, generate_embeddings_for_chunks
from src.retrieval import load_chunks_and_embeddings, retrieve_numpy, retrieve_faiss, search
from src.evaluate import run_evaluation

PDF_PATH = ROOT / "data" / "2024-wttc-introduction-to-ai.pdf"
QUESTIONS_PATH = ROOT / "data" / "test_questions.json"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

def ensure_index_ready(chunk_sizes=(100, 300)):
    """
    Checks if chunks and embeddings exist; builds them if missing.
    """
    needed = []
    for s in chunk_sizes:
        c_path = ARTIFACTS_DIR / f"chunks_{s}.json"
        e_path = ARTIFACTS_DIR / f"embeddings_{s}.npy"
        if not c_path.exists() or not e_path.exists():
            needed.append(s)
            
    if needed:
        print(f"Building index artifacts for chunk sizes: {needed}...")
        build_and_save_chunks(chunk_sizes=chunk_sizes)
        build_all_embeddings(sizes=chunk_sizes)
        print("Index ready!")

def ask_question(question, chunk_words=100, top_k=5, use_faiss=False):
    """
    Retrieves top passages and optionally calls LLM to generate an answer.
    """
    ensure_index_ready([chunk_words])
    chunks, vectors = load_chunks_and_embeddings(chunk_words)
    model = get_embedding_model()
    
    q_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    
    if use_faiss:
        results = retrieve_faiss(q_vec, chunks, vectors, top_k=top_k)
        retrieval_backend = "FAISS (IndexFlatIP)"
    else:
        results = retrieve_numpy(q_vec, chunks, vectors, top_k=top_k)
        retrieval_backend = "NumPy (Cosine Similarity)"
        
    print("\n" + "=" * 65)
    print(f"QUERY: {question}")
    print(f"Engine: {retrieval_backend} | Chunk size: {chunk_words} words | Top-k: {top_k}")
    print("=" * 65)
    
    print("\n--- TOP RETRIEVED PASSAGES ---")
    for r in results:
        print(f"\n[Rank {r['rank']}] Score: {r['score']:.4f} | Source Page: {r['page']} | Chunk: {r['chunk_id']}")
        print(f"Text:\n{r['text']}")
        print("-" * 65)
        
    # Optional LLM generation
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None
            )
            context = "\n\n".join(
                f"[Source: PDF Page {r['page']}, Chunk {r['chunk_id']}]\n{r['text']}"
                for r in results
            )
            llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            print(f"\nGenerating grounded answer using LLM ({llm_model})...")
            
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict, factual Question-Answering assistant. "
                            "Answer the user's question using ONLY the retrieved source passages below. "
                            "Always cite the exact PDF Page number(s) for every fact you state. "
                            "If the passages do not contain enough information to answer the question, "
                            "clearly state: 'The provided document passages do not contain this information.'"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nRetrieved Passages:\n{context}"
                    }
                ],
                temperature=0.1
            )
            answer = completion.choices[0].message.content
            print("\n" + "=" * 65)
            print("GENERATED ANSWER (Grounded with Citations):")
            print("=" * 65)
            print(answer)
            print("=" * 65)
        except Exception as e:
            print(f"\nNote: LLM answer generation failed: {e}")
            print("Retrieved passages above contain the source evidence.")
    else:
        print("\n[Optional LLM Answer Generation]")
        print("To generate a synthesis with an LLM, set environment variables:")
        print("  $env:OPENAI_API_KEY='sk-...'")
        print("  $env:LLM_MODEL='gpt-4o-mini'  # optional")
        print("  $env:OPENAI_BASE_URL='...'     # optional for non-OpenAI endpoints")
        print("Pure vector retrieval output is shown above without requiring an LLM.")

def main():
    parser = argparse.ArgumentParser(
        description="RAG from Scratch: Question-Answering over WTTC 2024 AI Report"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run comprehensive retrieval evaluation for 100-word and 300-word chunks"
    )
    parser.add_argument(
        "--ask",
        type=str,
        help="Ask a natural language question over the report"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Extract PDF and regenerate chunk files and vector embeddings"
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        choices=[100, 300],
        default=100,
        help="Chunk size in words (default: 100)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve (default: 5)"
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use FAISS IndexFlatIP instead of NumPy cosine similarity"
    )
    
    args = parser.parse_args()
    
    if args.build_index:
        build_and_save_chunks()
        build_all_embeddings()
    elif args.evaluate:
        ensure_index_ready([100, 300])
        run_evaluation(chunk_sizes=(100, 300), top_k=args.top_k)
    elif args.ask:
        ask_question(
            args.ask,
            chunk_words=args.chunk_words,
            top_k=args.top_k,
            use_faiss=args.use_faiss
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
