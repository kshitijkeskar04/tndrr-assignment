"""
Module: src/retrieval.py
Vector similarity retrieval over document chunks using pure NumPy or FAISS.
Reads data strictly from artifacts/.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from src.embeddings import get_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def load_chunks_and_embeddings(chunk_size):
    """
    Loads chunks json and embeddings npy strictly from artifacts/.
    """
    chunk_path = ARTIFACTS_DIR / f"chunks_{chunk_size}.json"
    emb_path = ARTIFACTS_DIR / f"embeddings_{chunk_size}.npy"
        
    if not chunk_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for chunk size {chunk_size} in {ARTIFACTS_DIR}. "
            f"Run 'python rag.py --build-index' first."
        )
        
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    embeddings = np.load(emb_path)
    return chunks, embeddings

def retrieve_numpy(query_vector, chunks, vectors, top_k=5):
    """
    Retrieves top_k most similar chunks using pure NumPy dot product.
    Since vectors and query_vector are L2-normalized, dot product == cosine similarity.
    """
    scores = vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "page": chunk["page"],
            "chunk_id": chunk.get("chunk_id", f"idx_{idx}"),
            "text": chunk["text"]
        })
    return results

def retrieve_faiss(query_vector, chunks, vectors, top_k=5):
    """
    Retrieves top_k most similar chunks using FAISS IndexFlatIP.
    Falls back to NumPy if faiss is not installed.
    """
    try:
        import faiss
    except ImportError:
        return retrieve_numpy(query_vector, chunks, vectors, top_k)
        
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))
    
    q = np.expand_dims(query_vector.astype("float32"), axis=0)
    scores, indices = index.search(q, top_k)
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append({
            "rank": rank,
            "score": float(score),
            "page": chunk["page"],
            "chunk_id": chunk.get("chunk_id", f"idx_{idx}"),
            "text": chunk["text"]
        })
    return results

def search(query, chunk_size=100, top_k=5, method="numpy", model=None):
    """
    High-level search helper given a text query.
    """
    if model is None:
        model = get_embedding_model()
        
    chunks, vectors = load_chunks_and_embeddings(chunk_size)
    query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    
    if method == "faiss":
        return retrieve_faiss(query_vector, chunks, vectors, top_k=top_k)
    else:
        return retrieve_numpy(query_vector, chunks, vectors, top_k=top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval")
    parser.add_argument("--query", type=str, default="What are the four Vs of Big Data?")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--method", choices=["numpy", "faiss"], default="numpy")
    args = parser.parse_args()
    
    print(f"\nQuery: {args.query}")
    print(f"Retrieval Method: {args.method.upper()} | Chunk size: {args.chunk_size} words | Top-k: {args.top_k}")
    print("=" * 60)
    
    hits = search(args.query, chunk_size=args.chunk_size, top_k=args.top_k, method=args.method)
    for h in hits:
        print(f"\n[Rank {h['rank']}] Score: {h['score']:.4f} | Page: {h['page']} | Chunk: {h['chunk_id']}")
        print(f"Passage: {h['text'][:300]}...")
