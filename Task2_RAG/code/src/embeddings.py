"""
Module: src/embeddings.py
Generates normalized dense embeddings using SentenceTransformers for document chunks.
Saves all embeddings into the artifacts/ directory.
"""
import os
import json
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_embedding_model(model_name=DEFAULT_MODEL_NAME):
    """
    Loads SentenceTransformer model with graceful handling for cached/offline mode.
    """
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            del os.environ["HF_HUB_OFFLINE"]
            model = SentenceTransformer(model_name)
        else:
            raise
    return model

def generate_embeddings_for_chunks(chunk_size, model=None):
    """
    Loads chunks_{chunk_size}.json from artifacts/, computes L2-normalized embeddings,
    and saves to artifacts/embeddings_{chunk_size}.npy.
    """
    if model is None:
        model = get_embedding_model()
        
    chunk_file = ARTIFACTS_DIR / f"chunks_{chunk_size}.json"
    if not chunk_file.exists():
        raise FileNotFoundError(f"Could not find {chunk_file}. Run chunking first.")
        
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks of size {chunk_size}...")
    
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    
    out_file = ARTIFACTS_DIR / f"embeddings_{chunk_size}.npy"
    np.save(out_file, embeddings)
    
    print(f"Saved {embeddings.shape} embeddings to {out_file}")
    return embeddings

def build_all_embeddings(sizes=(100, 300)):
    model = get_embedding_model()
    for size in sizes:
        generate_embeddings_for_chunks(size, model=model)

if __name__ == "__main__":
    build_all_embeddings()
