# Task 2 — Retrieval-Augmented Generation (RAG) from Scratch

A complete, lightweight Question-Answering (RAG) system built from first principles over the **World Travel & Tourism Council (WTTC) 2024 AI Report** (44 pages).

> **Pure Implementation Constraint**: This system is built **without** LangChain, LlamaIndex, Haystack, or any external RAG framework. It utilizes only `pypdf`, `sentence-transformers`, `numpy` (with optional `faiss-cpu`), and standard Python libraries, with an optional OpenAI-compatible LLM client for final document-grounded answer generation.

---

## 1. Project Directory Structure

```text
RAG_Task2_WTTC/
├── data/
│   ├── 2024-wttc-introduction-to-ai.pdf   # Public document source (44 pages)
│   └── test_questions.json                # 15 authored test questions with gold passages
├── artifacts/
│   ├── extracted_text.txt                 # Full cleaned extracted text (pages 3–43)
│   ├── chunks_100.json                    # 100-word chunks (20-word overlap, 198 chunks)
│   ├── chunks_300.json                    # 300-word chunks (20-word overlap, 73 chunks)
│   ├── embeddings_100.npy                 # (198, 384) L2-normalized dense embeddings
│   └── embeddings_300.npy                 # (73, 384) L2-normalized dense embeddings
├── results/
│   ├── summary.json                       # Comparative benchmark summary (Recall@5, MRR@5)
│   ├── evaluation_results.csv             # Per-query evaluation breakdown in tabular format
│   ├── evaluation_detailed.json           # Comprehensive per-query retrieved passages & scores
│   ├── chunk_100_evaluation.json          # Detailed evaluation metrics for 100-word chunks
│   └── chunk_300_evaluation.json          # Detailed evaluation metrics for 300-word chunks
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── chunking.py                        # PDF page extraction & sliding-window chunk generator
│   ├── embeddings.py                      # Vector embedding computation & caching
│   ├── retrieval.py                       # Pure NumPy dot-product & FAISS retrieval engines
│   └── evaluate.py                        # Metric evaluation (Recall@5, MRR@5, rank analysis)
├── tests/
│   ├── check_pdf.py                       # Diagnostic script for PDF text extraction & stats
│   └── inspect_pages.py                   # Diagnostic script verifying chunk-to-question mappings
├── rag.py                                 # Main CLI entry point for evaluation, querying, and indexing
├── evaluate.py                            # Convenience runner for evaluation
├── requirements.txt                       # Project dependencies
├── SUBMISSION_REPORT.md                   # Full comprehensive submission report addressing all deliverables
└── README.md                              # Project documentation & execution guide
```

---

## 2. Dataset Suitability

- **Document**: World Travel & Tourism Council (WTTC), *Introduction to Artificial Intelligence (AI) Technology: Guide for Travel & Tourism Leaders* (January 2024).
- **Page Count**: 44 physical PDF pages (Main report, annexes, and terminology glossary span printed pages 3–43). This strictly fulfills the 30-page minimum requirement.
- **Content Coverage**: Covers AI history (Alan Turing), the 3 core pillars (algorithms, data, compute), machine learning vs. deep learning, learning paradigms (supervised, unsupervised, reinforcement), Big Data (4 Vs, storage), hardware acceleration (GPU vs. TPU), AI capability taxonomies (ANI, AGI, ASI), generative AI, the digital divide, AI hallucinations, and comprehensive AI terminology.

---

## 3. Architecture & Pipeline

```text
[PDF Document (44 pages)]
         │
         ▼  pypdf text extraction (pages 3–43)
[Clean Extracted Text]
         │
         ▼  Sliding-window word splitting (20-word overlap)
[Chunks (100w / 300w)]
         │
         ▼  SentenceTransformers (all-MiniLM-L6-v2)
[Normalized Dense Vectors (N x 384)]
         │
         ├─── Query: "What are the four Vs of Big Data?"
         │       │
         │       ▼  model.encode(query, normalize=True)
         │    [Query Vector (1 x 384)]
         │       │
         ▼       ▼  NumPy Dot Product: scores = vectors @ q_vec
[Top-k Retrieved Passages (Rank, Score, Page)]
         │
         ▼  (Optional) OpenAI-compatible API
[Grounded Answer with Page Citations]
```

1. **Extraction**: `src/chunking.py` extracts text page-by-page (pages 3–43). In this document, physical PDF pages match printed page numbers directly.
2. **Chunking**: Chunks are generated using word-level sliding windows (100 words and 300 words) with a 20-word overlap to prevent semantic boundary truncation.
3. **Embeddings**: `src/embeddings.py` uses `sentence-transformers/all-MiniLM-L6-v2` to compute 384-dimensional dense vectors. All embeddings are L2-normalized so that inner product equals cosine similarity.
4. **Retrieval**: `src/retrieval.py` calculates cosine similarity via pure NumPy matrix multiplication (`scores = vectors @ query_vector`), sorting the scores in descending order. Optional FAISS `IndexFlatIP` is also supported.
5. **Evaluation**: `src/evaluate.py` measures whether the gold source page appears in the top 5 retrieved chunks for each of 15 authored questions, tracking Recall@5, hit ranks, and Mean Reciprocal Rank (MRR@5).
6. **Answer Generation**: `rag.py` formats the top passages into a strict grounding prompt for an OpenAI-compatible LLM with mandated page citations.

---

## 4. Setup Instructions

### Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
venv\Scripts\Activate.ps1
# On Linux/macOS:
# source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

> **Note on Windows / FAISS**: `faiss-cpu` is optional. If installation of `faiss-cpu` fails on Windows, the system runs with pure NumPy cosine similarity by default.

---

## 5. Usage & Commands

### A. Run Full Retrieval Evaluation
Compare 100-word and 300-word chunks across all 15 test questions:
```powershell
python rag.py --evaluate
# Or directly:
python evaluate.py
```
Outputs:
- Terminal comparison table of Recall@5 and MRR@5.
- Formatted reports in `results/summary.json`, `results/evaluation_results.csv`, and `results/evaluation_detailed.json`.

### B. Ask a Natural Language Question (Vector Search)
```powershell
# Retrieve using 100-word chunks (default)
python rag.py --ask "What are the four Vs of Big Data?"

# Retrieve using 300-word chunks
python rag.py --ask "How does a TPU differ from a GPU?" --chunk-words 300

# Retrieve top 3 passages
python rag.py --ask "What is Explainable AI?" --top-k 3
```

### C. Ask with Grounded LLM Answer Generation
Set your API key (and optional endpoint/model):
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:LLM_MODEL="gpt-4o-mini"
# Optional for third-party OpenAI-compatible endpoints:
# $env:OPENAI_BASE_URL="https://api.groq.com/openai/v1"

python rag.py --ask "What are the three capability-based categories of AI?"
```

### D. Rebuilding Chunks & Embeddings
```powershell
python rag.py --build-index
# Or modularly:
python src/chunking.py
python src/embeddings.py
```

---

## 6. Evaluation Results Summary

Evaluation was performed over the 15 authored questions with gold passages across printed pages 8 through 38.

| Chunk Size | Total Chunks | Hits in Top 5 | Recall@5 | MRR@5 | Rank 1 Hits |
|:---|:---:|:---:|:---:|:---:|:---:|
| **100 words** | 198 | **14 / 15** | **93.33%** | **0.9000** | **13 / 15 (86.7%)** |
| **300 words** | 73 | **14 / 15** | **93.33%** | **0.7444** | **10 / 15 (66.7%)** |

### Key Findings
- **100-word chunks performed significantly better for retrieval precision and ranking**:
  - MRR@5 was **0.9000** for 100-word chunks versus **0.7444** for 300-word chunks.
  - **13 of 14 successful questions were ranked #1** with 100-word chunks, compared to only 10 with 300-word chunks.
- **Why**: 100-word chunks provide high semantic density. Compact factual definitions (Explainable AI, GPU parallelism, Big Data 4 Vs) are preserved without being diluted by surrounding paragraphs. 300-word chunks dilute specific factual sentences across broader context, lowering their cosine similarity for narrow factual queries.

---

## 7. Failure Cases & Analysis

1. **Question 1 under 100-word chunks**:
   - *Query*: *"What three components explain how modern AI systems can exhibit intelligent behaviour?"*
   - *Gold Page*: Page 8 (*"algorithms, data and computing power"*).
   - *Result*: Pushed outside top-5 by generic intro pages (Pages 7, 34, 35) that discussed "AI systems" and "intelligent behaviour".
   - *Fix*: **BM25 Hybrid Search** or **300-word chunks** (which successfully retrieved it at Rank 3).
2. **Question 10 under 300-word chunks**:
   - *Query*: *"What are the three capability-based categories of AI?"*
   - *Gold Page*: Page 21 (*"ANI, AGI, ASI"*).
   - *Result*: Diluted by surrounding discussion of machine learning types and compute on Page 21; pushed outside top-5.
   - *Fix*: **100-word chunks** (which retrieved it at Rank 1) or a **Cross-Encoder Re-Ranker**.

---

## 8. Summary of BM25 and Re-Ranker Concepts

- **BM25 vs. Dense Embeddings**: Dense embeddings excel at synonyms, paraphrase, and conceptual similarity, but suffer from "soft match" errors on exact keywords, acronyms (TPU, GPU, ANI, AGI), numbers, and rare proper nouns. BM25 provides exact keyword matching weighted by Inverse Document Frequency (IDF) and term saturation. Hybrid search (BM25 + dense via Reciprocal Rank Fusion) delivers the best of both worlds.
- **What a Re-Ranker Does**: Dense vector retrieval uses bi-encoders where query and passage are embedded separately without token cross-attention. A re-ranker (cross-encoder) takes the top-20 candidate passages from stage 1 and passes each `[Query, Passage]` pair jointly through all transformer layers. Full cross-attention enables the re-ranker to accurately evaluate syntactic relationships, negations, and factual relevance, placing the true answer passage at Rank 1 before passing to the LLM.
