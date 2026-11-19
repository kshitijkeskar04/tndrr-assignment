# Task 2: Retrieval-Augmented Generation (RAG) from Scratch

This folder contains the complete submission for **Task 2 (RAG from Scratch)** over the 44-page World Travel & Tourism Council (WTTC) *Introduction to AI* report.

As requested, the deliverables are organized into separate folders for the **report** and the **code**:

```text
Task2_RAG/
├── report/
│   ├── SUBMISSION_REPORT.md         # Comprehensive markdown report with formulas, tables, and full analysis
│   └── Task2_report.pdf             # Executive presentation PDF report
└── code/
    ├── data/                        # WTTC AI PDF (44 pages) & 15 authored test questions
    ├── artifacts/                   # Extracted text, 100w/300w chunks, and precomputed embeddings
    ├── results/                     # Benchmark results (summary.json, CSV, detailed logs)
    ├── src/                         # Core Python modules (chunking, embeddings, retrieval, evaluate)
    ├── tests/                       # Diagnostics (check_pdf.py, inspect_pages.py)
    ├── rag.py                       # Unified CLI interface (--ask, --evaluate, --build-index)
    ├── evaluate.py                  # Evaluation benchmark runner
    ├── requirements.txt             # Dependencies
    └── README.md                    # Technical code documentation and setup guide
```

---

## 1. Reports (`report/`)
- **[SUBMISSION_REPORT.md](report/SUBMISSION_REPORT.md)**:
  - The 15 authored test questions, expected answers, and exact supporting passages.
  - Empirical Recall@5 (93.33%) and MRR@5 comparison between 100-word (0.9000) and 300-word (0.7444) chunks.
  - Root-cause analysis and fixes for the two failure cases (Q1 on 100w, Q10 on 300w).
  - Detailed explanations of what BM25 adds to dense retrieval and what a cross-encoder re-ranker does.
- **`Task2_report.pdf`**:
  - 3-page executive summary PDF report.

---

## 2. Code Implementation (`code/`)
- **No LangChain or LlamaIndex**: Built from scratch using `sentence-transformers/all-MiniLM-L6-v2`, NumPy matrix operations, and an optional OpenAI client for grounded answering.
- Quick start:
  ```powershell
  cd code
  pip install -r requirements.txt

  # Run evaluation benchmark across both chunk sizes
  python evaluate.py

  # Query the RAG system
  python rag.py --ask "What are the four Vs of Big Data?" --chunk-words 100
  ```
- See **[code/README.md](code/README.md)** for full documentation.
