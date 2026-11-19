"""
Module: src/evaluate.py
Evaluates retrieval accuracy (Recall@5, MRR@5, and Rank distribution)
comparing 100-word vs 300-word chunk sizes over 15 authored test questions.
Writes all outputs strictly into results/.
"""
import os
import json
import csv
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from src.embeddings import get_embedding_model
from src.retrieval import load_chunks_and_embeddings, retrieve_numpy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "test_questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_chunk_size(chunk_size, questions, model, top_k=5):
    """
    Evaluates a specific chunk size over the list of test questions.
    """
    chunks, vectors = load_chunks_and_embeddings(chunk_size)
    
    hits = 0
    reciprocal_ranks = []
    question_results = []
    
    for q in questions:
        q_id = q["id"]
        query = q["question"]
        gold_page = int(q["page"])
        
        q_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        retrieved = retrieve_numpy(q_vec, chunks, vectors, top_k=top_k)
        
        matched_rank = None
        matched_page = None
        matched_score = None
        
        for r in retrieved:
            if r["page"] == gold_page:
                matched_rank = r["rank"]
                matched_page = r["page"]
                matched_score = r["score"]
                break
                
        success = matched_rank is not None
        if success:
            hits += 1
            reciprocal_ranks.append(1.0 / matched_rank)
        else:
            reciprocal_ranks.append(0.0)
            
        status = f"PASS (rank {matched_rank})" if success else "FAIL (not in top-5)"
        print(f"[{chunk_size}w] Q{q_id:02d}: {status:<20} | Gold page: {gold_page:<2} | Top-5 pages: {[r['page'] for r in retrieved]}")
        
        question_results.append({
            "chunk_size": chunk_size,
            "question_id": q_id,
            "question": query,
            "correct_answer": q.get("answer", ""),
            "gold_page": gold_page,
            "source_passage": q.get("source_passage", ""),
            "matched_rank": matched_rank if matched_rank else "Not found",
            "matched_score": round(matched_score, 4) if matched_score else None,
            "success": success,
            "top_5_results": retrieved
        })
        
    total = len(questions)
    recall_at_k = hits / total if total > 0 else 0.0
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    
    # Save specific chunk size evaluation json
    chunk_eval_file = RESULTS_DIR / f"chunk_{chunk_size}_evaluation.json"
    with open(chunk_eval_file, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_size": chunk_size,
            "total_questions": total,
            "hits_at_k": hits,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "results": question_results
        }, f, indent=2, ensure_ascii=False)
        
    return {
        "chunk_size": chunk_size,
        "num_chunks": len(chunks),
        "total_questions": total,
        "hits_at_k": hits,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "results": question_results
    }

def run_evaluation(chunk_sizes=(100, 300), top_k=5):
    """
    Runs full comparative evaluation across both chunk sizes.
    """
    print("\n" + "=" * 65)
    print("STARTING RETRIEVAL EVALUATION")
    print(f"Questions: {QUESTIONS_PATH}")
    print(f"Top-k: {top_k} | Comparing chunk sizes: {chunk_sizes}")
    print("=" * 65 + "\n")
    
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    model = get_embedding_model()
    
    summary_data = []
    all_results = {}
    csv_rows = []
    
    for size in chunk_sizes:
        print(f"\n--- Evaluating Chunk Size: {size} words ---")
        eval_metrics = evaluate_chunk_size(size, questions, model, top_k=top_k)
        all_results[f"{size}_word_chunks"] = eval_metrics["results"]
        
        summary_data.append({
            "chunk_words": size,
            "overlap_words": 20,
            "num_chunks": eval_metrics["num_chunks"],
            "questions": eval_metrics["total_questions"],
            "hits_at_5": eval_metrics["hits_at_k"],
            "recall_at_5": round(eval_metrics["recall_at_k"], 4),
            "mrr_at_5": round(eval_metrics["mrr"], 4)
        })
        
        for r in eval_metrics["results"]:
            csv_rows.append({
                "chunk_size": r["chunk_size"],
                "question_id": r["question_id"],
                "question": r["question"],
                "correct_answer": r["correct_answer"],
                "correct_page": r["gold_page"],
                "matched_rank": r["matched_rank"],
                "matched_score": r["matched_score"] if r["matched_score"] is not None else "",
                "success": r["success"]
            })
            
    # Write summary.json
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # Write detailed json
    detailed_file = RESULTS_DIR / "evaluation_detailed.json"
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    # Write CSV
    csv_file = RESULTS_DIR / "evaluation_results.csv"
    fieldnames = [
        "chunk_size", "question_id", "question", "correct_answer",
        "correct_page", "matched_rank", "matched_score", "success"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
        
    # Print formatted comparative report
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 65)
    print(f"{'Chunk Size':<12} | {'Chunks':<8} | {'Hits@5':<8} | {'Recall@5':<12} | {'MRR@5':<10}")
    print("-" * 65)
    for s in summary_data:
        print(f"{s['chunk_words']} words    | {s['num_chunks']:<8} | {s['hits_at_5']}/{s['questions']}    | {s['recall_at_5']:<12.2%} | {s['mrr_at_5']:<10.4f}")
    print("-" * 65)
    
    print("\nSaved files:")
    print(f"  - {summary_file}")
    print(f"  - {detailed_file}")
    print(f"  - {csv_file}")
    print(f"  - {RESULTS_DIR / 'chunk_100_evaluation.json'}")
    print(f"  - {RESULTS_DIR / 'chunk_300_evaluation.json'}")
    print("\nEvaluation successfully completed!")

if __name__ == "__main__":
    run_evaluation()
