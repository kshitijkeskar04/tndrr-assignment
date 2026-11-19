"""
Diagnostic script: Inspect chunk text against test questions source passages.
Can be run from tests/ or project root.
"""
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "test_questions.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

for size in [100, 300]:
    chunk_file = ARTIFACTS_DIR / f"chunks_{size}.json"
    if not chunk_file.exists():
        print(f"Missing {chunk_file}")
        continue
        
    with open(chunk_file, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n{'='*55}")
    print(f"COMPLETE PAGE MAPPING: {size}-WORD CHUNKS")
    print(f"{'='*55}")

    for q in questions:
        source_words = re.findall(r"\w+", q["source_passage"].lower())[:6]
        matches = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk["text"].lower()
            if all(w in chunk_lower for w in source_words):
                matches.append({"chunk": i, "page": chunk["page"]})

        print(f"Q{q['id']:02d} | Expected page: {q['page']:<2} | Matching chunks: {matches}")
