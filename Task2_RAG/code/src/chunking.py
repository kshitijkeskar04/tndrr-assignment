"""
Module: src/chunking.py
Extracts text from the WTTC AI PDF report and generates word-based chunks with overlap.
Saves all artifacts into the artifacts/ directory.
"""
import json
import re
from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "2024-wttc-introduction-to-ai.pdf"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHUNK_SIZES = [100, 300]
DEFAULT_OVERLAP = 20

def extract_pages(pdf_path=PDF_PATH, min_page=3, max_page=43):
    """
    Extracts text page-by-page from the PDF report.
    Physical page numbers directly correspond to the printed page numbers in this document.
    Pages 3 to 43 encompass the full main report, annexes, and glossary.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    
    for physical_page, page in enumerate(reader.pages, start=1):
        if min_page <= physical_page <= max_page:
            raw_text = page.extract_text() or ""
            # Clean consecutive whitespace while preserving paragraph structure
            cleaned_text = re.sub(r"[ \t]+", " ", raw_text)
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
            
            words = cleaned_text.split()
            if words:
                pages.append({
                    "page": physical_page,
                    "text": cleaned_text,
                    "words": words
                })
    return pages

def create_chunks(pages, chunk_size, overlap=DEFAULT_OVERLAP):
    """
    Splits page text into sliding-window chunks of specified word count and overlap.
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    
    for page in pages:
        words = page["words"]
        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if not chunk_words:
                continue
            
            chunk = {
                "chunk_id": f"p{page['page']}_w{start}",
                "page": page["page"],
                "start_word": start,
                "word_count": len(chunk_words),
                "text": " ".join(chunk_words)
            }
            chunks.append(chunk)
            
            if start + chunk_size >= len(words):
                break
                
    return chunks

def build_and_save_chunks(chunk_sizes=None, overlap=DEFAULT_OVERLAP):
    """
    Extracts text, generates chunks for each size, and saves to artifacts directory.
    """
    if chunk_sizes is None:
        chunk_sizes = DEFAULT_CHUNK_SIZES
        
    pages = extract_pages()
    print(f"Extracted {len(pages)} pages from {PDF_PATH.name}")
    
    # Save full extracted text
    text_file = ARTIFACTS_DIR / "extracted_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n--- PAGE {p['page']} ---\n")
            f.write(p["text"] + "\n")
    print(f"Saved extracted text to {text_file}")
    
    results = {}
    for size in chunk_sizes:
        chunks = create_chunks(pages, chunk_size=size, overlap=overlap)
        out_file = ARTIFACTS_DIR / f"chunks_{size}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        print(f"Created {len(chunks)} chunks of size {size} (overlap={overlap}) -> {out_file}")
        results[size] = chunks
        
    return pages, results

if __name__ == "__main__":
    build_and_save_chunks()
