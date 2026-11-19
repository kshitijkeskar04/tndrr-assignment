"""
Diagnostic script: Inspect PDF extraction and statistics.
Can be run from tests/ or project root.
"""
from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "2024-wttc-introduction-to-ai.pdf"

reader = PdfReader(str(PDF_PATH))
print(f"PDF successfully loaded: {PDF_PATH}")
print(f"Total physical pages: {len(reader.pages)}")

# Extract main report pages (3 to 43)
all_text = []
for page_number, page in enumerate(reader.pages, start=1):
    if 3 <= page_number <= 43:
        text = page.extract_text() or ""
        all_text.append({
            "page": page_number,
            "text": text
        })

total_characters = sum(len(p["text"]) for p in all_text)
print(f"\nText extraction completed! Pages extracted: {len(all_text)}")
print(f"Total characters: {total_characters}")

out_path = PROJECT_ROOT / "artifacts" / "extracted_text.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for p in all_text:
        f.write(f"\n--- PAGE {p['page']} ---\n")
        f.write(p["text"])
        f.write("\n")

print(f"Saved extracted text to {out_path}")
