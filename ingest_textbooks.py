"""Extract text from PDF textbooks into corpus/ for BM25 retrieval and quote verification.

Usage:
    .venv/bin/python ingest_textbooks.py [--src Textbooks] [--dest corpus] [--max-pages 500]
"""
import argparse
import glob
import os
import re
from pypdf import PdfReader


def clean_text(text: str) -> str:
    # Normalize unicode quotes and dashes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    # Remove null bytes or control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def ingest_pdf(pdf_path: str, out_dir: str, max_pages: int | None = None) -> tuple[str, int, int]:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_file = os.path.join(out_dir, f"{base_name}.txt")

    print(f"Ingesting: {os.path.basename(pdf_path)}...")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages else total_pages

    extracted_pages = []
    word_count = 0

    for i in range(limit):
        try:
            page_text = reader.pages[i].extract_text() or ""
            if page_text.strip():
                extracted_pages.append(f"--- Page {i + 1} ---\n" + page_text.strip())
                word_count += len(page_text.split())
        except Exception as e:
            print(f"  [!] Error on page {i + 1}: {e}")

    full_text = clean_text("\n\n".join(extracted_pages))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"  -> Extracted {len(extracted_pages)}/{total_pages} pages ({word_count:,} words) -> {out_file}")
    return out_file, len(extracted_pages), word_count


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF textbooks into corpus text files")
    parser.add_argument("--src", default="Textbooks", help="Directory containing PDF textbooks")
    parser.add_argument("--dest", default="corpus", help="Directory to save extracted .txt files")
    parser.add_argument("--file", default=None, help="Specific PDF file to extract")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per book (optional)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "*.pdf")))

    if not files:
        print(f"No PDF files found in {args.src}")
        return

    print(f"Found {len(files)} textbooks in '{args.src}'. Ingesting into '{args.dest}/'...")
    total_words = 0
    for pdf in files:
        try:
            _, _, words = ingest_pdf(pdf, args.dest, args.max_pages)
            total_words += words
        except Exception as e:
            print(f"Failed to ingest {pdf}: {e}")

    print(f"\nDone! Ingested total {total_words:,} words into {args.dest}/")


if __name__ == "__main__":
    main()
