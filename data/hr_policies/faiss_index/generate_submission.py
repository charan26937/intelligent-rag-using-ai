from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag import MODEL_NAME, build_index, load_pdf_documents

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "submission_summary.json"


def main() -> None:
    documents, errors = load_pdf_documents(DATA_DIR)
    if errors:
        print("Skipped local documents:")
        for error in errors:
            print(f"- {error}")
    store = build_index(documents, SentenceTransformer(MODEL_NAME))
    chunks = store["chunks"]
    payload = {
        "total_chunks": len(chunks),
        "sources": sorted({chunk["source"] for chunk in chunks}),
        "skipped_documents": errors + store["invalid_documents"],
        "sample": chunks[:3],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(chunks)} chunks from {len(payload['sources'])} PDFs.")
    print(f"Saved summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
