"""
build.py  —  Run once to scrape Amenify and build the FAISS index.

Usage:
    python build.py
"""
import os
from pathlib import Path

if not os.environ.get("OPENAI_API_KEY"):
    raise EnvironmentError("Set OPENAI_API_KEY before running this script.")

from scraper import build_knowledge_base, DATA_PATH
from knowledge_base import build_index, CHUNKS_PATH

print("=" * 60)
print("Step 1/2 — Scraping amenify.com")
print("=" * 60)
build_knowledge_base(output_path=DATA_PATH)

print()
print("=" * 60)
print("Step 2/2 — Building FAISS vector index")
print("=" * 60)
build_index(chunks_path=CHUNKS_PATH)

print()
print("✅  Build complete. You can now start the server:")
print("    uvicorn main:app --reload")
