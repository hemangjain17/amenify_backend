"""
pinecone_store.py
-----------------
Pinecone vector store with Hugging Face Inference embeddings and Reranking.
Implements the hybrid dense + reranker flow.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent.parent
KB_RECORDS_PATH = BASE_DIR / "data" / "kb_records.json"

INDEX_NAME      = "amenify-kb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
RERANKER_MODEL  = "BAAI/bge-reranker-base"
EMBEDDING_DIM   = 384

CHUNK_SIZE      = 250
CHUNK_OVERLAP   = 50
EMBED_BATCH     = 8  # Keep low for free HF Inference API limits

# ---------------------------------------------------------------------------
# Hugging Face API Utilities
# ---------------------------------------------------------------------------
def _get_hf_client() -> InferenceClient:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token == "dummy_hf_token_override_later":
        raise ValueError("HF_TOKEN environment variable not set or invalid.")
    return InferenceClient(provider="hf-inference", api_key=hf_token)

def _embed_texts_hf(texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings via HF inference."""
    if not texts:
        return []
        
    client = _get_hf_client()
    all_vecs = []
    print(f"[HF Embed] Processing {len(texts)} chunks...")
    
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for attempt in range(3):
            try:
                # feature_extraction returns a list of floats (1D) for single text
                # or list of list of floats (2D) for batch.
                vecs = client.feature_extraction(text=batch, model=EMBEDDING_MODEL)
                
                # Convert numpy array or list format properly
                if hasattr(vecs, "tolist"):
                    vecs = vecs.tolist()
                
                if isinstance(vecs, list) and len(vecs) > 0:
                    if isinstance(vecs[0], float):
                        all_vecs.append(vecs)
                    else:
                        all_vecs.extend(vecs)
                break
            except Exception as e:
                print(f"[HF Embed] Attempt {attempt+1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            print(f"[ERROR] Failed to embed batch {i}-{i+len(batch)}. Inserting zeroes.")
            all_vecs.extend([[0.0]*EMBEDDING_DIM] * len(batch))
            
        time.sleep(0.5)  # rate limit safety
    return all_vecs

def _rerank_hf(query: str, texts: list[str]) -> list[float]:
    """Score pairs using a cross-encoder via HF text-classification."""
    if not texts:
        return []
    
    hf_token = os.environ.get("HF_TOKEN")
    api_url = f"https://router.huggingface.co/hf-inference/models/{RERANKER_MODEL}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # BAAI/bge-reranker-base is a text-classification model, expecting pairs.
    payload = {
        "inputs": [{"text": query, "text_pair": text} for text in texts]
    }
    
    import requests
    for attempt in range(3):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                res = response.json()
                # Typical response: [[{"label": "LABEL_0", "score": 0.8}, ...]]
                if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                    return [item.get("score", 0.0) for item in res[0]]
            elif response.status_code == 503:
                print(f"[HF Reranker] Loading. Wait...")
                time.sleep(15)
            else:
                print(f"[HF Reranker] Attempt {attempt+1} failed: {response.text}")
        except Exception as e:
            print(f"[HF Reranker] Attempt {attempt+1} failed: {e}")
            pass
        time.sleep(2)
        
    print("[HF Reranker] Failed. Falling back to default scores.")
    return [0.0] * len(texts)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words  = text.split()
    step   = size - overlap
    chunks = []
    for i in range(0, max(1, len(words) - overlap), step):
        segment = " ".join(words[i : i + size])
        if len(segment.split()) >= 30:
            chunks.append(segment)
    return chunks

def _clean_markdown(md: str) -> str:
    text = md
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)          
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)       
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"`[^`]+`", " ", text)                        
    text = re.sub(r"^[-*_]{3,}\s*$", " ", text, flags=re.MULTILINE)  
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------
def build_pinecone_index() -> int:
    with open(KB_RECORDS_PATH, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    if not records:
        print("[Pinecone] No records to index.")
        return 0
        
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "dummy_pinecone_key_override_later":
        raise ValueError("PINECONE_API_KEY must be set in .env")
        
    pc = Pinecone(api_key=api_key)
    
    # Check or create index
    existing_indexes = [info["name"] for info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"[Pinecone] Creating Serverless Index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for initialization
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)
    
    # Wipe old data (in serverless, best to delete all if you want a fresh start, 
    # but Pinecone native delete all takes effort. We'll simply upsert with stable IDs)
    
    rows_text: list[str] = []
    rows_meta: list[dict[str, Any]] = []
    rows_id: list[str] = []

    idx = 0
    for rec in records:
        raw_md = rec.get("raw_markdown", "")
        clean  = _clean_markdown(raw_md)
        chunks = _chunk_text(clean) if clean else []

        if not chunks:
            chunks = [rec.get("summary", rec.get("page_title", ""))]

        ctas_json = json.dumps(rec.get("cta_links", []), ensure_ascii=False)
        services_str = json.dumps(rec.get("services", []), ensure_ascii=False)

        for chunk in chunks:
            rows_text.append(chunk)
            # Pinecone metadata values must be strings, numbers, or lists of strings
            rows_meta.append({
                "text":           chunk, # Store text in metadata for RAG
                "url":            rec["url"][:1024],
                "page_title":     (rec.get("page_title") or "")[:512],
                "summary":        (rec.get("summary")    or "")[:4096],
                "services_json":  services_str[:4096],
                "cta_links_json": ctas_json[:8192],
            })
            rows_id.append(f"doc_{idx}")
            idx += 1

    print(f"[Pinecone] Embedding {len(rows_text)} chunks using HF Inference API...")
    vectors = _embed_texts_hf(rows_text)

    # Upsert
    BATCH_SIZE = 100
    for i in range(0, len(rows_text), BATCH_SIZE):
        batch_ids = rows_id[i : i + BATCH_SIZE]
        batch_vecs = vectors[i : i + BATCH_SIZE]
        batch_meta = rows_meta[i : i + BATCH_SIZE]
        
        to_upsert = zip(batch_ids, batch_vecs, batch_meta)
        index.upsert(vectors=to_upsert)
        print(f"  Upserted rows {i}-{i+len(batch_ids)-1}")

    print(f"[Pinecone] Index built — {len(rows_text)} vectors in '{INDEX_NAME}'.")
    return len(rows_text)

# ---------------------------------------------------------------------------
# PineconeKB Agentic Search
# ---------------------------------------------------------------------------
class PineconeKB:
    def __init__(self, top_k: int = 15, score_threshold: float = 0.5) -> None:
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._ready = False
        
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key or api_key == "dummy_pinecone_key_override_later":
            print("[Pinecone] PINECONE_API_KEY not configured. KB Search disabled.")
            self._index = None
            return
            
        try:
            self._pc = Pinecone(api_key=api_key)
            self._index = self._pc.Index(INDEX_NAME)
            # Wake it up
            stats = self._index.describe_index_stats()
            if stats.total_vector_count > 0:
                self._ready = True
                print(f"[Pinecone] Ready — {stats.total_vector_count} vectors loaded.")
        except Exception as exc:
            print(f"[PineconeKB] Error initializing: {exc}")

    def is_ready(self) -> bool:
        return self._ready

    def search(self, query: str) -> list[dict[str, Any]]:
        if not self.is_ready() or not self._index:
            return []
            
        try:
            vecs = _embed_texts_hf([query])
            if not vecs: return []
            q_vec = vecs[0]
            
            # Step 1: Base Retrieval (Top 15)
            res = self._index.query(
                vector=q_vec,
                top_k=self.top_k,
                include_metadata=True
            )
            
            matches = res.get('matches', [])
            if not matches: return []
            
            # Step 2: Reranking Selection (Disabled per user request)
            # texts_to_score = [m['metadata']['text'] for m in matches]
            # rerank_scores = _rerank_hf(query, texts_to_score)
            
            # Use base Pinecone similarity scores instead of hitting the HF API
            scored_matches = [(m, m.get('score', 0.0)) for m in matches]
            
            # Sort descending (though Pinecone already returns them sorted)
            scored_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top 3 absolute best
            hits = []
            for match, rerank_score in scored_matches[:3]:
                # If absolute score is terrible, break out early
                # NOTE: HF cross encoders output logits, so scores can be negative or > 1.
                # Usually anything > 0 is decent relevance in logit space. 
                # Our agent threshold logic handles the exact "I don't know".
                meta = match['metadata']
                
                try: cta_links = json.loads(meta.get("cta_links_json", "[]"))
                except: cta_links = []
                    
                hits.append({
                    "text":       meta.get("text", ""),
                    "source_url": meta.get("url", ""),
                    "page_title": meta.get("page_title", ""),
                    "summary":    meta.get("summary", ""),
                    "page_links": cta_links,
                    "score":      rerank_score,
                })
            return hits
            
        except Exception as exc:
            print(f"[PineconeKB] Search error: {exc}")
            return []

    async def hot_reload(self) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        print("[PineconeKB] Hot-reload: uploading to Pinecone cloud...")
        try:
            count = await loop.run_in_executor(None, build_pinecone_index)
            self._ready = count > 0
        except Exception as exc:
            print(f"[PineconeKB][ERROR] Hot-reload failed: {exc}")

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("  Amenify Pinecone Cloud Index Builder")
    print("=" * 60)
    if not KB_RECORDS_PATH.exists():
        print("[ERROR] kb_records.json not found. Run python scraper.py first.")
        sys.exit(1)
    # total = build_pinecone_index()
    print("Make sure you defined .env keys before running.")
