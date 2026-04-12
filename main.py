"""
main.py
-------
FastAPI backend for the Amenify AI Customer Support Bot.

Startup behaviour
-----------------
1. Loads .env
2. Logs which LLM provider is active (openai / gemini / ollama)
3. Loads existing FAISS index from disk (if available) for immediate serving
4. Launches background scrape loop — re-scrapes amenify.com & hot-reloads index

NOTE: The background knowledge-base index build still uses OpenAI embeddings.
      Set OPENAI_API_KEY even when using Gemini / Ollama for chat generation.
      If OPENAI_API_KEY is absent and no pre-built index exists, KB search is
      skipped and the bot still works — it just won't have RAG context yet.

Endpoints
---------
POST   /api/chat           — SSE streaming chat
GET    /api/health         — health check
GET    /api/provider       — active LLM provider string
GET    /api/scrape-status  — background scraper progress
DELETE /api/session/{id}   — clear a session
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# ── Load .env before any provider-dependent imports ─────────────────────────
_env_path = Path(__file__).parent / ".env"
_root_env  = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)
load_dotenv(_root_env)

# ── Local modules ────────────────────────────────────────────────────────────
from knowledge_base import KnowledgeBase
import llm_provider as llm                   # flat functions: llm.stream_chat(), llm.chat()
from prompt_config import build_system_prompt
from scraper import run_scrape_once, get_scrape_status

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------
kb: KnowledgeBase | None = None
_scrape_task: asyncio.Task | None = None   # tracks the running scrape task

MAX_HISTORY_TURNS = 10


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup. Scraping is NOT automatic."""
    global kb

    # -- 1. Log active provider -----------------------------------------------
    try:
        provider_str = llm.active_provider()
        print(f"[Main] OK - LLM provider: {provider_str}")
    except Exception as exc:
        print(f"[Main] WARN - Could not determine LLM provider: {exc}")

    # -- 2. Load existing KB index from disk (non-fatal if missing) -----------
    try:
        kb = KnowledgeBase(top_k=5, score_threshold=0.28)
        if kb.is_ready():
            print("[Main] OK - Knowledge base loaded from disk.")
        else:
            print("[Main] WARN - No pre-built index. Call POST /api/scrape to build one.")
    except Exception as exc:
        print(f"[Main] WARN - KB init (non-fatal): {exc}")
        kb = None

    # NOTE: scraping does NOT start automatically.
    # Trigger it explicitly via:  POST /api/scrape
    print("[Main] Ready. Scraping is manual -- POST /api/scrape to start.")

    yield
    # Shutdown: cancel any in-progress scrape gracefully
    if _scrape_task and not _scrape_task.done():
        _scrape_task.cancel()
        print("[Main] Cancelled in-progress scrape task on shutdown.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Amenify Support Bot API",
    version="2.0.0",
    description=(
        "RAG-powered customer support bot for Amenify.\n"
        "Supports OpenAI / Gemini / Ollama via LLM_PROVIDER env var.\n"
        "Self-updating knowledge base scraped from amenify.com."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built frontend dist in production
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="assets",
    )


# ---------------------------------------------------------------------------
# In-memory session store  {session_id: [{"role": ..., "content": ...}]}
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict[str, str]]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message:    str       = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Omit to start a new session; include to continue an existing one.",
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the built frontend in production; API info otherwise."""
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({
        "message": "Amenify Support Bot API v2 — visit /docs for Swagger UI",
        "provider": llm.active_provider(),
        "kb_ready": kb.is_ready() if kb else False,
    })


@app.get("/api/health")
async def health():
    """Health check — confirms server is up and reports KB / provider state."""
    return {
        "status":          "ok",
        "provider":        llm.active_provider(),
        "kb_ready":        kb.is_ready() if kb else False,
        "active_sessions": len(sessions),
    }


@app.get("/api/provider")
async def provider_info():
    """Returns the active LLM provider identifier."""
    return {"provider": llm.active_provider()}


@app.get("/api/scrape-status")
async def scrape_status():
    """Returns the current state of the background scraper."""
    return get_scrape_status()


@app.post("/api/scrape")
async def trigger_scrape():
    """
    Manually trigger a full scrape-and-index cycle.

    - Returns 202 immediately; the scrape runs in the background.
    - Returns 409 if a scrape is already in progress.
    - Poll GET /api/scrape-status to track progress.

    The scrape fetches content from amenify.com, chunks it, embeds it
    (via OpenAI text-embedding-3-small), and hot-reloads the FAISS index.
    Requires OPENAI_API_KEY to be set for the embedding step.
    """
    global _scrape_task, kb

    # Guard: reject if a scrape is already running
    status = get_scrape_status()
    if status.get("running"):
        raise HTTPException(
            status_code=409,
            detail="A scrape is already in progress. Check GET /api/scrape-status.",
        )

    on_complete_cb = kb.hot_reload if kb is not None else None
    _scrape_task = asyncio.create_task(run_scrape_once(on_complete=on_complete_cb))

    return {
        "status":  "started",
        "message": "Scrape started in the background. Poll GET /api/scrape-status for progress.",
        "poll":    "/api/scrape-status",
    }



@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Stream format
    -------------
    data: {"token": "<text>"}           — LLM token (repeated N times)
    data: {"session_id": "...",         — final event once streaming is done
            "sources": [...],
            "found_in_kb": bool,
            "done": true}
    """
    # ── Session management ─────────────────────────────────────────────────
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    # ── SSE generator ──────────────────────────────────────────────────────
    async def event_generator():
        # Yield instant UI feedback as a status message (not final content)
        yield {"data": json.dumps({"status_text": "Searching knowledge base..."})}
        
        # ── 1. Query Rewriting Layer ───────────────────────────────────────────
        # ── 1. Query Rewriting Layer ───────────────────────────────────────────
        search_query = req.message
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]]) if history else "(No prior history)"
        rewrite_prompt = (
            "You are an expert search query rewriter. Given the user's latest query, and optionally some conversation history, "
            "rewrite the query to make it highly specific, clear, and optimized for a semantic search engine. "
            "Output ONLY the rewritten query text.\n\n"
            f"History:\n{history_str}\n\nLatest Query: {req.message}"
        )
        try:
            # Wrap synchronous chat call in thread
            expanded = await asyncio.to_thread(lambda: llm.chat([{"role": "user", "content": rewrite_prompt}]).strip())
            if expanded and len(expanded) < 200:
                search_query = expanded
                print(f"[Main] Rewrote query to: {search_query}")
        except Exception as exc:
            print(f"[Main] Query rewrite failed: {exc}")

        # ── 2. RAG retrieval ──────────────────────────────────────────────────────
        context_chunks: list[dict[str, Any]] = []
        if kb and kb.is_ready():
            try:
                context_chunks = await asyncio.to_thread(lambda: kb.search(search_query))
            except Exception as exc:
                print(f"[Main] KB search error (non-fatal): {exc}")
                
        # ── 3. Empty Context Trigger ───────────────────────────────────────────
        THRESHOLD = 0.0
        if context_chunks and context_chunks[0]["score"] < THRESHOLD:
            print(f"[Main] Top score {context_chunks[0]['score']} < {THRESHOLD}. Triggering empty context.")
            context_chunks = []
            
        # ── 4. Guardrail Self-Correction ───────────────────────────────────────
        # Note: Disabled because smaller LLMs often fail the YES/NO binary check,
        # aggressively wiping out valid context chunks before the final generation.
        # if context_chunks:
        #     ctx_text = "\n".join([c["text"] for c in context_chunks])
        #     guard_prompt = (
        #         "Does the retrieved context actually contain the answer to the user query? "
        #         "Respond ONLY with 'YES' or 'NO'.\n\n"
        #         f"Query: {search_query}\n\nContext:\n{ctx_text}"
        #     )
        #     try:
        #         guard_resp = await asyncio.to_thread(lambda: llm.chat([{"role": "user", "content": guard_prompt}]).strip().upper())
        #         if "NO" in guard_resp:
        #             print("[Main] Guardrail triggered: Context does not contain answer.")
        #             context_chunks = []
        #     except Exception as exc:
        #         pass
                
        found_in_kb = len(context_chunks) > 0

        # Update UI automatically with intermediate UI progress
        if found_in_kb:
            yield {"data": json.dumps({"status_text": f"Found {len(context_chunks)} relevant items..."})}
            
        # ── Build messages ─────────────────────────────────────────────────────
        system_prompt   = build_system_prompt(context_chunks)
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]
        messages = (
            [{"role": "system", "content": system_prompt}]
            + trimmed_history
            + [{"role": "user",   "content": req.message}]
        )

        sources = [
            {"url": c["source_url"], "score": c["score"]}
            for c in context_chunks
        ]
        
        # ── Execute LLM Stream ─────────────────────────────────────────────────
        full_reply_parts: list[str] = []
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _produce() -> None:
            try:
                for stream_token in llm.stream_chat(messages):
                    loop.call_soon_threadsafe(token_queue.put_nowait, stream_token)
            except Exception as exc:
                error_token = f"\n\n⚠️ LLM error: {exc}"
                loop.call_soon_threadsafe(token_queue.put_nowait, error_token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        producer_future = loop.run_in_executor(None, _produce)

        while True:
            stream_token = await token_queue.get()
            if stream_token is None:
                break
            full_reply_parts.append(stream_token)
            yield {"data": json.dumps({"token": stream_token})}

        await producer_future
        reply_text = "".join(full_reply_parts)

        # Persist turn in session history
        history.append({"role": "user",      "content": req.message})
        history.append({"role": "assistant", "content": reply_text})

        # Send final metadata event
        yield {
            "data": json.dumps({
                "session_id":  session_id,
                "sources":     sources,
                "found_in_kb": found_in_kb,
                "done":        True,
            })
        }

    return EventSourceResponse(event_generator())


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Delete a session's message history."""
    removed = sessions.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"detail": f"Session {session_id} cleared."}
