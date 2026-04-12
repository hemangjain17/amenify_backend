"""
scraper.py
----------
Crawls Amenify pages using Firecrawl batch scraping with dual extraction:
  - markdown  : raw page content for chunking + embedding
  - json      : structured fields (summary, services, CTAs, FAQs) per page

Output: data/kb_records.json  — one rich record per URL.

Run directly to rebuild the knowledge base:
    python scraper.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

load_dotenv()

# ---------------------------------------------------------------------------
# Pages to scrape
# ---------------------------------------------------------------------------
BASE_URL = "https://www.amenify.com"

AMENIFY_URLS: list[str] = [
    "https://www.amenify.com",
    "https://www.amenify.com/resident-services",
    "https://www.amenify.com/cleaningservices1",
    "https://www.amenify.com/choreservices1",
    "https://www.amenify.com/handymanservices1",
    "https://www.amenify.com/professional-moving-services",
    "https://www.amenify.com/movingoutservices1",
    "https://www.amenify.com/groceryservices1",
    "https://www.amenify.com/dog-walking-services",
    "https://www.amenify.com/acommerce",
    "https://www.amenify.com/property-managers-2",
    "https://www.amenify.com/autogifts",
    "https://www.amenify.com/leasing-concession",
    "https://www.amenify.com/commercialcleaning1",
    "https://www.amenify.com/providers-1",
    "https://www.amenify.com/amenify-platform",
    "https://www.amenify.com/merchant-landing",
    "https://www.amenify.com/amenify-technology",
    "https://www.amenify.com/about-us",
    "https://www.amenify.com/news-articles",
    "https://www.amenify.com/blog",
    "https://www.amenify.com/contact-us",
    "https://www.amenify.com/resident-protection-plan",
    "https://www.amenify.com/faq",
    "https://www.amenify.com/sign-in-sign-up",
    "https://www.amenify.com/amenify-app",
    "https://www.amenify.com/tech-platform",
    "https://www.amenify.com/about",
    "https://www.amenify.com/services",
    "https://www.amenify.com/pricing",
    "https://www.amenify.com/careers",
    "https://www.amenify.com/technology",
    "https://www.amenify.com/cleaning-services",
    "https://www.amenify.com/chores-services",
    "https://www.amenify.com/handyman-services",
    "https://www.amenify.com/food-grocery-service",
    "https://www.amenify.com/move-out-cleaning-services",
]



POLL_INTERVAL = 3       # seconds between Firecrawl status polls
WAIT_TIMEOUT  = 300     # seconds before giving up

DATA_DIR      = Path(__file__).parent.parent / "data"
KB_RECORDS_PATH = DATA_DIR / "kb_records.json"   # rich structured output

# ---------------------------------------------------------------------------
# Firecrawl JSON schema — what we want extracted per page
# ---------------------------------------------------------------------------
PAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_title": {
            "type": "string",
            "description": "The main title of the page",
        },
        "summary": {
            "type": "string",
            "description": (
                "2-3 sentence plain-English summary of what this page is about "
                "and what a visitor can do or learn here"
            ),
        },
        "services": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of services, products, or features mentioned on this page",
        },
        "key_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Important facts, statistics, prices, or commitments mentioned "
                "(e.g. '100K+ five-star reviews', '$50 signup credit')"
            ),
        },
        "cta_links": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "url":   {"type": "string"},
                },
                "required": ["label", "url"],
            },
            "description": (
                "All call-to-action or navigation links found on the page "
                "(buttons, 'Book Now', 'Learn More', 'Sign Up', etc.) "
                "with their exact hrefs"
            ),
        },
        "faq": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer":   {"type": "string"},
                },
                "required": ["question", "answer"],
            },
            "description": "FAQ items found on this page, if any",
        },
    },
    "required": ["page_title", "summary"],
}

# ---------------------------------------------------------------------------
# Shared scrape-status dict
# ---------------------------------------------------------------------------
_status: dict[str, Any] = {
    "running":            False,
    "last_completed_utc": None,
    "records_indexed":    0,
    "next_run_seconds":   None,
    "cycle":              0,
    "triggered_by":       None,
    "error":              None,
}


def get_scrape_status() -> dict[str, Any]:
    return dict(_status)


# ---------------------------------------------------------------------------
# Firecrawl client
# ---------------------------------------------------------------------------

def _get_firecrawl_client() -> FirecrawlApp:
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "FIRECRAWL_API_KEY is not set. Add it to backend/.env and retry."
        )
    return FirecrawlApp(api_key=api_key)


# ---------------------------------------------------------------------------
# Parallel fetch (markdown + explicit JSON schema single-url extraction)
# ---------------------------------------------------------------------------

def fetch_single_page(client: FirecrawlApp, url: str) -> dict[str, Any] | None:
    try:
        data = client.scrape(
            url,
            only_main_content=False,
            max_age=172800000,
            formats=[
                "markdown",
                {
                    "type": "json",
                    "schema": PAGE_JSON_SCHEMA
                }
            ]
        )
        return data
    except Exception as e:
        print(f"  [WARN] Scrape error for {url}: {e}")
        return None

def batch_fetch_pages(urls: list[str]) -> dict[str, dict[str, Any]]:
    """
    Submit URLs to Firecrawl using individual app.scrape() calls run in parallel,
    requesting both markdown and structured JSON extraction via the nested payload.

    Returns {normalized_url -> {"markdown": str, "json": dict, "metadata": dict}}
    """
    import concurrent.futures
    client = _get_firecrawl_client()

    print(f"[Scraper] Submitting {len(urls)} individual URLs to Firecrawl using ThreadPool...")

    results: dict[str, dict[str, Any]] = {}
    completed = 0
    total = len(urls)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_single_page, client, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page = future.result()
                completed += 1
                if page:
                    # Firecrawl Python SDK returns an object, not a dict
                    meta = getattr(page, "metadata", None)
                    raw_url = url
                    if meta:
                        raw_url = (
                            getattr(meta, "source_url", None)
                            or getattr(meta, "url", None)
                            or url
                        )
                    norm_url = raw_url.rstrip("/")

                    markdown_content = getattr(page, "markdown", None) or ""
                    json_data = getattr(page, "json", None) or {}

                    results[norm_url] = {
                        "markdown": markdown_content.strip(),
                        "json":     json_data,
                        "metadata": {
                            "title":       getattr(meta, "title", "") if meta else "",
                            "description": getattr(meta, "description", "") if meta else "",
                            "og_url":      getattr(meta, "og_url", "") if meta else "",
                            "language":    getattr(meta, "language", "") if meta else "",
                        },
                    }
                    print(f"[Scraper] Progress: {completed}/{total} - {norm_url} (SUCCESS)")
                else:
                    print(f"[Scraper] Progress: {completed}/{total} - {url} (FAILED)")
            except Exception as exc:
                completed += 1
                print(f"[Scraper] Exception fetching {url}: {exc}")

    print(f"[Scraper] Received data for {len(results)} pages.")
    return results


# ---------------------------------------------------------------------------
# Link extraction from markdown
# ---------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[([^\]]{2,120})\]\((https?://[^)]+)\)")


def _extract_md_links(markdown: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    links = []
    for text, href in _MD_LINK_RE.findall(markdown):
        href = href.strip()
        if href in seen:
            continue
        seen.add(href)
        links.append({"label": text.strip(), "url": href})
    return links


# ---------------------------------------------------------------------------
# Build knowledge base records
# ---------------------------------------------------------------------------

def build_knowledge_base(
    urls:        list[str] | None = None,
    output_path: Path             = KB_RECORDS_PATH,
) -> list[dict[str, Any]]:
    """
    Full pipeline:
    1. Load existing kb_records.json (incremental upsert).
    2. Batch-fetch all target URLs via Firecrawl (markdown + JSON schema).
    3. Merge structured JSON fields + markdown + link list into one record per URL.
    4. Save to output_path.
    5. Return full record list.
    """
    urls = urls or AMENIFY_URLS

    # ── Load existing records ───────────────────────────────────────────────
    existing: list[dict[str, Any]] = []
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            print(f"[Scraper] Loaded {len(existing)} existing records.")
        except (json.JSONDecodeError, OSError):
            print("[Scraper][WARN] Existing data corrupt — starting fresh.")

    urls_being_scraped = {u.rstrip("/") for u in urls}
    preserved = [r for r in existing if r.get("url", "").rstrip("/") not in urls_being_scraped]
    print(f"[Scraper] Preserved {len(preserved)} records from un-touched URLs.")

    # ── Fetch via Firecrawl ─────────────────────────────────────────────────
    page_map = batch_fetch_pages(urls)

    # ── Build rich records ──────────────────────────────────────────────────
    new_records: list[dict[str, Any]] = []
    scraped_at = datetime.now(tz=timezone.utc).isoformat()

    for url in urls:
        lookup_key = url.rstrip("/")
        page_data  = page_map.get(lookup_key)
        if not page_data:
            print(f"  [WARN] No data for {url}")
            continue

        markdown  = page_data["markdown"]
        extracted = page_data["json"]       # structured fields from Firecrawl
        meta      = page_data["metadata"]

        # Merge CTA links: prefer Firecrawl-extracted, supplement from markdown
        json_ctas = extracted.get("cta_links") or []
        md_links  = _extract_md_links(markdown)
        # De-duplicate by url
        seen_hrefs = {c["url"] for c in json_ctas}
        extra_links = [l for l in md_links if l["url"] not in seen_hrefs]
        all_ctas = json_ctas + extra_links[:20]   # cap at 20 extra

        record: dict[str, Any] = {
            "url":           url,
            "page_title":    extracted.get("page_title") or meta.get("title", ""),
            "summary":       extracted.get("summary", ""),
            "services":      extracted.get("services") or [],
            "key_facts":     extracted.get("key_facts") or [],
            "cta_links":     all_ctas,
            "faq":           extracted.get("faq") or [],
            "raw_markdown":  markdown,
            "og_description": meta.get("description", ""),
            "scraped_at":    scraped_at,
        }

        new_records.append(record)
        print(
            f"  -> {url}\n"
            f"     title='{record['page_title'][:60]}' | "
            f"services={len(record['services'])} | "
            f"ctas={len(all_ctas)} | "
            f"faqs={len(record['faq'])}"
        )

    all_records = preserved + new_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\n[Scraper] Saved {len(all_records)} records -> {output_path}")
    return all_records


# ---------------------------------------------------------------------------
# Async trigger (called from POST /api/scrape)
# ---------------------------------------------------------------------------

async def run_scrape_once(
    on_complete: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """
    Run one scrape + index cycle asynchronously without blocking the server.
    Launch via: asyncio.create_task(run_scrape_once(kb.hot_reload))
    """
    global _status
    loop = asyncio.get_event_loop()

    _status["running"]      = True
    _status["triggered_by"] = "manual"
    _status["error"]        = None
    _status["cycle"]        = _status.get("cycle", 0) + 1
    cycle                   = _status["cycle"]

    print(f"\n[Scraper] Manual scrape cycle #{cycle} starting...")

    try:
        records = await loop.run_in_executor(None, build_knowledge_base)
        _status["records_indexed"]    = len(records)
        _status["last_completed_utc"] = datetime.now(tz=timezone.utc).isoformat()
        print(f"[Scraper] Cycle #{cycle} complete — {len(records)} records.")
    except Exception as exc:
        _status["error"] = str(exc)
        print(f"[Scraper][ERROR] Cycle #{cycle} failed: {exc}")
    finally:
        _status["running"]          = False
        _status["next_run_seconds"] = None

    if on_complete:
        try:
            await on_complete()
        except Exception as exc:
            print(f"[Scraper][ERROR] KB hot-reload failed: {exc}")

    print(f"[Scraper] Manual scrape cycle #{cycle} finished.")


# ---------------------------------------------------------------------------
# Direct execution — python scraper.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  Amenify Knowledge Base Builder")
    print("  Firecrawl batch scrape -> JSON records")
    print("=" * 60)
    print(f"  URLs to scrape : {len(AMENIFY_URLS)}")
    print(f"  Output path    : {KB_RECORDS_PATH}")
    print("=" * 60)

    _status["triggered_by"] = "direct"
    _status["running"]      = True
    start_time = time.time()

    try:
        records = build_knowledge_base()
        elapsed = time.time() - start_time
        _status["records_indexed"]    = len(records)
        _status["last_completed_utc"] = datetime.now(tz=timezone.utc).isoformat()
        print()
        print("=" * 60)
        print(f"  Done!  {len(records)} records saved in {elapsed:.1f}s")
        print(f"  Next:  python knowledge_base.py   (embed + index)")
        print("=" * 60)
        sys.exit(0)
    except EnvironmentError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        _status["error"] = str(exc)
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    finally:
        _status["running"] = False
