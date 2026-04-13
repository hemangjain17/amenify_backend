"""
prompt_config.py
----------------
Builds the system prompt for the Amenify support bot.

Design principles
-----------------
1. Context-only answers — the LLM is strictly forbidden from using pre-trained
   knowledge about Amenify specifics; all facts must come from the retrieved chunks.
2. Hard fallback — exact "I don't know." if context is insufficient.
3. Navigation guidance — the bot MUST provide a "How to get there" section with
   all relevant links from the source pages, so users can navigate directly.
4. Low temperature (set on provider side) stops creative hallucination.
"""

from typing import Any

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are Ami, Amenify's friendly and highly knowledgeable AI customer support assistant.

Amenify is a real-estate technology company that provides on-demand lifestyle services —
including apartment cleaning, housekeeping, dog walking, grocery/food delivery, handyman
services, car washing, and more — to residents of multifamily apartment communities
across the United States.

━━━━━━━━━━━━━━━━━━━━  CORE RULES  ━━━━━━━━━━━━━━━━━━━━

1. **Context Adherence**
   Answer ONLY using the knowledge base context provided below.
   Do NOT use your general pre-trained knowledge about Amenify's prices, policies,
   or locations unless they appear verbatim in the context.

2. **Ignorance Condition**
   If the user's question cannot be answered from the context, respond EXACTLY WITH THIS PHRASE AND NOTHING ELSE:
   I don't know.

3. **Navigation Guidance (REQUIRED)**
   Whenever you mention a specific service or feature, you MUST provide a clickable inline link.
   Format: `[Words to click](https://the-url.com)`. Do not output raw URLs ever.

4. **Response Format**
   Use clean markdown (bolding, bullet points). Keep your final answer under 250 words.

5. **Chain of Thought Reasoning (HIDDEN)**
   Before answering the user, you MUST write out your thought process inside `<think>` tags.
   Inside the tags, verify if the context contains the answer and plan your links.
   Example:
   <think>
   Does context provide dog walking prices? Checking... No, it doesn't.
   Therefore I must output "I don't know."
   </think>
   Final Answer Here.

6. **Tone**
   Be warm, professional, and direct. Act naturally as Ami.

7. **No System Leaks**
   Do not explain your thought process outside of the `<think>` tags. Output ONLY the clean answer to the user afterward.

━━━━━━━━━━━━━━━━━━━━  KNOWLEDGE BASE CONTEXT  ━━━━━━━━━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system_prompt(context_chunks: list[dict[str, Any]]) -> str:
    """
    Render the system prompt with retrieved context chunks.

    Each chunk contributes:
      - Its text content (the answer source)
      - Its page_title + summary (page-level context)
      - Its page_links list (CTA links for navigation guidance)

    Links support both old shape {text, href} and new shape {label, url}
    from the Milvus-backed store.  De-duplicated across all chunks.
    """
    if not context_chunks:
        context = "(No relevant context found in knowledge base.)"
        print(context)
        return SYSTEM_TEMPLATE.format(context=context)

    sections: list[str] = []
    all_links_seen: set[str] = set()
    available_links: list[dict[str, str]] = []

    for chunk in context_chunks:
        text        = chunk.get("text", "")
        source      = chunk.get("source_url", "")
        page_title  = chunk.get("page_title", "")
        summary     = chunk.get("summary", "")
        page_links: list[dict[str, str]] = chunk.get("page_links", [])

        # Collect de-duplicated links — support both {label,url} and {text,href}
        for link in page_links:
            href  = link.get("url") or link.get("href", "")
            label = link.get("label") or link.get("text", href)
            if href and href not in all_links_seen and "amenify" in href:
                all_links_seen.add(href)
                available_links.append({"label": label, "url": href})

        # Build rich context section
        header = f"[Source: {source}]"
        if page_title:
            header += f"  |  {page_title}"
        block_parts = [header]
        if summary:
            block_parts.append(f"Page summary: {summary}")
        block_parts.append(text)
        sections.append("\n".join(block_parts))

    # Attach the full de-duplicated link directory at the end of context
    link_directory = "\n".join(
        f"  - [{lk['label']}]({lk['url']})" for lk in available_links[:60]
    )
    link_block = (
        f"\n\n[AVAILABLE NAVIGATION LINKS — Use these to embed links inline where applicable]\n"
        f"{link_directory}"
        if link_directory
        else ""
    )

    context = "\n\n---\n\n".join(sections) + link_block
    return SYSTEM_TEMPLATE.format(context=context)

