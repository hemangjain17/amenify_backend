"""
llm_provider.py
---------------
Flat, separate inference functions for each LLM provider.
No classes — each provider has its own chat() and stream_chat() functions.

Providers
---------
  openai  — chat_openai()  / stream_openai()   (set OPENAI_API_KEY)
  gemini  — chat_gemini()  / stream_gemini()   (set GEMINI_API_KEY)
  ollama  — chat_ollama()  / stream_ollama()   (needs Ollama server running)

Dispatcher
----------
  chat(messages)         — calls the right provider based on LLM_PROVIDER env var
  stream_chat(messages)  — same, but streams tokens
  active_provider()      — returns "openai/gpt-4o-mini" style string for logging
"""

from __future__ import annotations

import json
import os
from typing import Iterator

# ---------------------------------------------------------------------------
# ── OpenAI ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _openai_client():
    """Lazy-init OpenAI client. Raises EnvironmentError if key is missing."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("Run: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return OpenAI(api_key=api_key)


def chat_openai(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Blocking OpenAI chat completion.
    Returns the full assistant reply as a string.
    """
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def stream_openai(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """
    Streaming OpenAI chat completion.
    Yields string tokens as they arrive from the API.
    """
    client = _openai_client()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ---------------------------------------------------------------------------
# ── Gemini ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _gemini_init(model_name: str = "gemini-1.5-flash"):
    """Configure Gemini SDK and return a GenerativeModel instance."""
    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "Run: pip install google-generativeai"
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def _split_messages_for_gemini(
    messages: list[dict],
) -> tuple[str, list[dict], str]:
    """
    Split OpenAI-style messages into:
      - system_text: concatenated system messages
      - history: prior user/model turns (all but the final user message)
      - last_user_text: the final user message content

    Gemini's start_chat() takes history, then send_message() takes the next prompt.
    """
    system_parts: list[str] = []
    history: list[dict] = []

    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

    # The final entry must be a user turn — pop it off as the "next" prompt
    last_user_text = ""
    if history and history[-1]["role"] == "user":
        last_user_text = history.pop()["parts"][0]

    system_text = "\n\n".join(system_parts)
    # Prepend system instruction to the user prompt (Gemini has no system role)
    full_prompt = f"{system_text}\n\n{last_user_text}" if system_text else last_user_text

    return full_prompt, history


def chat_gemini(
    messages: list[dict],
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Blocking Gemini chat completion.
    Returns the full assistant reply as a string.
    """
    import google.generativeai.types as gtypes

    gemini_model = _gemini_init(model)
    config = gtypes.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    full_prompt, history = _split_messages_for_gemini(messages)
    chat_session = gemini_model.start_chat(history=history)
    response = chat_session.send_message(full_prompt, generation_config=config)
    return response.text.strip()


def stream_gemini(
    messages: list[dict],
    model: str = "gemini-1.5-flash",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """
    Streaming Gemini chat completion.
    Yields string tokens as they arrive.
    """
    import google.generativeai.types as gtypes

    gemini_model = _gemini_init(model)
    config = gtypes.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    full_prompt, history = _split_messages_for_gemini(messages)
    chat_session = gemini_model.start_chat(history=history)

    for chunk in chat_session.send_message(
        full_prompt, generation_config=config, stream=True
    ):
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# ── Ollama ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def chat_ollama(
    messages: list[dict],
    model: str = "glm-5.1:cloud",
    base_url: str = "http://localhost:11434",
) -> str:
    """
    Blocking Ollama chat via its REST API.
    Requires a running Ollama server (ollama serve).
    Returns the full assistant reply as a string.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("Run: pip install httpx") from e

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        resp = client.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def stream_ollama(
    messages: list[dict],
    model: str = "glm-5.1:cloud",
    base_url: str = "http://localhost:11434",
) -> Iterator[str]:
    """
    Streaming Ollama chat via its REST API.
    Yields string tokens as they arrive from the local server.
    Requires a running Ollama server (ollama serve).
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("Run: pip install httpx") from e

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        with client.stream(
            "POST",
            "/api/chat",
            json={"model": model, "messages": messages, "stream": True},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done"):
                    break


# ---------------------------------------------------------------------------
# ── Dispatcher ──────────────────────────────────────────────────────────────
# Read LLM_PROVIDER env var and route to the right provider functions.
# ---------------------------------------------------------------------------

def _get_provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").lower().strip()


def active_provider() -> str:
    """
    Returns a human-readable string identifying the active LLM.
    e.g. "openai/gpt-4o-mini", "gemini/gemini-1.5-flash", "ollama/llama3.1:8b"
    """
    p = _get_provider_name()
    if p == "openai":
        return f"openai/{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}"
    if p == "gemini":
        return f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')}"
    if p == "ollama":
        return f"ollama/{os.environ.get('OLLAMA_MODEL', 'glm-5.1:cloud')}"
    return p


def chat(messages: list[dict]) -> str:
    """
    Dispatch to the correct chat function based on LLM_PROVIDER env var.
    Blocking — returns the complete response string.
    """
    p = _get_provider_name()
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

    if p == "openai":
        return chat_openai(
            messages,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    if p == "gemini":
        return chat_gemini(
            messages,
            model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
        )
    if p == "ollama":
        return chat_ollama(
            messages,
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER: {p!r}. Choose: openai | gemini | ollama"
    )


def stream_chat(messages: list[dict]) -> Iterator[str]:
    """
    Dispatch to the correct streaming function based on LLM_PROVIDER env var.
    Yields string tokens as they arrive from the active provider.
    """
    p = _get_provider_name()
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

    if p == "openai":
        yield from stream_openai(
            messages,
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )
    elif p == "gemini":
        yield from stream_gemini(
            messages,
            model=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            temperature=temperature,
        )
    elif p == "ollama":
        yield from stream_ollama(
            messages,
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {p!r}. Choose: openai | gemini | ollama"
        )
