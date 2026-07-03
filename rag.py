import os
import time
from typing import Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI, NotFoundError, RateLimitError

from vector_store import get_context_and_sources, semantic_search
from web_search import web_search

load_dotenv()

# ── Provider registry ─────────────────────────────────────────────────────────

PROVIDERS: dict[str, dict] = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPEN_ROUTER_API_KEY",
        "models": [
            "openai/gpt-oss-120b:free",
            "anthropic/claude-3-haiku",
            "google/gemini-flash-1.5-8b",
            "meta-llama/llama-3.1-8b-instruct:free",
        ],
    },
    "Anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "key_env": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    },
    "Ollama (local)": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "models": ["llama3.2", "mistral", "qwen2.5", "phi3"],
    },
}

_clients: dict[str, OpenAI] = {}

# ── Prompts ───────────────────────────────────────────────────────────────────

_RAG_SYSTEM = (
    "You are a helpful assistant for retrieval-augmented generation (RAG).\n"
    "Answer in the same language as the question.\n"
    "Answer ONLY using the provided context. "
    "If the answer is not found in the context, say so in the same language as the question."
)

_SUMMARY_SYSTEM = (
    "Write a concise summary (3–5 sentences) of the following document excerpt. "
    "Use the same language as the document."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_client(provider: str = "OpenRouter") -> OpenAI:
    if provider not in _clients:
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else "no-key"
        if cfg["key_env"] and not key:
            raise RuntimeError(
                f"{cfg['key_env']} is not set. Add it to your .env file."
            )
        _clients[provider] = OpenAI(base_url=cfg["base_url"], api_key=key or "no-key")
    return _clients[provider]


def get_ollama_models() -> list[str]:
    """Fetch models actually installed in the local Ollama instance."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def get_models(provider: str) -> list[str]:
    if provider == "Ollama (local)":
        models = get_ollama_models()
        return models if models else ["(no models installed — run: ollama pull <name>)"]
    return PROVIDERS.get(provider, {}).get("models", [])


def _call_llm(
    messages: list[dict],
    model: str,
    provider: str,
    on_token: Callable | None,
    on_retry: Callable | None,
) -> str:
    delays = [5, 15, 30]
    for attempt, delay in enumerate(delays, 1):
        try:
            client = get_client(provider)
            if on_token:
                stream = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=0.2, max_tokens=1024, stream=True,
                )
                full = ""
                for chunk in stream:
                    token = (
                        chunk.choices[0].delta.content
                        if chunk.choices and chunk.choices[0].delta.content
                        else ""
                    )
                    if token:
                        full += token
                        on_token(token)
                return full
            else:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=0.2, max_tokens=1024,
                )
                return resp.choices[0].message.content.strip()
        except NotFoundError:
            raise RuntimeError(
                f"Model '{model}' not found.\n"
                + (f"Run:  ollama pull {model}" if provider == "Ollama (local)" else
                   "Check the model name in the toolbar.")
            )
        except RateLimitError:
            if attempt == len(delays):
                raise
            if on_retry:
                on_retry(f"Rate limited — retrying in {delay}s ({attempt}/{len(delays)}) …")
            time.sleep(delay)
    raise RuntimeError("unreachable")


# ── Public API ────────────────────────────────────────────────────────────────

def rag_answer(
    query: str,
    n_results: int = 4,
    on_retry: Callable | None = None,
    on_token: Callable | None = None,
    history: list[dict] | None = None,
    provider: str = "OpenRouter",
    model: str | None = None,
    use_web_fallback: bool = False,
) -> tuple[str, list[dict]]:
    if model is None:
        model = PROVIDERS[provider]["models"][0]

    results = semantic_search(query, n_results=n_results)
    context, sources = get_context_and_sources(results)

    if not context.strip() and use_web_fallback:
        if on_retry:
            on_retry("No document context found — searching the web …")
        context, sources = web_search(query, max_results=3)

    if not context.strip():
        return "No relevant context found in the loaded documents.", []

    messages: list[dict] = [{"role": "system", "content": _RAG_SYSTEM}]
    if history:
        messages.extend(history[-6:])  # last 3 turns
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:",
    })

    answer = _call_llm(messages, model, provider, on_token, on_retry)
    return answer, sources


def summarize_document(
    chunks: list[str],
    on_token: Callable | None = None,
    provider: str = "OpenRouter",
    model: str | None = None,
) -> str:
    if not chunks:
        return ""
    if model is None:
        model = PROVIDERS[provider]["models"][0]
    text = "\n\n".join(chunks)[:4000]
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": text},
    ]
    return _call_llm(messages, model, provider, on_token, None)
