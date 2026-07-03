from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 3) -> tuple[str, list[dict]]:
    """Search DuckDuckGo and return (joined context, list of source dicts)."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "", []
        context = "\n\n".join(r.get("body", "") for r in results if r.get("body"))
        sources = [
            {
                "label": r.get("title", r.get("href", "")),
                "text": r.get("body", ""),
                "type": "web",
                "url": r.get("href", ""),
            }
            for r in results if r.get("body")
        ]
        return context, sources
    except Exception:
        return "", []
