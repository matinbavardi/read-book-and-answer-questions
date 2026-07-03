import webbrowser
from datetime import datetime
from pathlib import Path


_SESSIONS_DIR = Path(__file__).parent / "qa_sessions"


class SessionLogger:
    """Writes all Q&A pairs from one session to a dated text file."""

    def __init__(self, document_name: str):
        stem = Path(document_name).stem
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        folder = _SESSIONS_DIR / stem
        folder.mkdir(parents=True, exist_ok=True)

        self.file_path = folder / f"{stem}_{timestamp}.txt"
        self._pairs: list[dict] = []
        self._document_name = document_name
        self._started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.file_path.open("w", encoding="utf-8") as f:
            f.write(f"Document : {document_name}\n")
            f.write(f"Session  : {self._started}\n")
            f.write("=" * 60 + "\n\n")

    def log(self, question: str, answer: str, sources: list) -> None:
        source_labels = [
            s["label"] if isinstance(s, dict) else s for s in sources
        ]
        self._pairs.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        })
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(f"Q: {question}\n\n")
            f.write(f"A: {answer}\n\n")
            if source_labels:
                f.write("Sources:\n")
                for label in source_labels:
                    f.write(f"  - {label}\n")
            f.write("-" * 40 + "\n\n")

    def export_html(self) -> Path:
        """Generate a styled HTML export and open it in the browser."""
        html_path = self.file_path.with_suffix(".html")

        pairs_html = ""
        for p in self._pairs:
            src_items = ""
            for s in p["sources"]:
                if isinstance(s, dict):
                    url = s.get("url") or ""
                    label = s.get("label", "")
                    if s.get("type") == "web" and url:
                        src_items += f'<li><a href="{url}" target="_blank">{label}</a></li>'
                    else:
                        src_items += f"<li>{label}</li>"
                else:
                    src_items += f"<li>{s}</li>"

            sources_block = (
                f'<ul class="sources">{src_items}</ul>' if src_items else ""
            )
            q = p["question"].replace("<", "&lt;").replace(">", "&gt;")
            a = p["answer"].replace("<", "&lt;").replace(">", "&gt;")
            pairs_html += f"""
            <div class="qa-pair">
                <div class="question">{q}</div>
                <div class="answer">{a}</div>
                {sources_block}
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="auto">
<head>
<meta charset="UTF-8">
<title>Q&amp;A — {self._document_name}</title>
<style>
  body {{
    font-family: 'Noto Naskh Arabic', 'Segoe UI', Arial, sans-serif;
    max-width: 900px; margin: 40px auto; padding: 0 24px;
    background: #fafafa; color: #333; line-height: 1.7;
  }}
  h1 {{ color: #4a7fc1; border-bottom: 2px solid #4a7fc1; padding-bottom: 8px; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 32px; }}
  .qa-pair {{
    background: white; border-radius: 8px; padding: 20px 24px;
    margin: 16px 0; box-shadow: 0 1px 4px rgba(0,0,0,0.08); direction: auto;
  }}
  .question {{
    font-weight: 600; color: #4a7fc1; margin-bottom: 10px; font-size: 1.05em;
  }}
  .question::before {{ content: "Q: "; }}
  .answer {{ color: #222; white-space: pre-wrap; }}
  .sources {{ margin-top: 12px; font-size: 0.85em; color: #888; padding-left: 16px; }}
  .sources a {{ color: #4a7fc1; }}
</style>
</head>
<body>
<h1>Document Q&amp;A Session</h1>
<div class="meta">
  Document: <strong>{self._document_name}</strong><br>
  Session: {self._started}
</div>
{pairs_html}
</body>
</html>"""

        html_path.write_text(html, encoding="utf-8")
        webbrowser.open(html_path.as_uri())
        return html_path


def load_all_questions() -> list[str]:
    """Return every distinct question from all saved session files, newest first."""
    seen: set[str] = set()
    questions: list[str] = []
    if not _SESSIONS_DIR.exists():
        return questions
    for txt_file in sorted(_SESSIONS_DIR.rglob("*.txt"), reverse=True):
        try:
            for line in txt_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("Q: "):
                    q = line[3:].strip()
                    if q and q not in seen:
                        seen.add(q)
                        questions.append(q)
        except Exception:
            pass
    return questions
