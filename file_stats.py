from pathlib import Path

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # make detection deterministic

_LANG_NAMES: dict[str, str] = {
    "fa": "Persian",
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "ur": "Urdu",
    "he": "Hebrew",
}


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        for enc in ("utf-8", "utf-8-sig", "windows-1256", "cp1256", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, ValueError):
                continue
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")
            text = "\n".join(pages)
            if text.strip():
                return text
        except ImportError:
            pass
        from langchain_community.document_loaders import PyPDFLoader
        return "\n".join(d.page_content for d in PyPDFLoader(str(path)).load())
    if suffix in (".doc", ".docx"):
        from langchain_community.document_loaders import Docx2txtLoader
        return "\n".join(d.page_content for d in Docx2txtLoader(str(path)).load())
    return ""


def _recommend(char_count: int) -> tuple[int, int]:
    """Return (chunk_size, overlap) targeting a reasonable number of chunks."""
    if char_count < 5_000:
        target = 15
    elif char_count < 30_000:
        target = 40
    elif char_count < 100_000:
        target = 75
    elif char_count < 500_000:
        target = 150
    else:
        target = 300

    raw = max(300, min(2000, char_count / max(target, 1)))
    chunk_size = round(raw / 50) * 50
    overlap = max(50, round(chunk_size * 0.15 / 50) * 50)
    return chunk_size, overlap


def _human_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def analyze_text(text: str) -> dict:
    """Return statistics computed from raw text (used for indexed documents)."""
    char_count = len(text)
    word_count = len(text.split())
    try:
        lang_code = detect(text[:5_000]) if text.strip() else "?"
        language = _LANG_NAMES.get(lang_code, lang_code.upper())
    except LangDetectException:
        language = "Unknown"
    chunk_size, overlap = _recommend(char_count)
    return {
        "size": "~" + _human_size(char_count),
        "char_count": char_count,
        "word_count": word_count,
        "language": language,
        "recommended_chunk_size": chunk_size,
        "recommended_overlap": overlap,
    }


def analyze_file(path: Path) -> dict:
    """Return statistics and chunk recommendations for the given file."""
    size_bytes = path.stat().st_size
    text = _extract_text(path)
    char_count = len(text)
    word_count = len(text.split())

    try:
        lang_code = detect(text[:5_000]) if text.strip() else "?"
        language = _LANG_NAMES.get(lang_code, lang_code.upper())
    except LangDetectException:
        lang_code = "?"
        language = "Unknown"

    chunk_size, overlap = _recommend(char_count)

    return {
        "size": _human_size(size_bytes),
        "char_count": char_count,
        "word_count": word_count,
        "language": language,
        "lang_code": lang_code,
        "recommended_chunk_size": chunk_size,
        "recommended_overlap": overlap,
    }
