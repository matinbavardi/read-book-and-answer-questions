from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Persian/Arabic punctuation added so chunks break at natural sentence boundaries
_SEPARATORS = ["\n\n", "\n", ".", "!", "?", "؟", "،", "؛", " ", ""]


def _load_txt(file_path: Path) -> list[Document]:
    """Try common encodings so Persian/Arabic Windows files load correctly."""
    for enc in ("utf-8", "utf-8-sig", "windows-1256", "cp1256", "latin-1"):
        try:
            text = file_path.read_text(encoding=enc)
            return [Document(page_content=text, metadata={"source": str(file_path)})]
        except (UnicodeDecodeError, ValueError):
            continue
    # last resort — replace undecodable bytes
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": str(file_path)})]


def _load_pdf(file_path: Path) -> list[Document]:
    """Try pdfplumber first (better RTL support), fall back to PyPDFLoader."""
    try:
        import pdfplumber
        docs = []
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": str(file_path), "page": i},
                    ))
        if docs:
            return docs
    except ImportError:
        pass
    # fallback
    return PyPDFLoader(str(file_path)).load()


def load_document(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(file_path)
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix in (".docx", ".doc"):
        return Docx2txtLoader(str(file_path)).load()
    raise ValueError(f"Unsupported file type: {suffix}")


def make_document_chain(chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )
    return RunnableLambda(load_document) | RunnableLambda(splitter.split_documents)


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python document_pipeline.py <file>")
        sys.exit(1)

    chunks = make_document_chain().invoke(path)
    print(f"Loaded and split into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:300])
