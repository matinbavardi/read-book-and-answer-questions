from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter


_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
}

def load_document(file_path: Path) -> list[Document]:
    loader_cls = _LOADERS.get(file_path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return loader_cls(str(file_path)).load()


def make_document_chain(chunk_size: int = 1000, chunk_overlap: int = 150):
    """Return a LCEL chain: Path -> list[Document] with the given splitter settings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
