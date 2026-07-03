from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


_DB_PATH = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(_DB_PATH))

collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=DefaultEmbeddingFunction(),
)


def store_documents(docs: list[Document]) -> int:
    texts, metadatas, ids = [], [], []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        file_name = Path(source).name
        texts.append(doc.page_content)
        metadatas.append({"source": file_name, "chunk": i})
        ids.append(f"{file_name}_chunk_{i}")
    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
    return len(docs)


def semantic_search(query: str, n_results: int = 4) -> dict:
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"],
    )


def get_context_and_sources(results: dict) -> tuple[str, list[dict]]:
    if not results or not results.get("documents") or not results["documents"][0]:
        return "", []
    texts = results["documents"][0]
    metas = results["metadatas"][0]
    context = "\n\n".join(texts)
    seen: set[str] = set()
    sources: list[dict] = []
    for text, meta in zip(texts, metas):
        label = f"{meta.get('source', '?')} (chunk {meta.get('chunk', '?')})"
        if label not in seen:
            seen.add(label)
            sources.append({"label": label, "text": text, "type": "document", "url": None})
    return context, sources


def list_indexed_documents() -> list[str]:
    """Return unique document names stored in ChromaDB."""
    try:
        result = collection.get(include=["metadatas"])
        seen: set[str] = set()
        docs: list[str] = []
        for meta in result["metadatas"]:
            name = meta.get("source", "unknown")
            if name not in seen:
                seen.add(name)
                docs.append(name)
        return sorted(docs)
    except Exception:
        return []


def remove_document(source_name: str) -> int:
    """Delete all chunks for the given source. Returns number deleted."""
    try:
        result = collection.get(where={"source": source_name})
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def get_document_text(source_name: str) -> str:
    """Return all chunk text for a document concatenated (for stats analysis)."""
    try:
        result = collection.get(where={"source": source_name}, include=["documents"])
        return "\n\n".join(result.get("documents", []))
    except Exception:
        return ""


def get_first_n_chunks(n: int = 5) -> list[str]:
    """Return text of the first N chunks in the collection (for summarization)."""
    try:
        result = collection.get(limit=n, include=["documents"])
        return result.get("documents", [])
    except Exception:
        return []


store_chain = RunnableLambda(store_documents)
