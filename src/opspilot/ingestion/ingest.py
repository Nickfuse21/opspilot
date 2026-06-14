"""Ingestion pipeline: load -> chunk -> embed -> store.

This file builds the searchable index OpsPilot answers from. We build it one
function at a time. Piece 1 below just loads documents off disk.
"""

from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opspilot import config
from opspilot.embeddings import embed_texts


def load_documents(kb_dir: Path = config.KNOWLEDGE_BASE_DIR) -> list[dict]:
    """Read every .md file in the knowledge base into memory.

    Returns a list of dicts, one per document, each shaped like:
        {"source": "refund_policy.md", "text": "<full file contents>"}

    We keep "source" (the filename) because it becomes the CITATION later --
    every chunk needs to remember which document it came from.
    """
    documents: list[dict] = []
    for path in sorted(kb_dir.glob("*.md")):  # sorted = stable, predictable order
        text = path.read_text(encoding="utf-8")
        documents.append({"source": path.name, "text": text})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document into smaller, overlapping chunks.

    Returns a list of chunk dicts, each shaped like:
        {"id": "refund_policy.md::chunk-0",   # unique id (Chroma needs unique ids)
         "source": "refund_policy.md",         # filename -> becomes the citation
         "text": "<the chunk text>"}
    """
    # The splitter tries natural boundaries (paragraphs, then lines, then words)
    # so chunks don't cut mid-sentence. chunk_overlap repeats a little text
    # between neighbours so a boundary-straddling idea survives intact.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    for doc in documents:
        pieces = splitter.split_text(doc["text"])  # one document -> several strings
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f"{doc['source']}::chunk-{i}",
                    "source": doc["source"],
                    "text": piece,
                }
            )
    return chunks


def build_index(chunks: list[dict]) -> int:
    """Embed every chunk and store it in a persistent Chroma collection.

    Each stored record holds: the vector (for searching), the text (to feed the
    LLM later), the source filename (for citations), and a unique id.
    Returns the number of chunks now in the collection.
    """
    # PersistentClient saves the DB to disk so we build the index once and reuse it.
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # rank results by cosine similarity
    )

    embeddings = embed_texts([c["text"] for c in chunks])  # all chunks at once (GPU)

    # upsert = insert-or-overwrite by id, so re-running ingestion won't duplicate.
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    return collection.count()


def ingest() -> None:
    """Run the full pipeline: load -> chunk -> embed -> store."""
    docs = load_documents()
    chunks = chunk_documents(docs)
    total = build_index(chunks)
    print(f"Indexed {len(chunks)} chunks ({total} total in collection) into {config.CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
