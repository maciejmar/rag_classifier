from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from uuid import uuid5, NAMESPACE_URL

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class Chunk:
    source: str
    text: str


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def _read_documents(docs_dir: str) -> List[tuple[str, str]]:
    root = Path(docs_dir)
    if not root.exists():
        return []

    docs: List[tuple[str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        content = p.read_text(encoding="utf-8", errors="ignore")
        docs.append((str(p), content))
    return docs


def _embed(ollama_client: OllamaClient, model: str, text: str) -> List[float]:
    response = ollama_client.embed(model=model, input=text)
    return response["embeddings"][0]


def _iter_points(
    docs: Iterable[tuple[str, str]],
    *,
    ollama_client: OllamaClient,
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> Iterable[PointStruct]:
    for source, text in docs:
        for idx, part in enumerate(_split_text(text, chunk_size=chunk_size, overlap=overlap)):
            point_id = str(uuid5(NAMESPACE_URL, f"{source}:{idx}"))
            yield PointStruct(
                id=point_id,
                vector=_embed(ollama_client, embedding_model, part),
                payload={
                    "source": source,
                    "text": part,
                },
            )


def build_index(
    *,
    ollama_client: OllamaClient,
    qdrant_client: QdrantClient,
    embedding_model: str,
    docs_dir: str,
    collection_name: str,
    chunk_size: int,
    overlap: int,
) -> int:
    docs = _read_documents(docs_dir)
    points = list(
        _iter_points(
            docs,
            ollama_client=ollama_client,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )
    if not points:
        return 0

    vector_size = len(points[0].vector)
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    qdrant_client.upsert(collection_name=collection_name, points=points)
    return len(points)


def retrieve(
    *,
    ollama_client: OllamaClient,
    qdrant_client: QdrantClient,
    embedding_model: str,
    collection_name: str,
    query: str,
    top_k: int,
) -> List[Chunk]:
    query_vector = _embed(ollama_client, embedding_model, query)
    try:
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        return []

    chunks: List[Chunk] = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text")
        source = payload.get("source")
        if isinstance(text, str) and isinstance(source, str):
            chunks.append(Chunk(source=source, text=text))
    return chunks
