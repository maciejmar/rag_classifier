from __future__ import annotations

from dataclasses import dataclass
from typing import List
from uuid import NAMESPACE_URL, uuid5

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


@dataclass
class Chunk:
    source: str
    text: str


def split_text(text: str, *, chunk_size: int, overlap: int) -> List[str]:
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


def embed_text(ollama_client: OllamaClient, embedding_model: str, text: str) -> List[float]:
    response = ollama_client.embed(model=embedding_model, input=text)
    return response["embeddings"][0]


def ensure_collection(qdrant_client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if qdrant_client.collection_exists(collection_name=collection_name):
        return
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def index_document(
    *,
    ollama_client: OllamaClient,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_model: str,
    text: str,
    source: str,
    user_id: int,
    document_id: int,
    chunk_size: int,
    overlap: int,
) -> int:
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    first_vector = embed_text(ollama_client, embedding_model, chunks[0])
    ensure_collection(qdrant_client, collection_name, len(first_vector))

    points: List[PointStruct] = [
        PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{user_id}:{document_id}:{idx}")),
            vector=(first_vector if idx == 0 else embed_text(ollama_client, embedding_model, chunk)),
            payload={
                "user_id": user_id,
                "document_id": document_id,
                "source": source,
                "text": chunk,
            },
        )
        for idx, chunk in enumerate(chunks)
    ]
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
    user_id: int,
) -> List[Chunk]:
    query_vector = embed_text(ollama_client, embedding_model, query)
    try:
        hits = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        return []

    output: List[Chunk] = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text")
        source = payload.get("source")
        if isinstance(text, str) and isinstance(source, str):
            output.append(Chunk(source=source, text=text))
    return output
