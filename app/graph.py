from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from app.config import settings
from app.rag import Chunk, retrieve


class RAGState(TypedDict, total=False):
    user_id: int
    question: str
    retrieved: List[Chunk]
    answer: str
    label: str


def retrieve_node(
    state: RAGState,
    *,
    ollama_client: OllamaClient,
    qdrant_client: QdrantClient,
) -> Dict[str, Any]:
    chunks = retrieve(
        ollama_client=ollama_client,
        qdrant_client=qdrant_client,
        embedding_model=settings.embedding_model,
        collection_name=settings.qdrant_collection,
        query=state["question"],
        top_k=settings.top_k,
        user_id=state["user_id"],
    )
    return {"retrieved": chunks}


def generate_node(state: RAGState, *, ollama_client: OllamaClient) -> Dict[str, Any]:
    question = state["question"]
    context = "\n\n".join(c.text for c in state.get("retrieved", []))
    prompt = (
        "Use only context from company documents. "
        "If data is insufficient, answer exactly: BRAK_DANYCH.\n\n"
        f"Pytanie: {question}\n\n"
        f"Kontekst:\n{context}"
    )
    result = ollama_client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "Jestes asystentem do analiz i raportow firmowych."},
            {"role": "user", "content": prompt},
        ],
    )
    return {"answer": result["message"]["content"].strip()}


def classify_node(state: RAGState) -> Dict[str, Any]:
    answer = state.get("answer", "")
    return {"label": "NO_ANSWER" if answer == "BRAK_DANYCH" else "ANSWERED"}


def build_graph(*, ollama_client: OllamaClient, qdrant_client: QdrantClient):
    graph = StateGraph(RAGState)
    graph.add_node(
        "retrieve",
        lambda s: retrieve_node(s, ollama_client=ollama_client, qdrant_client=qdrant_client),
    )
    graph.add_node("generate", lambda s: generate_node(s, ollama_client=ollama_client))
    graph.add_node("classify", classify_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "classify")
    graph.add_edge("classify", END)
    return graph.compile()
