from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from app.config import settings
from app.rag import Chunk, build_index, retrieve


class RAGState(TypedDict, total=False):
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
    question = state["question"]
    chunks = retrieve(
        ollama_client=ollama_client,
        qdrant_client=qdrant_client,
        embedding_model=settings.embedding_model,
        collection_name=settings.qdrant_collection,
        query=question,
        top_k=settings.top_k,
    )
    return {"retrieved": chunks}


def generate_node(state: RAGState, *, ollama_client: OllamaClient) -> Dict[str, Any]:
    question = state["question"]
    context = "\n\n".join(c.text for c in state.get("retrieved", []))

    prompt = (
        "Answer only using the provided context. "
        "If the context is insufficient, say exactly: BRAK_DANYCH.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )

    result = ollama_client.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You are a precise assistant for company analyses and reports."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = result["message"]["content"].strip()
    return {"answer": answer}


def classify_node(state: RAGState) -> Dict[str, Any]:
    answer = state.get("answer", "")
    label = "NO_ANSWER" if answer == "BRAK_DANYCH" else "ANSWERED"
    return {"label": label}


def build_app_graph(*, ollama_client: OllamaClient, qdrant_client: QdrantClient):
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


def main() -> None:
    ollama_client = OllamaClient(host=settings.ollama_host)
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    indexed_chunks = build_index(
        ollama_client=ollama_client,
        qdrant_client=qdrant_client,
        embedding_model=settings.embedding_model,
        docs_dir=settings.docs_dir,
        collection_name=settings.qdrant_collection,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    app = build_app_graph(ollama_client=ollama_client, qdrant_client=qdrant_client)

    print(f"RAG gotowy. Zindeksowano chunki: {indexed_chunks}")
    print("Wpisz pytanie (exit aby zakończyć).")
    while True:
        question = input("\nPytanie: ").strip()
        if question.lower() == "exit":
            break
        out = app.invoke({"question": question})
        print(f"\nLabel: {out.get('label')}")
        print(f"Odpowiedź: {out.get('answer')}")
        sources = [c.source for c in out.get("retrieved", [])]
        if sources:
            print("Źródła:")
            for src in dict.fromkeys(sources):
                print(f"- {src}")


if __name__ == "__main__":
    main()
