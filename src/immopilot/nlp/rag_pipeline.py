"""Online RAG pipeline: retrieve relevant chunks, format prompt, call LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from immopilot import config
from immopilot.nlp.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    url: str
    score: float


@dataclass
class RagAnswer:
    answer: str
    chunks: list[RetrievedChunk]
    provider: str
    model: str


SYSTEM_PROMPT = """You are ImmoPilot, an assistant specialized in apartments and neighborhoods of Zurich (Switzerland).

Rules:
- Answer ONLY using the CONTEXT below. If the answer is not in the context, say so plainly.
- Quote brief facts in your own words, do not copy long passages.
- Cite sources as [Source N] inline, where N matches the numbered context.
- If the user asks for legal or financial advice, add a one-line disclaimer.
- Reply in the language of the user question."""

USER_TEMPLATE = """CONTEXT
{context}

USER QUESTION
{question}

Answer concisely (≤ 5 sentences) and cite sources as [Source N]."""


@lru_cache(maxsize=1)
def _load_index_and_meta():
    index = faiss.read_index(str(config.RAG_DIR / "faiss.index"))
    chunks = json.loads((config.RAG_DIR / "chunks.json").read_text(encoding="utf-8"))
    embedder_name = (config.RAG_DIR / "embedder.txt").read_text().strip()
    return index, chunks, embedder_name


@lru_cache(maxsize=2)
def _load_embedder(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


def retrieve(query: str, top_k: int = config.RAG_TOP_K) -> list[RetrievedChunk]:
    index, chunks, embedder_name = _load_index_and_meta()
    embedder = _load_embedder(embedder_name)
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    out: list[RetrievedChunk] = []
    for score, i in zip(D[0].tolist(), I[0].tolist()):
        if i < 0:
            continue
        c = chunks[i]
        out.append(
            RetrievedChunk(
                text=c["text"],
                source=c["source"],
                title=c["title"],
                url=c["url"],
                score=float(score),
            )
        )
    return out


def answer(question: str, top_k: int = config.RAG_TOP_K, provider: str | None = None) -> RagAnswer:
    chunks = retrieve(question, top_k=top_k)
    context = "\n\n".join(
        f"[Source {i+1}] {c.title}\n{c.text}\n(URL: {c.url or 'n/a'})"
        for i, c in enumerate(chunks)
    )
    client = LLMClient(provider=provider)  # type: ignore[arg-type]
    resp = client.complete(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE.format(context=context, question=question),
    )
    return RagAnswer(answer=resp.text, chunks=chunks, provider=resp.provider, model=resp.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "Wie ist Kreis 6 zum Wohnen?"
    a = answer(q)
    print(a.answer)
    print("\nSources:")
    for c in a.chunks:
        print(f"  - {c.title} ({c.source}) score={c.score:.3f}")
