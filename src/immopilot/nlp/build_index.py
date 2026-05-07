"""Build the FAISS index for the RAG corpus.

Sources: text snapshots in ``data/raw/rag_corpus/``. Each ``.txt`` or ``.md`` file
should contain text about one Zurich district or topic. Frontmatter (`---`) is
parsed as YAML for metadata (``title``, ``url``, ``district``).

Usage:
    python -m immopilot.nlp.build_index
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from immopilot import config
from immopilot.nlp.text_utils import chunk_text as _chunk_text

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    url: str
    district: str | None
    chunk_id: int


def _parse_doc(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.DOTALL)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        meta, body = {}, raw
    return meta, body


def collect_chunks(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = sorted(list(corpus_dir.glob("**/*.md")) + list(corpus_dir.glob("**/*.txt")))
    if not files:
        raise FileNotFoundError(f"No corpus files in {corpus_dir}")
    for f in files:
        meta, body = _parse_doc(f)
        for i, ch in enumerate(_chunk_text(body, config.RAG_CHUNK_SIZE, config.RAG_CHUNK_OVERLAP)):
            chunks.append(
                Chunk(
                    text=ch,
                    source=str(f.relative_to(corpus_dir)),
                    title=meta.get("title", f.stem),
                    url=meta.get("url", ""),
                    district=str(meta.get("district")) if meta.get("district") else None,
                    chunk_id=i,
                )
            )
    logger.info("Collected %d chunks from %d files", len(chunks), len(files))
    return chunks


def build_index(model_name: str = config.EMBEDDING_MODEL_DEFAULT) -> None:
    config.set_global_seed()
    corpus_dir = config.RAW_DIR / "rag_corpus"
    chunks = collect_chunks(corpus_dir)
    texts = [c.text for c in chunks]

    logger.info("Loading embedder %s …", model_name)
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    out_dir = config.RAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # Persist chunk metadata aligned with index ids
    import json

    (out_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "id": i,
                    "text": c.text,
                    "source": c.source,
                    "title": c.title,
                    "url": c.url,
                    "district": c.district,
                    "chunk_id": c.chunk_id,
                }
                for i, c in enumerate(chunks)
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "embedder.txt").write_text(model_name)
    logger.info("Wrote FAISS index + metadata to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_index()
