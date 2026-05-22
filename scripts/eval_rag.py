"""Quantitative evaluation of the RAG block over a hand-curated gold set.

Metrics:
  - Hit-Rate@k : fraction of questions where at least one expected source file
    appears in the top-k retrieved chunks (retrieval quality).
  - MRR        : mean reciprocal rank of the first correct source (how high the
    right document is ranked; 1.0 = always rank 1).
  - Citation rate : fraction of generated answers that cite at least one
    [Source N] marker (a light, deterministic proxy for grounding — full
    faithfulness scoring would need a second LLM judge, noted as future work).

The gold set (scripts/rag_gold_set.json) is curated by hand: each question is
tagged with the corpus file(s) that should answer it. Retrieval is evaluated
without calling the LLM (fast, free, deterministic); the citation check calls
the LLM once per question and is therefore optional via --no-llm.

Usage:
  python scripts/eval_rag.py            # full eval (retrieval + citation)
  python scripts/eval_rag.py --no-llm   # retrieval only (no API calls)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from immopilot import config
from immopilot.nlp.rag_pipeline import answer as rag_answer
from immopilot.nlp.rag_pipeline import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GOLD_PATH = Path(__file__).resolve().parent / "rag_gold_set.json"
OUT_DIR = config.DOCS_DIR / "rag_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITE_RE = re.compile(r"\[Source\s+\d+\]", re.IGNORECASE)


def load_gold() -> list[dict]:
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def eval_retrieval(gold: list[dict], top_k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    per_q = []
    for item in gold:
        chunks = retrieve(item["question"], top_k=top_k)
        retrieved_sources = [c.source for c in chunks]
        expected = set(item["expected_source"])

        # rank of first correct source (1-indexed); 0 if not found
        rank = 0
        for idx, src in enumerate(retrieved_sources, start=1):
            if src in expected:
                rank = idx
                break
        hit = rank > 0
        hits += int(hit)
        rr_sum += (1.0 / rank) if rank else 0.0
        per_q.append({
            "id": item["id"],
            "question": item["question"],
            "expected": sorted(expected),
            "retrieved": retrieved_sources,
            "hit": hit,
            "rank": rank,
        })
    n = len(gold)
    return {
        "n": n,
        "top_k": top_k,
        "hit_rate": hits / n,
        "mrr": rr_sum / n,
        "per_question": per_q,
    }


def eval_citations(gold: list[dict]) -> dict:
    cited = 0
    per_q = []
    for item in gold:
        a = rag_answer(item["question"])
        has_citation = bool(CITE_RE.search(a.answer))
        cited += int(has_citation)
        per_q.append({"id": item["id"], "has_citation": has_citation})
    n = len(gold)
    return {"n": n, "citation_rate": cited / n, "per_question": per_q}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the citation check (no API calls)")
    ap.add_argument("--top-k", type=int, default=config.RAG_TOP_K)
    args = ap.parse_args()

    gold = load_gold()
    logger.info("Gold set: %d questions", len(gold))

    retrieval = eval_retrieval(gold, top_k=args.top_k)
    logger.info("Hit-Rate@%d: %.3f | MRR: %.3f", args.top_k, retrieval["hit_rate"], retrieval["mrr"])

    results = {"retrieval": retrieval}

    if not args.no_llm:
        citations = eval_citations(gold)
        logger.info("Citation rate: %.3f", citations["citation_rate"])
        results["citations"] = citations

    (OUT_DIR / "rag_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Console summary
    print("\n" + "=" * 50)
    print("RAG EVALUATION")
    print("-" * 50)
    print(f"Gold questions          : {retrieval['n']}")
    print(f"Hit-Rate@{args.top_k}             : {retrieval['hit_rate']:.1%}")
    print(f"MRR                     : {retrieval['mrr']:.3f}")
    if not args.no_llm:
        print(f"Citation rate           : {results['citations']['citation_rate']:.1%}")
    print("=" * 50)

    # Show any misses for transparency
    misses = [q for q in retrieval["per_question"] if not q["hit"]]
    if misses:
        print("\nRetrieval misses (expected source not in top-k):")
        for m in misses:
            print(f"  Q{m['id']}: {m['question']}")
            print(f"      expected {m['expected']}, got {m['retrieved'][:3]}…")
    else:
        print("\nNo retrieval misses — every question retrieved an expected source.")
    print(f"\nSaved: {OUT_DIR / 'rag_eval.json'}")


if __name__ == "__main__":
    main()
