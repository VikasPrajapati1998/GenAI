"""
HybridRAG.py — Production-style "Advanced RAG"
==============================================

    question ─► [ BM25 (lexical) ] ─┐
                                    ├─► Reciprocal Rank Fusion ─► ACL filter ─► LLM rerank ─► grounded answer
    question ─► [ dense vectors  ] ─┘

What this file adds on top of RAG.py (section 4.2 of the guide)
--------------------------------------------------------------
1. BM25 keyword search (implemented from scratch, ~25 lines) — catches IDs like "PO-2026-0412" and
   part numbers that embeddings blur.
2. Reciprocal Rank Fusion (RRF) — merges two ranked lists without score calibration.
3. Permission (ACL) filtering BEFORE the LLM sees anything — never let the model "hide" documents.
4. LLM-as-reranker — the model scores each candidate 0-10 for relevance (structured JSON output).
5. Optional *contextual retrieval* (--contextual): the LLM writes a 1-line context prefix per chunk
   before indexing (Anthropic, 2024). Slower ingest, better recall.

Run
---
    ollama pull granite3.3:2b && ollama pull nomic-embed-text:v1.5
    python HybridRAG.py
    python HybridRAG.py --role employee      # see the ACL filter remove finance-only docs
    python HybridRAG.py --contextual --chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
CANDIDATES = 8     # how many fused candidates go to the reranker
FINAL_K = 3        # how many reranked chunks go to the answer prompt
RRF_K = 60         # standard RRF constant

# Each chunk carries metadata: source + allowed roles (ACL). In production this comes from the source system.
CHUNKS = [
    {"id": "c1", "source": "policy.md", "acl": ["employee", "finance"],
     "text": "Purchases above INR 10,00,000 need CFO approval and a three-quote comparison."},
    {"id": "c2", "source": "policy.md", "acl": ["employee", "finance"],
     "text": "Purchases between INR 2,00,001 and INR 10,00,000 need Finance Controller approval."},
    {"id": "c3", "source": "policy.md", "acl": ["employee", "finance"],
     "text": "Only APPROVED vendors with valid GST and ISO 9001 may receive purchase orders."},
    {"id": "c4", "source": "po_register.csv", "acl": ["finance"],
     "text": "PO-2026-0412: vendor Zenith Systems, item RTX 6000 Ada x2, total INR 12,40,000, status OPEN."},
    {"id": "c5", "source": "po_register.csv", "acl": ["finance"],
     "text": "PO-2026-0398: vendor ABC Technologies, item Dell Precision 7960, total INR 4,10,000, status CLOSED."},
    {"id": "c6", "source": "vendor_master.md", "acl": ["employee", "finance"],
     "text": "Nova Compute was BLACKLISTED on 2026-02-10 for repeated late deliveries."},
    {"id": "c7", "source": "it_standard.md", "acl": ["employee", "finance"],
     "text": "Every GPU purchase needs an IT Security review before the PO is raised."},
    {"id": "c8", "source": "it_standard.md", "acl": ["employee", "finance"],
     "text": "Servers and workstations require a minimum 3-year on-site warranty."},
    {"id": "c9", "source": "finance_memo.md", "acl": ["finance"],
     "text": "Q3 FY26 capex budget for AI hardware is INR 85,00,000, of which INR 31,00,000 is committed."},
]


# ──────────────────────────────────────────────────────────────────────────────
# LLM / embedding helpers
# ──────────────────────────────────────────────────────────────────────────────
def chat(messages, **kw) -> str:
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


def embed(texts, kind):
    if "nomic" in EMBED_MODEL:
        texts = [("search_document: " if kind == "document" else "search_query: ") + t for t in texts]
    v = np.array(ollama.embed(model=EMBED_MODEL, input=texts).embeddings, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# BM25 from scratch (Okapi BM25, k1=1.5, b=0.75)
# ──────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    # keep IDs like po-2026-0412 together; lowercase everything
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        for i, doc in enumerate(self.toks):
            tf = Counter(doc)
            for t in q:
                if t in tf:
                    num = tf[t] * (self.k1 + 1)
                    den = tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                    out[i] += self.idf[t] * num / den
        return out


# ──────────────────────────────────────────────────────────────────────────────
class HybridRAG:
    def __init__(self, chunks, contextual: bool = False):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        if contextual:
            # Contextual retrieval: prepend an LLM-written situating sentence to every chunk.
            print("[ingest] writing contextual prefixes (one LLM call per chunk) …")
            for c in chunks:
                c["context"] = chat([{"role": "user", "content":
                    f"Document '{c['source']}' contains this chunk:\n{c['text']}\n\n"
                    "Write ONE short sentence that situates this chunk within the document, "
                    "to improve search retrieval. Reply with the sentence only."}])
            texts = [f"{c['context']}\n{c['text']}" for c in chunks]
        self.index_texts = texts
        self.bm25 = BM25(texts)
        self.vectors = embed(texts, "document")
        print(f"[ingest] indexed {len(chunks)} chunks (BM25 + {EMBED_MODEL})")

    def hybrid_search(self, question: str) -> list[dict]:
        bm = self.bm25.scores(question)
        dense = self.vectors @ embed([question], "query")[0]
        bm_rank = np.argsort(-bm)
        dense_rank = np.argsort(-dense)
        # RRF: score = Σ 1 / (k + rank). Robust to the two scorers having different scales.
        fused = Counter()
        for ranking in (bm_rank, dense_rank):
            for r, idx in enumerate(ranking):
                fused[idx] += 1.0 / (RRF_K + r + 1)
        order = [i for i, _ in fused.most_common()]
        print(f"[search] BM25 top: {[self.chunks[i]['id'] for i in bm_rank[:3]]}  "
              f"dense top: {[self.chunks[i]['id'] for i in dense_rank[:3]]}  "
              f"fused: {[self.chunks[i]['id'] for i in order[:5]]}")
        return [self.chunks[i] for i in order]

    @staticmethod
    def acl_filter(cands, role):
        allowed = [c for c in cands if role in c["acl"]]
        removed = len(cands) - len(allowed)
        if removed:
            print(f"[acl] removed {removed} chunk(s) not visible to role '{role}'")
        return allowed

    def rerank(self, question, cands):
        """Pointwise LLM reranking with a JSON schema so small models return parseable output."""
        schema = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"]}
        scored = []
        for c in cands:
            raw = chat([{"role": "user", "content":
                f"Question: {question}\nPassage: {c['text']}\n"
                "Rate how useful the passage is for answering the question, 0 (useless) to 10 (answers it). "
                'Return JSON like {"score": 7}.'}], format=schema, options={"temperature": 0})
            try:
                s = int(json.loads(raw)["score"])
            except (ValueError, KeyError, TypeError):
                s = 0
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        print("[rerank]", ", ".join(f"{c['id']}={s}" for s, c in scored))
        return [c for _, c in scored]

    def answer(self, question, role="finance"):
        cands = self.acl_filter(self.hybrid_search(question), role)[:CANDIDATES]
        top = self.rerank(question, cands)[:FINAL_K]
        ctx = "\n".join(f"[{c['id']}] ({c['source']}) {c['text']}" for c in top)
        return chat([
            {"role": "system", "content": "Answer only from the sources. Cite chunk ids like [c4]. "
                                          "If the answer is not in the sources, say you don't know."},
            {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}],
            options={"temperature": 0})


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--role", default="finance", choices=["finance", "employee"])
    ap.add_argument("--contextual", action="store_true")
    ap.add_argument("--question")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    try:
        rag = HybridRAG(CHUNKS, contextual=a.contextual)
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}\nRun `ollama pull {EMBED_MODEL}` and `ollama pull {CHAT_MODEL}`.")
    qs = [a.question] if a.question else [
        "What is the status of PO-2026-0412 and who approves it?",   # BM25 wins on the exact ID
        "Is the blacklisted supplier allowed for new orders?",       # dense wins on paraphrase
        "How much AI hardware budget is left this quarter?",         # finance-only (try --role employee)
    ]
    if a.chat:
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            print("Bot:", rag.answer(q, a.role))
        return
    for q in qs:
        print(f"\nQ ({a.role}): {q}\nA: {rag.answer(q, a.role)}")


if __name__ == "__main__":
    main()
