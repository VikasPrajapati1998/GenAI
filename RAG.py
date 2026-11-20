"""
RAG.py — Classic Retrieval-Augmented Generation (dense vector search)
=====================================================================

Pipeline (see section 4 of the guide):

    INGEST:  documents ─► chunk ─► embed (nomic-embed-text) ─► in-memory vector index
    QUERY :  question ─► embed ─► cosine top-k ─► prompt with numbered sources ─► LLM answer + citations

What this file teaches
----------------------
* Structure-aware chunking with overlap.
* Asymmetric embeddings: nomic-embed-text expects "search_document:" / "search_query:" prefixes.
* Cosine similarity with NumPy (no vector DB needed for a demo; swap in Qdrant/pgvector/Chroma later).
* A grounded prompt that forces citations and an "I don't know" path.

Run
---
    ollama pull granite3.3:2b
    ollama pull nomic-embed-text:v1.5
    python RAG.py                       # runs demo questions
    python RAG.py --chat                # interactive
    python RAG.py --model gpt-oss:120b-cloud --question "Which GPUs need CFO approval?"
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (override with environment variables or CLI flags)
# ──────────────────────────────────────────────────────────────────────────────
CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")      # small, follows instructions well
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
TOP_K = 3                 # number of chunks passed to the LLM
CHUNK_SIZE = 400          # characters per chunk (small corpus → small chunks)
CHUNK_OVERLAP = 80        # overlap keeps sentences that straddle a boundary retrievable

# ──────────────────────────────────────────────────────────────────────────────
# Sample knowledge base: a tiny procurement (P2P) corpus
# ──────────────────────────────────────────────────────────────────────────────
DOCUMENTS = {
    "policy_procurement.md": """
# Procurement Policy v3.2
## Approval thresholds
Purchases up to INR 2,00,000 need approval from the department head.
Purchases between INR 2,00,001 and INR 10,00,000 need approval from the Finance Controller.
Purchases above INR 10,00,000 need approval from the CFO and a three-quote comparison.
## Vendor eligibility
Only vendors with status APPROVED in the vendor master may receive purchase orders.
Vendors must hold a valid GST registration and an ISO 9001 certificate for hardware supply.
Blacklisted vendors cannot be used for 24 months after blacklisting.
""",
    "policy_it_hardware.md": """
# IT Hardware Standard 2026
GPU workstations must use NVIDIA RTX PRO 6000 Blackwell or RTX 6000 Ada class cards.
Every GPU purchase requires a security review by the IT Security team before the PO is raised.
Laptops are refreshed every 4 years. Developers receive 32 GB RAM minimum.
Warranty must be at least 3 years on-site for all servers and workstations.
""",
    "vendor_notes.md": """
# Vendor notes (category: compute hardware)
ABC Technologies — APPROVED, ISO 9001 valid until 2027-03. Quoted RTX 6000 Ada at INR 6,50,000.
Zenith Systems — APPROVED, ISO 9001 valid until 2026-12. Quoted RTX 6000 Ada at INR 6,20,000.
Nova Compute — BLACKLISTED on 2026-02-10 for repeated late deliveries. Quoted INR 5,90,000.
Orbit Infotech — PENDING onboarding, GST verified, ISO certificate not yet submitted.
""",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on markdown headings first (structure-aware), then by size with overlap."""
    sections, current = [], []
    for line in text.strip().splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))

    # Pack whole lines into chunks of ~size chars; carry the last line(s) over as overlap
    chunks = []
    for sec in sections:
        lines, buf = sec.splitlines(), []
        heading = lines[0] if lines and lines[0].startswith("#") else ""
        for line in lines:
            if buf and len("\n".join(buf + [line])) > size:
                chunks.append("\n".join(buf))
                tail = buf[-1] if len(buf[-1]) <= overlap and buf[-1] != heading else ""
                buf = [x for x in (heading, tail) if x]  # repeat heading for context + overlap line
            buf.append(line)
        if buf:
            chunks.append("\n".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def embed(texts: list[str], kind: str) -> np.ndarray:
    """Embed a batch. kind = 'document' | 'query' (nomic models need task prefixes)."""
    if "nomic" in EMBED_MODEL:
        prefix = "search_document: " if kind == "document" else "search_query: "
        texts = [prefix + t for t in texts]
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    vecs = np.array(resp.embeddings, dtype=np.float32)
    # L2-normalise so a dot product equals cosine similarity
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def chat(messages: list[dict], **kw) -> str:
    """Non-streaming chat. Tries think=False (faster on qwen3/gpt-oss); falls back if unsupported."""
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# The RAG system
# ──────────────────────────────────────────────────────────────────────────────
class SimpleRAG:
    def __init__(self, docs: dict[str, str]):
        self.chunks: list[dict] = []
        for source, text in docs.items():
            for i, c in enumerate(chunk_text(text)):
                self.chunks.append({"id": f"{source}#{i}", "source": source, "text": c})
        print(f"[ingest] {len(docs)} documents → {len(self.chunks)} chunks; embedding with {EMBED_MODEL} …")
        self.matrix = embed([c["text"] for c in self.chunks], kind="document")

    def retrieve(self, question: str, k: int = TOP_K) -> list[tuple[float, dict]]:
        q = embed([question], kind="query")[0]
        scores = self.matrix @ q                       # cosine similarity (vectors are normalised)
        best = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in best]

    def answer(self, question: str) -> str:
        hits = self.retrieve(question)
        print("[retrieve]", ", ".join(f"{h['id']} ({s:.2f})" for s, h in hits))
        context = "\n\n".join(f"[{n}] ({h['source']})\n{h['text']}" for n, (_, h) in enumerate(hits, 1))
        system = (
            "You are a procurement assistant. Answer ONLY from the numbered sources. "
            "Cite sources inline like [1] or [2]. If the sources do not contain the answer, "
            "say 'I don't know based on the provided documents.' Be concise."
        )
        user = f"Sources:\n{context}\n\nQuestion: {question}"
        return chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    options={"temperature": 0})


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    global CHAT_MODEL
    ap = argparse.ArgumentParser(description="Classic dense-vector RAG over a procurement corpus")
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question")
    ap.add_argument("--chat", action="store_true", help="interactive mode")
    args = ap.parse_args()
    CHAT_MODEL = args.model

    try:
        rag = SimpleRAG(DOCUMENTS)
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}\nIs Ollama running, and did you `ollama pull {EMBED_MODEL}`?")

    questions = [args.question] if args.question else [
        "What approval is needed for a purchase of INR 6,50,000?",
        "Can we buy from Nova Compute?",
        "What warranty is required on GPU workstations?",
    ]
    if args.chat:
        print("Interactive RAG. Type 'exit' to quit.")
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            print("Bot:", rag.answer(q))
        return
    for q in questions:
        print(f"\nQ: {q}")
        print(textwrap.fill("A: " + rag.answer(q), 100))


if __name__ == "__main__":
    main()
