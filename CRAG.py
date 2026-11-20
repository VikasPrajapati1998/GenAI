"""
CRAG.py — Corrective + Self-Reflective RAG (CRAG / Self-RAG style control loop)
==============================================================================

    question ─► retrieve ─► GRADE each chunk (relevant?) ──all irrelevant──► REWRITE query ─► retrieve again
                                   │ some relevant                                 (max N rewrites, then fallback)
                                   ▼
                              generate answer ─► GROUNDEDNESS check ──not grounded──► regenerate (strict)
                                                        │ grounded
                                                        ▼
                                                     answer

Why: plain RAG trusts whatever the retriever returns. CRAG (arXiv 2401.15884) adds a retrieval evaluator
and corrective actions; Self-RAG (arXiv 2310.11511) adds reflection on the generated answer. The original
Self-RAG fine-tunes a model to emit reflection tokens; here we emulate both with prompted graders that
return structured JSON, which is how most teams build this today.

Run
---
    ollama pull granite3.3:2b && ollama pull nomic-embed-text:v1.5
    python CRAG.py
    python CRAG.py --question "What is the capital of France?"   # watch the fallback path
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
TOP_K = 4
MAX_REWRITES = 2

DOCS = [
    "Purchases above INR 10,00,000 need CFO approval and three competitive quotes.",
    "Purchases between INR 2,00,001 and INR 10,00,000 need Finance Controller approval.",
    "Only APPROVED vendors with GST and ISO 9001 certificates may receive hardware purchase orders.",
    "Nova Compute was blacklisted on 2026-02-10; blacklisted vendors are barred for 24 months.",
    "Every GPU purchase needs an IT Security review before the purchase order is raised.",
    "Zenith Systems quoted INR 6,20,000 for an RTX 6000 Ada; ABC Technologies quoted INR 6,50,000.",
    "Expense claims must be filed within 30 days with original receipts.",
    "The office cafeteria is open from 8 AM to 8 PM on weekdays.",
]

YES_NO = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]},
                                           "reason": {"type": "string"}}, "required": ["verdict"]}


def chat(messages, **kw) -> str:
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


def judge(prompt: str) -> bool:
    """Ask a yes/no question with a JSON schema; default to 'no' if the small model returns junk."""
    raw = chat([{"role": "user", "content": prompt + '\nReturn JSON: {"verdict": "yes" or "no", "reason": "..."}'}],
               format=YES_NO, options={"temperature": 0})
    try:
        return json.loads(raw).get("verdict", "no").lower() == "yes"
    except json.JSONDecodeError:
        return "yes" in raw.lower()[:10]


def embed(texts, kind):
    if "nomic" in EMBED_MODEL:
        texts = [("search_document: " if kind == "document" else "search_query: ") + t for t in texts]
    v = np.array(ollama.embed(model=EMBED_MODEL, input=texts).embeddings, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class CorrectiveRAG:
    def __init__(self, docs):
        self.docs = docs
        self.vecs = embed(docs, "document")

    def retrieve(self, q):
        s = self.vecs @ embed([q], "query")[0]
        return [self.docs[i] for i in np.argsort(-s)[:TOP_K]]

    def grade(self, q, chunks):
        """Retrieval evaluator: keep only chunks the grader judges relevant."""
        keep = []
        for c in chunks:
            ok = judge(f"Question: {q}\nDocument: {c}\nDoes the document contain information that helps answer the question?")
            print(f"   [grade] {'KEEP' if ok else 'drop'}: {c[:70]}")
            if ok:
                keep.append(c)
        return keep

    def rewrite(self, q):
        new = chat([{"role": "user", "content":
            f"Rewrite this search query to be more specific for a company procurement/finance policy search. "
            f"Return only the rewritten query.\nQuery: {q}"}], options={"temperature": 0.3})
        return new.splitlines()[0].strip('" ') or q

    def generate(self, q, ctx, strict=False):
        rule = ("Use ONLY facts stated in the context; if something is missing, say it is not covered."
                if strict else "Answer from the context and be concise.")
        return chat([{"role": "system", "content": rule},
                     {"role": "user", "content": "Context:\n- " + "\n- ".join(ctx) + f"\n\nQuestion: {q}"}],
                    options={"temperature": 0})

    def answer(self, question):
        query = question
        for attempt in range(MAX_REWRITES + 1):
            print(f"[retrieve] attempt {attempt + 1}: '{query}'")
            relevant = self.grade(question, self.retrieve(query))
            if relevant:
                break
            if attempt < MAX_REWRITES:
                query = self.rewrite(query)
                print(f"[correct] nothing relevant → rewritten query: '{query}'")
        else:
            # Corrective fallback. CRAG would call web search here; we refuse instead of hallucinating.
            return "I couldn't find this in the company knowledge base. (CRAG fallback: web search / human handoff.)"

        draft = self.generate(question, relevant)
        grounded = judge(f"Context:\n{relevant}\n\nAnswer: {draft}\n\nIs every claim in the answer supported by the context?")
        print(f"[reflect] grounded = {grounded}")
        if not grounded:
            draft = self.generate(question, relevant, strict=True)
            print("[reflect] regenerated with strict grounding instructions")
        return draft


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    try:
        bot = CorrectiveRAG(DOCS)
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")
    if a.chat:
        while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
            print("Bot:", bot.answer(q))
        return
    for q in [a.question] if a.question else ["Which vendor is cheapest for the RTX 6000 Ada and can we use it?",
                                              "Who won the cricket world cup?"]:
        print(f"\nQ: {q}")
        print("A:", bot.answer(q))


if __name__ == "__main__":
    main()
