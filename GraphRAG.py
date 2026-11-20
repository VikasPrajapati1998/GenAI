"""
GraphRAG.py — Graph Retrieval-Augmented Generation (Microsoft GraphRAG pattern, simplified)
==========================================================================================

    INDEX : documents ─► LLM extracts (entities, relations) as JSON ─► NetworkX knowledge graph
                     ─► Louvain community detection ─► LLM writes a summary report per community
    QUERY :
      LOCAL  — question ─► embed ─► nearest entities ─► 1-hop neighbourhood + source text ─► answer
      GLOBAL — question ─► MAP: each community summary gives a partial answer + score
                        ─► REDUCE: combine the best partial answers into one response

When to use: questions that depend on relationships spread across documents ("who is connected to whom")
and corpus-wide "themes" questions that no single chunk answers. For single-fact lookup, plain RAG is
cheaper. Real GraphRAG uses Leiden + hierarchical communities; Louvain (built into NetworkX) is used here.

Run
---
    ollama pull granite3.3:2b && ollama pull nomic-embed-text:v1.5
    pip install networkx
    python GraphRAG.py                      # builds graph, runs a local and a global question
    python GraphRAG.py --mode global --question "What are the main supplier risks?"
    python GraphRAG.py --rebuild            # ignore the cached graph_index.json
Tip: graph extraction quality depends heavily on the model; try --model gpt-oss:120b-cloud.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:v1.5")
CACHE = Path(__file__).with_name("graph_index.json")   # extraction is expensive: cache it

DOCS = {
    "d1": "ABC Technologies supplies NVIDIA RTX 6000 Ada GPUs to our Bengaluru data centre. "
          "ABC Technologies is a subsidiary of Helix Holdings.",
    "d2": "Zenith Systems is an approved vendor for compute hardware and partners with Dell for workstations. "
          "Zenith Systems quoted the lowest price for the RTX 6000 Ada.",
    "d3": "Nova Compute was blacklisted after late deliveries to the Pune office. "
          "Nova Compute is also owned by Helix Holdings.",
    "d4": "Helix Holdings is under investigation by the Competition Commission for bid rigging in hardware tenders.",
    "d5": "The Bengaluru data centre runs the AI training cluster managed by the ML Platform team.",
    "d6": "The ML Platform team requested eight RTX PRO 6000 Blackwell GPUs for the 2027 expansion.",
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["ORGANIZATION", "PRODUCT", "LOCATION", "TEAM", "EVENT", "OTHER"]},
            "description": {"type": "string"}}, "required": ["name", "type"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"},
            "relation": {"type": "string"}}, "required": ["source", "target", "relation"]}},
    },
    "required": ["entities", "relations"],
}


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


def norm(name: str) -> str:
    """Entity resolution, minimal version: case/whitespace normalisation."""
    return " ".join(name.strip().split()).title()


# ──────────────────────────────────────────────────────────────────────────────
# INDEXING
# ──────────────────────────────────────────────────────────────────────────────
def extract(doc_id: str, text: str) -> dict:
    prompt = (
        "Extract a knowledge graph from the text. List every organisation, product, location, team and "
        "event as entities, and every relationship between them as relations (use short verb phrases like "
        "'supplies', 'subsidiary of', 'located in'). Use entity names exactly as written.\n\n"
        f"Text: {text}"
    )
    raw = chat([{"role": "user", "content": prompt}], format=EXTRACT_SCHEMA, options={"temperature": 0})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"entities": [], "relations": []}
    print(f"[extract] {doc_id}: {len(data.get('entities', []))} entities, {len(data.get('relations', []))} relations")
    return data


def build_index(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        print(f"[index] loading cached graph from {CACHE.name} (use --rebuild to re-extract)")
        return json.loads(CACHE.read_text())

    G = nx.Graph()
    for doc_id, text in DOCS.items():
        data = extract(doc_id, text)
        for e in data.get("entities", []):
            n = norm(e.get("name", ""))
            if not n:
                continue
            node = G.nodes.get(n)
            if node is None:
                G.add_node(n, type=e.get("type", "OTHER"), desc=e.get("description", ""), docs=[doc_id])
            elif doc_id not in node["docs"]:
                node["docs"].append(doc_id)
        for r in data.get("relations", []):
            s, t = norm(r.get("source", "")), norm(r.get("target", ""))
            if s and t and s != t:
                for n in (s, t):
                    if n not in G:
                        G.add_node(n, type="OTHER", desc="", docs=[doc_id])
                G.add_edge(s, t, relation=r.get("relation", "related to"), doc=doc_id)

    print(f"[graph] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if G.number_of_nodes() == 0:
        sys.exit("No entities extracted. Try a stronger model, e.g. --model gpt-oss:120b-cloud")

    # Community detection (GraphRAG uses hierarchical Leiden; Louvain is the NetworkX equivalent)
    comms = nx.community.louvain_communities(G, seed=42)
    communities = []
    for i, members in enumerate(sorted(comms, key=len, reverse=True)):
        sub = G.subgraph(members)
        facts = [f"{u} —{d['relation']}→ {v}" for u, v, d in sub.edges(data=True)]
        summary = chat([{"role": "user", "content":
            "Write a 2-3 sentence report about this group of related entities: who they are, how they are "
            "connected, and any risks or notable facts.\nEntities: " + ", ".join(members) +
            "\nRelationships:\n" + "\n".join(facts or ["(none)"])}], options={"temperature": 0})
        communities.append({"id": i, "members": sorted(members), "facts": facts, "summary": summary})
        print(f"[community {i}] {', '.join(sorted(members))}")

    index = {"graph": nx.node_link_data(G, edges="edges"), "communities": communities}
    CACHE.write_text(json.dumps(index, indent=1))
    return index


# ──────────────────────────────────────────────────────────────────────────────
# QUERYING
# ──────────────────────────────────────────────────────────────────────────────
class GraphRAG:
    def __init__(self, index: dict):
        self.G = nx.node_link_graph(index["graph"], edges="edges")
        self.communities = index["communities"]
        self.names = list(self.G.nodes)
        self.vecs = embed([f"{n}: {self.G.nodes[n].get('desc', '')}" for n in self.names], "document")

    def local_search(self, q: str, k: int = 3) -> str:
        s = self.vecs @ embed([q], "query")[0]
        seeds = [self.names[i] for i in np.argsort(-s)[:k]]
        print(f"[local] seed entities: {seeds}")
        facts, docs = set(), set()
        for n in seeds:
            docs.update(self.G.nodes[n].get("docs", []))
            for nb in self.G.neighbors(n):                     # 1-hop expansion
                d = self.G.edges[n, nb]
                facts.add(f"{n} —{d['relation']}→ {nb}")
                docs.add(d.get("doc"))
        context = ("Graph facts:\n" + "\n".join(sorted(facts)) +
                   "\n\nSource text:\n" + "\n".join(f"[{d}] {DOCS[d]}" for d in sorted(x for x in docs if x in DOCS)))
        return chat([{"role": "system", "content": "Answer using the graph facts and source text. "
                                                   "Explain the chain of relationships. Cite [d#]."},
                     {"role": "user", "content": f"{context}\n\nQuestion: {q}"}], options={"temperature": 0})

    def global_search(self, q: str) -> str:
        schema = {"type": "object", "properties": {"answer": {"type": "string"},
                                                   "score": {"type": "integer"}}, "required": ["answer", "score"]}
        partials = []
        for c in self.communities:                             # MAP
            raw = chat([{"role": "user", "content":
                f"Community report:\n{c['summary']}\n\nQuestion: {q}\n"
                "Give a partial answer using only this report, and a helpfulness score 0-10 (0 if irrelevant)."}],
                format=schema, options={"temperature": 0})
            try:
                p = json.loads(raw)
                partials.append((int(p.get("score", 0)), p.get("answer", "")))
            except (json.JSONDecodeError, ValueError):
                pass
        partials = [p for p in sorted(partials, reverse=True) if p[0] > 0][:5]
        print(f"[global] {len(partials)} useful community answers (scores {[s for s, _ in partials]})")
        return chat([{"role": "user", "content":                 # REDUCE
            "Combine these partial answers into one coherent answer. Remove duplicates.\n\n" +
            "\n".join(f"- (score {s}) {a}" for s, a in partials) + f"\n\nQuestion: {q}"}],
            options={"temperature": 0})


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--mode", choices=["local", "global", "both"], default="both")
    ap.add_argument("--question")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    try:
        bot = GraphRAG(build_index(a.rebuild))
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")

    local_q = a.question or "Is there any connection between our GPU supplier ABC Technologies and the blacklisted vendor?"
    global_q = a.question or "What are the main risks across our supplier relationships?"
    if a.mode in ("local", "both"):
        print(f"\nLOCAL Q: {local_q}\nA: {bot.local_search(local_q)}")
    if a.mode in ("global", "both"):
        print(f"\nGLOBAL Q: {global_q}\nA: {bot.global_search(global_q)}")


if __name__ == "__main__":
    main()
