"""
KAG.py — Knowledge-Augmented Generation (OpenSPG-KAG style, simplified)
======================================================================

    KNOWLEDGE BUILDER                                 KNOWLEDGE SOLVER
    ─────────────────                                 ────────────────
    schema (typed entities + relations)               question
      + structured facts (vendor master, quotes)        ─► LLM PLANNER writes a *logical form*: a JSON list of operator calls
      + text chunks (policies)                          ─► EXECUTOR runs operators deterministically over the KG
      + MUTUAL INDEX: every fact ⇄ its source text         (lookup · filter · argmin · rule check · graph hop · text retrieve)
                                                        ─► GENERATOR explains the result, citing the facts and source text

How KAG differs from GraphRAG (section 5.3 of the guide)
--------------------------------------------------------
* Schema-first: entity types and properties are defined up front, not guessed by an LLM.
* Reasoning is done by operators (exact maths, rule checks), not by the LLM "reading" summaries.
* Facts and text are linked both ways, so every computed answer can be traced to evidence.
The real framework (github.com/OpenSPG/KAG) adds semantic alignment, a full logical-form language and
hybrid retrieval; this file shows the core idea in ~250 lines.

Run
---
    ollama pull granite3.3:2b
    python KAG.py
    python KAG.py --question "Who is the cheapest eligible vendor for the RTX 6000 Ada and who must approve buying two?"
Small models sometimes write invalid plans; the executor then falls back to a keyword-based plan and
tells you so. Stronger planners: --model gpt-oss:120b-cloud or qwen3:4b (ollama pull qwen3:4b).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
TODAY = date(2026, 11, 20)   # fixed "today" so certificate-expiry checks are reproducible

# ──────────────────────────────────────────────────────────────────────────────
# 1. SCHEMA + KNOWLEDGE (the "builder" output). Typed entities, relations, and links to text.
# ──────────────────────────────────────────────────────────────────────────────
TEXT_CHUNKS = {
    "pol-1": "Policy 3.2 §1: Up to INR 2,00,000 needs department head approval. INR 2,00,001–10,00,000 needs "
             "Finance Controller approval. Above INR 10,00,000 needs CFO approval and three quotes.",
    "pol-2": "Policy 3.2 §4: Only APPROVED vendors may receive purchase orders. Hardware vendors need valid GST "
             "registration and an unexpired ISO 9001 certificate.",
    "pol-3": "Policy 3.2 §5: Blacklisted vendors are barred for 24 months. Vendors owned by a company under "
             "regulatory investigation require Legal clearance.",
    "vm-abc": "Vendor master: ABC Technologies, status APPROVED, GST yes, ISO 9001 valid to 2027-03-31, parent Helix Holdings.",
    "vm-zen": "Vendor master: Zenith Systems, status APPROVED, GST yes, ISO 9001 valid to 2026-10-31.",
    "vm-nova": "Vendor master: Nova Compute, status BLACKLISTED since 2026-02-10, parent Helix Holdings.",
    "vm-orb": "Vendor master: Orbit Infotech, status APPROVED, GST yes, ISO 9001 valid to 2028-01-15.",
    "news-1": "Helix Holdings is under investigation by the Competition Commission for bid rigging.",
}

SCHEMA = {
    "Vendor": ["status", "gst", "iso_valid_until"],
    "Company": ["under_investigation"],
    "Product": ["category"],
    "Quote": ["vendor", "product", "unit_price_inr", "valid_until"],
    "relations": ["OWNED_BY(Vendor→Company)", "QUOTED(Vendor→Quote)", "FOR(Quote→Product)"],
}

ENTITIES = {
    "ABC Technologies": {"type": "Vendor", "status": "APPROVED", "gst": True, "iso_valid_until": "2027-03-31", "src": ["vm-abc"]},
    "Zenith Systems":   {"type": "Vendor", "status": "APPROVED", "gst": True, "iso_valid_until": "2026-10-31", "src": ["vm-zen"]},
    "Nova Compute":     {"type": "Vendor", "status": "BLACKLISTED", "gst": True, "iso_valid_until": None, "src": ["vm-nova"]},
    "Orbit Infotech":   {"type": "Vendor", "status": "APPROVED", "gst": True, "iso_valid_until": "2028-01-15", "src": ["vm-orb"]},
    "Helix Holdings":   {"type": "Company", "under_investigation": True, "src": ["news-1"]},
    "RTX 6000 Ada":     {"type": "Product", "category": "GPU", "src": []},
    "RTX PRO 6000 Blackwell": {"type": "Product", "category": "GPU", "src": []},
}
QUOTES = [
    {"vendor": "ABC Technologies", "product": "RTX 6000 Ada", "unit_price_inr": 650000, "valid_until": "2026-12-31"},
    {"vendor": "Zenith Systems",   "product": "RTX 6000 Ada", "unit_price_inr": 620000, "valid_until": "2026-12-15"},
    {"vendor": "Nova Compute",     "product": "RTX 6000 Ada", "unit_price_inr": 590000, "valid_until": "2026-12-31"},
    {"vendor": "Orbit Infotech",   "product": "RTX 6000 Ada", "unit_price_inr": 668000, "valid_until": "2027-01-31"},
    {"vendor": "Orbit Infotech",   "product": "RTX PRO 6000 Blackwell", "unit_price_inr": 910000, "valid_until": "2027-01-31"},
]
EDGES = [("ABC Technologies", "OWNED_BY", "Helix Holdings"), ("Nova Compute", "OWNED_BY", "Helix Holdings")]


# ──────────────────────────────────────────────────────────────────────────────
# 2. OPERATORS (the "solver"). Deterministic, testable, and each returns evidence ids.
# ──────────────────────────────────────────────────────────────────────────────
def _match_product(name: str) -> str:
    name = (name or "").lower()
    for p, e in ENTITIES.items():
        if e["type"] == "Product" and (name in p.lower() or p.lower() in name):
            return p
    raise ValueError(f"unknown product '{name}'")


def op_get_quotes(product: str) -> dict:
    p = _match_product(product)
    live = [q for q in QUOTES if q["product"] == p and q["valid_until"] >= TODAY.isoformat()]
    return {"value": live, "evidence": [f"quote:{q['vendor']}" for q in live]}


def op_check_eligibility(vendor: str) -> dict:
    """Policy §4 + §5 encoded as rules. The LLM never decides eligibility; this function does."""
    e = ENTITIES[vendor]
    reasons, ev = [], list(e["src"]) + ["pol-2"]
    if e["status"] != "APPROVED":
        reasons.append(f"status is {e['status']} (§4/§5)")
        ev.append("pol-3")
    if not e.get("gst"):
        reasons.append("no GST registration (§4)")
    if not e.get("iso_valid_until") or e["iso_valid_until"] < TODAY.isoformat():
        reasons.append(f"ISO 9001 not valid on {TODAY} (expires {e.get('iso_valid_until')}) (§4)")
    for s, rel, t in EDGES:
        if s == vendor and rel == "OWNED_BY" and ENTITIES[t].get("under_investigation"):
            reasons.append(f"parent {t} is under investigation → needs Legal clearance (§5)")
            ev += ["pol-3"] + ENTITIES[t]["src"]
    return {"value": {"vendor": vendor, "eligible": not reasons, "issues": reasons}, "evidence": sorted(set(ev))}


def op_filter_eligible(quotes: list) -> dict:
    out, ev = [], []
    for q in quotes:
        r = op_check_eligibility(q["vendor"])
        ev += r["evidence"]
        if r["value"]["eligible"]:
            out.append(q)
    return {"value": out, "evidence": sorted(set(ev))}


def op_argmin(items: list, field: str = "unit_price_inr") -> dict:
    if not items:
        return {"value": None, "evidence": []}
    best = min(items, key=lambda x: x[field])
    return {"value": best, "evidence": [f"quote:{best['vendor']}"]}


def op_approval_level(amount_inr: float) -> dict:
    a = float(amount_inr)
    level = ("Department head" if a <= 200000 else
             "Finance Controller" if a <= 1000000 else "CFO + three quotes")
    return {"value": {"amount_inr": a, "approver": level}, "evidence": ["pol-1"]}


def op_multiply(a: float, b: float) -> dict:
    return {"value": float(a) * float(b), "evidence": []}


def op_related(entity: str, relation: str = "OWNED_BY") -> dict:
    hits = [(s, r, t) for s, r, t in EDGES if entity in (s, t) and r == relation]
    return {"value": [f"{s} {r} {t}" for s, r, t in hits], "evidence": [x for _, _, t in hits for x in ENTITIES[t]["src"]]}


OPERATORS = {
    "get_quotes": op_get_quotes, "filter_eligible": op_filter_eligible, "argmin": op_argmin,
    "check_eligibility": op_check_eligibility, "approval_level": op_approval_level,
    "multiply": op_multiply, "related": op_related,
}
OP_DOC = """
get_quotes(product)            -> list of valid quotes for a product
filter_eligible(quotes)        -> keep quotes whose vendor passes policy checks
argmin(items, field)           -> item with the lowest field (use field "unit_price_inr")
check_eligibility(vendor)      -> {eligible, issues} for one vendor
approval_level(amount_inr)     -> who must approve a purchase of that amount
multiply(a, b)                 -> a*b (e.g. unit price x quantity)
related(entity, relation)      -> graph neighbours, relation e.g. "OWNED_BY"
Reference an earlier step's output as "$<out>" or a field of it as "$<out>.<field>".
"""

PLAN_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
    "op": {"type": "string", "enum": list(OPERATORS)}, "args": {"type": "object"}, "out": {"type": "string"}},
    "required": ["op", "args", "out"]}}}, "required": ["steps"]}

EXAMPLE = {"steps": [
    {"op": "get_quotes", "args": {"product": "RTX 6000 Ada"}, "out": "q"},
    {"op": "filter_eligible", "args": {"quotes": "$q"}, "out": "ok"},
    {"op": "argmin", "args": {"items": "$ok", "field": "unit_price_inr"}, "out": "best"},
    {"op": "multiply", "args": {"a": "$best.unit_price_inr", "b": 2}, "out": "total"},
    {"op": "approval_level", "args": {"amount_inr": "$total"}, "out": "approval"}]}


# ──────────────────────────────────────────────────────────────────────────────
# 3. PLANNER (LLM) + EXECUTOR (code) + GENERATOR (LLM)
# ──────────────────────────────────────────────────────────────────────────────
def chat(messages, **kw) -> str:
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


def plan(question: str) -> dict:
    prompt = (f"You plan queries over a procurement knowledge graph.\nSchema: {json.dumps(SCHEMA)}\n"
              f"Products: {[k for k, v in ENTITIES.items() if v['type'] == 'Product']}\n"
              f"Vendors: {[k for k, v in ENTITIES.items() if v['type'] == 'Vendor']}\n"
              f"Operators:{OP_DOC}\nExample plan for 'cheapest eligible vendor for 2 RTX 6000 Ada and approver':\n"
              f"{json.dumps(EXAMPLE)}\n\nWrite the plan (JSON with 'steps') for: {question}")
    raw = chat([{"role": "user", "content": prompt}], format=PLAN_SCHEMA, options={"temperature": 0})
    return json.loads(raw)


def fallback_plan(question: str) -> dict:
    """Keyword-based plan used when the LLM planner fails (keeps the demo robust on 1-2B models)."""
    q = question.lower()
    for v, e in ENTITIES.items():
        if e["type"] == "Vendor" and v.lower().split()[0] in q and "cheap" not in q:
            return {"steps": [{"op": "check_eligibility", "args": {"vendor": v}, "out": "elig"},
                              {"op": "related", "args": {"entity": v, "relation": "OWNED_BY"}, "out": "parents"}]}
    plan_ = json.loads(json.dumps(EXAMPLE))
    plan_["steps"][0]["args"]["product"] = "RTX PRO 6000 Blackwell" if "blackwell" in q else "RTX 6000 Ada"
    qty = next((int(w) for w in q.replace("x", " ").split() if w.isdigit()), 1)
    plan_["steps"][3]["args"]["b"] = qty
    return plan_


def resolve(arg, memory):
    if isinstance(arg, str) and arg.startswith("$"):
        name, _, field = arg[1:].partition(".")
        val = memory[name]
        return val[field] if field else val
    return arg


def execute(p: dict) -> tuple[dict, list]:
    memory, evidence = {}, []
    for i, step in enumerate(p["steps"], 1):
        fn = OPERATORS[step["op"]]
        args = {k: resolve(v, memory) for k, v in step.get("args", {}).items()}
        res = fn(**args)
        memory[step["out"]] = res["value"]
        evidence += res["evidence"]
        print(f"   step {i}: {step['op']}({', '.join(f'{k}={str(v)[:40]}' for k, v in step['args'].items())}) "
              f"→ {str(res['value'])[:90]}")
    return memory, sorted(set(evidence))


def answer(question: str) -> str:
    try:
        p = plan(question)
        print("[planner] LLM logical form:", json.dumps(p)[:300])
        memory, evidence = execute(p)
    except Exception as e:                      # invalid JSON, unknown op, bad args, missing vars …
        print(f"[planner] LLM plan failed ({type(e).__name__}: {str(e)[:80]}) → using fallback plan")
        p = fallback_plan(question)
        memory, evidence = execute(p)
    # Mutual index: map evidence ids back to source text
    texts = [f"[{e}] {TEXT_CHUNKS[e]}" for e in evidence if e in TEXT_CHUNKS]
    quotes = [e for e in evidence if e.startswith("quote:")]
    return chat([
        {"role": "system", "content": "Explain the computed result to a procurement user. The RESULTS are "
                                      "authoritative (computed by code); do not recalculate or contradict them. "
                                      "Cite policy/vendor ids like [pol-2]."},
        {"role": "user", "content": f"Question: {question}\nRESULTS: {json.dumps(memory, default=str)}\n"
                                    f"Quote evidence: {quotes}\nSupporting text:\n" + "\n".join(texts)}],
        options={"temperature": 0})


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    qs = [a.question] if a.question else [
        "Which eligible vendor offers the lowest price for 2 RTX 6000 Ada GPUs, and who must approve it?",
        "Can we place an order with ABC Technologies?",
    ]
    for q in qs:
        print(f"\nQ: {q}")
        try:
            print("A:", answer(q))
        except (ConnectionError, ollama.ResponseError) as e:
            sys.exit(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
