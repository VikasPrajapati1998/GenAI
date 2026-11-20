"""
MultiAgent.py — Orchestrator–worker multi-agent system with a critic
===================================================================

                         ┌──────────────────────────┐
     question ─────────► │ ORCHESTRATOR (planner)   │  splits the task into sub-questions, assigns specialists
                         └────────────┬─────────────┘
              ┌───────────────────────┼────────────────────────┐   run in PARALLEL, each with its own
              ▼                       ▼                        ▼   isolated context (no shared history)
     ┌────────────────┐     ┌──────────────────┐     ┌──────────────────┐
     │ POLICY agent   │     │ PRICING agent    │     │ RISK agent       │
     │ policy corpus  │     │ quotes table     │     │ vendor intel     │
     └───────┬────────┘     └────────┬─────────┘     └────────┬─────────┘
             └──────── compressed findings ───────────────────┘
                                      ▼
                         ┌──────────────────────────┐      ┌──────────────┐
                         │ SYNTHESIZER (lead)       │ ───► │ CRITIC        │ ── issues? ──► revise once
                         └──────────────────────────┘      └──────────────┘

Why (section 8.3 of the guide): sub-agents give *context isolation* (each explores in a clean window and
returns a short result) and parallelism. Costs: more tokens and coordination; use when the task really
has independent parts. Specialists here only have documents; give them tools (see AgenticRAG.py /
MCPAgent.py) in a real system.

Run
---
    ollama pull granite3.3:2b
    python MultiAgent.py
    python MultiAgent.py --model gpt-oss:120b-cloud --worker-model qwen3:0.6b   # big lead, small workers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEAD_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")     # orchestrator, synthesizer, critic
WORKER_MODEL = os.getenv("WORKER_MODEL", LEAD_MODEL)      # specialists (can be smaller/cheaper)

SPECIALISTS = {
    "policy": {
        "role": "You are the POLICY specialist. Answer only from the policy text.",
        "knowledge": "§1 Up to INR 2,00,000: department head. INR 2,00,001–10,00,000: Finance Controller. "
                     "Above INR 10,00,000: CFO approval + three quotes.\n§4 Only APPROVED vendors with an "
                     "unexpired ISO 9001 certificate may receive hardware POs.\n§6 GPU purchases need an "
                     "IT Security review before the PO.",
    },
    "pricing": {
        "role": "You are the PRICING specialist. Use only the quotes table; do exact arithmetic.",
        "knowledge": "Quotes for RTX 6000 Ada (INR per unit, valid to): ABC Technologies 6,50,000 (2026-12-31); "
                     "Zenith Systems 6,20,000 (2026-12-15); Nova Compute 5,90,000 (2026-12-31); "
                     "Orbit Infotech 6,68,000 (2027-01-31).",
    },
    "risk": {
        "role": "You are the VENDOR RISK specialist. Use only the vendor intelligence notes. Today is 2026-11-20.",
        "knowledge": "Nova Compute: BLACKLISTED since 2026-02-10. Zenith Systems: ISO 9001 expired 2026-10-31, "
                     "renewal pending. ABC Technologies: parent Helix Holdings is under Competition Commission "
                     "investigation; Legal clearance needed. Orbit Infotech: APPROVED, ISO valid to 2028-01-15, "
                     "no adverse news.",
    },
}

PLAN_SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {"type": "object", "properties": {
    "specialist": {"type": "string", "enum": list(SPECIALISTS)}, "question": {"type": "string"}},
    "required": ["specialist", "question"]}}}, "required": ["tasks"]}
CRITIC_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "issues": {"type": "string"}},
                 "required": ["ok", "issues"]}


def chat(model, messages, **kw) -> str:
    try:
        r = ollama.chat(model=model, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=model, messages=messages, **kw)
    return r.message.content.strip()


def orchestrate(question: str) -> list[dict]:
    raw = chat(LEAD_MODEL, [{"role": "user", "content":
        f"Break this request into 2-4 sub-questions, each answerable by ONE specialist: "
        f"policy (approval rules, eligibility rules), pricing (quotes, totals), risk (vendor status, red flags).\n"
        f"Request: {question}"}], format=PLAN_SCHEMA, options={"temperature": 0})
    try:
        tasks = [t for t in json.loads(raw)["tasks"] if t.get("specialist") in SPECIALISTS and t.get("question")]
    except (json.JSONDecodeError, KeyError):
        tasks = []
    if len(tasks) < 2:      # robust default for small models
        tasks = [{"specialist": "pricing", "question": "List all quotes from cheapest to most expensive and the total for the requested quantity."},
                 {"specialist": "risk", "question": "Which vendors have red flags that block or delay an order?"},
                 {"specialist": "policy", "question": "Which approval level and checks apply to this purchase?"}]
    return tasks


def run_worker(task: dict, original: str) -> dict:
    spec = SPECIALISTS[task["specialist"]]
    out = chat(WORKER_MODEL, [{"role": "system", "content": spec["role"] + "\n\nKNOWLEDGE:\n" + spec["knowledge"]},
                              {"role": "user", "content": f"Overall request: {original}\nYour sub-question: "
                                                          f"{task['question']}\nAnswer in at most 4 bullet points."}],
               options={"temperature": 0})
    return {**task, "finding": out}


def synthesize(question, findings, feedback=None) -> str:
    notes = "\n\n".join(f"[{f['specialist'].upper()}] Q: {f['question']}\n{f['finding']}" for f in findings)
    extra = f"\nA reviewer found these issues in your previous draft; fix them: {feedback}" if feedback else ""
    return chat(LEAD_MODEL, [{"role": "system", "content": "You are the lead agent. Combine specialist findings into "
                              "one recommendation: chosen vendor, total price, eligibility, approver, next steps. "
                              "Do not add facts that no specialist reported." + extra},
                             {"role": "user", "content": f"Request: {question}\n\nFindings:\n{notes}"}],
                options={"temperature": 0})


def critique(question, findings, draft) -> dict:
    raw = chat(LEAD_MODEL, [{"role": "user", "content":
        f"Request: {question}\nSpecialist findings:\n" + "\n".join(f"- {f['finding']}" for f in findings) +
        f"\n\nDraft answer:\n{draft}\n\nDoes the draft contradict the findings, miss a blocking risk, or contain "
        "arithmetic errors? Return JSON {\"ok\": true/false, \"issues\": \"...\"}."}], format=CRITIC_SCHEMA,
        options={"temperature": 0})
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"ok": True, "issues": ""}


def main():
    global LEAD_MODEL, WORKER_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LEAD_MODEL)
    ap.add_argument("--worker-model", default=None)
    ap.add_argument("--question", default="We need 2 RTX 6000 Ada GPUs. Which vendor should we buy from, "
                                          "what will it cost, and who has to approve it?")
    a = ap.parse_args()
    LEAD_MODEL, WORKER_MODEL = a.model, a.worker_model or a.model
    try:
        q = a.question
        print(f"Q: {q}\n")
        tasks = orchestrate(q)
        print("[orchestrator] plan:")
        for t in tasks:
            print(f"   → {t['specialist']}: {t['question']}")
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:   # parallel, isolated workers
            findings = list(pool.map(lambda t: run_worker(t, q), tasks))
        for f in findings:
            print(f"\n[{f['specialist']} agent]\n{f['finding']}")
        draft = synthesize(q, findings)
        review = critique(q, findings, draft)
        print(f"\n[critic] ok={review.get('ok')} issues={review.get('issues', '')[:200]}")
        if not review.get("ok", True):
            draft = synthesize(q, findings, feedback=review.get("issues"))
            print("[lead] revised after critique")
        print(f"\nFINAL ANSWER:\n{draft}")
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
