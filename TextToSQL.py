"""
TextToSQL.py — Text-to-SQL / BI chatbot with a semantic layer, validation and self-correction
============================================================================================

    question ─► schema + semantic layer + golden examples ─► LLM writes SQL
             ─► VALIDATE (single SELECT · table allow-list · LIMIT · EXPLAIN dry run)
             ─► EXECUTE on a read-only connection ──error──► feed error back, retry (max 3)
             ─► LLM explains the rows (numbers quoted from the result, SQL shown)

Principle (section 7 of the guide): numbers must come from queries, never from text similarity.
The "semantic layer" here is a dict of business definitions (what "spend" or "eligible vendor" means)
injected into the prompt; production systems use dbt Semantic Layer / Cube / LookML instead.

Run
---
    ollama pull granite3.3:2b
    python TextToSQL.py
    python TextToSQL.py --chat
    python TextToSQL.py --model qwen2.5-coder:3b     # code models do well (ollama pull qwen2.5-coder:3b)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile

import ollama

# Windows consoles: make sure arrows/box characters in the logs print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHAT_MODEL = os.getenv("CHAT_MODEL", "granite3.3:2b")
MAX_RETRIES = 3
ALLOWED_TABLES = {"vendors", "products", "quotes", "purchase_orders"}

SCHEMA_SQL = """
CREATE TABLE vendors(id INTEGER PRIMARY KEY, name TEXT, status TEXT CHECK(status IN ('APPROVED','PENDING','BLACKLISTED')),
                     city TEXT, iso_valid_until DATE);
CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT);
CREATE TABLE quotes(id INTEGER PRIMARY KEY, vendor_id INTEGER REFERENCES vendors(id), product_id INTEGER REFERENCES products(id),
                    unit_price_inr INTEGER, quote_date DATE, valid_until DATE);
CREATE TABLE purchase_orders(id TEXT PRIMARY KEY, vendor_id INTEGER REFERENCES vendors(id), product_id INTEGER REFERENCES products(id),
                    quantity INTEGER, total_inr INTEGER, status TEXT, created_on DATE, department TEXT);
"""
DATA_SQL = """
INSERT INTO vendors VALUES (1,'ABC Technologies','APPROVED','Bengaluru','2027-03-31'),(2,'Zenith Systems','APPROVED','Pune','2026-10-31'),
 (3,'Nova Compute','BLACKLISTED','Delhi',NULL),(4,'Orbit Infotech','APPROVED','Hyderabad','2028-01-15'),(5,'Kite Peripherals','PENDING','Chennai',NULL);
INSERT INTO products VALUES (1,'RTX 6000 Ada','GPU'),(2,'RTX PRO 6000 Blackwell','GPU'),(3,'Dell Precision 7960','Workstation'),
 (4,'ThinkPad P16','Laptop');
INSERT INTO quotes VALUES (1,1,1,650000,'2026-10-02','2026-12-31'),(2,2,1,620000,'2026-10-05','2026-12-15'),(3,3,1,590000,'2026-09-28','2026-12-31'),
 (4,4,1,668000,'2026-11-01','2027-01-31'),(5,4,2,910000,'2026-11-03','2027-01-31'),(6,1,3,410000,'2026-08-11','2026-11-30'),
 (7,2,4,245000,'2026-10-20','2027-01-20');
INSERT INTO purchase_orders VALUES ('PO-2026-0398',1,3,1,410000,'CLOSED','2026-08-20','ML Platform'),
 ('PO-2026-0412',2,1,2,1240000,'OPEN','2026-10-10','ML Platform'),('PO-2026-0431',2,4,5,1225000,'OPEN','2026-10-25','Engineering'),
 ('PO-2026-0450',4,2,1,910000,'APPROVED','2026-11-05','Research'),('PO-2026-0377',1,1,1,650000,'CLOSED','2026-07-14','Research');
"""

# The semantic layer: business meaning the raw schema does not tell the model.
SEMANTIC_LAYER = {
    "today": "2026-11-20 (use this literal date, not date('now'))",
    "spend": "SUM(purchase_orders.total_inr) where status IN ('OPEN','APPROVED','CLOSED')",
    "eligible vendor": "vendors.status = 'APPROVED' AND vendors.iso_valid_until >= '2026-11-20'",
    "valid quote": "quotes.valid_until >= '2026-11-20'",
    "open PO": "purchase_orders.status IN ('OPEN','APPROVED')",
}

# Golden (verified) question→SQL pairs used as few-shot examples
GOLDEN = [
    ("Total spend by department",
     "SELECT department, SUM(total_inr) AS spend_inr FROM purchase_orders GROUP BY department ORDER BY spend_inr DESC;"),
    ("Valid quotes for laptops",
     "SELECT v.name, p.name, q.unit_price_inr FROM quotes q JOIN vendors v ON v.id=q.vendor_id "
     "JOIN products p ON p.id=q.product_id WHERE p.category='Laptop' AND q.valid_until >= '2026-11-20';"),
]


def build_db() -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """Create the DB file, then open a second READ-ONLY connection for model-generated SQL."""
    path = os.path.join(tempfile.gettempdir(), "texttosql_demo.db")
    if os.path.exists(path):
        os.remove(path)
    rw = sqlite3.connect(path)
    rw.executescript(SCHEMA_SQL + DATA_SQL)
    rw.commit()
    ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True)    # defence in depth: DB rejects writes
    return rw, ro


def chat(messages, **kw) -> str:
    try:
        r = ollama.chat(model=CHAT_MODEL, messages=messages, think=False, **kw)
    except ollama.ResponseError as e:
        if "think" not in str(e).lower():
            raise
        r = ollama.chat(model=CHAT_MODEL, messages=messages, **kw)
    return r.message.content.strip()


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    m2 = re.search(r"\b(select|with)\b.*", sql, re.S | re.I)     # drop any chatter before SELECT
    return (m2.group(0) if m2 else sql).strip().rstrip(";")


def validate(sql: str, conn: sqlite3.Connection) -> str:
    """Return a cleaned SQL string or raise ValueError with a message the model can act on."""
    low = sql.lower()
    if not re.match(r"^\s*(select|with)\b", low):
        raise ValueError("Only SELECT (or WITH … SELECT) queries are allowed.")
    if ";" in sql:
        raise ValueError("Only one statement is allowed.")
    if re.search(r"\b(insert|update|delete|drop|alter|attach|pragma|create)\b", low):
        raise ValueError("Write/DDL keywords are not allowed.")
    tables = set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", low))
    bad = tables - ALLOWED_TABLES
    if bad:
        raise ValueError(f"Unknown tables {sorted(bad)}. Allowed: {sorted(ALLOWED_TABLES)}.")
    if not re.search(r"\blimit\s+\d+", low):
        sql += " LIMIT 50"                                     # protect the DB and the context window
    conn.execute("EXPLAIN QUERY PLAN " + sql)                 # dry run: catches syntax / column errors early
    return sql


def ask(question: str, conn: sqlite3.Connection) -> str:
    system = (
        "You translate questions into a single SQLite SELECT query.\n"
        f"Schema:\n{SCHEMA_SQL}\nBusiness definitions (semantic layer):\n{json.dumps(SEMANTIC_LAYER, indent=1)}\n"
        "Verified examples:\n" + "\n".join(f"Q: {q}\nSQL: {s}" for q, s in GOLDEN) +
        "\nReturn ONLY the SQL inside a ```sql code block."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    for attempt in range(1, MAX_RETRIES + 1):
        raw = chat(messages, options={"temperature": 0})
        sql = extract_sql(raw)
        try:
            sql = validate(sql, conn)
            cur = conn.execute(sql)
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            print(f"[sql attempt {attempt}] {sql}\n[rows] {rows[:5]}{' …' if len(rows) > 5 else ''}")
            break
        except (ValueError, sqlite3.Error) as e:
            print(f"[sql attempt {attempt}] {sql}\n[error] {e}")
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That query failed: {e}\nFix it and return only the SQL."}]
    else:
        return "Sorry, I couldn't produce a valid query for that question."
    return chat([{"role": "system", "content": "Explain the query result briefly. Quote numbers exactly as they "
                                               "appear in the rows (INR). Do not invent rows."},
                 {"role": "user", "content": f"Question: {question}\nSQL: {sql}\nRows (JSON): {json.dumps(rows)}"}],
                options={"temperature": 0}) + f"\n\n(SQL used: {sql})"


def main():
    global CHAT_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--question")
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    CHAT_MODEL = a.model
    _rw, ro = build_db()
    qs = [a.question] if a.question else [
        "Which eligible vendor has the cheapest valid quote for the RTX 6000 Ada?",
        "How much has each department spent on open POs?",
        "List vendors whose ISO certificate has expired or is missing.",
    ]
    try:
        if a.chat:
            while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
                print("Bot:", ask(q, ro))
            return
        for q in qs:
            print(f"\nQ: {q}")
            print("A:", ask(q, ro))
    except (ConnectionError, ollama.ResponseError) as e:
        sys.exit(f"Ollama error: {e}")


if __name__ == "__main__":
    main()
