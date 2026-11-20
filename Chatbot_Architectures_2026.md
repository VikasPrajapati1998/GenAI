# GenAI Chatbot & Agent Architectures: Engineer's Field Guide (November 2026)

> **Who this is for:** a working GenAI engineer who already knows basic RAG and wants an accurate, current map of the whole stack: retrieval, graphs, caching, agents, protocols, memory, evaluation, security, and cost.
>
> **How to read it:** sections 1 to 3 give the big picture. Sections 4 to 13 go deep on each layer. Section 14 reworks the procurement (P2P) example end to end. Sections 15 to 17 give the decision guide, comparison table, and roadmap. Section 18 lists corrections to the earlier ChatGPT version.
>
> **Diagrams:** all diagrams are plain-text (box-drawing) and render in any markdown viewer; view with a monospace font. A Word version with colour diagrams is available alongside this file.
>
> **Freshness flags:** items marked ⚡ change month to month (versions, prices, leaderboards). Check the linked source before you rely on them. Items marked ⚠️ are claims I could not fully verify from a primary source.

---

## Table of contents

1. [The 2026 mental model: it is a system, not a technique](#1-the-2026-mental-model)
2. [What changed since 2024–25](#2-what-changed-since-202425)
3. [Taxonomy of chatbot types](#3-taxonomy-of-chatbot-types)
4. [Retrieval (RAG) done properly in 2026](#4-retrieval-rag-done-properly-in-2026)
5. [Graph-based retrieval: GraphRAG, LazyGraphRAG, LightRAG, HippoRAG, KAG](#5-graph-based-retrieval)
6. [Long context, CAG, and caching](#6-long-context-cag-and-caching)
7. [Structured data: text-to-SQL and BI chatbots](#7-structured-data-text-to-sql-and-bi-chatbots)
8. [Agents: loops, workflows, multi-agent](#8-agents-loops-workflows-multi-agent)
9. [Protocols: MCP, A2A, AGENTS.md, Agent Skills](#9-protocols-mcp-a2a-agentsmd-agent-skills)
10. [Context engineering and memory](#10-context-engineering-and-memory)
11. [Reasoning models and model customisation](#11-reasoning-models-and-model-customisation)
12. [Voice and realtime chatbots](#12-voice-and-realtime-chatbots)
13. [Evaluation, observability, security, and cost](#13-evaluation-observability-security-and-cost)
14. [Worked example: the P2P procurement assistant, rebuilt](#14-worked-example-the-p2p-procurement-assistant-rebuilt)
15. [Decision guide](#15-decision-guide)
16. [Master comparison table](#16-master-comparison-table)
17. [Learning roadmap (12 weeks)](#17-learning-roadmap-12-weeks)
18. [Corrections to the earlier version](#18-corrections-to-the-earlier-version)
19. [References](#19-references)

---

## 1. The 2026 mental model

The earlier document framed RAG, CAG, KAG and GraphRAG as alternatives. In production they are **layers of one system**. The question is no longer "which one?" but "which context does the model need for this turn, where does it come from, and who decides?"

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    CLIENT / CHANNEL      chat · voice · API · Slack / Teams                    │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│         GATEWAY      SSO / authN-Z · rate limiting · input guardrails · PII redaction          │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║                          ORCHESTRATOR  (workflow graph or agent loop)                          ║
║                router · planner · context assembler · state store · checkpoints                ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
                                                 │
         ┌───────────────────┬───────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                   ▼                   ▼                   ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ RETRIEVAL      │  │ GRAPH          │  │ SQL / BI       │  │ TOOLS          │  │ MEMORY         │
│ BM25 + dense   │  │ GraphRAG · KAG │  │ text-to-SQL    │  │ MCP servers    │  │ profile        │
│ rerank · ACL   │  │ multi-hop      │  │ semantic layer │  │ A2A agents     │  │ episodic/graph │
└────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
         │                   │                   │                   │                   │
         └───────────────────┴───────────────────┼───────────────────┴───────────────────┘
                                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CONTEXT WINDOW  (assembled per step)                              │
│                                                                                                │
│             ┌───────────────────────────┐     ┌──────────────────────────────────┐             │
│             │    CACHED STATIC PREFIX   │     │         DYNAMIC EVIDENCE         │             │
│             │ system · tools · policies │  +  │ chunks · SQL rows · tool results │             │
│             │    (CAG / prompt cache)   │     │          memory recalls          │             │
│             └───────────────────────────┘     └──────────────────────────────────┘             │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
╔════════════════════════════════════════════════════════════════════════════════════════════════╗
║         LLM  (routed)      reasoning model for hard turns · fast model for easy turns          ║
╚════════════════════════════════════════════════════════════════════════════════════════════════╝
                                                 │
                                                 ▼
┌──────────────────────────────────────────┐     ┌──────────────────────────────────────┐
│             OUTPUT CONTROLS              │     │          RESPONSE / ACTION           │
│       guardrails · citation check        │ ──► │           answer + sources           │
│        human approval for actions        │     │             audit trail              │
└──────────────────────────────────────────┘     └──────────────────────────────────────┘

  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
  OBSERVABILITY:  every span traced (OpenTelemetry) ──► online evals ──► CI regression gate
```

Four ideas carry most of the weight:

- **Context engineering** replaced "prompt engineering" as the core skill: choosing what goes into a finite context window on each step, and what stays out ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
- **Agentic retrieval** replaced fixed retrieve-then-generate for anything non-trivial. The model decides when and how to search, and can search again.
- **Protocols** (MCP for tools/data, A2A for agent-to-agent) replaced bespoke integrations.
- **Evals and traces** are part of the architecture, not an afterthought.

---

## 2. What changed since 2024–25

| Area | 2024 view | Late-2026 reality |
|---|---|---|
| Retrieval | Vector DB + top-k chunks | Hybrid (BM25 + dense) + reranker + contextualised chunks; agentic multi-step search; visual (page-image) retrieval for PDFs |
| Graphs | Microsoft GraphRAG is expensive to index | Cheaper variants: LazyGraphRAG (defer summarisation to query time), LightRAG, HippoRAG 2; graphs used selectively |
| Long context | "RAG is dead" debate | Million-token windows are common, but quality degrades with length ("context rot"). Long context + caching for bounded corpora, retrieval for everything else |
| Caching | Niche optimisation | Standard on every major API; drives prompt layout. CAG on hosted APIs ≈ big static prefix + prompt caching |
| Agents | ReAct demos, AutoGen experiments | Production agent loops with tool search, sub-agents, compaction, long-running tasks; consolidated frameworks |
| Integration | Custom function-calling per vendor | **MCP** (spec 2026-07-28, stateless) and **A2A v1.0**, both under the Linux Foundation |
| Models | One model per app | Routed: reasoning models for planning/hard turns, small fast models for classification, extraction, voice |
| Memory | Chat history in a buffer | Dedicated memory layers (profile, episodic, semantic), often graph-backed, with explicit write/forget policies |
| Security | Jailbreak filters | Indirect prompt injection, tool poisoning, "lethal trifecta" design rules, OWASP Agentic Top 10 |
| Evaluation | RAGAS on a toy set | Trace-based evals, LLM-as-judge with calibration, agent trajectory evals, CI regression gates |

---

## 3. Taxonomy of chatbot types

It helps to name the bot you are building, because each type has a different "centre of gravity".

| Type | Centre of gravity | Typical core architecture | Hard part |
|---|---|---|---|
| **FAQ / policy bot** | Small stable corpus | Long context + prompt caching (CAG-style), or simple hybrid RAG | Keeping answers in sync with policy changes |
| **Enterprise knowledge assistant** | Large, permissioned, changing documents | Hybrid RAG + reranking + ACL filtering + agentic search | Permissions, freshness, citation accuracy |
| **Analytics / BI bot** | Tables and metrics | Text-to-SQL over a semantic layer, code execution for charts | Correct joins, metric definitions, trust |
| **Transactional / task bot** | Actions in systems of record | Tool calling via MCP, workflows, human approval | Safety, idempotency, auth, audit |
| **Research / "deep research" agent** | Open-ended multi-source synthesis | Planner + parallel sub-agents + web/doc search + citation verifier | Cost, runtime, hallucinated citations |
| **Coding / ops agent** | Repos, terminals, infra | Agent loop + sandbox + AGENTS.md / Skills + MCP servers | Sandboxing, review, blast radius |
| **Personal assistant / companion** | The user over time | Memory layer + tools + personalization | What to remember, privacy, drift |
| **Voice agent / contact centre** | Latency and turn-taking | Speech-to-speech model or cascaded STT→LLM→TTS, with tools | Sub-second latency, interruptions, telephony |
| **Multimodal document bot** | Scans, diagrams, tables | Visual retrieval (ColPali-style) + VLM answering | Layout, tables, OCR errors |
| **Multi-agent system** | Several specialised agents | Orchestrator–worker or A2A federation | Coordination cost, error compounding |

---

## 4. Retrieval (RAG) done properly in 2026

### 4.1 Reference pipeline

```text
┌─ [1] INGEST · offline / streaming ───────────────────────────────────────────────────────────┐
│ Sources    ──► Parse     ──► Structure ──► Chunk      ──► Contextualise ──► Index            │
│ PDF · DOCX     Docling /     headings,     structure-     LLM 1–2 line      dense + BM25     │
│ HTML · DB      VLM parse     tables→MD     aware, P/C     prefix/chunk      + ACL, date      │
└─────────────────────────────────────────────────────────────────────┬────────────────────────┘
                                                                      │  writes hybrid index
                                                                      ▼
┌─ [2] QUERY · online ─────────────────────────────────────────────────────────────────────────┐
│ Question  ──► Route        ──► Rewrite      ──► Hybrid search ──► Fuse                       │
│ user turn     docs / SQL /     multi-query,     BM25 ∪ dense      RRF                        │
│               graph / none     HyDE (opt.)      top 50–200                                   │
│                                                                                              │
│ ╰─► ACL filter  ──► Rerank        ──► Assemble         ──► Generate    ──► Verify            │
│     metadata +      cross-encoder     dedupe, parent       with inline     grounding         │
│     permissions     or LLM            expand, cite IDs     citations       check             │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

  Impact order:  parsing quality > hybrid search > reranking > contextual chunks > ACL filtering
```

### 4.2 Techniques that matter most (ranked by typical impact)

1. **Parsing quality.** Most "RAG doesn't work" failures are table and layout failures upstream. Use layout-aware parsers ([Docling](https://github.com/docling-project/docling)) or VLM-based parsing for complex PDFs.
2. **Hybrid search + reciprocal rank fusion.** BM25 catches IDs, SKUs, part numbers and names that embeddings blur. Nearly every production system now uses both.
3. **Reranking.** A cross-encoder or LLM reranker over the top 50–200 candidates is usually the single biggest quality win per line of code.
4. **Contextual retrieval.** Prepend a short, LLM-generated context sentence to each chunk before embedding and BM25 indexing. Anthropic reported a 49% reduction in failed retrievals, and 67% with reranking ([Anthropic, Sept 2024](https://www.anthropic.com/news/contextual-retrieval)). Prompt caching makes this cheap at ingest.
5. **Metadata and permission filtering.** Filter by ACL *before* the LLM sees anything. Never rely on the model to hide documents.
6. **Chunking strategy.** Structure-aware chunks (by heading/section), parent-child expansion, and optionally **late chunking** (embed the whole document with a long-context embedder, then pool per chunk) ([Jina, 2024](https://arxiv.org/abs/2409.04701)).
7. **Query transformation** (multi-query, HyDE, decomposition). Useful, but it adds latency. In agentic systems the agent does this naturally.

### 4.3 Late-interaction and multimodal retrieval

- **ColBERT / ColBERTv2** keep one vector *per token* and score with MaxSim. They are more accurate than single-vector embeddings on hard queries, at higher storage cost ([ColBERTv2](https://arxiv.org/abs/2112.01488)).
- **ColPali / ColQwen** apply late interaction to **page images**: embed screenshots of PDF pages with a VLM, skipping OCR and chunking. This is strong for slides, forms, charts and scanned documents ([ColPali paper](https://arxiv.org/abs/2407.01449)). The answer step then uses a VLM over the retrieved page images.
- **MUVERA** (Google) turns multi-vector sets into fixed-dimensional encodings so late-interaction retrieval can use standard ANN indexes; it is used to scale ColBERT/ColPali-style search ([MUVERA](https://arxiv.org/abs/2405.19504), [example](https://blog.mixpeek.com/colqwen2-muvera-multimodal-late-interaction/)).
- Vector DBs with native multi-vector support (Vespa, Qdrant, Weaviate, LanceDB and others) make this practical. ⚡ Check current support per vendor.

### 4.4 The RAG variants, reclassified

The earlier table listed them flatly. They fall into four groups:

| Group | Techniques | Note |
|---|---|---|
| **Index-time** | Contextual retrieval, parent-child, late chunking, RAPTOR (recursive summary tree), graph extraction | Improve what is stored |
| **Query-time** | Multi-query, HyDE, step-back, decomposition, routing | Improve what is asked |
| **Post-retrieval** | Reranking, compression, dedupe, MMR diversity | Improve what is passed |
| **Control-loop** | Self-RAG, CRAG, Adaptive-RAG, agentic RAG, deep research | Decide *whether / again / where* to retrieve |

Self-RAG in its original form needs a model **fine-tuned** to emit reflection tokens ([paper](https://arxiv.org/abs/2310.11511)). Most teams now get the same behaviour by prompting a capable model inside an agent loop. CRAG adds a retrieval evaluator that triggers corrective web search or query refinement ([paper](https://arxiv.org/abs/2401.15884)).

### 4.5 Agentic RAG and "deep research"

```text
                              ┌──────────────────┐
                              │     QUESTION     │
                              └──────────────────┘
                                        │
                                        ▼
╔═ AGENT LOOP (token / step budget) ════════════════════════════════════════════╗
║ [1] Plan      ──► [2] Search        ──► [3] Read     ──► [4] Sufficient?      ║
║ facts needed?     vector · BM25         take notes       evidence check       ║
║                   web · SQL · graph     cite sources                          ║
║                                                                               ║
║  ▲                                                             │ no           ║
║  └─────────────────────────────────────────────────────────────┘              ║
║    refine query · switch source · spawn parallel sub-agent (isolated context) ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                        │  yes
                                        ▼
              Draft answer ──► Citation check ──► Answer + sources

  Deep research = orchestrator + N parallel sub-agents, each running this loop.
```

- The model receives search tools (vector, keyword, web, SQL, graph) and decides the sequence.
- **Deep research** is this pattern at scale: an orchestrator spawns parallel sub-agents, each with its own context window, and merges compressed findings. Anthropic's write-up of its multi-agent research system is the best public engineering account ([Anthropic, June 2025](https://www.anthropic.com/engineering/multi-agent-research-system)). Multi-agent runs used about 15× the tokens of a chat turn in that account, so reserve the pattern for high-value queries.
- Agentic retrieval often removes the need for elaborate query-transformation code: a strong model with good search tools does better than a hand-tuned pipeline on multi-hop questions, at higher latency.

---

## 5. Graph-based retrieval

### 5.1 When a graph helps

Graphs help when the answer depends on **relationships the text never states in one place**: "Which suppliers of component X also share a parent company with a sanctioned entity?" or "What are the main themes across 5,000 incident reports?" (a *global* question that no single chunk answers).

They rarely help for single-fact lookup. A good hybrid retriever is cheaper there.

### 5.2 Variants

```text
┌─ MICROSOFT GRAPHRAG · INDEXING (expensive, up front) ──────────────────────────────────────┐
│ Documents  ──► LLM extraction ──► Knowledge graph ──► Leiden       ──► Summaries           │
│ text units     entities +         nodes, edges,       hierarchical     LLM report per      │
│                relations          claims              communities      community level     │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────┐  ┌────────────────────────────┐  ┌────────────────────────────┐
│ LOCAL SEARCH               │  │ GLOBAL SEARCH              │  │ DRIFT SEARCH               │
│ matched entities →         │  │ map-reduce over            │  │ global primer →            │
│ neighbourhood + text       │  │ community summaries        │  │ local follow-ups           │
│ "What does X supply?"      │  │ "Main themes overall?"     │  │ broad + specific           │
└────────────────────────────┘  └────────────────────────────┘  └────────────────────────────┘

  Cheaper / alternative designs
┌────────────────────────────┐  ┌────────────────────────────┐  ┌────────────────────────────┐
│ LazyGraphRAG               │  │ LightRAG · HippoRAG 2      │  │ KAG (OpenSPG)              │
│ NLP noun-phrase graph;     │  │ incremental entity graph;  │  │ schema-first KG ⇄ text     │
│ LLM work deferred to       │  │ dual-level keys / PPR      │  │ mutual index; logical-     │
│ query time                 │  │ over triples               │  │ form planner + operators   │
└────────────────────────────┘  └────────────────────────────┘  └────────────────────────────┘
```

| System | Index | Query | Strength | Cost profile |
|---|---|---|---|---|
| **Microsoft GraphRAG** ([paper](https://arxiv.org/abs/2404.16130), [docs](https://microsoft.github.io/graphrag/)) | LLM extracts entities/relations → Leiden communities → LLM summaries per community level | *Local* (entity neighbourhood), *Global* (map-reduce over community summaries), *DRIFT* (global primer then local follow-ups) | Global "sensemaking" questions | Expensive indexing (many LLM calls); re-index on change |
| **LazyGraphRAG** ([MSR blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)) | Cheap NLP noun-phrase graph, no up-front summaries | Defers LLM summarisation to query time, iterative deepening | GraphRAG-like quality at a fraction of index cost | Index cost ≈ vector RAG; query cost tunable |
| **LightRAG** ([paper](https://arxiv.org/abs/2410.05779)) | Entity/relation extraction with key-value profiles; incremental updates | Dual-level (low = entities, high = themes) | Simpler, incremental, popular OSS | Moderate |
| **HippoRAG 2** ([paper](https://arxiv.org/abs/2502.14802)) | Open KG of triples linked to passages | Personalized PageRank from query-matched nodes | Multi-hop association, "memory-like" continual learning | Moderate |
| **KAG (OpenSPG / Ant Group)** ([paper](https://arxiv.org/abs/2409.13731), [repo](https://github.com/OpenSPG/KAG)) | Schema-constrained KG (SPG) mutually indexed with text chunks | Logical-form-guided hybrid solver: planning, retrieval, graph reasoning, maths operators | Professional domains with rules and schemas (finance, medical, government) | High modelling effort |
| **Neo4j / property-graph RAG** | Your own schema, often from structured sources | Cypher generation (text-to-Cypher) + vector index on nodes | Deterministic traversal over known entities | Depends on data engineering |

### 5.3 GraphRAG vs KAG (the comparison the earlier document deferred)

| Dimension | Microsoft GraphRAG | OpenSPG KAG |
|---|---|---|
| Schema | Schema-light; LLM decides entity types | Schema-first (SPG semantic types, constraints) |
| Main goal | Summarise and connect themes across a corpus | Accurate, rule-respecting reasoning in a domain |
| Retrieval unit | Community summaries, entity neighbourhoods, text units | Logical forms executed over graph + chunks |
| Reasoning | LLM reasons over retrieved summaries | Planner decomposes into operators (retrieve, deduce, compute) |
| Best at | "What are the main…?" global questions | Multi-hop questions with numeric or logical constraints |
| Weakness | Index cost, stale summaries | Up-front schema and knowledge-modelling work |
| Recent work | LazyGraphRAG, DRIFT | **KAG-Thinker** (2025): trains the model to do structured interactive thinking with KAG ([paper](https://arxiv.org/abs/2506.17728)) |

**Practical advice:** if your structured facts already live in a database (vendors, prices, approvals), you usually do not need an LLM to *extract* a graph from text. Query the database, and use a graph only for relationships that live in documents.

---

## 6. Long context, CAG, and caching

### 6.1 Long context vs retrieval

Context windows of hundreds of thousands to millions of tokens are common across frontier models ⚡. But:

- **Accuracy degrades with input length**, and position matters ("lost in the middle", [Liu et al.](https://arxiv.org/abs/2307.03172); "context rot", [Chroma, 2025](https://research.trychroma.com/context-rot)).
- **Cost and latency scale with tokens.** Even with caching, a 500k-token prompt per turn is expensive.
- **Permissions** are hard to enforce if the whole corpus sits in one prompt.

**Rule of thumb:** if the relevant corpus is bounded (under roughly 100–200k tokens), stable, and readable by every user, put it in context and cache it. Otherwise retrieve. Many systems do both: a cached static prefix (policies, schemas, instructions) plus retrieved dynamic evidence.

### 6.2 CAG, precisely

The CAG paper ("Don't Do RAG", [arXiv 2412.15605](https://arxiv.org/abs/2412.15605)) preloads all documents into a **locally hosted** model, precomputes the KV cache once, and reuses it for every query (resetting appended tokens between queries). That needs control over the inference runtime.

| Where you run | How "CAG" is actually implemented |
|---|---|
| Self-hosted (vLLM, SGLang, TensorRT-LLM) | Automatic prefix caching (vLLM's PagedAttention-based [APC](https://docs.vllm.ai/), SGLang's RadixAttention [paper](https://arxiv.org/abs/2312.07104)); KV-cache offload to CPU/SSD (e.g. LMCache) to hold large prefixes |
| Hosted APIs | **Prompt caching**: you cannot hold the KV state yourself, but the provider caches the processed prefix. Same effect, less control |

### 6.3 Prompt caching across providers ⚡

Mechanics are broadly similar, details differ. Snapshot from an October 2026 cross-provider summary ([developersdigest](https://www.developersdigest.tech/blog/prompt-caching-across-ai-providers-2026)); verify on each provider's pricing page before you build cost models.

| Provider | Mode | TTL | Pricing shape |
|---|---|---|---|
| Anthropic | Explicit `cache_control` breakpoints (up to 4) or automatic mode | 5 min or 1 h, refreshed on hit | Write premium (1.25× for 5 min, 2× for 1 h); reads ≈ 10% of input or lower on newer models |
| OpenAI | Implicit (automatic) on prompts above ~1,024 tokens; newer models add explicit breakpoints and longer retention | Minutes by default; extended retention options | Cached reads heavily discounted; check whether your model charges a write premium |
| Google Gemini | Implicit caching (automatic) plus explicit `CachedContent` objects | Explicit default 1 h, configurable | Reads discounted; explicit caches billed for storage per hour |
| DeepSeek, Qwen, Mistral, xAI | Mostly automatic prefix caching | Varies, often undocumented | Large read discounts |

```text
┌───────────────┬──────────────────┬────────────────────────┬───────────────┬───────────┐
│ System prompt │ Tool definitions │ Static docs / policies │  Conversation │  New turn │
└───────────────┴──────────────────┴────────────────────────┴───────────────┴───────────┘
 ◄───────────────────── CACHED PREFIX ─────────────────────►◄───────── FRESH ──────────►
 written once (write premium), then read at ~10% or less     recomputed each turn

 Order: most stable ─────────────────────────────────────────────────────► most volatile
 Rule : a change at position k invalidates the cache for every token after k

 ✗ breaks cache : timestamp/user ID at top · reordering tools · editing system prompt
 ✓ keeps warm   : stable prefix · on-demand tools appended later · track cache-hit rate
```

**Design rules that apply everywhere:**
- Put **stable content first** (system prompt → tool definitions → static documents → conversation → new turn). Any change early in the prefix invalidates everything after it.
- Do not inject timestamps, random IDs or per-user data at the top of the prompt.
- Keep tool definitions stable during a session. Adding or reordering tools breaks the cache. Tool-search patterns (load tool schemas on demand, appended later) help here.
- Measure cache-hit rate as a first-class metric.

### 6.4 The caching hierarchy

| Layer | What is cached | Correctness risk |
|---|---|---|
| Response cache | Final answer for identical query | Stale or wrong for personalised/permissioned answers |
| Semantic cache | Answer for *similar* query (embedding match) | High: near-duplicate questions may need different answers |
| Retrieval cache | Search results for a query | Medium (freshness) |
| Prompt / prefix cache | Processed prompt prefix (KV) | None for correctness; output is identical to uncached |
| KV cache (in-request) | Attention states during decoding | None |

---

## 7. Structured data: text-to-SQL and BI chatbots

The earlier document lists "SQL-RAG" in one row. For enterprise chatbots it deserves its own architecture, because **numbers must come from queries, not from text similarity**.

```text
┌─ UNDERSTAND ───────────────────────────────────────────────────────────────────────────┐
│ Question ──► Entity linking ──► Schema retrieval ──► Semantic layer                    │
│              "GPU", "Q3"        tables, cols,        metric defs:                      │
│              → cols/values      join paths           "spend", "approved"               │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
╔═ GENERATE & GUARD ═════════════════════════════════════════════════════════════════════╗
║ Generate SQL ──► Validate           ──► Execute                                        ║
║ or semantic      parse · allow-list     read replica,                                  ║
║ layer query      RLS · LIMIT · cost     read-only creds                                ║
║  ▲                                         │ error                                     ║
║  └─────────────────────────────────────────┘ feed DB error back · bounded retries      ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
                                             │
                                             ▼
┌─ ANSWER ───────────────────────────────────────────────────────────────────────────────┐
│ Result rows ──► Code exec (opt.) ──► Answer         ──► Golden queries                 │
│                 chart / stats        numbers quoted     save verified                  │
│                                      + SQL shown        Q→SQL pairs                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Key points:

- **Use a semantic layer** (dbt Semantic Layer, Cube, LookML, Snowflake/Databricks metric views ⚡) where you can. The LLM picks metrics and dimensions, and the layer generates correct SQL. This removes most join and definition errors.
- **Benchmarks:** Spider 1.0 is effectively saturated. **BIRD** tests dirty real-world values. **Spider 2.0** tests enterprise workflows across large schemas and dialects, and scores are far lower ([Spider 2.0 paper](https://arxiv.org/abs/2411.07763)). ⚡ Leaderboards move monthly; agentic approaches (explore schema, run probe queries, self-correct) lead.
- **Security:** read-only credentials, row-level security enforced by the database, query cost limits, and never interpolate model output into privileged statements.
- **Trust:** show the generated SQL and let analysts save verified question→SQL pairs as few-shot examples ("golden queries").

---

## 8. Agents: loops, workflows, multi-agent

### 8.1 Workflows vs agents

Anthropic's distinction ([Building effective agents, Dec 2024](https://www.anthropic.com/engineering/building-effective-agents)) is now standard vocabulary:

- **Workflows:** code defines the path; LLMs fill steps. Patterns: prompt chaining, routing, parallelisation, orchestrator–workers, evaluator–optimiser.
- **Agents:** the LLM decides the path, in a loop with tools, until it decides it is done.

Start with the simplest thing that works. Many "agents" in production are routers plus two or three fixed workflows.

### 8.2 The agent loop

```text
╔═ AGENT LOOP ══════════════════════════════════════════════════════════════════════════════════╗
║ [1] Assemble    ──► [2] LLM decides  ──► [3] Authorise  ──► [4] Execute       ──► [5] Observe ║
║ system, tools,      text · tool          allow-list,        sandbox, timeout,     truncate /  ║
║ memory, history     call(s) · finish     human approval     idempotency key       summarise   ║
║  ▲                                                                                     │      ║
║  └─────────────────────────────────────────────────────────────────────────────────────┘      ║
║    repeat · compact history near the limit · checkpoint state for resume                      ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
                             │ finish / budget exhausted
                             ▼
                     ┌─────────────────────┐
                     │ RESULT + full trace │
                     └─────────────────────┘
```

Pseudocode:

```
while not done and budget_left:
    context = assemble(system, tools, memory, history_compacted, new_observations)
    action  = llm(context)            # text, tool call(s), or "finish"
    if action.is_tool_call:
        check_permissions(action)     # allow-list, human approval for risky ones
        obs = execute(action)         # sandboxed, timeout, idempotency key
        history.append(action, truncate_or_summarise(obs))
    else:
        done = True
    maybe_compact(history)            # summarise old turns when near the limit
```

Things that changed in practice:

- **Parallel tool calls** are normal; design tools to be independent and idempotent.
- **Tool design is product design.** Fewer, well-named tools with clear descriptions and compact outputs beat many thin API wrappers. Return IDs plus short summaries, not 50 KB JSON.
- **Tool search / progressive disclosure.** With hundreds of tools (common with MCP), load full schemas only when needed.
- **Code execution as a tool.** Letting the agent write and run code in a sandbox (for maths, data transforms, calling many APIs) is often more reliable and cheaper than many individual tool calls.
- **Long-running agents** need checkpointing, resumability and durable execution (LangGraph persistence, Temporal-style engines, MCP Tasks).

### 8.3 Multi-agent patterns

| Pattern | Shape | Use when | Watch out |
|---|---|---|---|
| Orchestrator–worker | One lead spawns sub-agents with isolated contexts | Parallel research, breadth-first tasks | Token cost; lead must merge well |
| Pipeline / handoff | Agent A hands to B with state | Clear stages (triage → specialist) | Lost context at handoff |
| Supervisor / hierarchical | Manager routes to team agents | Many domains | Router errors cascade |
| Debate / critic | Generator + reviewer | Quality-critical outputs | Can converge on shared mistakes |
| Federated (A2A) | Independent agents owned by different teams/vendors | Cross-organisation or cross-platform | Trust, auth, versioning |

Sub-agents' main benefit is **context isolation**: each explores in a fresh window and returns a compressed result.

### 8.4 Frameworks landscape ⚡

| Framework | Model | Notes (late 2026) |
|---|---|---|
| **LangGraph** | Graph/state-machine with checkpoints | De facto for explicit control flow and human-in-the-loop; 1.x line |
| **OpenAI Agents SDK** | Agents, handoffs, guardrails, tracing | Pairs with the Responses API. The **Assistants API shut down on 26 Aug 2026** ([migration guide](https://developers.openai.com/api/docs/assistants/migration)) |
| **Claude Agent SDK** | The agent harness behind Claude Code as a library | Built-in context management, sub-agents, MCP, Skills |
| **Google ADK** | Code-first agents, multi-agent, A2A-native | Deploys to Vertex AI Agent Engine |
| **Microsoft Agent Framework** | Merges Semantic Kernel + AutoGen; .NET and Python | **1.0 GA around April 2026** ([summary](https://jasonmoon-dev.pages.dev/blog/2026-04-17-microsoft-agent-framework-1-0-ga)) ⚠️ secondary source |
| **CrewAI, Pydantic AI, LlamaIndex Workflows, smolagents, Strands (AWS), Mastra (TS)** | Various | Pick by language and how much control you want |
| **Managed runtimes** | AWS Bedrock AgentCore, Vertex AI Agent Engine, Azure AI Foundry Agent Service | Hosting, identity, memory, observability as services |

Advice: learn one explicit-control framework (LangGraph or equivalent) deeply, and understand one vendor agent SDK. The concepts transfer; the APIs churn.

---

## 9. Protocols: MCP, A2A, AGENTS.md, Agent Skills

```text
╔══════════════════════════╗                              ┌──────────────────────────┐
║        YOUR AGENT        ║           A2A v1.0           │      PARTNER AGENT       │
║  host app + MCP client   ║ ◄═════════════════════════►  │   other team / vendor    │
║  LangGraph / ADK / SDKs  ║   Signed Agent Card · Task   │     opaque internals     │
╚══════════════════════════╝    streaming · artifacts     └──────────────────────────┘
                                JSON-RPC · gRPC · REST
              │  MCP 2026-07-28 · stateless HTTP
              ▼  tools · resources · prompts
┌──────────────────────────────────────────────────────────────────────┐
│   MCP GATEWAY  authN-Z · tool allow-list · audit log · rate limits   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌─────────────────┬─────────┴───────┬─────────────────┐
        ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     ERP      │  │     SQL      │  │  CRM / Git   │  │    Search    │
│ quotes, POs  │  │  warehouse   │  │   actions    │  │  docs · web  │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘

 Also: AGENTS.md (repo instructions) · Agent Skills / SKILL.md (procedures loaded on demand)
       MCP, goose, AGENTS.md governed by the Agentic AI Foundation (Linux Foundation)
```

### 9.1 MCP (Model Context Protocol)

- **Purpose:** a standard way for an AI application (host/client) to use **tools, resources and prompts** exposed by servers. Think "USB-C for tools and data".
- **Governance:** donated by Anthropic to the **Agentic AI Foundation (AAIF)** under the Linux Foundation, announced 9 Dec 2025 with MCP, Block's goose and OpenAI's AGENTS.md as founding projects ([WorkOS summary](https://workos.com/blog/the-linux-foundation-launches-the-aaif)).
- **Current spec: 2026-07-28** ([release post](https://blog.modelcontextprotocol.io/posts/2026-07-28/)). The big changes:
  - **Stateless core:** the `initialize` handshake and `Mcp-Session-Id` are removed. Each request carries version, client identity and capabilities in `_meta`. Requests can hit any instance behind a load balancer. Optional `server/discover`.
  - **Routing headers:** `Mcp-Method` and `Mcp-Name` are required on Streamable HTTP so gateways can route and apply policy.
  - **Multi Round-Trip Requests (MRTR):** sampling and elicitation (server asking the user for input) now work by returning `input_required` and having the client retry, instead of a held-open stream.
  - **Extensions framework:** versioned extensions, including **Tasks** (long-running, pollable operations), **MCP Apps** (interactive UI) and **Enterprise Managed Authorization**.
  - **Auth hardening:** RFC 9207 issuer validation; Dynamic Client Registration deprecated in favour of Client ID Metadata Documents.
  - **Deprecations** with a 12-month minimum window: Roots, Sampling, Logging, and the old HTTP+SSE transport.
  - Tier-1 SDKs: TypeScript, Python, Go, C#.
- **Engineering implications:** design MCP servers as stateless HTTP services; put an MCP gateway in front for auth, allow-listing and audit; treat every third-party server as untrusted code (see 13.3).

### 9.2 A2A (Agent2Agent)

- **Purpose:** interoperability **between agents**, which may be opaque to each other. MCP connects an agent to tools; A2A connects agents to agents.
- Originated at Google (April 2025), now a Linux Foundation project. **v1.0, the first stable spec**, adds multi-protocol support, multi-tenancy and **Signed Agent Cards**. Over 150 supporting organisations as of April 2026, with integrations in Azure AI Foundry, Copilot Studio and Bedrock AgentCore ([LF press release](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)).
- Concepts: **Agent Card** (capabilities, endpoint, auth, published at a well-known URL), **Task** lifecycle with streaming updates, **Messages/Parts**, **Artifacts** as outputs. Transports include JSON-RPC, gRPC and REST ([spec](https://a2a-protocol.org/latest/)).
- Payments extension **AP2** exists for agent-initiated payments.

### 9.3 AGENTS.md and Agent Skills

- **AGENTS.md:** a plain markdown file in a repo telling coding agents how to build, test and follow conventions. Now an AAIF project.
- **Agent Skills (SKILL.md):** folders of instructions, scripts and resources that an agent loads **on demand** (only a short description sits in context until needed). Introduced by Anthropic, published as an open standard and adopted across many agent tools ([agentskills.io](https://agentskills.io/)). Skills are a context-engineering tool: they keep procedural knowledge out of the prompt until it is relevant.

---

## 10. Context engineering and memory

### 10.1 Context engineering

The context window is a scarce, degrading resource. Common strategies, often summarised as **write, select, compress, isolate**:

| Strategy | Techniques |
|---|---|
| **Write** (persist outside the window) | Scratchpads, todo files, notes, memory stores, checkpoints |
| **Select** (pull in only what's relevant) | Retrieval, tool search, Skills, memory recall, just-in-time file reads |
| **Compress** | Compaction (summarise old turns), tool-result clearing, trimming large observations |
| **Isolate** | Sub-agents with their own windows, sandboxes holding big data outside the context |

Practical rules: keep the system prompt at the "right altitude" (clear principles, not brittle if-else lists); prefer just-in-time retrieval (give the agent `grep`/search tools and file paths) over preloading; clear stale tool outputs; and keep the prefix stable for caching ([Anthropic, 2025](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

### 10.2 Memory architecture

```text
┌────────────────────────┐                          ╔═══════════════════════════════════════╗
│     WORKING MEMORY     │   extract (post-turn)    ║ LONG-TERM MEMORY STORE                ║
│                        │    ─────────────────►    ║                                       ║
│    context window:     │  facts · prefs · events  ║ profile      user facts, preferences  ║
│ system · recent turns  │                          ║ episodic     dated events, outcomes   ║
│     recalled facts     │    recall (pre-turn)     ║ semantic     temporal knowledge graph ║
└────────────────────────┘    ◄─────────────────    ║ procedural   skills, learned rules    ║
                             similarity · recency   ╚═══════════════════════════════════════╝

┌──────────────────────────────┐  ┌───────────────────────────────┐  ┌────────────────────────────┐
│         CONSOLIDATE          │  │            PROTECT            │  │          EXAMPLES          │
│  dedupe · resolve conflicts  │  │ tenant isolation · provenance │  │   Letta (MemGPT) · Mem0    │
│ validity intervals · expire  │  │  user view/delete · poisoning │  │ Zep / Graphiti · provider  │
└──────────────────────────────┘  └───────────────────────────────┘  └────────────────────────────┘
```

| System | Idea |
|---|---|
| **MemGPT / Letta** ([paper](https://arxiv.org/abs/2310.08560)) | OS-style paging between core memory (in context) and archival memory; the agent edits its own memory via tools |
| **Mem0** ([paper](https://arxiv.org/abs/2504.19413)) | Extract-and-update pipeline that adds, updates or deletes discrete memories; optional graph variant |
| **Zep / Graphiti** ([paper](https://arxiv.org/abs/2501.13956)) | Temporal knowledge graph: facts have validity intervals, so "used to work at X" is handled correctly |
| **Provider memory** | Memory features in vendor APIs and agent SDKs (e.g. a memory tool where the agent reads/writes files) ⚡ |
| **Benchmarks** | LoCoMo ([paper](https://arxiv.org/abs/2402.17753)), LongMemEval ([paper](https://arxiv.org/abs/2410.10813)). ⚠️ Vendor-published comparisons conflict; run your own |

Design questions to answer explicitly: what is worth remembering, who can see it (tenant and user isolation), how users inspect and delete it, how conflicting facts are resolved (temporal validity), and how poisoned memories are prevented (memory is a prompt-injection persistence vector).

---

## 11. Reasoning models and model customisation

### 11.1 Reasoning models

Most frontier families now ship "thinking" modes: the model spends extra tokens reasoning before answering, often with a controllable budget or effort level ⚡. Architecture implications:

- **Route by difficulty.** Use reasoning for planning, multi-hop analysis, code and maths. Use fast models for classification, extraction, rewriting and voice.
- **Interleaved thinking with tools** (reasoning between tool calls) makes agent loops much more reliable than plain ReAct prompting.
- **Latency/cost:** reasoning tokens are billed and add seconds. Set budgets per route.
- **Prompting changes:** do not hand-write chain-of-thought scaffolds for reasoning models; give goals, constraints and success criteria.

### 11.2 Customising models

| Method | Changes | Use for |
|---|---|---|
| Prompting + few-shot | Nothing | Default; always try first |
| SFT / LoRA / QLoRA | Weights (or adapters) | Format, tone, domain jargon, small-model specialisation |
| Preference tuning (DPO etc.) | Weights | Style and preference alignment |
| **Reinforcement fine-tuning (RFT)** | Weights, via graded rewards | Task-specific reasoning where outputs can be verified (classification, extraction, code) |
| Distillation | Small model trained on large-model outputs | Cutting cost/latency for a narrow task |
| RAG / tools / memory | Nothing in weights | Knowledge that changes, must be cited, or is permissioned |

Fine-tuning teaches **behaviour**; retrieval supplies **knowledge**. Fine-tuning facts into weights is unreliable and hard to update or cite.

### 11.3 Self-hosting and serving

- Open-weight models are competitive for many enterprise tasks ⚡.
- Serving stack: **vLLM**, **SGLang**, TensorRT-LLM, llama.cpp/Ollama for local. Key features: continuous batching, PagedAttention, prefix caching, speculative decoding, quantisation (FP8, INT4/AWQ/GPTQ), disaggregated prefill/decode at scale.
- Use an **LLM gateway** (LiteLLM, Portkey, cloud gateways) for routing, fallbacks, budgets and logging.

---

## 12. Voice and realtime chatbots

```text
┌─ CASCADED ───────────────────────────────────────────────────────────────────────────────────┐
│ Mic        ──► VAD + turn det. ──► STT       ──► LLM + tools ──► TTS       ──► Speaker       │
│ WebRTC/SIP     semantic            streaming     streaming       streaming                   │
│                end-of-turn                       tokens                                      │
│                                                                                              │
│ + any LLM, full control, text guardrails between stages, easy transcript evals               │
│ − latency accumulates across stages                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

╔═ SPEECH-TO-SPEECH (native audio) ════════════════════════════════════════════════════════════╗
║ Mic        ──► Realtime speech-to-speech model ──► Speaker                                   ║
║ WebRTC/SIP     audio in → audio out                                                          ║
║                native tool calls                                                             ║
║                                                                                              ║
║ + lowest latency, natural prosody, robust barge-in handling                                  ║
║ − less control, fewer model choices, guardrails need a parallel transcript                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

 Target ≈ <800 ms voice-to-voice · filler speech while tools run · LiveKit Agents / Pipecat
```

- Target **under ~800 ms** voice-to-voice for natural conversation ⚠️ (commonly quoted industry target).
- Key components: streaming at every stage, semantic turn detection (not just silence), barge-in handling, filler/acknowledgement while tools run, WebRTC for browsers, SIP for telephony.
- Frameworks: LiveKit Agents, Pipecat; provider realtime APIs (OpenAI Realtime, Gemini Live) ⚡ ([comparison](https://deepgram.com/learn/speech-to-speech-vs-cascade-voice-agent-architecture)).
- Hybrid is common: native speech-to-speech for conversation, with a text LLM agent behind tools for complex work.

---

## 13. Evaluation, observability, security, and cost

### 13.1 Evaluation

| Level | What to measure | Tools/methods |
|---|---|---|
| Retrieval | Recall@k, MRR/nDCG, context precision | Labelled query→doc sets; synthetic queries from your corpus |
| Generation | Faithfulness/groundedness, answer relevance, citation correctness | RAGAS ([paper](https://arxiv.org/abs/2309.15217)), DeepEval, TruLens, LLM-as-judge |
| Agent | Task success, trajectory quality, tool-call accuracy, steps/cost per task | Scenario suites, simulated users, τ-bench-style environments |
| Safety | Injection resistance, data leakage, policy adherence | Red-team sets, garak, PyRIT, Promptfoo |
| Production | Thumbs, escalations, abandonment, latency p95, cost per conversation | Online evals on sampled traces |

Rules: build a **golden set from real traffic** early; calibrate LLM judges against human labels; evaluate components separately *and* end to end; gate releases on regression evals in CI.

### 13.2 Observability

- Trace every LLM call, retrieval, tool call and sub-agent as spans. **OpenTelemetry GenAI semantic conventions** define attributes for model calls, tokens and agent/tool spans; ⚠️ they are still marked experimental/in development, so pin versions ([OTel GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)).
- Platforms: Langfuse, LangSmith, Arize Phoenix, Braintrust, W&B Weave, Datadog/New Relic LLM observability ⚡.
- Track: tokens (input / cached / output / reasoning), cache-hit rate, tool error rate, retries, latency per stage, cost per task.

### 13.3 Security and guardrails

**Threats specific to LLM apps and agents:**

| Threat | Description | Mitigation |
|---|---|---|
| Direct prompt injection / jailbreak | User tries to override instructions | Input classifiers, strong system prompts, model choice; never treat as solved |
| **Indirect prompt injection** ([Greshake et al.](https://arxiv.org/abs/2302.12173)) | Instructions hidden in retrieved docs, web pages, emails, tool results | Treat all retrieved content as data; privilege separation; output checks |
| **Tool poisoning / rug pulls (MCP)** | Malicious tool descriptions or a server that changes behaviour after approval | Pin and review servers, gateway allow-lists, hash/sign tool definitions, least privilege |
| Data exfiltration | Agent leaks data via links, images, tool calls | Block untrusted outbound URLs, render-safe markdown, egress controls |
| Excessive agency | Agent takes destructive or costly actions | Human approval for irreversible actions, scoped credentials, budgets |
| Memory poisoning | Injected content persisted into long-term memory | Provenance on memories, review/expiry, isolation |
| Insecure output handling | Model output passed to SQL, shell, HTML | Parameterise, sandbox, escape |

- **The "lethal trifecta"** ([Simon Willison, 2025](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)): an agent that has (1) access to private data, (2) exposure to untrusted content, and (3) a way to communicate externally can be made to exfiltrate data. Remove at least one leg per execution context.
- **Design patterns:** dual-LLM / quarantined-LLM, plan-then-execute, and **CaMeL** (Google DeepMind), which separates control flow from untrusted data with capability tracking ([paper](https://arxiv.org/abs/2503.18813)).
- **Frameworks:** OWASP Top 10 for LLM Applications and the **OWASP Top 10 for Agentic Applications (2026)** ([overview](https://speakeasy.com/blog/owasp-agentic-top-10-explained)); NIST AI RMF; EU AI Act obligations phasing in ⚡.
- **Guardrail tooling:** NeMo Guardrails, Guardrails AI, Llama Guard / Prompt Guard, cloud guardrail services, provider safety classifiers.

### 13.4 Cost and deployment

Cost levers, roughly in order of impact:

1. **Model routing** (small model for easy turns).
2. **Prompt caching** (stable prefix; measure hit rate).
3. **Context discipline** (rerank to fewer chunks; compact history; trim tool outputs).
4. **Batch APIs** for offline work (ingest, contextualisation, evals), usually around half price ⚡.
5. **Reasoning budgets** per route.
6. **Self-hosting** for high, steady volume with open-weight models.

Rough per-turn token budget for an enterprise RAG chat: 2–6k cached system/tools + 2–8k retrieved evidence + history + ~500 output. Agentic and multi-agent turns can be 10–100× that.

---

## 14. Worked example: the P2P procurement assistant, rebuilt

**Question:** "Which vendor can supply an NVIDIA RTX 6000-class GPU at the lowest price, and is that vendor eligible under our procurement policy?"

(Note: the RTX 6000 Ada used in the earlier document is the previous generation. NVIDIA's RTX PRO 6000 Blackwell launched in 2025, so a real catalogue may list both. That ambiguity is itself a good test for entity resolution.)

### 14.1 Why the earlier per-architecture answers were all partial

- *Lowest price* is an **aggregation over transactional data**. That is SQL, not vector similarity, graph traversal or a cached prompt.
- *Eligibility* is a **rule evaluation** over vendor status (approved, blacklisted, certifications, spend thresholds). The rules live in policy text, but the facts live in systems of record.
- *Supporting evidence* (policy clause, quotation document) is **document retrieval** with citations.

### 14.2 Target architecture

```text
             ┌──────────────┐     ┌──────────────────────┐     ╔══════════════════╗
             │     USER     │     │       GATEWAY        │     ║      ROUTER      ║
             │ role via SSO │ ──► │ authN-Z · guardrails │ ──► ║ classify request ║
             └──────────────┘     └──────────────────────┘     ╚══════════════════╝
                                                                         │
                 ┌──────────────────────────────┬────────────────────────┴─────┐
                 ▼                              ▼                              ▼
   ┌──────────────────────────┐   ╔══════════════════════════╗   ┌──────────────────────────┐
   │        POLICY Q&A        │   ║    ANALYTICAL / MIXED    ║   │      ACTION REQUEST      │
   │   cached policy prefix   │   ║        agent loop        ║   │  "raise PR for 4 GPUs"   │
   │       + hybrid RAG       │   ║      (this example)      ║   │     → approval flow      │
   └──────────────────────────┘   ╚══════════════════════════╝   └──────────────────────────┘
                                                │
                                                ▼
┌─ TOOLS VIA MCP (scoped credentials) ───────────────────────────────────────────────────────────┐
│ [1] catalog.resolve_item   "RTX 6000" → SKUs (Ada vs Blackwell); ask if ambiguous              │
│ [2] erp.sql                MIN(latest valid quote) per vendor via semantic layer               │
│ [3] rules.evaluate         eligibility: approved status, category, spend limit (deterministic) │
│ [4] policy.search          hybrid RAG + ACL → clause to cite                                   │
│ [5] graph.related          ownership / sanctions links (optional)                              │
│ [6] erp.create_pr          write action → human approval required                              │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                            ANSWER                                            ║
║                vendor · lowest valid price · quote date · eligibility verdict                ║
║                   cited policy clause · SQL used · data-freshness warning                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

### 14.3 Step by step

1. **Resolve the entity.** Map "RTX 6000" to SKUs (Ada vs Blackwell). Ask the user if ambiguous.
2. **Query prices** through the semantic layer: latest valid quote per vendor for the SKU, in INR, within validity date. Deterministic `MIN()`, not the LLM.
3. **Evaluate eligibility** with a rules engine or a well-tested function over vendor master data (approved status, category, spend limit, compliance docs current). The LLM explains the result; it does not decide it.
4. **Retrieve the policy clause** for citation (hybrid RAG over policy docs; policies can also sit in the cached prefix if small).
5. **Optional graph check** for related-party or sanctions relationships extracted from documents.
6. **Compose** the answer with citations, the query used, and data freshness.
7. **Actions** (creating a purchase requisition) go through MCP tools with scoped credentials and explicit human approval.

### 14.4 Where each architecture fits

| Need | Mechanism |
|---|---|
| Static policies and approval matrix (small) | Long context + prompt caching (CAG-style) |
| Large policy/contract corpus | Hybrid RAG + rerank + ACL |
| Prices, POs, vendor status | Text-to-SQL over semantic layer |
| Eligibility decision | Deterministic rules engine, explained by LLM |
| Hidden relationships across documents | GraphRAG / KAG, selectively |
| Multi-step orchestration | Agent loop or LangGraph workflow |
| Integration with ERP | MCP servers behind a gateway |
| Cross-team agents (e.g. finance's agent) | A2A |
| User preferences (default cost centre) | Memory layer |

---

## 15. Decision guide

```text
 START
   │
   ▼
   ◆ Answer is a number / aggregate from systems of record?
   ├──yes──►  Text-to-SQL / semantic layer (+ code exec)
   │ no, or also needed
   ▼
   ◆ Corpus small (<~150k tok), stable, visible to all users?
   ├──yes──►  Long context + prompt caching (CAG-style)
   │
   ▼
   ◆ Large, changing or permissioned documents?
   ├──yes──►  Hybrid RAG + rerank + ACL + contextual chunks
   │
   ▼
   ◆ PDFs heavy with layout, charts, scans?
   ├──yes──►  Add visual retrieval (ColPali/ColQwen) + VLM
   │
   ▼
   ◆ Multi-hop or vague questions?
   ├──yes──►  Agentic retrieval loop / deep research
   │
   ▼
   ◆ "Themes across the whole corpus" questions?
   ├──yes──►  GraphRAG / LazyGraphRAG
   │
   ▼
   ◆ Strict schema + rules + multi-hop logic?
   ├──yes──►  KAG-style schema graph or graph + rules engine
   │
   ▼
   ◆ Must take actions in other systems?
   ├──yes──►  MCP tools · least privilege · human approval
   │
   ▼
   ◆ Must remember users across sessions?
   ├──yes──►  Memory layer with write / forget policy
   │
   ▼
   ◆ Voice channel?
   ├──yes──►  Speech-to-speech (latency) or cascaded (control)
   │
   ▼
   ◆ Several teams' agents must cooperate?
   ├──yes──►  A2A between agents; MCP inside each
   │
   ▼
 END: combine every "yes"; most production systems use 3–5 of these together.
```

---

## 16. Master comparison table

| Approach | Knowledge form | Update cost | Query latency | Multi-hop | Exact numbers | Citations | Infra complexity | Best for |
|---|---|---|---|---|---|---|---|---|
| Long context + prompt cache (CAG-style) | Raw text in prompt | Low (edit prefix; cache rebuilds) | Low after warm cache | Medium (degrades with size) | Weak | Medium | Low | Small stable corpora |
| Naive vector RAG | Chunks + embeddings | Low (re-embed changed docs) | Low | Weak | Weak | Good | Low–Medium | Prototypes |
| Hybrid RAG + rerank | Chunks + BM25 + dense | Low | Low–Medium | Weak–Medium | Weak | Good | Medium | Default for documents |
| Agentic RAG / deep research | Any searchable sources | Low | High (seconds–minutes) | Strong | Medium (with tools) | Good if verified | Medium–High | Complex research questions |
| Visual RAG (ColPali) | Page images, multi-vector | Medium | Medium | Weak | Weak | Page-level | Medium–High | Slides, scans, charts |
| Microsoft GraphRAG | LLM-built graph + community summaries | High (re-index) | Medium–High (global) | Strong | Weak | Medium | High | Corpus-wide themes |
| LazyGraphRAG / LightRAG / HippoRAG 2 | Lightweight graph + passages | Medium | Medium | Strong | Weak | Good | Medium | Multi-hop on a budget |
| KAG (OpenSPG) | Schema KG + text, mutual index | High (modelling) | Medium–High | Strong | Medium (operators) | Good | High | Rule-heavy professional domains |
| Text-to-SQL / semantic layer | Tables | None (live data) | Low–Medium | Medium (joins) | **Strong** | SQL as evidence | Medium | Metrics, transactions |
| Tool / MCP agents | Live systems | None | Medium | Strong | Strong (deterministic tools) | Tool outputs | Medium–High | Actions and live data |
| Memory layer | Extracted facts, graph | Continuous | Low | Medium | n/a | Weak | Medium | Personalisation |
| Fine-tuning / RFT | Weights | High (retrain) | Lowest | n/a | n/a | None | Medium | Behaviour, format, narrow skills |

---

## 17. Learning roadmap (12 weeks)

Each phase ends with something you build and **measure**.

| Weeks | Topic | Build | Measure |
|---|---|---|---|
| 1–2 | Retrieval fundamentals done properly: parsing, chunking, hybrid search, RRF, reranking, contextual retrieval, ACL filtering | Document chatbot over a messy real corpus (PDFs with tables) | Recall@k and faithfulness on a 100-question golden set; ablate each technique |
| 3 | Visual and late-interaction retrieval | Add ColPali/ColQwen page retrieval for slide/scan docs | Compare against text-only on layout-heavy questions |
| 4 | Long context and caching | Same bot as cached long-context variant; vLLM prefix caching locally | Cost, latency, accuracy vs RAG as corpus size grows (find your crossover) |
| 5 | Text-to-SQL and semantic layers | BI bot over a sample warehouse (dbt + semantic layer) | Execution accuracy on your own question set; try a BIRD/Spider 2.0 subset |
| 6 | Graphs | GraphRAG and LightRAG (or LazyGraphRAG) on the same corpus; try OpenSPG KAG on a schema-rich domain | Global vs local question quality, index cost |
| 7–8 | Agents | Agent loop from scratch (no framework), then in LangGraph and one vendor SDK; sub-agents; compaction | Task success rate, steps and tokens per task |
| 9 | Protocols | Write a stateless MCP server (2026-07-28 spec) for your ERP-like API; expose an agent over A2A | Interop with two different clients |
| 10 | Memory and context engineering | Add a memory layer (Mem0, Zep/Graphiti or Letta); Skills for procedures | LongMemEval-style tests on your own scenarios |
| 11 | Security | Red-team your agent: indirect injection via documents, malicious MCP tool description, exfiltration via links | Attack success rate before/after mitigations |
| 12 | Production | Tracing with OpenTelemetry → Langfuse/Phoenix; CI eval gate; model routing; cost dashboard; optional voice front end (LiveKit/Pipecat) | p95 latency, cost per conversation, regression pass rate |

**Capstone:** the P2P procurement assistant from section 14, with SQL for prices, a rules function for eligibility, cited policy retrieval, MCP tools, human approval for actions, and an eval suite.

**Papers worth reading in full:** RAG (Lewis 2020), Lost in the Middle, Self-RAG, CRAG, RAPTOR, GraphRAG, LightRAG, HippoRAG 2, KAG, CAG, ColPali, MemGPT, Zep, ReAct, CaMeL, Spider 2.0. Engineering posts: Anthropic's *Building effective agents*, *Effective context engineering*, *Multi-agent research system*, *Contextual retrieval*.

---

## 18. Corrections to the earlier version

| Earlier claim / gap | Correction |
|---|---|
| RAG, CAG, KAG, GraphRAG presented as competing choices | They are layers; most production systems combine 3–5 mechanisms (section 1, 14) |
| CAG described generically as "reuse KV cache" | The paper's CAG needs a self-hosted runtime holding a precomputed KV cache. On hosted APIs the practical equivalent is long context + **prompt caching**, with different control and pricing (section 6.2) |
| "Long-context LLM: analyze entire books" | True but incomplete: accuracy degrades with length and position ("context rot"); cost scales with tokens; permissions are hard (section 6.1) |
| Self-RAG "model evaluates when and what to retrieve" | Original Self-RAG requires a fine-tuned model emitting reflection tokens; most teams emulate it with agentic prompting |
| GraphRAG described only as Microsoft's community-summary approach | Missing LazyGraphRAG, DRIFT, LightRAG, HippoRAG 2, and the cost trade-off (section 5) |
| KAG vs GraphRAG "worth studying next" | Covered in section 5.3; also KAG-Thinker (2025) |
| SQL-RAG as one table row | Deserves its own architecture with a semantic layer, validation and security (section 7) |
| P2P example answered per architecture | The correct answer needs SQL for price + deterministic rules for eligibility + RAG for citations; no single "RAG type" solves it (section 14) |
| RTX 6000 Ada as the GPU | Previous generation; RTX PRO 6000 Blackwell launched 2025. Good test for entity resolution |
| MCP mentioned only in the roadmap | MCP is now core infrastructure; spec 2026-07-28 is stateless with Tasks, extensions and new auth (section 9) |
| No A2A, AGENTS.md, Skills | Added (section 9) |
| No memory architecture beyond one row | Added (section 10.2) |
| Evaluation = "RAGAS, DeepEval" | Needs retrieval, generation, agent-trajectory, safety and production evals, plus tracing (section 13) |
| No security section | Prompt injection, tool poisoning, lethal trifecta, OWASP Agentic Top 10 (section 13.3) |
| Orchestration = "LangGraph or Google ADK" | Broader landscape incl. OpenAI Agents SDK, Claude Agent SDK, Microsoft Agent Framework 1.0; OpenAI Assistants API is shut down (section 8.4) |
| No voice, multimodal or reasoning-model coverage | Added (sections 4.3, 11, 12) |
| Formatting artefacts (`&#x20;`, "GitHub+1", duplicated sections, image placeholder links) | Removed; all links here point to primary or named sources |

---

## 19. References

**Retrieval and RAG**
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP* — https://arxiv.org/abs/2005.11401
- Anthropic, *Introducing Contextual Retrieval* (2024) — https://www.anthropic.com/news/contextual-retrieval
- Self-RAG — https://arxiv.org/abs/2310.11511
- Corrective RAG — https://arxiv.org/abs/2401.15884
- RAPTOR — https://arxiv.org/abs/2401.18059
- HyDE — https://arxiv.org/abs/2212.10496
- Late chunking (Jina) — https://arxiv.org/abs/2409.04701
- ColBERTv2 — https://arxiv.org/abs/2112.01488
- ColPali — https://arxiv.org/abs/2407.01449
- MUVERA — https://arxiv.org/abs/2405.19504
- Docling — https://github.com/docling-project/docling

**Graphs and knowledge**
- GraphRAG (From Local to Global) — https://arxiv.org/abs/2404.16130 · docs https://microsoft.github.io/graphrag/
- LazyGraphRAG — https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
- DRIFT search — https://microsoft.github.io/graphrag/query/drift_search
- LightRAG — https://arxiv.org/abs/2410.05779
- HippoRAG 2 — https://arxiv.org/abs/2502.14802
- KAG — https://arxiv.org/abs/2409.13731 · repo https://github.com/OpenSPG/KAG
- KAG-Thinker — https://arxiv.org/abs/2506.17728

**Long context and caching**
- CAG, *Don't Do RAG* — https://arxiv.org/abs/2412.15605
- Lost in the Middle — https://arxiv.org/abs/2307.03172
- Chroma, *Context Rot* — https://research.trychroma.com/context-rot
- SGLang / RadixAttention — https://arxiv.org/abs/2312.07104
- vLLM / PagedAttention — https://arxiv.org/abs/2309.06180
- Cross-provider prompt caching snapshot (Oct 2026) ⚡ — https://www.developersdigest.tech/blog/prompt-caching-across-ai-providers-2026

**Structured data**
- Spider 2.0 — https://arxiv.org/abs/2411.07763
- Text-to-SQL benchmarks collection — https://github.com/uiuc-kang-lab/text_to_sql_benchmarks

**Agents, protocols, context**
- Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents
- Anthropic, *Effective context engineering for AI agents* — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic, *How we built our multi-agent research system* — https://www.anthropic.com/engineering/multi-agent-research-system
- ReAct — https://arxiv.org/abs/2210.03629
- MCP 2026-07-28 release — https://blog.modelcontextprotocol.io/posts/2026-07-28/
- Claude support for MCP 2026-07-28 — https://claude.com/blog/bringing-mcp-2026-07-28-to-claude
- A2A one-year / v1.0 press release — https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year
- A2A spec — https://a2a-protocol.org/latest/
- Agentic AI Foundation launch — https://workos.com/blog/the-linux-foundation-launches-the-aaif
- Agent Skills standard — https://agentskills.io/
- OpenAI Assistants → Responses migration — https://developers.openai.com/api/docs/assistants/migration
- Microsoft Agent Framework 1.0 summary ⚠️ — https://jasonmoon-dev.pages.dev/blog/2026-04-17-microsoft-agent-framework-1-0-ga

**Memory**
- MemGPT — https://arxiv.org/abs/2310.08560
- Mem0 — https://arxiv.org/abs/2504.19413
- Zep / Graphiti — https://arxiv.org/abs/2501.13956
- LoCoMo — https://arxiv.org/abs/2402.17753
- LongMemEval — https://arxiv.org/abs/2410.10813

**Evaluation, observability, security**
- RAGAS — https://arxiv.org/abs/2309.15217
- OpenTelemetry GenAI semantic conventions — https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Indirect prompt injection (Greshake et al.) — https://arxiv.org/abs/2302.12173
- The lethal trifecta (Willison) — https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- CaMeL (Google DeepMind) — https://arxiv.org/abs/2503.18813
- OWASP Agentic Top 10 explainer — https://speakeasy.com/blog/owasp-agentic-top-10-explained

**Voice**
- Cascaded vs speech-to-speech (Deepgram) — https://deepgram.com/learn/speech-to-speech-vs-cascade-voice-agent-architecture

---

*Compiled 20 November 2026. Fast-moving items (⚡) were checked against sources dated mid-to-late 2026; re-verify versions and prices before making commitments.*
