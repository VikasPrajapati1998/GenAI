# Chatbot architectures, runnable with Ollama

One self-contained Python file per architecture from the guide. Every file has its own small dataset
(the same procurement / P2P scenario throughout, so you can compare approaches on the same questions),
a model-name constant at the top, heavy comments, and a `--model` flag.

| File | Architecture | What it demonstrates |
|---|---|---|
| `RAG.py` | Classic dense RAG | structure-aware chunking, nomic task prefixes, cosine top-k, cited answers |
| `HybridRAG.py` | Advanced / hybrid RAG | BM25 from scratch + dense, RRF fusion, ACL filtering, LLM reranker, optional contextual retrieval |
| `CRAG.py` | Corrective + self-reflective RAG | relevance grader, query rewrite loop, fallback, groundedness check |
| `CAG.py` | Cache-Augmented Generation | whole knowledge base in a fixed prefix, `keep_alive` + KV-cache reuse, prints tokens evaluated per call |
| `GraphRAG.py` | GraphRAG | LLM entity/relation extraction → NetworkX graph → Louvain communities → summaries; local and global (map-reduce) search |
| `KAG.py` | Knowledge-Augmented Generation | schema-first KG, mutual index to text, LLM planner writes a logical form, deterministic operators execute it |
| `AgenticRAG.py` | Agentic RAG | native Ollama tool calling: policy search, read-only SQL, vendor lookup, safe calculator |
| `TextToSQL.py` | Text-to-SQL / BI bot | semantic layer, golden examples, SQL validation, read-only DB, self-correcting retries |
| `MemoryChatbot.py` | Memory-augmented chatbot | long-term memory store with extract / dedupe-update / recall / forget, rolling summary compaction |
| `MCPAgent.py` + `mcp_procurement_server.py` | MCP tool agent | real MCP server (Python SDK v2) over stdio, tool discovery, tool allow-list, human-confirmation gate |
| `MultiAgent.py` | Orchestrator–worker multi-agent | planner, parallel isolated specialists, synthesizer, critic loop |

## Setup (Windows)

```powershell
# 1. Python 3.10+ and the packages
pip install -r requirements.txt

# 2. Models (you already have these two)
ollama pull granite3.3:2b
ollama pull nomic-embed-text:v1.5

# 3. Run any file
python RAG.py
python HybridRAG.py --role employee
python KAG.py --model gpt-oss:120b-cloud
```

Every file also accepts `--question "..."`, and most accept `--chat` for interactive mode.
To change the default model for every file in a session:

```powershell
$env:CHAT_MODEL = "qwen3:4b"          # PowerShell
set CHAT_MODEL=qwen3:4b               # cmd.exe
$env:EMBED_MODEL = "qwen3-embedding:0.6b"
```

## Which model to use

| Need | Your installed models | Optional extra pulls |
|---|---|---|
| Default for all files (follows instructions, JSON, tools) | `granite3.3:2b` | `qwen3:4b` is noticeably better at tool calling and JSON plans |
| Best quality (planning in KAG, extraction in GraphRAG, agents) | `gpt-oss:120b-cloud`, `gemma4:31b-cloud` (need `ollama signin`) | — |
| Fastest / lowest RAM | `qwen3:0.6b`, `gemma3:1b`, `llama3.2:1b` | — |
| SQL generation | `granite3.3:2b` | `qwen2.5-coder:3b` |
| Embeddings | `nomic-embed-text:v1.5` (default), `qwen3-embedding:0.6b` | — |

Notes:
* Tool calling (`AgenticRAG.py`, `MCPAgent.py`) needs a model that supports tools in Ollama: granite3.3,
  llama3.2, qwen3 and gpt-oss do; gemma3 does not.
* 1-2B models are fine for RAG answers but weak at JSON plans and graph extraction. `KAG.py` and
  `MultiAgent.py` fall back to a built-in default plan when the model's plan is invalid, and say so in the log.
* Every file sends `think=False` (faster on thinking models such as qwen3 and gpt-oss) and retries without
  it for models that don't support the parameter.
* If you change `EMBED_MODEL`, the nomic `search_document:` / `search_query:` prefixes are applied only
  for nomic models automatically.

## Files the scripts create

* `GraphRAG.py` caches its extracted graph in `graph_index.json` (delete it or pass `--rebuild` after changing models).
* `MemoryChatbot.py` stores memories in `memory_store.json` (use `/memories`, `/forget <id>`, `/reset` in `--chat`).
* `TextToSQL.py` writes a throwaway SQLite file in your temp folder.

## Testing status

These files were run end to end in a Linux sandbox that could not download Ollama models. The tests used a mock
Ollama server that implements the same HTTP API (`/api/chat`, `/api/embed`, tool calls, JSON-schema `format`)
with canned answers. That checks the code paths: parsing, tool-call loops, SQL validation and retries, graph
building, the MCP client/server round trip with the real `mcp` 2.x SDK, memory persistence, and the `think`
fallback. **Answer quality with real models was not tested here**, so expect to tune prompts for very small models.
