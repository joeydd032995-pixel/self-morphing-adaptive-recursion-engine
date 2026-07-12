# Self-Morphing Adaptive Recursion Engine

**A hybrid neurosymbolic reasoning engine with polymorphic morphing nodes, PoG planning, production RAG, a real knowledge-graph layer (SQLite + Neo4j), a learned GNN, autonomous self-teaching, and production deployment (FastAPI + Docker + Kubernetes + Prometheus/Grafana).**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20hypothesis-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

> Evolved from a fuzzy semantic parser into a verifiable cognitive architecture: deterministic safety and auditability alongside probabilistic flexibility and continuous self-improvement.

---

## Quick Start

```bash
# 1. Install (base deps are light; heavy features are optional — see "Dependencies")
pip install -r deployment/requirements.txt

# 2. Try it from the CLI
python cli.py demo
python cli.py pog "How can I build a self-improving reasoning agent?" --max-hops 3

# 3. Or run the API (ENGINE_API_KEY is REQUIRED — the server refuses to start without it)
export ENGINE_API_KEY=your-secret-key
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

Docker (full stack with Neo4j) and Kubernetes are in [Deployment](#deployment). Observability (Prometheus + Grafana) is a one-line compose overlay.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Use Cases & Capabilities](#use-cases--capabilities)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [FastAPI Server & CLI](#fastapi-server--cli)
- [Knowledge Graph & Neo4j](#knowledge-graph--neo4j)
- [Learned GNN](#learned-gnn)
- [Observability](#observability)
- [Testing](#testing)
- [Deployment](#deployment)
- [Persistence & Outputs](#persistence--outputs)
- [Design Notes & Limitations](#design-notes--limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Overview

The engine reasons over a graph of **polymorphic nodes** (`MorphicTextNode`) that morph between `linear`, `tree`, `nested`, and `indirect` execution strategies at runtime, driven by a hybrid similarity signal. Around that core it layers a production RAG pipeline, a knowledge-graph layer, PoG-style adaptive planning, an autonomous self-teaching loop, and a learned GNN — all behind a hardened FastAPI service and a CLI.

A guiding principle throughout: **heavy or infrastructure-bound capabilities are optional and degrade gracefully.** FAISS, sentence-transformers, cross-encoder re-rankers, Neo4j, PyTorch-Geometric, Prometheus, and rate limiting are each guarded by an availability flag, so the base install stays light and the test suite runs anywhere. When an optional dependency is absent, the engine falls back to a working (if simpler) implementation rather than failing.

`organized_self_morphing_engine.py` is the authoritative implementation (imported by `cli.py`, `fastapi_server.py`, and the tests). `archive/final_self_morphing_engine.py` is an earlier reference snapshot kept only for comparison — it is not part of the runtime.

## Key Features

### Polymorphic morphing execution
- **Node types**: `linear` (tail-recursion flattening), `tree` (branching with multi-criteria tie-breakers), `nested` (encapsulated parameter extraction), `indirect` (mutual-routine / external routing).
- **Trampoline safety**: a heap-allocated execution stack (`process_query_stream`) prevents Python recursion-depth errors and is provably terminating even on cyclic graphs (see the property tests).
- **Explosion tracking**: cascading sub-questions are logged for audit.

### Hybrid similarity & typo-robust embeddings
- **Levenshtein** scored on both the raw and synonym-canonicalized forms (so a typo is never penalized against its expanded target token), combined with a **vector** cosine into `hybrid_similarity` (all scores clamped to `[0, 100]`).
- **Deterministic character-trigram hashed embedding** as the always-available fallback — typo-robust *and* stable across processes (uses a fixed hash, not Python's per-process-randomized `hash()`), so a persisted FAISS index stays valid across runs.
- **Optional sentence-transformers** for true semantic embeddings; the model is selectable via `ENGINE_EMBEDDING_MODEL`.

### Advanced RAG pipeline
- **Chunking** (`advanced_chunk_text`): `fixed` (sliding window), `semantic` (embedding-based boundary detection between sentences), and `recursive` (hierarchical paragraph → sentence → word → char). All strategies honor overlap and never emit empty chunks.
- **Production FAISS** (`build_faiss_index`): exact `IndexFlatIP` for small corpora, approximate **HNSW** above a configurable size threshold; disk-persisted and auto-loaded.
- **Advanced retrieval** (`retrieve_context_advanced`): LLM **query rewriting**, optional **cross-encoder re-ranking**, and **agentic multi-hop** expansion (the LLM proposes follow-up queries until the context is sufficient).
- **Entity extraction** on ingest seeds the knowledge graph (`kg_entities` + co-occurrence `kg_relations`).

### Knowledge graph (SQLite + real Neo4j)
- Rich SQLite schema: `kg_entities` (with embeddings), `kg_relations` (typed, confidence-scored), `kg_metadata`.
- **Real Neo4j integration**: `connect_neo4j` opens an actual driver (verified connectivity), a **parameterized Cypher builder** (`build_entity_merge` / `build_relation_merge`, with sanitized relationship types), and **bidirectional sync** (`kg_sync_to_neo4j` / `kg_sync_from_neo4j`). Falls back to the SQLite KG when the driver or server is absent.

### PoG adaptive planning & self-teaching
- **PoG** (`pog_plan_and_reason`): task decomposition → multi-hop KG exploration → memory update → reflection/self-correction via the symbolic verifier + self-auditor.
- **Self-teaching loop**: proposes mappings/nodes via LLM, verifies through multiple layers, learns validated synonyms, and (when a graph is attached) uses GNN signals to prioritize and place new nodes. Tracks proposals, acceptances, rejection rate, average confidence, and learning history.

### Learned GNN (PyTorch Geometric)
- Builds a typed PyG graph from the morphic node structure and trains a **GraphSAGE** model for **node classification** and **link prediction** (dynamic path prediction). NumPy mean-aggregation propagation is the fallback when PyG is not installed.

### Production surface
- **FastAPI**: lifespan-managed engine, dependency-injected engine accessor, non-blocking (threadpool-offloaded) handlers, optional rate limiting, structured request logging with `X-Request-ID`, API-key auth, Pydantic-validated bodies, and truthful capability flags.
- **Observability**: a Prometheus `/metrics` exposition endpoint and provisioned Grafana dashboards.
- **CLI**, Docker/Compose, Kubernetes manifests, and audit/DOT-graph exports.

## Use Cases & Capabilities

### Where it fits

| Use case | How the engine helps | Key pieces |
|---|---|---|
| **Domain Q&A / knowledge assistant** | Ingest your docs, then answer questions grounded in retrieved context with confidence scoring. | `ingest_documents` → `retrieve_context_advanced` → `pog_plan_and_reason` |
| **Troubleshooting / root-cause analysis** | Decompose a problem into sub-objectives and walk multi-hop entity/relation paths in the knowledge graph. | PoG planning + `kg_relations` + agentic multi-hop retrieval |
| **Self-improving agent / router** | Route queries by hybrid similarity and let the background loop learn new vocabulary and graph structure over time, gated by verification. | `hybrid_similarity` + `self_teaching_loop` + verifiers |
| **Structured knowledge base** | Extract entities/relations on ingest and mirror them to a real Neo4j graph for Cypher analytics and multi-hop queries. | entity extraction + `kg_sync_to_neo4j` / `kg_sync_from_neo4j` |
| **Graph-ML research on reasoning graphs** | Train a GNN over the live morphic graph to classify node roles and predict likely next links (dynamic paths). | `build_graph_tensors` + `train_gnn` + `gnn_predict_links` |
| **Production microservice** | Expose the whole thing as an authenticated, rate-limited, observable HTTP API. | FastAPI service + Prometheus `/metrics` + Grafana |

### Simple capabilities (a few lines, base install)

These work out of the box with only the required dependencies — no external services, no GPU, deterministic:

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine
engine = ProductionAdaptiveEngine(target_solution_text="Compute Analytics")

# 1. Fuzzy + semantic similarity (typo-robust, 0–100)
engine.hybrid_similarity("comput analytics", "compute analytics")   # ~92

# 2. Chunk and ingest text, then retrieve relevant context
engine.ingest_documents(["Neural networks power deep learning."], strategy="recursive")
engine.semantic_retrieve_context("deep learning", k=3)

# 3. Plan-on-Graph reasoning with confidence + verification
engine.pog_plan_and_reason("How to bound recursion explosions?", max_hops=2)

# 4. Inspect operational metrics
engine.get_metrics_snapshot()
```

Or entirely from the terminal:

```bash
python cli.py demo
python cli.py pog "How to scale self-improving agents?" --max-hops 3
python cli.py ingest notes.txt --strategy semantic
python cli.py metrics
```

### Advanced capabilities (opt-in dependencies / services)

Each unlocks a richer implementation and degrades gracefully to a simpler one when its dependency is absent:

- **Production semantic RAG** — `sentence-transformers` embeddings (model set by `ENGINE_EMBEDDING_MODEL`) + a persistent FAISS index that auto-scales from exact `IndexFlatIP` to approximate **HNSW** past `ENGINE_FAISS_HNSW_THRESHOLD`.
- **Advanced retrieval** — `retrieve_context_advanced(query, rewrite=True, rerank=True, agentic=True)`: LLM query rewriting, **cross-encoder re-ranking**, and an **agentic multi-hop** loop that gathers context until it's sufficient.
- **Real knowledge graph** — connect a live **Neo4j** and sync entities/relations both directions via a parameterized Cypher builder; run multi-hop Cypher analytics over what the engine learns.
- **Learned GNN** — with `torch-geometric`, train a GraphSAGE model for **node classification** and **link prediction**, and let those signals guide where the self-teaching loop attaches new nodes.
- **Real LLM reasoning** — set `OPENAI_API_KEY` or `GROQ_API_KEY` to replace the deterministic mock for decomposition, reflection, query rewriting, and proposal generation.
- **Production API** — authenticated, rate-limited (`slowapi`), non-blocking endpoints with structured request logging; a Prometheus `/metrics` endpoint and provisioned Grafana dashboards; Docker Compose and Kubernetes deployment.

> Rule of thumb: the **simple** capabilities give you a working reasoning/RAG engine on a laptop with `pip install`; the **advanced** capabilities turn it into a scalable, observable service backed by real vector search, a graph database, and learned models — without changing your calling code.

## Architecture

```
User Query / Task
       │
       ▼
ProductionAdaptiveEngine(target_solution_text, similarity_threshold)
       │
       ├── MorphicTextNode graph (linear / tree / nested / indirect)
       │        └── Trampoline stack (process_query_stream) — safe, terminating
       │
       ├── RAG (advanced_chunk_text → ingest_documents → FAISS → retrieve_context_advanced)
       │        ├── query rewriting · cross-encoder re-rank · agentic multi-hop
       │        └── entity extraction → Knowledge Graph
       │
       ├── Knowledge Graph (SQLite schema ↔ real Neo4j via Cypher builder, bidirectional)
       │
       ├── PoG planning (pog_plan_and_reason: decompose → explore → reflect → verify)
       │
       ├── Self-teaching loop (LLM propose → multi-layer verify → learn → GNN-guided attach)
       │
       ├── Learned GNN (build_graph_tensors → train_gnn → classify / predict links)
       │
       └── Service layer (FastAPI + CLI + Prometheus /metrics + admin controls)
```

## Project Structure

```
self-morphing-adaptive-recursion-engine/
├── organized_self_morphing_engine.py   # Authoritative engine implementation
├── fastapi_server.py                   # Production FastAPI service
├── cli.py                              # Command-line interface
├── test_engine.py                      # Example-based pytest suite
├── test_properties.py                  # Hypothesis property-based tests
├── README.md · USAGE.md · GETTING_STARTED.md
├── archive/
│   └── final_self_morphing_engine.py   # Earlier reference snapshot (not runtime)
└── deployment/
    ├── requirements.txt
    ├── Dockerfile · Dockerfile.full     # base image · full image (heavy deps preinstalled)
    ├── docker-compose.yml               # engine + Neo4j
    ├── docker-compose.full.yml          # engine (full image)
    ├── docker-compose.observability.yml # + Prometheus + Grafana overlay
    ├── k8s-deployment.yaml              # StatefulSet + PVC + Prometheus scrape annotations
    ├── build.sh · start.sh
    └── observability/
        ├── prometheus.yml
        └── grafana/                     # provisioned datasource + dashboard
```

Runtime artifacts — `engine_logs.db`, `faiss_index.bin`, `gnn_model.pt`, `learning_report.json`, `morphic_graph*.dot` — are generated on use and git-ignored. In containers they live under the persisted `/app/state` volume.

## Installation & Dependencies

**Prerequisites**: Python 3.11+.

```bash
pip install -r deployment/requirements.txt
```

**Required**: `torch`, `numpy`, `networkx`, `fastapi`, `uvicorn`, `requests`, `pytest`, `hypothesis`.

**Optional (graceful fallback if absent)**:

| Dependency | Enables | Fallback when absent |
|---|---|---|
| `sentence-transformers` | true semantic embeddings + cross-encoder re-ranking | deterministic char-trigram hashed embedding; base retrieval order |
| `faiss-cpu` | persistent ANN vector index | brute-force cosine over SQLite |
| `neo4j` | external graph store + bidirectional sync | rich SQLite KG |
| `torch-geometric` | learned GraphSAGE GNN | NumPy message-passing propagation |
| `prometheus-client` | `/metrics` Prometheus exposition | `/metrics` returns 501; `/metrics.json` still works |
| `slowapi` | API rate limiting | limiter is a no-op |
| `groq` / `openai` | real LLM calls | deterministic mock LLM |

The `deployment/Dockerfile.full` image preinstalls the heavy optional stack.

## Usage Examples

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine

engine = ProductionAdaptiveEngine(target_solution_text="Compute Analytics", similarity_threshold=80.0)

# PoG adaptive planning (decomposition + multi-hop + reflection)
plan = engine.pog_plan_and_reason("Diagnose multi-hop dependencies in a data pipeline", max_hops=3)
print(plan["result"], plan["confidence"], plan["verified"])

# RAG: ingest (returns the real chunk count) and retrieve
n = engine.ingest_documents(
    ["Neural networks power deep learning.", "Paris is the capital of France."],
    strategy="recursive",
)
contexts = engine.retrieve_context_advanced(
    "deep learning", k=3, rewrite=True, rerank=True, agentic=False
)

# Learned GNN over a morphic graph
losses = engine.train_gnn(root_node)            # no-op + NumPy fallback if PyG absent
links = engine.gnn_predict_links(root_node, top_k=5)

# Knowledge graph ↔ Neo4j (no-op with a clear message if not connected)
engine.connect_neo4j()      # uses NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
engine.kg_sync_to_neo4j()
engine.kg_sync_from_neo4j()

# Observability snapshot (the same data the Prometheus endpoint exposes)
print(engine.get_metrics_snapshot())
```

## Configuration

All configuration is environment-variable driven (nothing hardcoded that matters for deployment):

| Variable | Purpose | Default |
|---|---|---|
| `ENGINE_API_KEY` | **required** for the API (fails fast if unset) | — |
| `ENGINE_DB_PATH` | SQLite database path | `engine_logs.db` |
| `ENGINE_FAISS_INDEX_PATH` | persisted FAISS index path | `faiss_index.bin` |
| `ENGINE_FAISS_HNSW_THRESHOLD` | vector count above which HNSW is used | `10000` |
| `ENGINE_EMBEDDING_MODEL` | sentence-transformers model name | `all-MiniLM-L6-v2` |
| `ENGINE_RERANK_MODEL` | cross-encoder re-rank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `ENGINE_GNN_MODEL_PATH` | persisted GNN weights path | `gnn_model.pt` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j connection | `bolt://localhost:7687` / `neo4j` / `password` |
| `OPENAI_API_KEY` / `GROQ_API_KEY` | real LLM backend | mock LLM |
| `RATE_LIMIT` | API rate limit (slowapi) | `120/minute` |
| `LOG_LEVEL` | server log level | `INFO` |

Constructor knobs: `similarity_threshold` (resolved-match cutoff), and internal `flag_buffer` (low-confidence zone width).

## FastAPI Server & CLI

### FastAPI (`fastapi_server.py`)

```bash
export ENGINE_API_KEY=your-secret-key
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

- Swagger UI at `/docs`; API-key auth via the `X-API-Key` header.
- Blocking work is offloaded to a threadpool; optional rate limiting; structured logs with `X-Request-ID`.
- Endpoints: `/query`, `/pog/plan`, `/rag/ingest`, `/rag/retrieve`, `/rag/stats`, `/teach`, `/teach/status`, `/nodes/attach`, `/admin/mappings` (GET/POST), `/admin/resolve`, `/admin/halts`, `/graph/export`, `/graph/visualize`, `/system/info`, `/health`, `/metrics` (Prometheus), `/metrics.json` (authenticated JSON).
- `/metrics` is unauthenticated by convention (for Prometheus scraping) and exposes only non-sensitive counters; `/system/info` reports *real* runtime capability (installed optional deps and live Neo4j connection).

### CLI (`cli.py`)

```bash
python cli.py demo
python cli.py pog "How to scale self-improving agents?" --max-hops 3
python cli.py query "How to handle recursive explosions?"
python cli.py ingest doc1.txt doc2.txt --strategy semantic
python cli.py teach --iterations 3
python cli.py metrics
```

## Knowledge Graph & Neo4j

The SQLite KG (`kg_entities`, `kg_relations`, `kg_metadata`) is always available. With the `neo4j` driver installed and a reachable server:

- `connect_neo4j()` opens a verified connection (from `NEO4J_URI/USER/PASSWORD`);
- `kg_sync_to_neo4j()` pushes entities/relations via idempotent, parameterized `MERGE` Cypher (relationship types are sanitized to safe identifiers — Cypher cannot parameterize them);
- `kg_sync_from_neo4j()` pulls the graph back into SQLite, keeping both stores consistent.

The bundled `docker-compose.yml` includes a Neo4j service and a healthcheck so the engine only starts once Neo4j is ready.

## Learned GNN

`build_graph_tensors(root)` converts a `MorphicTextNode` graph (cycle-safely) into node features (embedding ⊕ one-hot node type) and typed edges (`linear` / `branch` / `inner_formula` / `mutual_routine`). `train_gnn(root)` trains a GraphSAGE encoder with node-classification and link-prediction heads (weights persisted to `ENGINE_GNN_MODEL_PATH`); `gnn_classify_nodes`, `gnn_predict_links`, and `gnn_node_relevance` provide inference. Without `torch-geometric`, a NumPy propagation fallback keeps all of this functional.

## Observability

- The engine exposes `get_metrics_snapshot()`; the FastAPI `/metrics` endpoint renders it in Prometheus text-exposition format (`morphic_*` gauges: proposals, accepted learnings, rejection rate, average confidence, explosion events, KB size, FAISS vectors, admin-queue size, …).
- Stand up the full monitoring stack with the overlay:

```bash
cd deployment
ENGINE_API_KEY=your-secret-key docker-compose -f docker-compose.yml -f docker-compose.observability.yml up --build
# Grafana: http://localhost:3000 (admin/admin)  ·  Prometheus: http://localhost:9090
```

Grafana is auto-provisioned with the Prometheus datasource and a dashboard (learning metrics + explosion rates). The Kubernetes manifest carries `prometheus.io/scrape` annotations.

## Testing

```bash
pytest -q                      # full suite
pytest test_properties.py -q   # property-based tests only
```

- `test_engine.py` — example-based tests across similarity, chunking, retrieval, entity extraction, Neo4j (mocked driver), GNN (fallback + PyG-gated), metrics, and the FastAPI surface.
- `test_properties.py` — **Hypothesis** property-based tests for morphing invariants: similarity bounds/symmetry/reflexivity, deterministic embeddings, chunking coverage (no dropped content, no empty chunks), cycle-safe & terminating graph traversal, cache invalidation after synonym mutation, and trampoline termination on deep/cyclic graphs.

Tests requiring an optional dependency `skipif` when it's absent, so the suite is green on a base install. (The property tests originally caught a real invariant bug — a similarity score exceeding 100 from floating-point rounding — now fixed by clamping.)

## Deployment

### Docker Compose (engine + Neo4j)
```bash
cd deployment
export ENGINE_API_KEY=your-secret-key
./start.sh
# API + Swagger: http://localhost:8000/docs · Neo4j: http://localhost:7474
```
Add `-f docker-compose.observability.yml` (see [Observability](#observability)) for Prometheus + Grafana. Use `docker-compose.full.yml` / `Dockerfile.full` for an image with the heavy optional stack preinstalled. Images run as a non-root user.

### Kubernetes
```bash
kubectl apply -f deployment/k8s-deployment.yaml
```
A `StatefulSet` with a `PersistentVolumeClaim` mounted at `/app/state` (where the DB and FAISS index live), readiness/liveness probes, resource limits, and Prometheus scrape annotations. `ENGINE_API_KEY` (and optional LLM keys) are read from a `morphic-secrets` Secret. For large multi-replica deployments, prefer an external vector DB (Qdrant/Milvus/pgvector) over the file-based FAISS index.

## Persistence & Outputs

- **`engine_logs.db`** (SQLite, WAL): `synonym_mappings`, `qa_audit_log`, `query_explosions`, `rag_chunks`, `kg_entities`/`kg_relations`/`kg_metadata`.
- **`faiss_index.bin`** (+ `.contents.json`): persisted vector index, auto-loaded on startup.
- **`gnn_model.pt`**: persisted GNN weights.
- **`learning_report.json`**: proposals / acceptance / confidence snapshot (`generate_learning_report`).
- **`morphic_graph*.dot`**: Graphviz DOT export of the morphic structure (`export_graph_viz`).

## Design Notes & Limitations

- **LLM dependence**: generative steps (proposals, PoG decomposition, query rewriting, agentic hops) use a real LLM when `OPENAI_API_KEY`/`GROQ_API_KEY` are set, otherwise a deterministic mock — sufficient for demos and tests but not for production reasoning quality.
- **In-memory graph**: `MorphicTextNode` graphs are in-memory per request; durable state lives in the DB/KG, not as a serialized graph.
- **Single-node vector search**: the file-based FAISS index suits single-node or small clusters; scale out with an external vector DB.
- **Security**: API-key auth + optional rate limiting are provided; add input sanitization and sandboxing before exposing LLM-generated content externally.

## Roadmap

The original productionization roadmap is **complete** and merged:

- ✅ Fleshed-out engine methods (real semantic/recursive chunking, GNN-guided self-teaching, production FastAPI migration)
- ✅ Real Neo4j driver + bidirectional sync + Cypher builder
- ✅ Pre-trained embeddings + production FAISS (HNSW) scaling
- ✅ Prometheus exporter + Grafana dashboards
- ✅ Advanced RAG: query rewriting, cross-encoder re-ranking, agentic retrieval
- ✅ Learned GNN (PyTorch Geometric) for path prediction / node classification
- ✅ Full pytest suite + Hypothesis property-based testing

Possible future directions: continuous online learning without catastrophic forgetting, formal verification of morphing-safety properties, multi-agent orchestration, and multi-tenant deployment with isolated graphs.

## Contributing

1. Branch from `main`.
2. Prefer non-breaking, optional-with-fallback additions (match the existing `*_AVAILABLE` guard pattern).
3. Add/update tests (`pytest`), including a property test for any new invariant.
4. Keep docstring coverage high and update this README when behavior changes.
5. Open a PR describing the morphing / RAG / KG / GNN impact.

---

*Explore the code starting from `organized_self_morphing_engine.py` (the authoritative implementation). `archive/final_self_morphing_engine.py` is an earlier snapshot kept only for comparison.*
