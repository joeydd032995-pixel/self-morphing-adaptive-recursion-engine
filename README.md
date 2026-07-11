# Self-Morphing Adaptive Recursion Engine

**A Hybrid, Self-Improving General-Purpose Reasoning Framework with Polymorphic Morphing Nodes, Advanced RAG, PoG/KG-LLM Integration, GNN-like Propagation, and Production-Ready Deployment**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- Update if added -->
[![Status](https://img.shields.io/badge/Status-Prototype%20%7C%20Enhanced-orange.svg)]()

> **Evolved from a fuzzy semantic parser into a sophisticated cognitive architecture** capable of deep contextual understanding, adaptive execution morphing, continuous self-teaching, verifiable hybrid intelligence (deterministic + probabilistic), and scalable deployment.

## Quick Start

### Local
```bash
pip install -r deployment/requirements.txt
python organized_self_morphing_engine.py
# or
python cli.py demo
```

### Docker Compose (Recommended)
```bash
cd deployment
./start.sh
# API + Swagger: http://localhost:8000/docs
```

### Kubernetes
```bash
kubectl apply -f deployment/k8s-deployment.yaml
```

See the **Comprehensive Setup Guide** and **Deployment** sections below for full details.

## Table of Contents
- [Overview](#overview)
- [Key Features & Capabilities](#key-features--capabilities)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start & Usage Examples](#quick-start--usage-examples)
- [Configuration](#configuration)
- [Core Components & API](#core-components--api)
- [PoG & KG-LLM Integration](#pog--kg-llm-integration)
- [Testing, Benchmarking & Demo](#testing-benchmarking--demo)
- [Deployment](#deployment)
- [Persistence, Outputs & Monitoring](#persistence-outputs--monitoring)
- [Limitations, Edge Cases & Known Issues](#limitations-edge-cases--known-issues)
- [Roadmap & Future Enhancements](#roadmap--future-enhancements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview

The **Self-Morphing Adaptive Recursion Engine** is a production-oriented prototype for general-purpose reasoning. It integrates:

- **Polymorphic graph execution** (nodes that dynamically morph between linear, tree, nested, and indirect strategies at runtime).
- **Hybrid similarity & embeddings** (Levenshtein distance + Torch-based vector embeddings for robust fuzzy + semantic matching).
- **Advanced RAG pipeline** (semantic/recursive chunking, hybrid retrieval, re-ranking).
- **Knowledge Graph (KG) layer** (SQLite with advanced schema + optional Neo4j sync for multi-hop reasoning and entity-relation persistence).
- **PoG (Plan-on-Graph) style adaptive planning** (task decomposition, guidance/memory/reflection loops for self-correcting multi-hop KG exploration).
- **Autonomous self-teaching** (background LLM-proposal + multi-layer verification loop with dynamic node generation and cycle detection).
- **GNN-like propagation** (graph message passing for node feature refinement and path optimization).
- **Safe recursion** (heap-allocated trampoline stack to avoid Python recursion limits).
- **Production scaffolding** (FastAPI server with Swagger + auth, CLI, distributed multi-process execution, metrics, Docker/K8s support, admin controls, audit logging).
- **Persistent Vector Database** (FAISS with disk-persisted index + metadata for fast semantic retrieval, integrated with RAG ingestion).

It began as a fuzzy semantic parser for query routing and evolved through iterative enhancements (self-teaching, vector embeddings, RAG, GNN simulation, PoG/KG-LLM, Neo4j integration) into a verifiable, extensible hybrid neurosymbolic engine. The design prioritizes **deterministic safety + auditability** alongside **probabilistic flexibility and self-improvement**, making it suitable for autonomous agents, knowledge systems, troubleshooting pipelines, and research into scalable reasoning architectures.

**Current State**: Core engine, similarity/embedding layers, LLM integration (mock + real SDK stubs), Neo4j stubs, advanced KG schema, PoG-style planning hooks, RAG foundation, and demo infrastructure are implemented. `organized_self_morphing_engine.py` is the authoritative, actively-used implementation (imported by `cli.py`, `fastapi_server.py`, and `test_engine.py`). `archive/final_self_morphing_engine.py` is an earlier, lighter-weight draft kept only for comparing engine evolution — it is not imported anywhere and should not be treated as a live entrypoint.

## Key Features & Capabilities

### Polymorphic Morphing Execution
- **Node Types**: `linear` (tail recursion flattening), `tree` (branching with sophisticated tie-breakers: similarity + length + ID + history), `nested` (encapsulated parameter extraction & multi-phase feedback), `indirect` (mutual routine jumps / external module routing).
- **Runtime Adaptation**: Execution path morphs based on input similarity, confidence zones, and admin overrides.
- **Trampoline Safety**: Heap-based stack prevents recursion depth errors while supporting deep nested/indirect flows.
- **Explosion Tracking**: Monitors cascading sub-questions (query explosions) for audit and optimization.

### Hybrid Semantic Matching & Embeddings
- **Levenshtein DP** (cached, vectorized) for fuzzy string matching.
- **Torch Bag-of-Words Embeddings** (512-dim, normalized) + Cosine similarity.
- **Hybrid Scoring**: Weighted combination (default 60% Levenshtein + 40% vector) for robust matching beyond pure syntax or semantics.
- **Synonym Auto-Learning**: Persistent dictionary with DB-backed mappings; low-confidence isolation + admin review queue.

### Advanced RAG Pipeline
- **Chunking Strategies**: Fixed-size (with overlap), semantic (sentence/paragraph-aware), recursive (hierarchical).
- **Ingestion & Storage**: Text chunking + embedding + metadata in SQLite `rag_chunks`.
- **Retrieval**: Top-k hybrid search (vector + keyword/graph signals); multi-hop KG traversal simulation.
- **Re-ranking & Augmentation**: Planned hooks for context re-ranking and query rewriting.

### Knowledge Graph (KG) Layer & PoG Integration
- **Advanced SQLite Schema**: `kg_entities` (with embeddings, source), `kg_relations` (typed, confidence-scored), `kg_metadata`.
- **Neo4j Sync Stubs**: `connect_neo4j()`, `kg_sync_to_neo4j()` for external persistence and Cypher queries.
- **PoG-Style Adaptive Planning**: Task decomposition into sub-objectives, adaptive path exploration on KG, memory updating (historical retrieval/reasoning), reflection/self-correction (via auditor + symbolic verifier). Guidance/Memory/Reflection loop for robust multi-hop reasoning and error recovery.
- **GNN-like Propagation**: Basic message-passing simulation on morphic graph for node refinement and intelligent tie-breaking/path prediction.

### Autonomous Self-Teaching & Verification
- **Background Loop**: Queries unresolved/low-confidence cases from audit logs; proposes mappings/nodes via LLM.
- **Multi-Layer Verification**: Self-Auditor (similarity + rules), Symbolic Verifier (syntax, consistency, domain rules, embedding cross-check), optional admin halt/override.
- **Dynamic Graph Evolution**: LLM-proposed nodes validated and attached (with cycle detection); synonym learning persisted.
- **Metrics Tracking**: Proposals, acceptance rate, avg confidence, rejection rate, learning history.

### Production & Distributed Features
- **FastAPI Server** (`fastapi_server.py`): Modern API with automatic Swagger docs, API key authentication, endpoints for reasoning, PoG planning, RAG ingest/retrieve, self-teaching, admin, metrics, and graph export.
- **CLI** (`cli.py`): Powerful command-line interface for querying (including PoG), ingesting documents, triggering self-teaching, viewing metrics, and running demos.
- **Distributed Execution**: `concurrent.futures` Thread/ProcessPool, graph sharding by ID hash, non-blocking task streams.
- **Observability**: Prometheus-compatible metrics, QA audit logs, explosion logs, learning reports, DOT graph visualization (`morphic_graph.dot`).
- **Safety Controls**: Admin review queue for low-confidence mappings, halt/resolution interface, symbolic conflict detection.
- **Persistence**: SQLite (WAL mode for concurrency) + **persistent FAISS vector index** (disk-saved for fast semantic search) + optional Neo4j; learning reports JSON.

### LLM Integration
- **Real SDK Ready**: Stubs for OpenAI/Groq with JSON mode, retry logic, structured outputs (commented for production).
- **Mock Fallback**: Deterministic simulation for demos/testing.
- **JSON-Mode Proposals**: Structured suggestions for mappings, new nodes, confidence scores.

## Architecture

```
User Query / Task
       │
       ▼
ProductionAdaptiveEngine (init with target, threshold, embeddings, KG schema, Neo4j stub)
       │
       ├── MorphicTextNode Graph (polymorphic: linear/tree/nested/indirect links + branches + sub-questions)
       │        │
       │        ├── Trampoline Execution Stack (safe deep recursion)
       │        │        ├── process_query_stream() → morph based on hybrid_similarity()
       │        │        ├── Tree tie-breakers + GNN-like propagation for path optimization
       │        │        └── Explosion tracking & audit
       │        │
       │        ├── RAG Layer (advanced_chunk_text, ingest, retrieve_context, kg_enhanced_retrieve)
       │        │        └── Multi-hop KG signals (SQLite + Neo4j sync)
       │        │
       │        └── PoG Adaptive Planning (pog_plan_and_reason)
       │                 ├── Task Decomposition (sub-objectives via LLM)
       │                 ├── Guidance / Memory / Reflection Loop (self-correction)
       │                 └── Path Exploration + Verification (auditor + symbolic)
       │
       ├── Self-Teaching Loop (background)
       │        ├── LLM proposals (llm_call JSON mode)
       │        ├── Multi-layer Verify (self_auditor_verify, symbolic_verifier)
       │        ├── Dynamic Node Attach (cycle detection) + Synonym Learning
       │        └── Metrics Update + Persistence
       │
       └── API / Admin / Metrics (start_api_server, admin_resolve_halt, export_graph_viz)
              └── Distributed Pools & Observability (logs, reports, DOT viz)
```

**Data Flow Nuances**:
- Low-confidence zones trigger auto-learning or admin halt.
- Tree execution uses multi-criteria tie-breakers (similarity primary, then length/ID/history).
- KG relations carry confidence; GNN propagation refines node features for better routing.
- All mutations (learning, node attach) go through verification layers for safety.
- Neo4j sync is non-breaking stub; falls back to rich SQLite schema.

## Project Structure

```
self-morphing-adaptive-recursion-engine/
├── organized_self_morphing_engine.py  # Authoritative full implementation (core + all enhancements)
├── fastapi_server.py                  # Production FastAPI server (Swagger + auth)
├── cli.py                             # Command-line interface
├── test_engine.py                     # Comprehensive pytest suite
├── README.md                          # This file
├── deployment/
│   ├── Dockerfile
│   ├── Dockerfile.full                # Full-featured image (sentence-transformers + faiss-cpu)
│   ├── docker-compose.yml
│   ├── docker-compose.full.yml
│   ├── k8s-deployment.yaml
│   ├── start.sh                       # One-command startup
│   ├── build.sh                       # Clean Docker build
│   └── requirements.txt
├── archive/
│   └── final_self_morphing_engine.py  # Superseded reference version, kept for comparison
├── engine_logs.db                     # SQLite + persistent FAISS index (generated at runtime)
├── faiss_index.bin                    # Persistent vector index (auto-managed)
├── learning_report.json
├── morphic_graph.dot
└── run_output.txt / output.log
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Optional: Neo4j server (or use Docker Compose)
- For production LLM: `OPENAI_API_KEY` or `GROQ_API_KEY` env vars

### Local Setup
```bash
# 1. Clone or copy the artifacts directory
cd /path/to/artifacts

# 2. (Recommended) Create virtualenv
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r deployment/requirements.txt
# Note: neo4j is stubbed (install real driver for full sync: pip install neo4j)

# 4. (Optional) Start Neo4j locally or via Docker
# docker run --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

# 5. Run demo
python cli.py demo
```

### Docker Setup (Recommended for full stack)
```bash
cd deployment
docker-compose up --build
# Access API on localhost:8000; Neo4j browser on localhost:7474 (user: neo4j, pass: password)
```

Environment variables (`.env` or docker-compose):
- `ENGINE_API_KEY` (**required** — the API refuses to start without it; no default is provided)
- `OPENAI_API_KEY`, `GROQ_API_KEY`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

## Quick Start & Usage Examples

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine

# Initialize (target objective drives routing/similarity baseline)
engine = ProductionAdaptiveEngine(target_solution_text="Compute Analytics", similarity_threshold=80.0)

# Basic hybrid reasoning query
result = engine.reasoning_query("How to solve complex data pipeline issues?")  # Uses RAG + morphing + PoG hooks
print(result)

# PoG-style adaptive planning (task decomposition + reflection)
pog_result = engine.pog_plan_and_reason("Analyze multi-hop dependencies in knowledge graph for troubleshooting")
print(pog_result)

# Ingest documents for RAG/KG
engine.ingest_documents(["path/to/doc1.txt", "path/to/doc2.pdf"])  # Advanced chunking + embedding + KG entity extraction

# Multi-hop KG-enhanced retrieve
context = engine.kg_enhanced_retrieve("root cause of query explosion in recursive systems", k=5, hops=2)
print(context)

# Start API server (for external integration)
# engine.start_api_server(port=8000)  # Endpoints: /query, /ingest, /metrics, /admin, /graph

# Neo4j sync (after configuring connection)
engine.kg_sync_to_neo4j()

# Run full demo (includes self-teaching iteration, benchmarks, API test)
engine.run_full_demo()
```

See code for `process_query_stream(task_id, start_node, live_user_input)` for low-level trampoline control and admin flows.

## Configuration

- **similarity_threshold** (default 80.0): Minimum % for "resolved" match; below triggers learning/halt.
- **flag_buffer** (5.0): Low-confidence zone width for auto-learning vs admin review.
- **vocab_size** (512): Embedding dimension.
- **DB paths**: Hardcoded to `engine_logs.db` (override in `_init_db`).
- **LLM**: Edit `llm_call` for real client or temperature/max_tokens.
- **Neo4j**: Params in `connect_neo4j()`; set env vars for compose.
- **Learning**: `self_teaching_loop(background=True, max_iterations=...)`.

## Core Components & API

### MorphicTextNode
Polymorphic container with runtime-morphable links. Tracks triggered sub-questions for explosion analysis.

### ProductionAdaptiveEngine (Main Class)
**Core Methods Implemented**:
- `__init__`, `_init_db` (advanced KG schema), `_load_synonyms_from_db`
- `calculate_similarity` (Levenshtein DP, cached), `_tokenize_synonyms`, `_compute_embedding`, `hybrid_similarity`
- `llm_call` (mock + real SDK stubs, JSON mode)
- `connect_neo4j`, `kg_sync_to_neo4j` (stubs with fallback)
- `run_full_demo`

**Referenced / Consolidated from Prior Iterations** (full logic preserved in `organized_*.py` or extensible):
- `process_query_stream`, `reasoning_query`, `pog_plan_and_reason`
- `ingest_documents`, `retrieve_context`, `kg_enhanced_retrieve`, `advanced_chunk_text`
- `self_teaching_loop`, `self_auditor_verify`, `symbolic_verifier`
- `attach_node` (with cycle detection), `gnn_propagate`
- `start_api_server`, `admin_*` methods, `export_graph_viz`, `generate_learning_report`
- `run_benchmarks`, `run_basic_tests` / full test suite

Admin interfaces, explosion logging, and verification layers provide production safety.

## PoG & KG-LLM Integration

**PoG (Plan-on-Graph)**: Self-correcting adaptive planning over KGs.
- Decomposes query into sub-objectives (Guidance) via `pog_plan_and_reason`.
- Iteratively explores reasoning paths on KG while updating Memory (retrieval history + reasoning state).
- Reflection step evaluates progress, detects errors, triggers self-correction or backtracking.
- Integrated via the dedicated `pog_plan_and_reason(query, max_hops=3)` method and hooks in reasoning flow + verifier layers.

**KG-LLM Hybrid Patterns**:
- LLM augments KG construction (entity/relation extraction during ingest).
- KG grounds LLM outputs (multi-hop retrieval, confidence-scored relations, GNN refinement).
- Bidirectional: LLM proposes KG updates; KG validates/propagates via GNN-like messages.
- Benefits: Reduced hallucinations, interpretable multi-hop reasoning, evolving structured memory.

These were added as non-breaking enhancements (stubs + schema + planning hooks) while preserving original morphing core.

## Persistent Vector Database (FAISS)

The engine includes a **persistent FAISS vector index** for fast semantic retrieval:
- Automatically builds and saves `faiss_index.bin` + metadata on disk.
- Loaded automatically on engine startup.
- Integrated with `ingest_documents()` (auto-rebuilds index) and `semantic_retrieve_context()`.
- Falls back gracefully if FAISS is not installed.
- Works alongside the advanced RAG chunking (`advanced_chunk_text`) and optional `sentence-transformers` embeddings.

This provides production-grade vector search without requiring an external service while remaining optional.

## Testing, Benchmarking & Demo

- `run_full_demo()`: End-to-end smoke test (RAG, reasoning, self-teaching iteration, API stubs, metrics).
- `run_benchmarks()`: Similarity timing, teaching loop, etc. (uses `timeit`).
- `run_basic_tests()`: Unit assertions on hybrid similarity, node morphing, verifiers.
- Audit logs + `learning_report.json` for post-run analysis.
- Graph export (`morphic_graph.dot`) for visual inspection (use Graphviz).

Extend with pytest for full coverage of verification layers and PoG loops.

## Deployment

### Docker Compose (Full Stack)
```bash
cd deployment
./start.sh          # Recommended one-liner
# or
docker-compose up --build -d
# FastAPI on :8000 (Swagger: /docs), Neo4j on :7474/:7687
```

### Kubernetes (Multi-Node Ready)
```bash
kubectl apply -f deployment/k8s-deployment.yaml
# Includes example PersistentVolumeClaim for shared storage.
# For production multi-replica: use ReadWriteMany storage + external vector DB recommended.
```

### Production Considerations
- Use real Neo4j driver + connection pooling.
- FastAPI is the recommended production API (`fastapi_server.py`).
- Add Prometheus exporter, structured logging, distributed tracing.
- Horizontal scaling via Ray/Dask; for true multi-node vector search, prefer external vector DB (pgvector, Qdrant, etc.) over file-based FAISS.
- Secrets via K8s secrets or Vault; rate limiting on API.
- CI/CD: Build image, run tests/benchmarks, deploy with rolling updates.

See `Dockerfile`, `docker-compose.yml`, `start.sh`, and `build.sh` in the `deployment/` folder.

## FastAPI Server & CLI

### FastAPI Server (`fastapi_server.py`)
Modern, documented API:
```bash
uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000
```
- Automatic Swagger UI at `/docs`
- API Key authentication (`X-API-Key` header)
- Key endpoints: `/query`, `/pog/plan`, `/rag/ingest`, `/rag/retrieve`, `/teach`, `/metrics`, `/admin/*`, `/graph/export`, `/rag/stats`

### CLI (`cli.py`)
```bash
python cli.py --help
python cli.py pog "How to scale self-improving agents?" --max-hops 3
python cli.py ingest doc1.txt doc2.txt --strategy semantic
python cli.py teach --iterations 3
python cli.py metrics
python cli.py demo
```

## Multi-Node Deployment

The engine supports multi-node setups with the following considerations:

**Docker Compose Scaling** (limited):
```yaml
# In docker-compose.yml
deploy:
  replicas: 3
```
Note: SQLite + file-based FAISS require shared storage (NFS volume) for consistency.

**Kubernetes (Recommended for Multi-Node)**:
- Replicas set to 3 in `k8s-deployment.yaml`
- Includes example `PersistentVolumeClaim` (ReadWriteMany)
- For production: Mount shared PVC for `/app` (DB + FAISS index) **or** migrate to an external vector database + shared Postgres/Neo4j.

**Best Practice for Scale**:
- Use the persistent FAISS index for single-node or small clusters.
- For large multi-node deployments, replace FAISS with a distributed vector DB (e.g., Milvus, Qdrant, or pgvector) while keeping the rest of the engine unchanged.
- Self-teaching and admin features remain functional across nodes when using shared persistent storage.

## Persistence, Outputs & Monitoring

- **engine_logs.db**: Full audit (synonym_mappings, qa_audit_log, query_explosions, rag_chunks, kg_entities/relations/metadata). WAL mode for concurrency.
- **learning_report.json**: Snapshot of proposals, acceptance, confidence metrics.
- **morphic_graph.dot**: DOT representation of current morphic structure (nodes = id/key_phrase/type; edges = links/branches).
- Logs: Console + `output.log` / `run_output.txt`.
- Metrics: In-memory `learning_metrics`; extend to Prometheus `/metrics` endpoint.

Query DB directly for analytics or build dashboards on explosion counts, learning trends, KG growth.

## Limitations, Edge Cases & Known Issues

- **LLM Dependency**: Generative parts (proposals, PoG decomposition) rely on external LLM or mock. Hallucinations possible without strong verification (mitigated by multi-layer checks).
- **Scale**: Single-process demo; distributed sharding/orchestration is stubbed. Large graphs or high query volume may need Ray + persistent KV store.
- **Embeddings**: Simple hash-based BoW (fast, no external models). Upgrade path to sentence-transformers or fine-tuned models exists in design.
- **Neo4j**: Full driver integration is stub; requires `pip install neo4j` + running server. Sync is one-way placeholder.
- **Final.py State**: Many advanced methods (complete RAG chunkers, full GNN training loop, exhaustive PoG reflection) are referenced from prior iterations or stubbed with comments. Use `organized_self_morphing_engine.py` as reference implementation or extend `final_`.
- **Edge Cases**:
  - Very short/empty queries → low similarity, triggers learning/halt.
  - High explosion depth → logged but may overwhelm stack if not pruned.
  - Conflicting synonym proposals → symbolic verifier flags; admin resolution required.
  - Cycle in dynamic node attachment → detected and rejected.
  - No Neo4j → graceful fallback to SQLite (rich schema still provides value).
- **No Persistence of Full Graph State** beyond DB tables and DOT export (in-memory nodes on restart).
- **Security**: Basic; add auth, input sanitization, sandboxing for LLM-generated code/nodes in prod.

## Roadmap & Future Enhancements

**Near-term**:
- Flesh out remaining stubbed methods in `organized_self_morphing_engine.py` (full `advanced_chunk_text`, complete `self_teaching_loop` with GNN signals, production FastAPI migration).
- Real Neo4j driver + bidirectional sync + Cypher query builder.
- Pre-trained embeddings + FAISS index for production RAG scale.
- Full test suite (pytest) + property-based testing for morphing invariants.
- Prometheus exporter + Grafana dashboards for learning metrics / explosion rates.

**Mid-term**:
- Learned GNN (PyTorch Geometric) for dynamic path prediction and node classification.
- Multi-agent orchestration hub (integrate with external agents like Manus/Goose/Claude/GPT etc.).
- Advanced RAG: query rewriting, re-ranking models, agentic retrieval.
- iOS / on-device companion (agentic shortcuts calling the engine API).

**Long-term / Research**:
- Continuous online learning without catastrophic forgetting.
- Formal verification of morphing safety properties.
- Production multi-tenant deployment with tenant-isolated graphs and billing.

Contributions in these areas (especially fleshing out stubs while preserving non-breaking nature) are highly valued.

## Contributing

1. Fork & branch from `main`.
2. Make focused changes (prefer non-breaking additions with feature flags or optional params).
3. Add/update tests and run `python -m pytest` (or extend `run_basic_tests`).
4. Update README and docstrings.
5. Submit PR with clear description of morphing/RAG/KG impact.

Issues/PRs welcome for bugs in verification layers, PoG reflection logic, embedding quality, or deployment.

## Acknowledgments

Built iteratively with hybrid neurosymbolic principles, drawing inspiration from Graph-of-Thoughts / Plan-on-Graph research, adaptive recursion patterns, and production agent frameworks. Special thanks to the iterative enhancement process that added PoG/KG-LLM, Neo4j stubs, advanced schema, and deployment scaffolding while keeping the polymorphic core intact.

**Explore the code** — start with `organized_self_morphing_engine.py` (the authoritative implementation); `archive/final_self_morphing_engine.py` is kept only as an earlier reference snapshot for comparison.

---

*Last refined: 2026-07-05. This README aims for completeness while highlighting actionable next steps for productionization and research extensions.*
