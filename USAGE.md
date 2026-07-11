# Self-Morphing Adaptive Recursion Engine - User Guide

This guide provides a practical workflow for new users to get started quickly and effectively with the engine.

## 1. Quickest Way to Try It (2 minutes)

### Option A: Using the CLI (Recommended for beginners)

```bash
cd /path/to/artifacts

# Install dependencies
pip install -r deployment/requirements.txt

# Run a PoG reasoning query
python cli.py pog "How can I build a self-improving AI agent?"

# Run the full demo
python cli.py demo
```

### Option B: Using Python directly

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine

engine = ProductionAdaptiveEngine(target_solution_text="Build better AI systems")

# Basic PoG reasoning
result = engine.pog_plan_and_reason(
    "Explain the advantages of polymorphic execution in reasoning engines",
    max_hops=3
)
print(result["result"])
print(f"Confidence: {result['confidence']:.1f}%")
```

---

## 2. Recommended Workflow for New Users

### Step 1: Start with PoG Planning (`pog_plan_and_reason`)

This is the most powerful and user-friendly feature.

```python
result = engine.pog_plan_and_reason(
    query="How to implement continuous self-improvement in AI agents?",
    max_hops=3
)
```

**What it does**:
- Decomposes your query into sub-objectives
- Explores the Knowledge Graph
- Reflects and self-corrects
- Returns structured output with confidence

### Step 2: Ingest Your Own Knowledge (RAG)

Make the engine knowledgeable about your domain.

```python
# From text
documents = [
    "Recursive systems can suffer from query explosions when not properly bounded.",
    "Polymorphic nodes allow dynamic switching between linear, tree, and nested execution strategies."
]

engine.ingest_documents(documents, strategy="semantic")

# Then query with context
contexts = engine.semantic_retrieve_context("query explosions in recursion", k=3)
print(contexts)
```

### Step 3: Use the Self-Teaching Loop

Let the engine improve itself over time.

```python
# Run self-teaching in background
engine.self_teaching_loop(background=True, max_iterations=5)

# Check status
print(engine.learning_metrics)
```

### Step 4: Use the FastAPI Server (for applications)

```bash
# Start the server
uvicorn fastapi_server:app --reload --port 8000
```

Set `ENGINE_API_KEY` before starting the server — there is no default key, and the server refuses to start without it:

```bash
export ENGINE_API_KEY=your-secret-key
uvicorn fastapi_server:app --reload --port 8000
```

Then call it from your application:

```bash
curl -X POST http://localhost:8000/pog/plan \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Design a multi-agent system", "max_hops": 4}'
```

---

## 3. Common Usage Patterns

### Pattern 1: Research / Exploration
```python
engine = ProductionAdaptiveEngine("Research Topic")
result = engine.pog_plan_and_reason("Your complex research question")
```

### Pattern 2: Knowledge-Augmented Reasoning (RAG + PoG)
```python
engine.ingest_documents(["your_knowledge_base.txt"])
result = engine.pog_plan_and_reason("Question about your domain")
```

### Pattern 3: Autonomous Improvement
```python
# Start background self-teaching
engine.self_teaching_loop(background=True)

# Periodically check learning progress
print(engine.learning_metrics)
```

### Pattern 4: Dynamic Graph Building
```python
parent = MorphicTextNode("root", "main concept")
child = MorphicTextNode("child1", "sub-concept", "tree")
engine.attach_node(parent, child, "left")
```

---

## 4. Using the CLI Effectively

The CLI is great for quick experiments:

```bash
# Reasoning
python cli.py pog "Your question here" --max-hops 3

# Ingest documents
python cli.py ingest knowledge1.txt knowledge2.txt --strategy semantic

# Self-teaching
python cli.py teach --iterations 3

# Check system state
python cli.py metrics
```

---

## 5. Using the FastAPI Server

Start the server:

```bash
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

Key endpoints:

| Endpoint              | Purpose                          | Example Use Case                     |
|-----------------------|----------------------------------|--------------------------------------|
| `POST /pog/plan`      | Advanced reasoning               | Complex multi-step questions         |
| `POST /query`         | General reasoning                | Quick answers                        |
| `POST /rag/ingest`    | Add knowledge                    | Domain adaptation                    |
| `GET /rag/retrieve`   | Semantic search                  | Retrieve relevant context            |
| `POST /teach`         | Trigger self-improvement         | Continuous learning                  |
| `GET /system/info`    | Capabilities overview            | Feature discovery                    |
| `GET /teach/status`   | Learning progress                | Monitoring                           |

---

## 6. Enabling Advanced Features

### Real LLM (OpenAI / Groq)

```bash
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
```

The engine will automatically use real models when keys are present.

### Better Embeddings (sentence-transformers + FAISS)

```bash
pip install sentence-transformers faiss-cpu
```

The engine will automatically use semantic embeddings and persistent FAISS index.

---

## 7. Deployment Workflows

### Docker Compose (Recommended for most users)

```bash
cd deployment
./start.sh
```

This starts:
- FastAPI server on port 8000
- Neo4j on ports 7474 / 7687
- Persistent volumes for DB and FAISS index

### Kubernetes

```bash
kubectl apply -f deployment/k8s-deployment.yaml
```

Uses a **StatefulSet** with persistent storage. Suitable for multi-replica deployments.

---

## 8. Best Practices for New Users

1. **Start simple** — Use `cli.py pog "question"` first.
2. **Ingest knowledge early** — The more relevant documents you add, the better the reasoning.
3. **Use PoG for complex questions** — Simple questions can use basic `hybrid_similarity`, but complex ones benefit greatly from `pog_plan_and_reason`.
4. **Monitor learning** — Regularly check `engine.learning_metrics` or `/teach/status`.
5. **Use the API for production** — The FastAPI server is the recommended way to integrate the engine into applications.
6. **Persist your data** — Use Docker volumes or Kubernetes PVCs so the FAISS index and learned synonyms survive restarts.

---

## 9. Example End-to-End Session

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine

# 1. Initialize
engine = ProductionAdaptiveEngine("Build autonomous AI systems")

# 2. Ingest domain knowledge
engine.ingest_documents([
    "Self-teaching loops allow agents to improve without human intervention.",
    "Polymorphic execution enables dynamic strategy selection at runtime."
])

# 3. Ask a complex question using PoG
result = engine.pog_plan_and_reason(
    "How can we design an AI system that continuously improves its reasoning?",
    max_hops=4
)
print(result["result"])

# 4. Trigger self-teaching
engine.self_teaching_loop(background=True, max_iterations=3)

# 5. Check what it learned
print(engine.learning_metrics)
```

---

## Need Help?

- Run `python cli.py --help`
- Visit Swagger UI at `http://localhost:8000/docs` when using the FastAPI server
- Check `/system/info` endpoint for available features

This engine is designed to be explored interactively. Start with the CLI and PoG queries — you'll quickly see the power of the hybrid symbolic + neural approach.