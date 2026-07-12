# Getting Started in 5 Minutes

Welcome! This guide will get you running the **Self-Morphing Adaptive Recursion Engine** in under 5 minutes.

## Prerequisites

- Python 3.11+
- Git (optional)

---

> **No API keys needed to start.** The CLI (Option 1) and Python usage (Option 2) work
> out of the box with a deterministic mock LLM. Set `OPENAI_API_KEY` or `GROQ_API_KEY`
> (see [`.env.example`](.env.example)) only if you want real PoG planning, self-teaching,
> and advanced RAG reasoning instead of the mock. The API server (Option 3) additionally
> requires `ENGINE_API_KEY`.

## Option 1: CLI (Fastest - Recommended)

```bash
# 1. Clone or download the project and go to the folder
cd /path/to/artifacts

# 2. Install dependencies
pip install -r deployment/requirements.txt

# 3. Ask a complex question using PoG planning
python cli.py pog "How can AI systems continuously improve themselves?"

# 4. Run the full demo
python cli.py demo
```

**Time: ~2 minutes**

---

## Option 2: Python (Interactive)

```bash
cd /path/to/artifacts
pip install -r deployment/requirements.txt

python3
```

Then paste this:

```python
from organized_self_morphing_engine import ProductionAdaptiveEngine

engine = ProductionAdaptiveEngine("Build intelligent systems")

# PoG Reasoning
result = engine.pog_plan_and_reason(
    "What are the benefits of polymorphic execution?", 
    max_hops=2
)
print(result["result"])
print(f"Confidence: {result['confidence']:.1f}%")

# Check what it learned
print(engine.learning_metrics)
```

**Time: ~3 minutes**

---

## Option 3: Docker (Zero Install)

```bash
cd deployment

# ENGINE_API_KEY is required - the API will not start without it
export ENGINE_API_KEY=your-secret-key

# One command to start everything
./start.sh
```

Then open:
- **API + Swagger UI**: http://localhost:8000/docs
- **Try a query** using the web interface or curl:

```bash
curl -X POST http://localhost:8000/pog/plan \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain self-teaching AI systems", "max_hops": 3}'
```

**Time: ~2 minutes** (after first build)

---

## What to Try Next (in the next 5 minutes)

### 1. Ingest Your Own Knowledge
```python
engine.ingest_documents([
    "Your first document about any topic.",
    "Another document with related information."
])

# Now ask questions about what you added
result = engine.pog_plan_and_reason("Question about your documents")
```

### 2. Trigger Self-Improvement
```python
engine.self_teaching_loop(background=True, max_iterations=2)
print(engine.learning_metrics)
```

### 3. Explore the Web Interface
Go to http://localhost:8000/docs and try the endpoints:
- `POST /pog/plan`
- `POST /rag/ingest`
- `GET /teach/status`

---

## Next Steps

- Read the full **[USAGE.md](USAGE.md)** for detailed workflows
- Check the **[README.md](README.md)** for architecture and advanced features
- Try the CLI commands: `python cli.py --help`

---

**You're now running a hybrid neurosymbolic reasoning engine with PoG planning, persistent vector search, and self-teaching capabilities.**

Enjoy exploring! 🚀

*Need help? Run `python cli.py metrics` or visit the Swagger UI.*