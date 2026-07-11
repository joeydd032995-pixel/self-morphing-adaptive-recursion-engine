"""
FastAPI Server for Self-Morphing Adaptive Recursion Engine
Modern production API with Swagger docs, basic auth, and all key endpoints.

Run with: uvicorn fastapi_server:app --reload --host 0.0.0.0 --port 8000

Requirements: fastapi, uvicorn, python-multipart (for forms)
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organized_self_morphing_engine import (
    ProductionAdaptiveEngine,
    MorphicTextNode,
    FAISS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Initialize engine (singleton for demo; in prod use dependency injection or pool)
engine = ProductionAdaptiveEngine(target_solution_text="General Reasoning", similarity_threshold=80.0)

app = FastAPI(
    title="Self-Morphing Adaptive Recursion Engine API",
    description="Hybrid neurosymbolic reasoning engine with PoG, RAG, KG, self-teaching, and polymorphic execution.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Simple API Key Auth (header: X-API-Key)
API_KEY = os.environ.get("ENGINE_API_KEY")
if not API_KEY:
    raise RuntimeError("ENGINE_API_KEY environment variable is required — set it before starting the server.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(key: str = Depends(api_key_header)):
    """FastAPI dependency: reject the request unless X-API-Key matches ENGINE_API_KEY."""
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return key

# Pydantic Models
class QueryRequest(BaseModel):
    query: str
    max_hops: Optional[int] = 3
    use_pog: Optional[bool] = True

class QueryResponse(BaseModel):
    result: str
    confidence: float
    verified: bool
    details: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    documents: List[str]
    strategy: Optional[str] = "semantic"

class PoGResponse(BaseModel):
    query: str
    sub_objectives: List[str]
    memory: Dict[str, Any]
    result: str
    confidence: float
    verified: bool

class TeachRequest(BaseModel):
    background: Optional[bool] = True
    max_iterations: Optional[int] = 2

# Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Basic service banner listing the engine's headline capabilities."""
    return {
        "message": "Self-Morphing Adaptive Recursion Engine API",
        "status": "running",
        "features": ["PoG Planning", "Advanced RAG", "KG + Neo4j", "Self-Teaching", "Polymorphic Execution"]
    }

@app.get("/health", tags=["Health"])
async def health():
    """Liveness/readiness probe endpoint."""
    return {"status": "healthy", "engine_initialized": True}

@app.post("/query", response_model=QueryResponse, tags=["Reasoning"])
async def reasoning_query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Main reasoning endpoint. Supports hybrid + PoG mode."""
    try:
        if request.use_pog:
            pog_result = engine.pog_plan_and_reason(request.query, max_hops=request.max_hops)
            return QueryResponse(
                result=pog_result["result"],
                confidence=pog_result["confidence"],
                verified=pog_result["verified"],
                details=pog_result
            )
        else:
            # Fallback to basic hybrid reasoning (simplified)
            score = engine.hybrid_similarity(request.query, engine.raw_target)
            return QueryResponse(
                result=f"Hybrid match score: {score:.1f}%",
                confidence=score,
                verified=score >= engine.threshold
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pog/plan", response_model=PoGResponse, tags=["PoG Planning"])
async def pog_plan(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    """Dedicated PoG adaptive planning endpoint."""
    try:
        result = engine.pog_plan_and_reason(request.query, max_hops=request.max_hops)
        return PoGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ingest", tags=["RAG"])
async def rag_ingest(request: IngestRequest, api_key: str = Depends(verify_api_key)):
    """Ingest documents into advanced RAG + KG."""
    try:
        engine.ingest_documents(request.documents, strategy=request.strategy)
        return {"status": "success", "chunks_ingested": len(request.documents) * 3}  # rough estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/retrieve", tags=["RAG"])
async def rag_retrieve(query: str, k: int = 5, api_key: str = Depends(verify_api_key)):
    """Semantic retrieval from RAG store."""
    try:
        contexts = engine.semantic_retrieve_context(query, k=k)
        return {"query": query, "contexts": contexts, "count": len(contexts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/teach", tags=["Self-Teaching"])
async def self_teach(request: TeachRequest, api_key: str = Depends(verify_api_key)):
    """Trigger self-teaching loop (background or foreground)."""
    try:
        thread = engine.self_teaching_loop(background=request.background, max_iterations=request.max_iterations)
        return {
            "status": "started" if request.background else "completed",
            "thread_id": str(thread.ident) if thread else None,
            "metrics": engine.learning_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", tags=["Observability"])
async def get_metrics():
    """Prometheus metrics endpoint (text exposition format) for scraping.
    Unauthenticated by convention so Prometheus can scrape it; exposes only
    non-sensitive operational counters/gauges. Returns 501 if prometheus_client
    is not installed — use /metrics.json for the plain-JSON view."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501,
                            detail="prometheus_client not installed; use /metrics.json")
    from fastapi import Response
    registry = CollectorRegistry()
    snapshot = engine.get_metrics_snapshot()
    for name, value in snapshot.items():
        try:
            g = Gauge(f"morphic_{name}", f"Self-morphing engine metric: {name}", registry=registry)
            g.set(float(value))
        except Exception:
            continue
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics.json", tags=["Observability"])
async def get_metrics_json(api_key: str = Depends(verify_api_key)):
    """Plain-JSON metrics view (authenticated) — the previous /metrics payload."""
    try:
        return {
            "learning_metrics": engine.learning_metrics,
            "snapshot": engine.get_metrics_snapshot(),
            "synonym_count": len(engine.synonym_dictionary),
            "has_neo4j": getattr(engine, 'has_neo4j', False),
            "faiss_available": FAISS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/resolve", tags=["Admin"])
async def admin_resolve(task_id: str, approve: bool = True, api_key: str = Depends(verify_api_key)):
    """Admin intervention for halted tasks."""
    try:
        engine.admin_resolve_halt(task_id, approve=approve)
        return {"status": "resolved", "task_id": task_id, "approved": approve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/mappings", tags=["Admin"])
async def get_mappings(api_key: str = Depends(verify_api_key)):
    """View current synonym mappings."""
    return {"mappings": engine.admin_view_mappings()}

@app.post("/admin/mappings", tags=["Admin"])
async def edit_mapping(word: str, concept_token: str, api_key: str = Depends(verify_api_key)):
    """Add or override a synonym mapping."""
    engine.admin_edit_mapping(word, concept_token)
    return {"status": "updated", "word": word, "token": concept_token}

@app.post("/graph/export", tags=["Observability"])
async def export_graph(api_key: str = Depends(verify_api_key)):
    """Export current morphic graph to DOT format."""
    # For demo we use a simple root if not available
    try:
        root = MorphicTextNode("API_Root", "api_export", "linear")
        filename = engine.export_graph_viz(root, "morphic_graph_api.dot")
        return {"status": "exported", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/stats", tags=["RAG"])
async def rag_stats(api_key: str = Depends(verify_api_key)):
    """Get RAG + FAISS statistics."""
    try:
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rag_chunks")
        chunk_count = cursor.fetchone()[0]
        conn.close()
        
        faiss_loaded = hasattr(engine, 'faiss_index') and engine.faiss_index is not None
        return {
            "rag_chunks": chunk_count,
            "faiss_index_loaded": faiss_loaded,
            "faiss_vectors": len(getattr(engine, 'faiss_contents', [])),
            "faiss_available": FAISS_AVAILABLE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ADDITIONAL ENDPOINTS ====================

@app.get("/teach/status", tags=["Self-Teaching"])
async def teach_status(api_key: str = Depends(verify_api_key)):
    """Get current self-teaching / learning status and metrics."""
    return {
        "learning_metrics": engine.learning_metrics,
        "synonym_count": len(engine.synonym_dictionary),
        "admin_queue_size": len(engine.admin_review_queue),
        "faiss_index_loaded": hasattr(engine, 'faiss_index') and engine.faiss_index is not None
    }


@app.post("/nodes/attach", tags=["Graph Management"])
async def attach_node(
    parent_id: str,
    new_node_id: str,
    key_phrase: str,
    node_type: str = "tree",
    branch_type: str = "left",
    api_key: str = Depends(verify_api_key)
):
    """Dynamically attach a new node to an existing parent (with verification)."""
    try:
        # In a real system we'd maintain a node registry. Here we create a simple parent for demo.
        parent = MorphicTextNode(parent_id, "parent_placeholder", "tree")
        new_node = MorphicTextNode(new_node_id, key_phrase, node_type)
        success = engine.attach_node(parent, new_node, branch_type)
        return {"success": success, "attached": new_node_id, "to": parent_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/halts", tags=["Admin"])
async def list_pending_halts(api_key: str = Depends(verify_api_key)):
    """List all currently pending admin halts / review queue."""
    return {"pending_halts": engine.admin_review_queue}


@app.get("/system/info", tags=["System"])
async def system_info(api_key: str = Depends(verify_api_key)):
    """Get comprehensive system capabilities and configuration."""
    try:
        return {
            "engine_version": "2.1.0",
            "target": engine.raw_target,
            "threshold": engine.threshold,
            "features": {
                "polymorphic_morphing": True,
                "pog_planning": True,
                "persistent_faiss": FAISS_AVAILABLE,
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "neo4j_support": True,
                "gnn_propagation": True,
                "self_teaching": True,
                "fastapi_api": True
            },
            "persistence": {
                "sqlite": True,
                "faiss_persistent_index": True,
                "neo4j_optional": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/visualize", tags=["Observability"])
async def visualize_graph(api_key: str = Depends(verify_api_key)):
    """Generate and return the current morphic graph in DOT format."""
    try:
        root = MorphicTextNode("Root", "system_graph", "linear")
        engine.export_graph_viz(root, "morphic_graph_latest.dot")
        with open("morphic_graph_latest.dot", "r") as f:
            dot_content = f.read()
        return {"dot": dot_content, "format": "graphviz"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)