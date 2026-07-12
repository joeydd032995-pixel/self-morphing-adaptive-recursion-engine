"""
FastAPI Server for Self-Morphing Adaptive Recursion Engine
Production API with Swagger docs, API-key auth, structured logging, rate
limiting, non-blocking (threadpool-offloaded) engine calls, and a lifespan-managed
engine with graceful teardown.

Run with: uvicorn fastapi_server:app --host 0.0.0.0 --port 8000

Requirements: fastapi, uvicorn. Optional: prometheus-client (/metrics), slowapi
(rate limiting).
"""

import os
import sys
import time
import uuid
import logging
import sqlite3
import asyncio
import threading
import queue as thread_queue
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.security import APIKeyHeader
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organized_self_morphing_engine import (
    ProductionAdaptiveEngine,
    MorphicTextNode,
    FAISS_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    NEO4J_AVAILABLE,
    PYG_AVAILABLE,
)

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("morphic.api")

# API key auth — required, no default (fail fast if unset).
API_KEY = os.environ.get("ENGINE_API_KEY")
if not API_KEY:
    raise RuntimeError("ENGINE_API_KEY environment variable is required — set it before starting the server.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Shared ceiling for client-controlled PoG hop counts across /query,
# /pog/plan, and /ws/pog/plan — bounds how many locked-SQLite hops a single
# request can force.
MAX_POG_HOPS = 20

# Max seconds /ws/pog/plan waits for the next hop event before treating the
# generator as stalled and closing the connection.
WS_QUEUE_TIMEOUT_SECONDS = float(os.getenv("WS_QUEUE_TIMEOUT_SECONDS", "60"))

# Module-level engine, exposed via a dependency accessor so endpoints depend on
# get_engine() rather than the global directly — this is the seam a future
# connection pool / per-request engine would slot into.
engine = ProductionAdaptiveEngine(target_solution_text="General Reasoning", similarity_threshold=80.0)


def get_engine() -> ProductionAdaptiveEngine:
    """Dependency accessor for the engine (enables pooling/DI later)."""
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage engine lifecycle: optionally connect Neo4j at startup (when the
    driver + env are present) and close it gracefully on shutdown."""
    if NEO4J_AVAILABLE and os.getenv("NEO4J_URI"):
        try:
            engine.connect_neo4j()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Neo4j connect at startup failed: %s", e)
    logger.info("Engine ready (faiss=%s, st=%s, neo4j=%s, gnn=%s)",
                FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE, NEO4J_AVAILABLE, PYG_AVAILABLE)
    yield
    try:
        engine.close_neo4j()
    except Exception:
        pass


def _rate_limit_key(request: Request) -> str:
    """Rate-limit bucket: per API key when present, else per client address."""
    return request.headers.get("X-API-Key") or get_remote_address(request)


_RATE_LIMIT_UNITS_PER_MINUTE = {"second": 60, "minute": 1, "hour": 1 / 60}


def _parse_rate_limit(spec: str) -> int:
    """Parse a slowapi-style '120/minute' (or '/second', '/hour') spec into a
    per-minute integer, matching how slowapi itself interprets the same env
    var for the HTTP middleware. Defaults to 120 on any unexpected format."""
    try:
        count_str, _, unit = spec.partition("/")
        count = int(count_str)
        multiplier = _RATE_LIMIT_UNITS_PER_MINUTE[unit.strip().lower()]
        return max(1, round(count * multiplier))
    except (ValueError, KeyError):
        return 120


class _TokenBucket:
    """Simple in-process per-key token bucket. slowapi's HTTP middleware
    doesn't cover WebSocket connections, so /ws/pog/plan is rate-limited
    manually here — independent of slowapi, not a reuse of it. Not
    distributed; fine for the single-process deployment this engine
    currently targets."""

    def __init__(self, rate_per_minute: int, capacity: Optional[int] = None):
        self.rate_per_minute = rate_per_minute
        self.capacity = capacity or rate_per_minute
        self._buckets: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * (self.rate_per_minute / 60.0))
            if tokens < 1:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1, now)
            return True


ws_rate_limiter = _TokenBucket(rate_per_minute=_parse_rate_limit(os.getenv("RATE_LIMIT", "120/minute")))


app = FastAPI(
    title="Self-Morphing Adaptive Recursion Engine API",
    description="Hybrid neurosymbolic reasoning engine with PoG, RAG, KG, self-teaching, and polymorphic execution.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Rate limiting (optional): a global default limit via middleware, no per-route
# decorators required. No-op when slowapi is not installed.
if SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=_rate_limit_key,
                      default_limits=[os.getenv("RATE_LIMIT", "120/minute")])
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Return HTTP 429 when a client exceeds the configured rate limit."""
        return Response(content="Rate limit exceeded", status_code=429)

    app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Structured request logging: assign a request id, time the handler, and log
    method/path/status/latency. The request id is echoed in the response header."""
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info("rid=%s %s %s -> %s (%.1fms)",
                request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    return response


def verify_api_key(key: str = Depends(api_key_header)):
    """FastAPI dependency: reject the request unless X-API-Key matches ENGINE_API_KEY."""
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return key


# ---- Pydantic request/response models ----
class QueryRequest(BaseModel):
    query: str
    max_hops: Optional[int] = Field(default=3, ge=1, le=MAX_POG_HOPS)
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
    task_id: Optional[str] = None
    stop_reason: Optional[str] = None

class TeachRequest(BaseModel):
    background: Optional[bool] = True
    max_iterations: Optional[int] = 2

class MappingRequest(BaseModel):
    word: str
    concept_token: str

class ResolveRequest(BaseModel):
    task_id: str
    approve: bool = True

class AttachRequest(BaseModel):
    parent_id: str
    new_node_id: str
    key_phrase: str
    node_type: str = "tree"
    branch_type: str = "left"


# ---- Endpoints ----
@app.get("/", tags=["Health"])
async def root():
    """Basic service banner listing the engine's headline capabilities."""
    return {
        "message": "Self-Morphing Adaptive Recursion Engine API",
        "status": "running",
        "features": ["PoG Planning", "Advanced RAG", "KG + Neo4j", "Self-Teaching", "Polymorphic Execution"],
    }

@app.get("/health", tags=["Health"])
async def health():
    """Liveness/readiness probe endpoint."""
    return {"status": "healthy", "engine_initialized": True}

@app.post("/query", response_model=QueryResponse, tags=["Reasoning"])
async def reasoning_query(request: QueryRequest, api_key: str = Depends(verify_api_key),
                          eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Main reasoning endpoint. Supports hybrid + PoG mode."""
    try:
        if request.use_pog:
            pog_result = await run_in_threadpool(eng.pog_plan_and_reason, request.query, request.max_hops)
            return QueryResponse(
                result=pog_result["result"],
                confidence=pog_result["confidence"],
                verified=pog_result["verified"],
                details=pog_result,
            )
        score = await run_in_threadpool(eng.hybrid_similarity, request.query, eng.raw_target)
        return QueryResponse(
            result=f"Hybrid match score: {score:.1f}%",
            confidence=score,
            verified=score >= eng.threshold,
        )
    except Exception as e:
        logger.exception("query failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/pog/plan", response_model=PoGResponse, tags=["PoG Planning"])
async def pog_plan(request: QueryRequest, api_key: str = Depends(verify_api_key),
                   eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Dedicated PoG adaptive planning endpoint."""
    try:
        result = await run_in_threadpool(eng.pog_plan_and_reason, request.query, request.max_hops)
        return PoGResponse(**result)
    except Exception as e:
        logger.exception("pog_plan failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.websocket("/ws/pog/plan")
async def pog_plan_ws(websocket: WebSocket, eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Real-time PoG visibility: streams one JSON event per hop as PoG explores
    the knowledge graph (mirroring _pog_hop_generator's 'hop'/'reflection'
    events), then a final 'done' event with the same result shape POST
    /pog/plan returns. This endpoint is purely additive — /pog/plan is
    unchanged and still fully drains the generator internally.

    Auth is folded into the first JSON message ({"api_key": ..., "query": ...,
    "max_hops": ..., "as_of": ...}) rather than a query-string key, since a
    key in the URL would leak into access/proxy logs and browser history;
    WebSocket handshakes can't carry a custom X-API-Key header the way plain
    HTTP requests do. Close codes: 4400 malformed/missing first message or
    query, 4401 bad api_key, 4429 rate-limited.
    """
    await websocket.accept()

    try:
        first_message = await websocket.receive_json()
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=4400)
        return
    if not isinstance(first_message, dict):
        await websocket.close(code=4400)
        return

    if first_message.get("api_key") != API_KEY:
        await websocket.close(code=4401)
        return

    if not ws_rate_limiter.allow(first_message.get("api_key")):
        await websocket.close(code=4429)
        return

    query = first_message.get("query")
    if not query or not isinstance(query, str):
        await websocket.close(code=4400)
        return
    try:
        max_hops = int(first_message.get("max_hops", 3))
    except (TypeError, ValueError):
        max_hops = 3
    if not 1 <= max_hops <= MAX_POG_HOPS:
        await websocket.close(code=4400)
        return
    as_of = first_message.get("as_of")
    task_id = str(uuid.uuid4())[:8]

    q: "thread_queue.Queue" = thread_queue.Queue()
    stop_producing = threading.Event()

    def _produce():
        """Runs in a worker thread (via run_in_executor): drains the sync
        generator and relays each event onto the thread-safe queue. A plain
        queue.Queue is used (not asyncio.Queue) because it's the only safe
        way to hand events from a worker thread to the event loop's task.
        Checks stop_producing between hops (the natural pause point for a
        generator) so an abandoned connection stops requesting further hops;
        a hop already in flight when stop is requested still runs to
        completion since Python can't preempt a worker thread mid-call."""
        try:
            hop_iter = iter(eng._pog_hop_generator(query, max_hops, as_of, task_id))
            while not stop_producing.is_set():
                try:
                    event = next(hop_iter)
                except StopIteration:
                    break
                q.put(event)
        except Exception as e:
            logger.exception("pog_plan_ws generator failed")
            q.put({"type": "error", "task_id": task_id, "detail": str(e)})
        finally:
            q.put(None)  # sentinel

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _produce)

    try:
        while True:
            try:
                # q.get(block=True, timeout=...) bounds the worker-thread wait
                # itself (unlike wrapping an untimed q.get() in
                # asyncio.wait_for, which would abandon the async wait but
                # leave the executor thread blocked indefinitely).
                event = await asyncio.to_thread(q.get, True, WS_QUEUE_TIMEOUT_SECONDS)
            except thread_queue.Empty:
                logger.warning("pog_plan_ws: timed out waiting for hop event, closing (task_id=%s)", task_id)
                await websocket.close(code=1011)
                return
            if event is None:
                break
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    finally:
        stop_producing.set()
        try:
            await websocket.close()
        except Exception as e:
            logger.debug("pog_plan_ws: close on already-closed socket: %s", e)

@app.post("/rag/ingest", tags=["RAG"])
async def rag_ingest(request: IngestRequest, api_key: str = Depends(verify_api_key),
                     eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Ingest documents into advanced RAG + KG. Returns the real chunk count."""
    try:
        chunks = await run_in_threadpool(eng.ingest_documents, request.documents, request.strategy)
        return {"status": "success", "chunks_ingested": int(chunks or 0)}
    except Exception as e:
        logger.exception("rag_ingest failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/rag/retrieve", tags=["RAG"])
async def rag_retrieve(query: str, k: int = 5, api_key: str = Depends(verify_api_key),
                       eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Semantic retrieval from RAG store."""
    try:
        contexts = await run_in_threadpool(eng.semantic_retrieve_context, query, k)
        return {"query": query, "contexts": contexts, "count": len(contexts)}
    except Exception as e:
        logger.exception("rag_retrieve failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/teach", tags=["Self-Teaching"])
async def self_teach(request: TeachRequest, api_key: str = Depends(verify_api_key),
                     eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Trigger self-teaching loop (background or foreground)."""
    try:
        thread = await run_in_threadpool(
            eng.self_teaching_loop, request.background, request.max_iterations)
        return {
            "status": "started" if request.background else "completed",
            "thread_id": str(thread.ident) if thread else None,
            "metrics": eng.learning_metrics,
        }
    except Exception as e:
        logger.exception("self_teach failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/metrics", tags=["Observability"])
async def get_metrics(eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Prometheus metrics endpoint (text exposition format) for scraping.
    Unauthenticated by convention so Prometheus can scrape it; exposes only
    non-sensitive operational counters/gauges. Returns 501 if prometheus_client
    is not installed — use /metrics.json for the plain-JSON view."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501,
                            detail="prometheus_client not installed; use /metrics.json")
    registry = CollectorRegistry()
    snapshot = eng.get_metrics_snapshot()
    for name, value in snapshot.items():
        try:
            g = Gauge(f"morphic_{name}", f"Self-morphing engine metric: {name}", registry=registry)
            g.set(float(value))
        except Exception:
            continue
    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics.json", tags=["Observability"])
async def get_metrics_json(api_key: str = Depends(verify_api_key),
                           eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Plain-JSON metrics view (authenticated)."""
    try:
        return {
            "learning_metrics": eng.learning_metrics,
            "snapshot": eng.get_metrics_snapshot(),
            "synonym_count": len(eng.synonym_dictionary),
            "has_neo4j": getattr(eng, 'has_neo4j', False),
            "faiss_available": FAISS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        }
    except Exception as e:
        logger.exception("metrics.json failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/admin/resolve", tags=["Admin"])
async def admin_resolve(request: ResolveRequest, api_key: str = Depends(verify_api_key),
                        eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Admin intervention for halted tasks."""
    try:
        eng.admin_resolve_halt(request.task_id, approve=request.approve)
        return {"status": "resolved", "task_id": request.task_id, "approved": request.approve}
    except Exception as e:
        logger.exception("admin_resolve failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/admin/mappings", tags=["Admin"])
async def get_mappings(api_key: str = Depends(verify_api_key),
                       eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """View current synonym mappings."""
    return {"mappings": eng.admin_view_mappings()}

@app.post("/admin/mappings", tags=["Admin"])
async def edit_mapping(request: MappingRequest, api_key: str = Depends(verify_api_key),
                       eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Add or override a synonym mapping (JSON body)."""
    eng.admin_edit_mapping(request.word, request.concept_token)
    return {"status": "updated", "word": request.word, "token": request.concept_token}

@app.post("/graph/export", tags=["Observability"])
async def export_graph(api_key: str = Depends(verify_api_key),
                       eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Export current morphic graph to DOT format."""
    try:
        root = MorphicTextNode("API_Root", "api_export", "linear")
        filename = await run_in_threadpool(eng.export_graph_viz, root, "morphic_graph_api.dot")
        return {"status": "exported", "filename": filename}
    except Exception as e:
        logger.exception("export_graph failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/rag/stats", tags=["RAG"])
async def rag_stats(api_key: str = Depends(verify_api_key),
                    eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Get RAG + FAISS statistics."""
    def _stats():
        """Blocking DB read, offloaded to a worker thread to keep the loop free."""
        conn = sqlite3.connect(eng.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rag_chunks")
        chunk_count = cursor.fetchone()[0]
        conn.close()
        faiss_loaded = hasattr(eng, 'faiss_index') and eng.faiss_index is not None
        return {
            "rag_chunks": chunk_count,
            "faiss_index_loaded": faiss_loaded,
            "faiss_vectors": len(getattr(eng, 'faiss_contents', [])),
            "faiss_available": FAISS_AVAILABLE,
        }
    try:
        return await run_in_threadpool(_stats)
    except Exception as e:
        logger.exception("rag_stats failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/cache/stats", tags=["Cache"])
async def cache_stats(api_key: str = Depends(verify_api_key),
                      eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Semantic cache hit/entry counts per cache_kind (llm_call, rag_retrieve)."""
    if eng.semantic_cache is None:
        return {"enabled": False}
    try:
        stats = await run_in_threadpool(eng.semantic_cache.stats)
        return {"enabled": True, "by_kind": stats}
    except Exception as e:
        logger.exception("cache_stats failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/cache/clear", tags=["Cache", "Admin"])
async def cache_clear(cache_kind: Optional[str] = None, api_key: str = Depends(verify_api_key),
                      eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Manually invalidate the semantic cache. Omit cache_kind to clear everything,
    or pass 'llm_call' / 'rag_retrieve' to clear just that kind."""
    if eng.semantic_cache is None:
        return {"enabled": False, "cleared": False}
    try:
        await run_in_threadpool(eng.semantic_cache.invalidate, cache_kind)
        return {"enabled": True, "cleared": True, "cache_kind": cache_kind}
    except Exception as e:
        logger.exception("cache_clear failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/teach/status", tags=["Self-Teaching"])
async def teach_status(api_key: str = Depends(verify_api_key),
                       eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Get current self-teaching / learning status and metrics."""
    return {
        "learning_metrics": eng.learning_metrics,
        "synonym_count": len(eng.synonym_dictionary),
        "admin_queue_size": len(eng.admin_review_queue),
        "faiss_index_loaded": hasattr(eng, 'faiss_index') and eng.faiss_index is not None,
    }

@app.post("/nodes/attach", tags=["Graph Management"])
async def attach_node(request: AttachRequest, api_key: str = Depends(verify_api_key),
                      eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Dynamically attach a new node to an existing parent (with verification)."""
    try:
        parent = MorphicTextNode(request.parent_id, "parent_placeholder", "tree")
        new_node = MorphicTextNode(request.new_node_id, request.key_phrase, request.node_type)
        success = await run_in_threadpool(eng.attach_node, parent, new_node, request.branch_type)
        return {"success": success, "attached": request.new_node_id, "to": request.parent_id}
    except Exception as e:
        logger.exception("attach_node failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/admin/halts", tags=["Admin"])
async def list_pending_halts(api_key: str = Depends(verify_api_key),
                             eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """List all currently pending admin halts / review queue."""
    return {"pending_halts": eng.admin_review_queue}

@app.get("/system/info", tags=["System"])
async def system_info(api_key: str = Depends(verify_api_key),
                      eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Comprehensive system capabilities — flags reflect REAL runtime availability
    (installed optional deps and, for Neo4j, a live connection) rather than
    hardcoded values."""
    try:
        return {
            "engine_version": "2.1.0",
            "target": eng.raw_target,
            "threshold": eng.threshold,
            "features": {
                "polymorphic_morphing": True,
                "pog_planning": True,
                "persistent_faiss": FAISS_AVAILABLE,
                "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
                "neo4j_driver_installed": NEO4J_AVAILABLE,
                "neo4j_connected": bool(getattr(eng, "has_neo4j", False)),
                "gnn_propagation": PYG_AVAILABLE,
                "prometheus_metrics": PROMETHEUS_AVAILABLE,
                "rate_limiting": SLOWAPI_AVAILABLE,
                "self_teaching": True,
                "fastapi_api": True,
            },
            "persistence": {
                "sqlite": True,
                "faiss_persistent_index": FAISS_AVAILABLE,
                "neo4j_optional": True,
            },
        }
    except Exception as e:
        logger.exception("system_info failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/graph/visualize", tags=["Observability"])
async def visualize_graph(api_key: str = Depends(verify_api_key),
                          eng: ProductionAdaptiveEngine = Depends(get_engine)):
    """Generate and return the current morphic graph in DOT format."""
    try:
        root = MorphicTextNode("Root", "system_graph", "linear")
        await run_in_threadpool(eng.export_graph_viz, root, "morphic_graph_latest.dot")
        with open("morphic_graph_latest.dot", "r") as f:
            dot_content = f.read()
        return {"dot": dot_content, "format": "graphviz"}
    except Exception as e:
        logger.exception("visualize_graph failed")
        raise HTTPException(status_code=500, detail="Internal server error") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
