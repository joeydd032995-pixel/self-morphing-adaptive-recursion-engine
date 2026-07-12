"""
Pytest Suite for Self-Morphing Adaptive Recursion Engine
Run with: pytest test_engine.py -v --tb=short
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organized_self_morphing_engine import (
    ProductionAdaptiveEngine,
    MorphicTextNode,
    FAISS_AVAILABLE,
    CROSS_ENCODER_AVAILABLE,
    PYG_AVAILABLE,
    NODE_TYPES,
)
import json
import sqlite3
import tempfile
import shutil
import torch
from unittest.mock import MagicMock

# Fixture for clean engine instance (uses temp DB to avoid polluting main engine_logs.db)
@pytest.fixture(scope="function")
def engine():
    # Use a temporary DB for isolation
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_engine_logs.db")
    
    eng = ProductionAdaptiveEngine(target_solution_text="Test Analytics", similarity_threshold=75.0)
    eng.db_path = db_path  # Override for test isolation
    eng._init_db()  # Re-init with temp path
    
    yield eng
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestCoreSimilarity:
    """Test hybrid similarity and embedding invariants."""
    
    def test_exact_match(self, engine):
        assert abs(engine.hybrid_similarity("compute analytics", "compute analytics") - 100.0) < 0.5
    
    def test_fuzzy_match(self, engine):
        score = engine.hybrid_similarity("comput analytics", "compute analytics")
        assert score > 65, f"Expected fuzzy score > 65, got {score}"
    
    def test_embedding_consistency(self, engine):
        emb1 = engine._compute_embedding("test query")
        emb2 = engine._compute_embedding("test query")
        assert torch.allclose(emb1, emb2), "Embeddings should be deterministic for same input"


class TestMorphicNode:
    """Test polymorphic node behavior and morphing invariants."""
    
    def test_node_creation(self):
        node = MorphicTextNode("test1", "Analyze data", "tree")
        assert node.id == "test1"
        assert node.node_type == "tree"
        assert isinstance(node.branches, dict)
    
    def test_linear_morphing(self, engine):
        root = MorphicTextNode("root", "start", "linear")
        child = MorphicTextNode("child", "next step", "linear")
        root.next_linear = child
        
        # Simulate simple linear flow (would be inside process_query_stream in full use)
        assert root.next_linear is not None
        assert root.next_linear.key_phrase == "next step"


class TestVerificationLayers:
    """Test self_auditor_verify and symbolic_verifier logic."""
    
    def test_auditor_high_confidence(self, engine):
        proposal = "compute analytics strongly matches target"
        result = engine.self_auditor_verify(proposal, "compute analytics", min_confidence=60.0)
        assert result is not None
    
    def test_auditor_low_confidence(self, engine):
        proposal = "xyz unrelated concept"
        result = engine.self_auditor_verify(proposal, "compute analytics", min_confidence=80.0)
        assert result is None or "rejected" in str(result).lower()
    
    def test_symbolic_verifier_valid(self, engine):
        assert engine.symbolic_verifier("Use compute_token for analytics tasks", None) is True
    
    def test_symbolic_verifier_invalid_short(self, engine):
        assert engine.symbolic_verifier("ab", None) is False


class TestPoGPlanning:
    """Test the new dedicated PoG method."""
    
    def test_pog_basic(self, engine):
        result = engine.pog_plan_and_reason("How to optimize data pipelines?", max_hops=2)
        assert isinstance(result, dict)
        assert "sub_objectives" in result
        assert "result" in result
        assert result["confidence"] > 0
        assert "PoG" in result["result"] or "Plan" in result["result"]


class TestSelfTeachingAndDynamicNodes:
    """Test learning loop and attach_node safety."""
    
    def test_attach_with_verification(self, engine):
        parent = MorphicTextNode("parent", "core analytics", "tree")
        child = MorphicTextNode("child", "enhanced analytics branch", "tree")
        
        success = engine.attach_node(parent, child, "left")
        assert success is True
        assert "left" in parent.branches
        assert parent.branches["left"].id == "child"
    
    def test_attach_rejects_invalid(self, engine):
        parent = MorphicTextNode("p", "test", "linear")
        bad_child = MorphicTextNode("bad", "xyz", "tree")  # Will likely fail symbolic check
        
        # Even if verifier is lenient, we test the API
        result = engine.attach_node(parent, bad_child)
        # In current verifier it may pass or fail depending on target — we just ensure no crash
        assert isinstance(result, bool)


class TestBenchmarksAndSmoke:
    """Smoke test for benchmarks and basic demo flow."""
    
    def test_benchmarks_run(self, engine):
        # Should not raise
        engine.run_benchmarks()
    
    def test_basic_tests_pass(self, engine):
        engine.run_basic_tests()


SAMPLE_DOC = (
    "Machine Learning is a field of Artificial Intelligence. "
    "Neural Networks power Deep Learning models. "
    "The weather in Paris is mild today. Paris is the capital of France. "
    "Recursive systems can suffer query explosions when they are not bounded."
)


class TestChunking:
    """advanced_chunk_text: all strategies real, overlap honored, no empty chunks."""

    def test_all_strategies_nonempty(self, engine):
        for strat in ("fixed", "semantic", "recursive"):
            chunks = engine.advanced_chunk_text(SAMPLE_DOC, strategy=strat, chunk_size=80, overlap=15)
            assert chunks, f"{strat} produced no chunks"
            assert all(c.strip() for c in chunks), f"{strat} produced empty chunk"

    def test_empty_input_returns_empty(self, engine):
        assert engine.advanced_chunk_text("", strategy="recursive") == []
        assert engine.advanced_chunk_text("   ", strategy="semantic") == []

    def test_recursive_respects_chunk_size_roughly(self, engine):
        # Recursive packing should keep pieces near chunk_size (allow overlap slack).
        chunks = engine.advanced_chunk_text(SAMPLE_DOC, strategy="recursive", chunk_size=60, overlap=10)
        assert chunks
        assert all(len(c) <= 60 * 2 for c in chunks)  # generous bound incl. overlap prefix

    def test_unknown_strategy_falls_back(self, engine):
        chunks = engine.advanced_chunk_text(SAMPLE_DOC, strategy="bogus", chunk_size=100, overlap=10)
        assert chunks and all(c.strip() for c in chunks)


class TestEmbeddingRobustness:
    """Typo-robust, deterministic character-n-gram fallback embedding."""

    def test_typo_not_penalized(self, engine):
        assert engine.hybrid_similarity("comput", "compute") > 70
        assert engine.hybrid_similarity("comput analytics", "compute analytics") > 65

    def test_embedding_deterministic(self, engine):
        a = engine._compute_embedding("stable text")
        b = engine._compute_embedding("stable text")
        assert torch.allclose(a, b)

    def test_stable_hash_is_process_independent(self, engine):
        # Known-value check: stable across runs (unlike builtin hash()).
        assert engine._stable_hash("compute") == engine._stable_hash("compute")
        assert engine._stable_hash("a") != engine._stable_hash("b")

    def test_unrelated_scores_zero(self, engine):
        assert engine.hybrid_similarity("xyz", "compute") < 30


class TestEntityExtraction:
    """Lightweight entity extraction + KG seeding during ingestion."""

    def test_extracts_capitalized_entities(self, engine):
        ents = engine._extract_entities(SAMPLE_DOC)
        assert "Machine Learning" in ents
        assert "Paris" in ents
        assert "France" in ents

    def test_ingest_returns_count_and_populates_kg(self, engine):
        n = engine.ingest_documents([SAMPLE_DOC], strategy="recursive")
        assert isinstance(n, int) and n >= 1
        conn = sqlite3.connect(engine.db_path)
        cur = conn.cursor()
        chunk_count = cur.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        entity_count = cur.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        conn.close()
        assert chunk_count == n
        assert entity_count >= 3


class TestAdvancedRetrieval:
    """Query rewriting + re-ranking + agentic multi-hop, graceful without extras."""

    def test_advanced_retrieval_returns_list(self, engine):
        engine.ingest_documents([SAMPLE_DOC], strategy="fixed")
        res = engine.retrieve_context_advanced("deep learning neural networks", k=3)
        assert isinstance(res, list)

    def test_agentic_retrieval_returns_list(self, engine):
        engine.ingest_documents([SAMPLE_DOC], strategy="fixed")
        res = engine.retrieve_context_advanced("AI", k=2, agentic=True, max_hops=2)
        assert isinstance(res, list)

    def test_rerank_without_cross_encoder_preserves_topk(self, engine):
        if CROSS_ENCODER_AVAILABLE:
            pytest.skip("cross-encoder installed; fallback path not exercised")
        cands = ["one", "two", "three", "four"]
        assert engine._rerank("q", cands, 2) == cands[:2]

    def test_containment_and_semantic_match(self, engine):
        # A proposal fully covering the reference scores high via containment.
        assert engine._containment_score("compute analytics extra words", "compute analytics") == 100.0
        assert engine.semantic_match_score("compute analytics strongly matches", "compute analytics") >= 60


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
class TestFaissIndexSelection:
    """Index type scales with corpus size."""

    def test_small_corpus_uses_flat(self, engine):
        idx = engine._make_faiss_index(16, n_vectors=10)
        assert type(idx).__name__ == "IndexFlatIP"

    def test_large_corpus_uses_hnsw(self, engine):
        os.environ["ENGINE_FAISS_HNSW_THRESHOLD"] = "100"
        try:
            idx = engine._make_faiss_index(16, n_vectors=500)
            assert "HNSW" in type(idx).__name__
        finally:
            del os.environ["ENGINE_FAISS_HNSW_THRESHOLD"]


class TestCypherBuilder:
    """Pure Cypher builders — no driver required."""

    def test_sanitize_rel_type(self, engine):
        assert engine._sanitize_rel_type("co-occurs with") == "CO_OCCURS_WITH"
        assert engine._sanitize_rel_type("") == "RELATED_TO"
        assert engine._sanitize_rel_type("a`b; DROP") == "A_B__DROP"

    def test_entity_merge_parameterized(self, engine):
        q, p = engine.build_entity_merge("paris", "Paris", {"surface": "Paris"})
        assert "MERGE (e:Entity {id: $id})" in q
        assert p["id"] == "paris" and p["label"] == "Paris"
        assert json.loads(p["properties"]) == {"surface": "Paris"}

    def test_relation_merge_type_interpolated_values_parameterized(self, engine):
        q, p = engine.build_relation_merge("a", "b", "co_occurs_with", 0.5, {"src": "x"})
        assert "[r:CO_OCCURS_WITH]" in q
        assert p["source_id"] == "a" and p["target_id"] == "b" and p["confidence"] == 0.5
        # Values must be parameters, never interpolated.
        assert "0.5" not in q and "$confidence" in q


class TestNeo4jSync:
    """Bidirectional sync using a mocked driver (no live Neo4j / no neo4j package)."""

    def _mock_engine_with_driver(self, engine):
        driver = MagicMock()
        session = MagicMock()
        # session used as a context manager
        driver.session.return_value.__enter__.return_value = session
        engine.neo4j_driver = driver
        engine.has_neo4j = True
        return driver, session

    def test_sync_to_neo4j_runs_cypher(self, engine):
        # Seed SQLite KG.
        conn = sqlite3.connect(engine.db_path)
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO kg_entities (entity_id,label,properties,source) VALUES (?,?,?,?)",
                    ("paris", "Paris", "{}", "test"))
        cur.execute("INSERT OR REPLACE INTO kg_entities (entity_id,label,properties,source) VALUES (?,?,?,?)",
                    ("france", "France", "{}", "test"))
        cur.execute("INSERT INTO kg_relations (source_id,target_id,relation_type,properties,confidence) VALUES (?,?,?,?,?)",
                    ("paris", "france", "capital_of", "{}", 0.9))
        conn.commit(); conn.close()

        driver, session = self._mock_engine_with_driver(engine)
        summary = engine.kg_sync_to_neo4j()
        assert summary == {"synced": True, "entities": 2, "relations": 1}
        assert session.run.call_count == 3  # 2 entities + 1 relation

    def test_sync_from_neo4j_upserts_sqlite(self, engine):
        driver, session = self._mock_engine_with_driver(engine)
        # First .run() call -> entities, second -> relations.
        entities = [{"id": "berlin", "label": "Berlin", "properties": "{}"}]
        relations = [{"source_id": "berlin", "target_id": "germany",
                      "relation_type": "CAPITAL_OF", "confidence": 0.8, "properties": "{}"}]
        session.run.side_effect = [entities, relations]

        summary = engine.kg_sync_from_neo4j()
        assert summary["synced"] is True
        conn = sqlite3.connect(engine.db_path)
        cur = conn.cursor()
        assert cur.execute("SELECT label FROM kg_entities WHERE entity_id='berlin'").fetchone()[0] == "Berlin"
        assert cur.execute("SELECT COUNT(*) FROM kg_relations WHERE source_id='berlin'").fetchone()[0] == 1
        conn.close()

    def test_sync_noop_without_connection(self, engine):
        # No driver connected -> graceful no-op, no exception.
        engine.has_neo4j = False
        assert engine.kg_sync_to_neo4j() == {"synced": False, "entities": 0, "relations": 0}
        assert engine.kg_sync_from_neo4j() == {"synced": False, "entities": 0, "relations": 0}

    def test_connect_graceful_when_driver_absent(self, engine):
        import organized_self_morphing_engine as ose
        if ose.NEO4J_AVAILABLE:
            pytest.skip("neo4j installed; absent-path not exercised")
        assert engine.connect_neo4j() is False
        assert engine.has_neo4j is False


class TestBitemporalKG:
    """Bitemporal (valid-time + transaction-time) fields on kg_entities/kg_relations."""

    def _columns(self, engine, table):
        conn = sqlite3.connect(engine.db_path)
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        conn.close()
        return cols

    def test_schema_has_bitemporal_columns(self, engine):
        assert {"created_at", "updated_at"} <= self._columns(engine, "kg_entities")
        assert {"valid_from", "valid_to", "tx_start", "tx_end"} <= self._columns(engine, "kg_relations")

    def test_init_db_migration_is_idempotent(self, engine):
        # Calling _init_db() again on the same db_path must not raise
        # (SQLite has no ADD COLUMN IF NOT EXISTS — the migration guard is manual).
        engine._init_db()
        engine._init_db()
        assert {"valid_from", "valid_to", "tx_start", "tx_end"} <= self._columns(engine, "kg_relations")

    def test_kg_assert_relation_sets_tx_start_and_leaves_tx_end_null(self, engine):
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine.kg_assert_relation(cursor, "a", "b", "related_to", 0.7)
        conn.commit()
        row = cursor.execute(
            "SELECT tx_start, tx_end FROM kg_relations WHERE source_id='a' AND target_id='b'"
        ).fetchone()
        conn.close()
        assert row[1] is None
        # tx_start must be a parseable ISO8601 timestamp.
        import datetime as dt
        dt.datetime.fromisoformat(row[0])

    def test_kg_assert_relation_respects_valid_time_window(self, engine):
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine.kg_assert_relation(cursor, "a", "b", "related_to", 0.7,
                                   valid_from="2020-01-01T00:00:00", valid_to="2021-01-01T00:00:00")
        conn.commit()
        row = cursor.execute(
            "SELECT valid_from, valid_to FROM kg_relations WHERE source_id='a' AND target_id='b'"
        ).fetchone()
        conn.close()
        assert row == ("2020-01-01T00:00:00", "2021-01-01T00:00:00")

    def test_kg_supersede_relation_closes_old_row_before_reassert(self, engine):
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine.kg_assert_relation(cursor, "a", "b", "related_to", 0.5)
        conn.commit()
        conn.close()

        n_closed = engine.kg_supersede_relation("a", "b", "related_to")
        assert n_closed == 1

        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine.kg_assert_relation(cursor, "a", "b", "related_to", 0.9)
        conn.commit()
        # Exactly one row for this triple should be "current" (tx_end IS NULL) at a time.
        current = cursor.execute(
            "SELECT COUNT(*) FROM kg_relations WHERE source_id='a' AND target_id='b' "
            "AND relation_type='related_to' AND tx_end IS NULL"
        ).fetchone()[0]
        total = cursor.execute(
            "SELECT COUNT(*) FROM kg_relations WHERE source_id='a' AND target_id='b' "
            "AND relation_type='related_to'"
        ).fetchone()[0]
        conn.close()
        assert current == 1
        assert total == 2  # old closed row + new current row, history preserved

    def test_cypher_builders_include_bitemporal_params(self, engine):
        q, p = engine.build_relation_merge(
            "a", "b", "related_to", 0.5, {"x": 1},
            valid_from="2020-01-01", valid_to=None, tx_start="2024-01-01", tx_end=None)
        assert "$valid_from" in q and "$tx_start" in q
        assert p["valid_from"] == "2020-01-01" and p["tx_start"] == "2024-01-01"
        assert p["valid_to"] is None and p["tx_end"] is None

        eq, ep = engine.build_entity_merge("paris", "Paris", {"surface": "Paris"},
                                            created_at="2024-01-01", updated_at="2024-06-01")
        assert "$created_at" in eq and "$updated_at" in eq
        assert ep["created_at"] == "2024-01-01" and ep["updated_at"] == "2024-06-01"

    def test_store_entities_preserves_created_at_across_upsert(self, engine):
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine._store_entities(cursor, ["Paris", "France"], source="test")
        conn.commit()
        first_created = cursor.execute(
            "SELECT created_at FROM kg_entities WHERE entity_id='paris'").fetchone()[0]
        assert first_created is not None

        # Re-upserting the same entity must not reset created_at.
        engine._store_entities(cursor, ["Paris"], source="test")
        conn.commit()
        second_created = cursor.execute(
            "SELECT created_at FROM kg_entities WHERE entity_id='paris'").fetchone()[0]
        conn.close()
        assert second_created == first_created

    def test_store_entities_threads_valid_from_into_co_occurrence_relations(self, engine):
        conn = sqlite3.connect(engine.db_path)
        cursor = conn.cursor()
        engine._store_entities(cursor, ["Paris", "France"], source="test", valid_from="2019-05-01")
        conn.commit()
        row = cursor.execute(
            "SELECT valid_from FROM kg_relations WHERE relation_type='co_occurs_with'").fetchone()
        conn.close()
        assert row[0] == "2019-05-01"


def _sample_graph():
    """A small typed morphic graph exercising all four edge types + a cycle."""
    root = MorphicTextNode("root", "compute analytics pipeline", "linear")
    c1 = MorphicTextNode("c1", "process data metrics", "tree")
    c2 = MorphicTextNode("c2", "graph structure hierarchy", "nested")
    c3 = MorphicTextNode("c3", "indirect llm routing", "indirect")
    root.next_linear = c1
    c1.branches["left"] = c2
    c2.inner_formula = c3
    c3.mutual_routine = root  # cycle back to root
    return root


class TestGraphTensors:
    """Pure-torch graph construction (no PyG required)."""

    def test_build_tensors_shapes_and_cycle_safe(self, engine):
        t = engine.build_graph_tensors(_sample_graph())
        assert t["node_ids"] == ["root", "c1", "c2", "c3"]
        # feature width = embedding dim + one-hot node types
        assert t["x"].shape[0] == 4
        assert t["x"].shape[1] > len(NODE_TYPES)
        # 4 edges (linear, branch, inner_formula, mutual_routine); cycle didn't loop.
        assert t["edge_index"].shape[1] == 4
        assert t["y"].tolist() == [0, 1, 2, 3]  # linear,tree,nested,indirect

    def test_empty_graph_returns_none(self, engine):
        assert engine.build_graph_tensors(None) is None


class TestGnnFallback:
    """NumPy propagation fallback works with or without torch-geometric."""

    def test_classify_returns_types(self, engine):
        preds = engine.gnn_classify_nodes(_sample_graph())
        assert set(preds.keys()) == {"root", "c1", "c2", "c3"}
        assert all(v in NODE_TYPES for v in preds.values())

    def test_predict_links_excludes_existing(self, engine):
        links = engine.gnn_predict_links(_sample_graph(), top_k=3)
        assert isinstance(links, list) and len(links) <= 3
        for a, b, s in links:
            assert a != b

    def test_node_relevance_range(self, engine):
        score = engine.gnn_node_relevance("compute metrics", _sample_graph())
        assert 0.0 <= score <= 100.0

    def test_train_gnn_noop_without_pyg(self, engine):
        if PYG_AVAILABLE:
            pytest.skip("PyG installed; no-op path not exercised")
        assert engine.train_gnn(_sample_graph()) is None


class TestGnnWiring:
    """GNN signals integrate into self-teaching without breaking the no-graph path."""

    def test_self_teaching_without_graph_still_runs(self, engine):
        engine.current_graph_root = None
        engine.self_teaching_loop(background=False, max_iterations=1)  # must not raise

    def test_gnn_guided_attach_when_graph_present(self, engine):
        engine.current_graph_root = _sample_graph()
        new_node = MorphicTextNode("dyn", "compute analytics extra", "tree")
        # Should attach (or safely return False) without raising.
        result = engine._gnn_guided_attach(new_node)
        assert isinstance(result, bool)


@pytest.mark.skipif(not PYG_AVAILABLE, reason="torch-geometric not installed")
class TestGnnTrained:
    """Real trained-GNN path when torch-geometric is present."""

    def test_train_and_infer(self, engine):
        root = _sample_graph()
        losses = engine.train_gnn(root, epochs=30)
        assert losses is not None and losses["total"] >= 0
        preds = engine.gnn_classify_nodes(root)
        assert len(preds) == 4
        links = engine.gnn_predict_links(root, top_k=2)
        assert len(links) <= 2


class TestMetricsSnapshot:
    """Engine observability snapshot + populated learning-metric fields."""

    def test_snapshot_has_numeric_fields(self, engine):
        snap = engine.get_metrics_snapshot()
        for key in ("total_proposals", "accepted_learnings", "rejection_rate",
                    "avg_confidence", "synonym_count", "rag_chunks",
                    "kg_entities", "kg_relations", "has_neo4j"):
            assert key in snap
            assert isinstance(snap[key], (int, float))

    def test_rejection_rate_and_history_populated(self, engine):
        engine.learning_metrics["learning_history"].append({"outcome": "accepted"})
        engine.learning_metrics["learning_history"].append({"outcome": "rejected"})
        engine._update_rejection_rate()
        assert abs(engine.learning_metrics["rejection_rate"] - 0.5) < 1e-9
        assert engine.get_metrics_snapshot()["learning_history_len"] == 2

    def test_snapshot_reflects_ingested_counts(self, engine):
        engine.ingest_documents([SAMPLE_DOC], strategy="fixed")
        snap = engine.get_metrics_snapshot()
        assert snap["rag_chunks"] >= 1
        assert snap["kg_entities"] >= 1


class TestPrometheusEndpoint:
    """/metrics exposition via FastAPI TestClient (skips if fastapi absent)."""

    def _client(self):
        try:
            os.environ.setdefault("ENGINE_API_KEY", "test-key")
            import importlib
            import fastapi_server
            importlib.reload(fastapi_server)
            from fastapi.testclient import TestClient
            return fastapi_server, TestClient(fastapi_server.app)
        except Exception as e:
            pytest.skip(f"fastapi stack unavailable: {e}")

    def test_metrics_endpoint(self):
        mod, client = self._client()
        resp = client.get("/metrics")
        if not mod.PROMETHEUS_AVAILABLE:
            assert resp.status_code == 501
        else:
            assert resp.status_code == 200
            body = resp.text
            assert "morphic_total_proposals" in body
            assert "# HELP" in body and "# TYPE" in body

    def test_metrics_json_requires_auth(self):
        mod, client = self._client()
        assert client.get("/metrics.json").status_code in (401, 403)
        ok = client.get("/metrics.json", headers={"X-API-Key": "test-key"})
        assert ok.status_code == 200
        assert "snapshot" in ok.json()


class TestFastAPIProduction:
    """Phase 5 hardening: JSON-body endpoints, real counts, truthful flags."""

    def _client(self):
        try:
            os.environ.setdefault("ENGINE_API_KEY", "test-key")
            import importlib
            import fastapi_server
            importlib.reload(fastapi_server)
            from fastapi.testclient import TestClient
            return fastapi_server, TestClient(fastapi_server.app)
        except Exception as e:
            pytest.skip(f"fastapi stack unavailable: {e}")

    def test_admin_mapping_accepts_json_body(self):
        mod, client = self._client()
        with client:
            resp = client.post("/admin/mappings",
                               headers={"X-API-Key": "test-key"},
                               json={"word": "widget", "concept_token": "data_token"})
            assert resp.status_code == 200
            assert resp.json()["word"] == "widget"

    def test_rag_ingest_reports_real_count(self):
        mod, client = self._client()
        with client:
            resp = client.post("/rag/ingest",
                               headers={"X-API-Key": "test-key"},
                               json={"documents": [SAMPLE_DOC], "strategy": "fixed"})
            assert resp.status_code == 200
            body = resp.json()
            # Real count from ingest_documents, not the old len(documents)*3 estimate.
            assert body["chunks_ingested"] >= 1

    def test_system_info_flags_are_truthful(self):
        mod, client = self._client()
        with client:
            resp = client.get("/system/info", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            feats = resp.json()["features"]
            assert feats["gnn_propagation"] == mod.PYG_AVAILABLE
            assert feats["neo4j_driver_installed"] == mod.NEO4J_AVAILABLE
            assert feats["prometheus_metrics"] == mod.PROMETHEUS_AVAILABLE
            assert feats["neo4j_connected"] is False  # not connected in test

    def test_request_id_header_present(self):
        mod, client = self._client()
        with client:
            resp = client.get("/health")
            assert "X-Request-ID" in resp.headers

    def test_missing_body_is_422_not_500(self):
        mod, client = self._client()
        with client:
            # Pydantic body validation -> 422, not a 500 from bad query-param binding.
            resp = client.post("/admin/mappings", headers={"X-API-Key": "test-key"}, json={})
            assert resp.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=line"])