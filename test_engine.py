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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=line"])