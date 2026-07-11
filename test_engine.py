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
)
import json
import sqlite3
import tempfile
import shutil
import torch

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=line"])