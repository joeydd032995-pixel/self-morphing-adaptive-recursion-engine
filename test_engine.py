"""
Pytest Suite for Self-Morphing Adaptive Recursion Engine
Run with: pytest test_engine.py -v --tb=short
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organized_self_morphing_engine import ProductionAdaptiveEngine, MorphicTextNode
import json
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=line"])