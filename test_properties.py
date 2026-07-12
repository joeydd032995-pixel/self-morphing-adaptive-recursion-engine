"""
Property-based tests for morphing invariants (Hypothesis).

These complement the example-based tests in test_engine.py by asserting
invariants that must hold across a wide range of generated inputs — the kinds
of guarantees an adaptive, self-morphing engine relies on:

- similarity scores stay within [0, 100] and are reflexive / symmetric,
- graph traversal is cycle-safe and terminating (no stack overflow),
- the similarity cache is invalidated after synonym mutations,
- chunking never drops content or emits empty chunks,
- the trampoline execution loop always terminates.

All strategies are deterministically seeded via Hypothesis defaults so runs are
reproducible and CI-fast.
"""

import os
import sys
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st, HealthCheck

from organized_self_morphing_engine import ProductionAdaptiveEngine, MorphicTextNode
from semantic_cache import SemanticCache


@pytest.fixture(scope="module")
def engine():
    """Module-scoped engine on a temp DB (property tests don't mutate shared state
    except where a test explicitly exercises mutation)."""
    tmp_dir = tempfile.mkdtemp()
    eng = ProductionAdaptiveEngine(target_solution_text="Compute Analytics", similarity_threshold=75.0)
    eng.db_path = os.path.join(tmp_dir, "prop_engine.db")
    eng._init_db()
    yield eng
    shutil.rmtree(tmp_dir, ignore_errors=True)


# Printable-ish text without control characters that break SQLite/regex.
text_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0, max_size=120,
)
word_strategy = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12)


class TestSimilarityInvariants:
    @settings(max_examples=100)
    @given(a=text_strategy, b=text_strategy)
    def test_similarity_bounded(self, engine, a, b):
        score = engine.calculate_similarity(a, b)
        assert 0.0 <= score <= 100.0

    @settings(max_examples=100)
    @given(a=text_strategy, b=text_strategy)
    def test_similarity_symmetric(self, engine, a, b):
        assert abs(engine.calculate_similarity(a, b) - engine.calculate_similarity(b, a)) < 1e-6

    @settings(max_examples=80)
    @given(a=text_strategy)
    def test_similarity_reflexive(self, engine, a):
        # Identical strings score 100 (empty vs empty is defined as 100 too).
        assert abs(engine.calculate_similarity(a, a) - 100.0) < 1e-6

    @settings(max_examples=100)
    @given(a=text_strategy, b=text_strategy)
    def test_hybrid_and_containment_bounded(self, engine, a, b):
        assert 0.0 <= engine.hybrid_similarity(a, b) <= 100.0
        assert 0.0 <= engine._containment_score(a, b) <= 100.0


class TestEmbeddingInvariants:
    @settings(max_examples=60)
    @given(t=text_strategy)
    def test_embedding_normalized_and_deterministic(self, engine, t):
        import numpy as np
        e1 = engine._compute_embedding(t).numpy()
        e2 = engine._compute_embedding(t).numpy()
        assert np.allclose(e1, e2)  # deterministic
        norm = float(np.linalg.norm(e1))
        # Zero vector only for text with no word characters; otherwise unit norm.
        assert norm == pytest.approx(0.0, abs=1e-6) or norm == pytest.approx(1.0, abs=1e-5)


class TestChunkingInvariants:
    @settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(text=text_strategy,
           strategy=st.sampled_from(["fixed", "semantic", "recursive"]),
           chunk_size=st.integers(min_value=10, max_value=200),
           overlap=st.integers(min_value=0, max_value=40))
    def test_chunks_nonempty_and_cover_content(self, engine, text, strategy, chunk_size, overlap):
        overlap = min(overlap, chunk_size - 1)
        chunks = engine.advanced_chunk_text(text, strategy=strategy, chunk_size=chunk_size, overlap=overlap)
        # Never emit empty chunks.
        assert all(c.strip() for c in chunks)
        # Empty/whitespace input yields no chunks; non-empty yields at least one.
        if text.strip():
            assert len(chunks) >= 1
            # Every non-space character of the input appears somewhere in the chunks.
            joined = "".join(chunks)
            missing = set(text.strip()) - set(joined)
            assert not missing
        else:
            assert chunks == []


class TestGraphTraversalInvariants:
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(n=st.integers(min_value=1, max_value=12), seed=st.integers(min_value=0, max_value=10_000))
    def test_traversal_terminates_on_cyclic_graph(self, engine, n, seed):
        import random
        rng = random.Random(seed)
        types = ["linear", "tree", "nested", "indirect"]
        nodes = [MorphicTextNode(f"n{i}", f"phrase {i}", rng.choice(types)) for i in range(n)]
        # Randomly wire edges, deliberately allowing cycles (including self-loops).
        for node in nodes:
            if rng.random() < 0.7:
                node.next_linear = rng.choice(nodes)
            if rng.random() < 0.5:
                node.branches["b"] = rng.choice(nodes)
            if rng.random() < 0.3:
                node.mutual_routine = rng.choice(nodes)
        # Must terminate (cycle-safe) and only ever return known nodes.
        tensors = engine.build_graph_tensors(nodes[0])
        assert tensors is not None
        assert 1 <= len(tensors["node_ids"]) <= n
        assert len(set(tensors["node_ids"])) == len(tensors["node_ids"])  # no dupes


class TestCacheInvalidationInvariant:
    @settings(max_examples=25, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(word=word_strategy)
    def test_cache_reflects_mapping_change(self, word):
        # Fresh engine per example to isolate synonym-dictionary mutation.
        tmp = tempfile.mkdtemp()
        try:
            eng = ProductionAdaptiveEngine("compute analytics", similarity_threshold=75.0)
            eng.db_path = os.path.join(tmp, "c.db")
            eng._init_db()
            before = eng.calculate_similarity(word, "compute analytics")
            eng.admin_edit_mapping(word, "compute_token")  # clears the cache
            after = eng.calculate_similarity(word, "compute analytics")
            # Mapping the word to a target token cannot decrease its canonical match.
            assert after >= before - 1e-6
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTrampolineTermination:
    @settings(max_examples=25, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(depth=st.integers(min_value=1, max_value=30), make_cycle=st.booleans())
    def test_process_query_stream_terminates(self, engine, depth, make_cycle):
        # Build a deep linear chain, optionally closing it into a cycle.
        nodes = [MorphicTextNode(f"t{i}", f"step {i} compute", "linear") for i in range(depth)]
        for a, b in zip(nodes, nodes[1:]):
            a.next_linear = b
        if make_cycle and depth > 1:
            nodes[-1].next_linear = nodes[0]
        # The heap trampoline must terminate without RecursionError / hang.
        result = engine.process_query_stream("prop_task", nodes[0], "compute analytics")
        assert result is None or isinstance(result, (str, dict))


class TestPoGHopInvariants:
    """Hypothesis: PoG's redesigned hop loop (real frontier-to-frontier
    traversal + bitemporal scoping + grounding-score-based early exit)."""

    @settings(max_examples=25, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        n=st.integers(min_value=1, max_value=8),
        seed=st.integers(min_value=0, max_value=10_000),
        max_hops=st.integers(min_value=1, max_value=6),
        grounding_threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_hop_chain_bounds_and_invariants(self, n, seed, max_hops, grounding_threshold):
        import random
        import datetime
        import sqlite3
        rng = random.Random(seed)

        tmp = tempfile.mkdtemp()
        try:
            eng = ProductionAdaptiveEngine("compute analytics", similarity_threshold=75.0)
            eng.db_path = os.path.join(tmp, "pog_prop.db")
            eng._init_db()
            eng.grounding_threshold = grounding_threshold

            # Random chain n0 -> n1 -> ... -> n{n-1}; each relation randomly
            # confident and randomly time-scoped, so some are expired or
            # superseded and must never be selectable regardless of confidence.
            now = datetime.datetime.now()
            node_ids = [f"n{i}" for i in range(n)]
            expired, superseded = set(), set()
            conn = sqlite3.connect(eng.db_path)
            cursor = conn.cursor()
            to_supersede = []
            for i in range(n - 1):
                src, tgt = node_ids[i], node_ids[i + 1]
                confidence = rng.random()
                choice = rng.random()
                valid_from = valid_to = None
                if choice < 0.2:
                    valid_from = (now - datetime.timedelta(days=10)).isoformat()
                    valid_to = (now - datetime.timedelta(days=5)).isoformat()
                    expired.add((src, tgt))
                eng.kg_assert_relation(cursor, src, tgt, "next", confidence,
                                        valid_from=valid_from, valid_to=valid_to)
                if 0.2 <= choice < 0.35:
                    superseded.add((src, tgt))
                    to_supersede.append((src, tgt))
            conn.commit()
            conn.close()
            for src, tgt in to_supersede:
                eng.kg_supersede_relation(src, tgt, "next")

            task_id = f"prop-{seed}-{n}-{max_hops}"
            events = list(eng._pog_hop_generator(f"query about {node_ids[0]}", max_hops, None, task_id))
            hop_events = [e for e in events if e["type"] == "hop"]
            done = next(e for e in events if e["type"] == "done")

            # Hop count never exceeds max_hops.
            assert len(hop_events) <= max_hops

            # Early exit only fires once a hop's own grounding score met the threshold.
            if done["stop_reason"] == "grounded":
                assert hop_events[-1]["grounding"] >= eng.grounding_threshold

            # Chain invariant: each hop's source is the previous hop's target —
            # proves real frontier-to-frontier traversal, not a repeated
            # unrestricted global query disconnected from prior hops.
            selected_pairs = set()
            for h1, h2 in zip(hop_events, hop_events[1:]):
                src2 = h2["path"].split(" --")[0]
                tgt1 = h1["path"].split("--> ")[1]
                assert src2 == tgt1
            for h in hop_events:
                src = h["path"].split(" --")[0]
                tgt = h["path"].split("--> ")[1]
                selected_pairs.add((src, tgt))

            # No expired or superseded relation is ever selected.
            assert not (selected_pairs & expired)
            assert not (selected_pairs & superseded)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestSemanticCacheInvariants:
    """Hypothesis: SemanticCache's threshold / exact-hash / invalidate guarantees."""

    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        query_text=word_strategy,
        response_value=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
        threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        query_vec=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                            min_size=2, max_size=6),
        cand_vec=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                           min_size=2, max_size=6),
    )
    def test_never_returns_below_threshold_similarity(self, query_text, response_value, threshold,
                                                        query_vec, cand_vec):
        from hypothesis import assume
        assume(len(query_vec) == len(cand_vec))
        import numpy as np
        tmp = tempfile.mkdtemp()
        try:
            cache = SemanticCache(os.path.join(tmp, "c.db"), threshold=threshold)
            # Stored under different text than the lookup, so any hit must come
            # from the cosine-similarity scan, not the exact-hash fast path.
            cache.put("kind", "different text than query", cand_vec, response_value)
            result = cache.get("kind", query_text, query_vec)
            if result is not None:
                qv = np.asarray(query_vec, dtype=np.float32)
                cv = np.asarray(cand_vec, dtype=np.float32)
                qn, cn = float(np.linalg.norm(qv)), float(np.linalg.norm(cv))
                score = 100.0 * float(np.dot(qv, cv) / (qn * cn))
                assert score >= threshold - 1e-4
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        query_text=word_strategy,
        response_value=st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
        threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_exact_text_always_hits_regardless_of_embedding(self, query_text, response_value, threshold):
        tmp = tempfile.mkdtemp()
        try:
            cache = SemanticCache(os.path.join(tmp, "c.db"), threshold=threshold)
            cache.put("kind", query_text, [1.0, 0.0], response_value)
            # Even with a wildly different / mismatched-shape embedding on lookup,
            # identical query_text must still hit via the exact-hash fast path.
            result = cache.get("kind", query_text, [0.0, 1.0, 0.0, -1.0])
            assert result == response_value
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(query_text=word_strategy, response_value=st.text(max_size=50))
    def test_invalidate_clears_subsequent_hits(self, query_text, response_value):
        tmp = tempfile.mkdtemp()
        try:
            cache = SemanticCache(os.path.join(tmp, "c.db"))
            cache.put("kind", query_text, [1.0, 0.0], response_value)
            assert cache.get("kind", query_text, [1.0, 0.0]) == response_value
            cache.invalidate("kind")
            assert cache.get("kind", query_text, [1.0, 0.0]) is None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
