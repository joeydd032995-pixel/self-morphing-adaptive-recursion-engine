"""
Semantic caching layer for expensive, side-effect-free engine calls
(llm_call, semantic_retrieve_context).

Two-tier lookup: an exact-hash fast path (identical query_text + params always
hits), then a brute-force cosine similarity scan over recent entries of the
same cache_kind. Deliberately conservative by default (high similarity
threshold) — a wrong hit silently returns a stale/wrong answer, which is
worse than a cache miss.

Kept as a standalone module (not methods on ProductionAdaptiveEngine) so the
cache is independently testable and doesn't grow the core engine file
further. Stored in its own SQLite table — mixing cached-query vectors into
the RAG document-chunk FAISS index would corrupt document retrieval.
"""

import sqlite3
import json
import hashlib
import datetime

import numpy as np


class SemanticCache:
    def __init__(self, db_path, ttl_seconds=3600, threshold=92.0):
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS semantic_cache (
                query_hash TEXT PRIMARY KEY,
                cache_kind TEXT,
                query_text TEXT,
                embedding BLOB,
                response TEXT,
                created_at TEXT,
                expires_at TEXT,
                hit_count INTEGER DEFAULT 0
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_semantic_cache_kind ON semantic_cache(cache_kind)')
        conn.commit()
        conn.close()

    @staticmethod
    def _key(cache_kind, query_text, params=None):
        payload = json.dumps({"kind": cache_kind, "text": query_text, "params": params or {}}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_kind, query_text, embedding, params=None):
        """Exact-hash fast path first (identical query_text + params always
        hits, regardless of the similarity threshold), else a brute-force
        cosine scan over rows of matching cache_kind; returns the best match
        if its score >= self.threshold, else None."""
        self.sweep_expired()
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        cursor = conn.cursor()

        key = self._key(cache_kind, query_text, params)
        row = cursor.execute(
            "SELECT response, hit_count FROM semantic_cache WHERE query_hash = ?", (key,)).fetchone()
        if row is not None:
            response, hit_count = row
            cursor.execute(
                "UPDATE semantic_cache SET hit_count = ? WHERE query_hash = ?", (hit_count + 1, key))
            conn.commit()
            conn.close()
            return json.loads(response)

        if embedding is None:
            conn.close()
            return None

        query_vec = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        best_score, best_hash, best_response = -1.0, None, None
        for row_hash, emb_blob, response in cursor.execute(
            "SELECT query_hash, embedding, response FROM semantic_cache WHERE cache_kind = ?", (cache_kind,)
        ):
            try:
                cand_vec = np.frombuffer(emb_blob, dtype=np.float32)
                if cand_vec.shape != query_vec.shape:
                    continue
                cand_norm = np.linalg.norm(cand_vec)
                if query_norm == 0 or cand_norm == 0:
                    continue
                score = 100.0 * float(np.dot(query_vec, cand_vec) / (query_norm * cand_norm))
            except Exception:
                continue
            if score > best_score:
                best_score, best_hash, best_response = score, row_hash, response

        if best_response is not None and best_score >= self.threshold:
            cursor.execute(
                "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE query_hash = ?", (best_hash,))
            conn.commit()
            conn.close()
            return json.loads(best_response)

        conn.close()
        return None

    def put(self, cache_kind, query_text, embedding, response, params=None, ttl_seconds=None):
        key = self._key(cache_kind, query_text, params)
        now = datetime.datetime.now()
        ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = (now + datetime.timedelta(seconds=ttl)).isoformat()
        emb_blob = np.asarray(embedding, dtype=np.float32).tobytes() if embedding is not None else None

        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('''
            INSERT OR REPLACE INTO semantic_cache
                (query_hash, cache_kind, query_text, embedding, response, created_at, expires_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT hit_count FROM semantic_cache WHERE query_hash = ?), 0))
        ''', (key, cache_kind, query_text, emb_blob, json.dumps(response), now.isoformat(), expires_at, key))
        conn.commit()
        conn.close()

    def invalidate(self, cache_kind=None):
        """Delete all entries, or only those of the given cache_kind."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        if cache_kind is None:
            conn.execute("DELETE FROM semantic_cache")
        else:
            conn.execute("DELETE FROM semantic_cache WHERE cache_kind = ?", (cache_kind,))
        conn.commit()
        conn.close()

    def sweep_expired(self):
        """Lazy expiry: called from get()/put() rather than a background thread."""
        now_iso = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute("DELETE FROM semantic_cache WHERE expires_at IS NOT NULL AND expires_at < ?", (now_iso,))
        conn.commit()
        conn.close()

    def stats(self):
        """Per-cache_kind row counts and cumulative hit counts."""
        self.sweep_expired()
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT cache_kind, COUNT(*), COALESCE(SUM(hit_count), 0) FROM semantic_cache GROUP BY cache_kind"
        ).fetchall()
        conn.close()
        return {kind: {"entries": count, "hits": hits} for kind, count, hits in rows}
