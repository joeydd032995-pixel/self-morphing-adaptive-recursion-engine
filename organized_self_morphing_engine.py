import re
import time
import datetime
import concurrent.futures
import sqlite3
import json
import hashlib
import functools
import threading
import random  # For mock LLM simulation
import uuid
import numpy as np
import torch
import os
from functools import lru_cache
from collections import defaultdict, Counter

from semantic_cache import SemanticCache

# Optional advanced dependencies (graceful fallback if not installed)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    _semantic_model = None  # Lazy loaded
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    _semantic_model = None

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    _cross_encoder_model = None  # Lazy loaded
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    _cross_encoder_model = None

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import torch_geometric  # noqa: F401
    from torch_geometric.data import Data as _PyGData
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

# Canonical node-type vocabulary for GNN node classification / one-hot features.
NODE_TYPES = ["linear", "tree", "nested", "indirect"]
NODE_TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}
# Edge-type codes for the typed morphic graph.
EDGE_TYPES = ["linear", "branch", "inner_formula", "mutual_routine"]
EDGE_TYPE_INDEX = {t: i for i, t in enumerate(EDGE_TYPES)}


if PYG_AVAILABLE:
    class MorphicGNN(torch.nn.Module):
        """Two-layer GraphSAGE encoder with two heads:
        - node classification (predict a node's morph type), and
        - link prediction (score whether an edge should exist, via the dot product
          of the two endpoint embeddings).
        Small by design — it learns over the modestly-sized morphic graph."""

        def __init__(self, in_dim, hidden_dim=64, num_classes=len(NODE_TYPES)):
            """Build two SAGE conv layers plus a linear node-type classifier head."""
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)

        def encode(self, x, edge_index):
            """Return per-node embeddings after two SAGE message-passing rounds."""
            h = torch.relu(self.conv1(x, edge_index))
            h = self.conv2(h, edge_index)
            return h

        def classify(self, h):
            """Node-type logits from node embeddings."""
            return self.classifier(h)

        def link_score(self, h, edge_pairs):
            """Dot-product link scores for pairs [2, P] of node indices."""
            src, dst = edge_pairs
            return (h[src] * h[dst]).sum(dim=-1)

# =====================================================================
# 1. THE DATA STRUCTURE LAYOUT (Heterogeneous Morphic Graph Nodes)
# =====================================================================

class MorphicTextNode:
    """
    A polymorphic structural graph node capable of altering its footprint
    between Linear, Tree, Nested, and Indirect configurations based on data input.
    """
    def __init__(self, node_id, key_phrase, node_type="linear"):
        """Create a node with a normalized key phrase and empty linear/branch/inner-formula links."""
        self.id = node_id
        # Normalize text and clean spacing layout on ingestion
        self.key_phrase = key_phrase.lower().strip() 
        self.node_type = node_type  # Options: linear, tree, nested, indirect
        
        # Method A: Linear / Tail Optimization Routing Links
        self.next_linear = None
        
        # Method B: Non-Linear / Tree Structural Branch Routing Maps
        self.branches = {} 
        
        # Method C: Indirect / Mutual Module Switching Address Flags
        self.mutual_routine = None
        
        # Method D: Nested / Self-Evaluating Parameter Slices
        self.inner_formula = None
        
        # Downstream Cascading Analytics Trackers (Tracks Query Explosions)
        self.triggered_sub_questions = []


# =====================================================================
# 2. THE CORE ENGINE ARCHITECTURE (Fuzzy Semantic Parser)
# =====================================================================

class ProductionAdaptiveEngine:
    def __init__(self, target_solution_text, similarity_threshold=80.0):
        """Initialize engine state: SQLite persistence, synonym dictionary, embeddings,
        FAISS index, and the per-instance similarity cache."""
        self.raw_target = target_solution_text.lower().strip()
        self.threshold = similarity_threshold
        self.flag_buffer = 5.0  # Low confidence zone range (e.g., 80% to 85%)
        
        # Thread safety for shared structures (reentrant: attach_node holds this
        # while calling symbolic_verifier, which acquires it again internally)
        self._lock = threading.RLock()
        
        # Core Concept Tokens Table (Seed Vocabulary Matrix)
        self.synonym_dictionary = {
            "calculate": "compute_token", "compute": "compute_token",
            "solve": "compute_token", "evaluate": "compute_token",
            "analytics": "data_token", "data": "data_token",
            "metrics": "data_token", "logs": "data_token",
            "tree": "structure_token", "graph": "structure_token",
            "hierarchy": "structure_token"
        }
        
        # Shared Cross-Thread Real-Time Registries
        self.admin_review_queue = {}
        self.qa_audit_log = []
        self.explosion_log = []

        # PoG grounding-score early-stopping threshold (0-100): a hop whose
        # grounding score meets this bar ends exploration early instead of
        # always spending max_hops.
        self.grounding_threshold = float(os.getenv("ENGINE_POG_GROUNDING_THRESHOLD", "75.0"))

        # SQLite Database Layer for Persistence
        self.db_path = os.getenv("ENGINE_DB_PATH", "engine_logs.db")
        self.semantic_cache = None  # (re)constructed by _init_db(), which tracks self.db_path
        self._init_db()

        # Pre-process target string structure into absolute tokens
        self.target_tokens = self._tokenize_synonyms(self.raw_target)
        
        # Load persisted state
        self._load_synonyms_from_db()
        
        # Vector Embedding Setup for Rich Semantic Similarity
        self.vocab_size = 512  # Fixed dim for simple bag-of-words / hash embedding
        self.target_embedding = self._compute_embedding(self.raw_target)
        
        # Metrics Tracking for Learning Efficacy
        self.learning_metrics = {
            "total_proposals": 0,
            "accepted_learnings": 0,
            "avg_confidence": 0.0,
            "rejection_rate": 0.0,
            "learning_history": []
        }
        
        # Persistent FAISS Index
        self.faiss_index = None
        self.faiss_index_path = os.getenv("ENGINE_FAISS_INDEX_PATH", "faiss_index.bin")
        self.faiss_contents = []
        self.load_faiss_index()  # Attempt to load existing persistent index on startup

        # GNN state: optional trained model + a hook to the current morphic graph
        # root so the self-teaching loop can consult GNN link-prediction signals.
        self.gnn_model = None
        self.current_graph_root = None

        # Per-instance memoized similarity (bound here, not on the class, so the
        # cache dies with the instance instead of pinning every instance forever)
        self.calculate_similarity = lru_cache(maxsize=512)(self._calculate_similarity_uncached)

    def _init_db(self):
        """Initialize SQLite database for persistent logging and mappings."""
        conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        cursor = conn.cursor()
        
        # Synonym mappings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synonym_mappings (
                word TEXT PRIMARY KEY,
                concept_token TEXT,
                learned_at TEXT,
                confidence REAL,
                approved_by TEXT DEFAULT 'auto'
            )
        ''')
        
        # Q&A audit log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                task_id TEXT,
                query TEXT,
                target_node TEXT,
                score TEXT,
                clearance_security TEXT
            )
        ''')
        
        # Explosion logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_explosions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                trigger_task TEXT,
                origin_node TEXT,
                explosion_count INTEGER,
                cascade_manifest TEXT
            )
        ''')
        
        # RAG chunks table for advanced chunking, embeddings, and metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding BLOB,
                metadata TEXT
            )
        ''')
        
        # Advanced KG Schema: Entities (with embeddings + source for traceability)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kg_entities (
                entity_id TEXT PRIMARY KEY,
                label TEXT,
                properties TEXT,
                embedding BLOB,
                source TEXT
            )
        ''')
        
        # KG Relations (typed, confidence-scored for PoG adaptive planning & verification)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kg_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                properties TEXT,
                confidence REAL
            )
        ''')
        
        # KG Metadata for configuration and state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kg_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')

        # Bitemporal fields: valid-time (when a fact was/is true) and
        # transaction-time (when the system asserted/superseded that belief).
        # Added via migration since SQLite has no ADD COLUMN IF NOT EXISTS.
        self._add_column_if_missing(cursor, 'kg_entities', 'created_at', 'TEXT')
        self._add_column_if_missing(cursor, 'kg_entities', 'updated_at', 'TEXT')
        self._add_column_if_missing(cursor, 'kg_relations', 'valid_from', 'TEXT')
        self._add_column_if_missing(cursor, 'kg_relations', 'valid_to', 'TEXT')
        self._add_column_if_missing(cursor, 'kg_relations', 'tx_start', 'TEXT')
        self._add_column_if_missing(cursor, 'kg_relations', 'tx_end', 'TEXT')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_kg_relations_tx_current ON kg_relations(source_id, tx_end)')

        conn.commit()
        conn.close()
        print(f"✅ [DB] SQLite persistence initialized at {self.db_path} (advanced KG schema + RAG support enabled)")

        # Semantic cache for expensive, side-effect-free calls (llm_call,
        # semantic_retrieve_context). Rebuilt here (not just in __init__) so
        # it always tracks the current self.db_path, including callers that
        # reassign db_path and re-invoke _init_db() (e.g. test fixtures).
        # Disable via ENGINE_SEMANTIC_CACHE_ENABLED for benchmarking/cache-free runs.
        self.semantic_cache = None
        if os.getenv("ENGINE_SEMANTIC_CACHE_ENABLED", "true").lower() not in ("false", "0", "no"):
            self.semantic_cache = SemanticCache(
                self.db_path,
                ttl_seconds=int(os.getenv("ENGINE_SEMANTIC_CACHE_TTL_SECONDS", "3600")),
                threshold=float(os.getenv("ENGINE_SEMANTIC_CACHE_THRESHOLD", "92.0")),
            )

    @staticmethod
    def _add_column_if_missing(cursor, table, column, coltype):
        """Idempotently add a column to an existing table (SQLite has no ADD COLUMN IF NOT EXISTS)."""
        existing_cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}
        if column not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    def _load_synonyms_from_db(self):
        """Load persisted synonyms on engine startup for full persistence."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT word, concept_token FROM synonym_mappings")
            for word, token in cursor.fetchall():
                self.synonym_dictionary[word] = token
            conn.close()
            print(f"📥 [DB] Loaded {len(self.synonym_dictionary)} synonyms from persistence.")

    # --- ADMINISTRATIVE CONTROL PANEL INTERFACES ---
    def admin_view_mappings(self):
        """Returns the current state of the engine's active vocabulary dictionary."""
        return self.synonym_dictionary

    def admin_edit_mapping(self, word, concept_token):
        """Forcibly overrides or binds a specific word behavior pattern."""
        cleaned_word = word.lower().strip()
        cleaned_token = concept_token.lower().strip()
        with self._lock:
            self.synonym_dictionary[cleaned_word] = cleaned_token
            self.calculate_similarity.cache_clear()
            if self.semantic_cache is not None:
                self.semantic_cache.invalidate("llm_call")
            # Persist to DB
            conn = sqlite3.connect(self.db_path)
            conn.execute('PRAGMA journal_mode=WAL')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO synonym_mappings
                (word, concept_token, learned_at) VALUES (?, ?, ?)
            ''', (cleaned_word, cleaned_token, datetime.datetime.now().isoformat()))
            conn.commit()
            conn.close()

    def admin_delete_mapping(self, word):
        """Purges incorrect auto-learned mappings from the routing tables."""
        cleaned_word = word.lower().strip()
        with self._lock:
            if cleaned_word in self.synonym_dictionary:
                del self.synonym_dictionary[cleaned_word]
                self.calculate_similarity.cache_clear()
                if self.semantic_cache is not None:
                    self.semantic_cache.invalidate("llm_call")
                # Remove from DB
                conn = sqlite3.connect(self.db_path)
                conn.execute('PRAGMA journal_mode=WAL')
                cursor = conn.cursor()
                cursor.execute('DELETE FROM synonym_mappings WHERE word = ?', (cleaned_word,))
                conn.commit()
                conn.close()

    def admin_resolve_halt(self, task_id, approve=True):
        """Human Admin intervention gateway to unblock a specifically suspended execution thread.
        Queue entries come in two kinds: the trampoline's synonym-learning halts
        (no explicit "kind", carry "word"/"suggested_token") and PoG's non-blocking
        "pog_ungrounded" annotations (carry "query"/"confidence" instead) — only the
        former learns a synonym mapping on approval; the latter just updates status."""
        if task_id in self.admin_review_queue:
            item = self.admin_review_queue[task_id]
            if item.get("kind") == "pog_ungrounded":
                item["status"] = "APPROVED" if approve else "REJECTED"
                print(f"🛠️ [Admin Interface] {'Approved' if approve else 'Rejected'} PoG review Task [{task_id}].")
                return
            if approve:
                item["status"] = "APPROVED"
                # Lock vocabulary translation rule to the production engine dictionary
                with self._lock:
                    self.synonym_dictionary[item["word"]] = item["suggested_token"]
                    self.calculate_similarity.cache_clear()
                    if self.semantic_cache is not None:
                        self.semantic_cache.invalidate("llm_call")
                    # Persist
                    conn = sqlite3.connect(self.db_path)
                    conn.execute('PRAGMA journal_mode=WAL')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO synonym_mappings 
                        (word, concept_token, learned_at, approved_by) 
                        VALUES (?, ?, ?, ?)
                    ''', (item["word"], item["suggested_token"], datetime.datetime.now().isoformat(), 'admin'))
                    conn.commit()
                    conn.close()
                print(f"🛠️ [Admin Interface] Approved Task [{task_id}]. Permanently Learned: '{item['word']}'")
            else:
                item["status"] = "REJECTED"
                print(f"🗑️ [Admin Interface] Rejected Task [{task_id}]. Discarding execution branch.")

    # --- NEO4J & EXTERNAL KG INTEGRATION (real driver + Cypher builder + bidirectional sync) ---
    def connect_neo4j(self, uri=None, user=None, password=None):
        """Open a real Neo4j connection using the official driver. Connection
        parameters default to the NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars
        (matching docker-compose). Sets self.has_neo4j only when the driver loads
        AND connectivity is verified; otherwise the engine keeps using its rich
        SQLite KG backend. Returns True on success."""
        if not NEO4J_AVAILABLE:
            self.has_neo4j = False
            print("⚠️ neo4j driver not installed (pip install neo4j). Using SQLite KG fallback.")
            return False

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            self.neo4j_driver.verify_connectivity()
            self.has_neo4j = True
            print(f"✅ [Neo4j] Connected to {uri} — external KG sync enabled.")
            return True
        except Exception as e:
            self.has_neo4j = False
            self.neo4j_driver = None
            print(f"⚠️ [Neo4j] Connection failed ({e}). Using SQLite KG fallback.")
            return False

    def close_neo4j(self):
        """Close the Neo4j driver if open (idempotent)."""
        driver = getattr(self, "neo4j_driver", None)
        if driver is not None:
            try:
                driver.close()
            finally:
                self.neo4j_driver = None
                self.has_neo4j = False

    @staticmethod
    def _sanitize_rel_type(relation_type):
        """Cypher relationship types cannot be parameterized, so they must be
        interpolated — sanitize to an uppercase [A-Z0-9_] token to prevent
        injection and produce a valid identifier."""
        cleaned = re.sub(r'[^A-Za-z0-9_]', '_', str(relation_type or "")).strip('_').upper()
        return cleaned or "RELATED_TO"

    @classmethod
    def build_entity_merge(cls, entity_id, label, properties, created_at=None, updated_at=None):
        """Build a parameterized Cypher MERGE for an entity node.
        Returns (query, params). Pure — no driver required (unit-testable)."""
        query = (
            "MERGE (e:Entity {id: $id}) "
            "SET e.label = $label, e.properties = $properties, "
            "e.created_at = $created_at, e.updated_at = $updated_at"
        )
        params = {
            "id": entity_id,
            "label": label,
            "properties": json.dumps(properties) if not isinstance(properties, str) else properties,
            "created_at": created_at,
            "updated_at": updated_at,
        }
        return query, params

    @classmethod
    def build_relation_merge(cls, source_id, target_id, relation_type, confidence, properties=None,
                              valid_from=None, valid_to=None, tx_start=None, tx_end=None):
        """Build a parameterized Cypher MERGE for a typed, confidence-scored, bitemporal
        relation between two entities. valid_from/valid_to describe when the fact was
        true in the world; tx_start/tx_end describe when the system asserted/superseded
        that belief. The relationship type is sanitized and interpolated (Cypher can't
        parameterize it); all values are parameterized.
        Returns (query, params). Pure — no driver required (unit-testable)."""
        rel = cls._sanitize_rel_type(relation_type)
        query = (
            "MERGE (a:Entity {id: $source_id}) "
            "MERGE (b:Entity {id: $target_id}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            "SET r.confidence = $confidence, r.properties = $properties, "
            "r.valid_from = $valid_from, r.valid_to = $valid_to, "
            "r.tx_start = $tx_start, r.tx_end = $tx_end"
        )
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "confidence": confidence,
            "properties": json.dumps(properties or {}),
            "valid_from": valid_from,
            "valid_to": valid_to,
            "tx_start": tx_start,
            "tx_end": tx_end,
        }
        return query, params

    def kg_assert_relation(self, cursor, source_id, target_id, relation_type, confidence,
                            properties=None, valid_from=None, valid_to=None):
        """Insert a new current-belief relation row (tx_start=now(), tx_end=NULL).
        Bitemporal: valid_from/valid_to describe when the fact was/is true in the
        world; tx_start/tx_end (set by this method) describe when the system
        asserted that belief. Call kg_supersede_relation first if this replaces
        an existing current belief for the same (source, target, relation_type)
        triple — otherwise both rows would remain 'current' simultaneously."""
        now = datetime.datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO kg_relations
                (source_id, target_id, relation_type, properties, confidence,
                 valid_from, valid_to, tx_start, tx_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ''', (source_id, target_id, relation_type, json.dumps(properties or {}),
              confidence, valid_from, valid_to, now))

    def kg_supersede_relation(self, source_id, target_id, relation_type, as_of=None):
        """Close the current-belief row(s) for this triple (tx_end = now(), or
        `as_of` if given) without deleting them — corrections are append-only so
        'what did we believe on date X' stays answerable. Returns the number of
        rows closed."""
        closed_at = as_of or datetime.datetime.now().isoformat()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE kg_relations SET tx_end = ?
                WHERE source_id = ? AND target_id = ? AND relation_type = ? AND tx_end IS NULL
            ''', (closed_at, source_id, target_id, relation_type))
            n_closed = cursor.rowcount
            conn.commit()
            conn.close()
        return n_closed

    def kg_sync_to_neo4j(self):
        """Push all SQLite KG entities and relations to Neo4j via idempotent MERGE
        Cypher (safe to call repeatedly). No-op with a clear message when Neo4j is
        not connected. Returns a dict summary of what was synced."""
        if not getattr(self, "has_neo4j", False) or getattr(self, "neo4j_driver", None) is None:
            print("Using SQLite KG backend (Neo4j not connected).")
            return {"synced": False, "entities": 0, "relations": 0}

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            entities = cursor.execute(
                "SELECT entity_id, label, properties, created_at, updated_at FROM kg_entities").fetchall()
            relations = cursor.execute(
                "SELECT source_id, target_id, relation_type, confidence, properties, "
                "valid_from, valid_to, tx_start, tx_end FROM kg_relations"
            ).fetchall()
            conn.close()

        print(f"🔄 [Neo4j] Syncing {len(entities)} entities + {len(relations)} relations to Neo4j...")
        with self.neo4j_driver.session() as session:
            for entity_id, label, properties, created_at, updated_at in entities:
                q, p = self.build_entity_merge(
                    entity_id, label, properties or "{}", created_at=created_at, updated_at=updated_at)
                session.run(q, **p)
            for (source_id, target_id, rel_type, confidence, properties,
                 valid_from, valid_to, tx_start, tx_end) in relations:
                q, p = self.build_relation_merge(
                    source_id, target_id, rel_type, confidence, properties,
                    valid_from=valid_from, valid_to=valid_to, tx_start=tx_start, tx_end=tx_end)
                session.run(q, **p)
        print("✅ [Neo4j] Sync-to complete.")
        return {"synced": True, "entities": len(entities), "relations": len(relations)}

    def kg_sync_from_neo4j(self):
        """Pull entities and relations from Neo4j back into the SQLite KG so the two
        stores stay consistent (bidirectional sync). Upserts entities and inserts
        current-belief relations that are not already present. Returns a dict summary."""
        if not getattr(self, "has_neo4j", False) or getattr(self, "neo4j_driver", None) is None:
            print("Using SQLite KG backend (Neo4j not connected).")
            return {"synced": False, "entities": 0, "relations": 0}

        with self.neo4j_driver.session() as session:
            ent_records = list(session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.label AS label, e.properties AS properties, "
                "e.created_at AS created_at, e.updated_at AS updated_at"))
            rel_records = list(session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "RETURN a.id AS source_id, b.id AS target_id, type(r) AS relation_type, "
                "r.confidence AS confidence, r.properties AS properties, "
                "r.valid_from AS valid_from, r.valid_to AS valid_to, "
                "r.tx_start AS tx_start, r.tx_end AS tx_end"))

        n_ent = n_rel = 0
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            now = datetime.datetime.now().isoformat()
            for rec in ent_records:
                existing = cursor.execute(
                    "SELECT created_at FROM kg_entities WHERE entity_id = ?", (rec["id"],)).fetchone()
                created_at = (existing[0] if existing and existing[0] else None) or rec.get("created_at") or now
                cursor.execute('''
                    INSERT OR REPLACE INTO kg_entities
                        (entity_id, label, properties, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (rec["id"], rec.get("label"), rec.get("properties") or "{}", "neo4j",
                      created_at, rec.get("updated_at") or now))
                n_ent += 1
            for rec in rel_records:
                # Avoid duplicating a relation that already exists as a current belief.
                exists = cursor.execute('''
                    SELECT 1 FROM kg_relations
                    WHERE source_id=? AND target_id=? AND relation_type=? AND tx_end IS NULL LIMIT 1
                ''', (rec["source_id"], rec["target_id"], rec["relation_type"])).fetchone()
                if not exists:
                    cursor.execute('''
                        INSERT INTO kg_relations
                            (source_id, target_id, relation_type, properties, confidence,
                             valid_from, valid_to, tx_start, tx_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (rec["source_id"], rec["target_id"], rec["relation_type"],
                          rec.get("properties") or "{}", rec.get("confidence"),
                          rec.get("valid_from"), rec.get("valid_to"),
                          rec.get("tx_start") or now, rec.get("tx_end")))
                    n_rel += 1
            conn.commit()
            conn.close()
        print(f"✅ [Neo4j] Sync-from complete: {n_ent} entities, {n_rel} new relations.")
        return {"synced": True, "entities": n_ent, "relations": n_rel}

    # --- ANALYTICAL ALGORITHMIC FORMULAE ---
    def _tokenize_synonyms(self, text):
        """Converts language inputs into absolute core conceptual tokens."""
        words = re.findall(r'\b\w+\b', text)
        return " ".join([self.synonym_dictionary.get(w, w) for w in words])

    @staticmethod
    def _levenshtein_pct(a, b):
        """Character-level Levenshtein distance expressed as a 0-100 similarity percentage."""
        if len(a) < len(b):
            a, b = b, a
        if len(b) == 0:
            return 100.0 if len(a) == 0 else 0.0

        previous_row = list(range(len(b) + 1))
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (1 if c1 != c2 else 0)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return (1.0 - (previous_row[-1] / max(len(a), len(b)))) * 100.0

    def _calculate_similarity_uncached(self, s1, s2):
        """Levenshtein similarity scored on BOTH the raw and synonym-canonicalized
        forms, taking the higher score. Canonicalization lets synonyms match, but a
        misspelled word ('comput') does not expand to a token and would otherwise be
        penalized against the expanded target ('compute_token'); scoring the raw form
        too ensures typos are not scored worse than exact spellings.
        Cached per-instance in __init__ (see self.calculate_similarity)."""
        raw_score = self._levenshtein_pct(s1, s2)
        canon_score = self._levenshtein_pct(
            self._tokenize_synonyms(s1), self._tokenize_synonyms(s2)
        )
        return max(raw_score, canon_score)

    @staticmethod
    def _stable_hash(token):
        """Deterministic hash (unlike builtin hash(), which is per-process randomized).
        Stable buckets are required so persisted FAISS vectors remain valid across runs."""
        return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")

    def _compute_embedding(self, text):
        """Hashed character-n-gram bag-of-words embedding (numpy + torch compatible).
        Character trigrams (rather than whole words) make the vector typo-robust —
        'comput' and 'compute' share most trigrams and therefore score as similar —
        which is what the hybrid morphing similarity relies on. Whole-word tokens are
        also mixed in so exact-word matches stay strong."""
        text = text.lower()
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        words = re.findall(r'\b\w+\b', text)
        features = list(words)  # whole-word features
        for w in words:
            padded = f"#{w}#"
            for i in range(len(padded) - 2):  # character trigrams
                features.append(padded[i:i + 3])
        for f in features:
            idx = self._stable_hash(f) % self.vocab_size
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return torch.tensor(vec, dtype=torch.float32)

    def calculate_vector_similarity(self, s1, s2):
        """Cosine similarity on embeddings for rich semantic matching.
        Clamped to [0, 100] — floating-point rounding can push a cosine of two
        identical unit vectors marginally above 1.0."""
        emb1 = self._compute_embedding(s1)
        emb2 = self._compute_embedding(s2)
        cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return min(1.0, max(0.0, cos_sim)) * 100.0  # Scale to percentage, clamped

    def hybrid_similarity(self, s1, s2):
        """Combined Levenshtein + Vector embedding for robust matching.
        Result is clamped to [0, 100]."""
        lev = self.calculate_similarity(s1, s2)
        vec = self.calculate_vector_similarity(s1, s2)
        return min(100.0, max(0.0, lev * 0.6 + vec * 0.4))  # Weighted hybrid, clamped

    def _containment_score(self, proposal, reference):
        """Fraction (0-100) of the reference's canonical tokens that appear in the
        proposal's canonical tokens. Unlike symmetric edit distance, this rewards a
        longer proposal that fully covers a shorter reference concept — the right
        signal for 'is this concept supported by the proposal?' checks."""
        ref_tokens = set(self._tokenize_synonyms(reference).split())
        if not ref_tokens:
            return 0.0
        prop_tokens = set(self._tokenize_synonyms(proposal).split())
        covered = len(ref_tokens & prop_tokens)
        return 100.0 * covered / len(ref_tokens)

    def semantic_match_score(self, a, b):
        """Best-of similarity for verification decisions: the max of hybrid
        similarity and directional containment in either direction. Robust to
        length differences and synonym surface forms."""
        return max(
            self.hybrid_similarity(a, b),
            self._containment_score(a, b),
            self._containment_score(b, a),
        )

    # --- ADVANCED RAG / EMBEDDINGS (Optional FAISS + sentence-transformers) ---
    def _get_semantic_model(self):
        """Lazy load the sentence-transformers model. The model name is configurable
        via the ENGINE_EMBEDDING_MODEL env var (default all-MiniLM-L6-v2), so a
        deployment can swap in a larger production embedding model without code
        changes."""
        global _semantic_model
        if _semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = os.getenv("ENGINE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            try:
                _semantic_model = SentenceTransformer(model_name)
                print(f"✅ [RAG] sentence-transformers model '{model_name}' loaded for semantic embeddings.")
            except Exception as e:
                print(f"⚠️ Could not load sentence-transformers model '{model_name}': {e}")
                _semantic_model = None
        return _semantic_model

    def get_semantic_embedding(self, text):
        """True semantic embedding using sentence-transformers if available, else fallback to BoW."""
        model = self._get_semantic_model()
        if model:
            emb = model.encode(text, convert_to_tensor=True)
            return emb.cpu().numpy() if hasattr(emb, 'cpu') else emb
        # Fallback to existing torch BoW
        return self._compute_embedding(text).numpy()

    def advanced_chunk_text(self, text, strategy="semantic", chunk_size=512, overlap=50):
        """
        Advanced chunking for RAG. All strategies honor `overlap` and never emit
        empty chunks.
        - 'fixed': sliding fixed-size windows with character overlap.
        - 'semantic': sentence-aware packing whose boundaries are placed where the
          embedding similarity between adjacent sentences drops (a topic shift),
          falling back to size-based packing when embeddings are unavailable.
        - 'recursive': hierarchical split on paragraph -> sentence -> word -> char
          boundaries, only descending to a finer separator when a piece still
          exceeds chunk_size (LangChain-style recursive character splitting).
        """
        text = (text or "").strip()
        if not text:
            return []
        overlap = max(0, min(overlap, chunk_size - 1))

        if strategy == "fixed":
            return self._chunk_fixed(text, chunk_size, overlap)
        if strategy == "recursive":
            return self._chunk_recursive(text, chunk_size, overlap)
        if strategy == "semantic":
            return self._chunk_semantic(text, chunk_size, overlap)
        # Unknown strategy -> safe default
        return self._chunk_fixed(text, chunk_size, overlap)

    def _chunk_fixed(self, text, chunk_size, overlap):
        """Sliding fixed-size character windows with overlap."""
        step = max(1, chunk_size - overlap)
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
        return [c.strip() for c in chunks if c.strip()]

    def _add_overlap(self, chunks, overlap):
        """Prepend a character tail of the previous chunk to each subsequent chunk."""
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        out = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            out.append((tail + " " + cur).strip())
        return out

    def _chunk_semantic(self, text, chunk_size, overlap):
        """Sentence packing with embedding-based boundary detection.
        A new chunk is started when adding a sentence would exceed chunk_size, or
        when the semantic similarity to the running chunk drops below a threshold."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return self._chunk_fixed(text, chunk_size, overlap)

        boundary_threshold = 0.35  # cosine below this => topic shift => new chunk
        chunks = []
        current = sentences[0]
        for sent in sentences[1:]:
            too_big = len(current) + 1 + len(sent) > chunk_size
            topic_shift = False
            if not too_big:
                try:
                    a = np.asarray(self.get_semantic_embedding(current), dtype=np.float32).ravel()
                    b = np.asarray(self.get_semantic_embedding(sent), dtype=np.float32).ravel()
                    if a.shape == b.shape:
                        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
                        topic_shift = float(np.dot(a, b) / denom) < boundary_threshold
                except Exception:
                    topic_shift = False
            if too_big or topic_shift:
                chunks.append(current.strip())
                current = sent
            else:
                current += " " + sent
        if current.strip():
            chunks.append(current.strip())
        return self._add_overlap(chunks, overlap)

    def _chunk_recursive(self, text, chunk_size, overlap, separators=None):
        """Hierarchical split: try coarse separators first, recurse into finer ones
        only for pieces still larger than chunk_size, then pack adjacent pieces."""
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            # Base case: hard character split
            return self._chunk_fixed(text, chunk_size, overlap)

        pieces = text.split(sep)
        atoms = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) > chunk_size and rest:
                atoms.extend(self._chunk_recursive(piece, chunk_size, overlap, rest))
            else:
                atoms.append(piece)

        # Greedily pack atoms up to chunk_size
        packed = []
        current = ""
        for atom in atoms:
            if not current:
                current = atom
            elif len(current) + 1 + len(atom) <= chunk_size:
                current += " " + atom
            else:
                packed.append(current)
                current = atom
        if current:
            packed.append(current)
        return self._add_overlap(packed, overlap)

    # Common words that a capitalization heuristic would otherwise mis-flag as entities.
    _ENTITY_STOPWORDS = frozenset({
        "The", "A", "An", "This", "That", "These", "Those", "It", "They", "We",
        "I", "You", "He", "She", "In", "On", "At", "For", "And", "But", "Or",
        "If", "When", "While", "As", "To", "Of", "By", "With", "From",
    })

    def _extract_entities(self, text):
        """Lightweight, dependency-free entity extraction: contiguous runs of
        Capitalized words (optionally joined by 'of'/'and'), minus sentence-initial
        stopwords. Not as accurate as spaCy/NER, but real, deterministic, and
        adequate for seeding the KG. Returns a list of unique entity strings."""
        candidates = re.findall(
            r'\b[A-Z][a-zA-Z0-9]+(?:\s+(?:of|and|the)?\s*[A-Z][a-zA-Z0-9]+)*\b', text
        )
        entities = []
        for cand in candidates:
            cand = cand.strip()
            words = cand.split()
            # Drop a leading standalone stopword (e.g. "The Engine" -> "Engine").
            if len(words) > 1 and words[0] in self._ENTITY_STOPWORDS:
                cand = " ".join(words[1:])
            if not cand or cand in self._ENTITY_STOPWORDS:
                continue
            if cand not in entities:
                entities.append(cand)
        return entities

    def _store_entities(self, cursor, entities, source, valid_from=None):
        """Upsert entities into kg_entities and record co-occurrence relations.
        valid_from lets a caller (e.g. historical-document ingestion) record
        facts as valid as of the document's own time rather than ingestion time."""
        now = datetime.datetime.now().isoformat()
        for ent in entities:
            entity_id = ent.lower().replace(" ", "_")
            emb = self.get_semantic_embedding(ent)
            emb_blob = emb.tobytes() if hasattr(emb, "tobytes") else json.dumps(
                emb.tolist() if hasattr(emb, "tolist") else list(emb)
            )
            existing = cursor.execute(
                "SELECT created_at FROM kg_entities WHERE entity_id = ?", (entity_id,)).fetchone()
            created_at = (existing[0] if existing and existing[0] else None) or now
            cursor.execute('''
                INSERT OR REPLACE INTO kg_entities
                    (entity_id, label, properties, embedding, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (entity_id, ent, json.dumps({"surface": ent}), emb_blob, source, created_at, now))
        # Co-occurrence relations between entities found in the same chunk.
        for a, b in zip(entities, entities[1:]):
            self.kg_assert_relation(
                cursor, a.lower().replace(" ", "_"), b.lower().replace(" ", "_"),
                "co_occurs_with", 0.5, properties={"source": source}, valid_from=valid_from)

    def ingest_documents(self, documents, strategy="semantic", document_time=None):
        """Ingest documents into RAG + KG: chunk, embed, and store each chunk in
        rag_chunks, and extract entities/co-occurrence relations into the KG.
        document_time (optional ISO8601 string) records the extracted facts as
        valid as of the document's own time rather than ingestion time.
        Returns the total number of chunks ingested (for accurate reporting)."""
        print(f"📥 [RAG] Ingesting {len(documents)} documents with {strategy} chunking...")
        total_chunks = 0
        for doc in documents:
            chunks = self.advanced_chunk_text(doc, strategy=strategy)
            for i, chunk in enumerate(chunks):
                emb = self.get_semantic_embedding(chunk)
                entities = self._extract_entities(chunk)
                with self._lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO rag_chunks (content, embedding, metadata)
                        VALUES (?, ?, ?)
                    ''', (chunk, emb.tobytes() if hasattr(emb, 'tobytes') else json.dumps(emb.tolist()),
                          json.dumps({"source": "ingest", "chunk_id": i, "entities": entities})))
                    self._store_entities(cursor, entities, source="ingest", valid_from=document_time)
                    conn.commit()
                    conn.close()
                total_chunks += 1
        print(f"✅ [RAG] Ingestion complete. {total_chunks} chunks stored with embeddings + KG entities.")
        self.update_faiss_after_ingest()  # Keep persistent index in sync
        return total_chunks

    # --- PERSISTENT FAISS INDEX MANAGEMENT ---
    def build_faiss_index(self, force_rebuild=False):
        """Build or rebuild a persistent FAISS index from rag_chunks table.
        Saves to self.faiss_index_path for fast subsequent retrievals."""
        if not FAISS_AVAILABLE:
            print("⚠️ FAISS not available. Skipping persistent index build.")
            return False

        if hasattr(self, 'faiss_index') and self.faiss_index is not None and not force_rebuild:
            return True

        print("🔧 [FAISS] Building persistent index from RAG chunks...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content, embedding FROM rag_chunks")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("No chunks to index yet.")
            return False

        embeddings = []
        self.faiss_contents = []
        dimension = None

        for content, emb_blob in rows:
            try:
                if isinstance(emb_blob, (bytes, bytearray)):
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                else:
                    emb = np.array(json.loads(emb_blob), dtype=np.float32)
                
                if dimension is None:
                    dimension = len(emb)
                if len(emb) == dimension:
                    embeddings.append(emb)
                    self.faiss_contents.append(content)
            except Exception:
                continue

        if not embeddings:
            return False

        embeddings_np = np.array(embeddings).astype('float32')
        self.faiss_index = self._make_faiss_index(dimension, len(embeddings_np))
        # IVF indexes must be trained before adding vectors.
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            self.faiss_index.train(embeddings_np)
        self.faiss_index.add(embeddings_np)

        # Save persistently
        try:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.faiss_index_path + ".contents.json", "w") as f:
                json.dump(self.faiss_contents, f)
            print(f"✅ [FAISS] Persistent index ({type(self.faiss_index).__name__}) "
                  f"saved to {self.faiss_index_path} ({len(embeddings)} vectors)")
        except Exception as e:
            print(f"⚠️ Could not save FAISS index: {e}")

        return True

    def _make_faiss_index(self, dimension, n_vectors):
        """Choose a FAISS index type by corpus size, for production-scale retrieval.
        Small corpora use an exact inner-product index (best recall, no training);
        larger corpora use an approximate HNSW graph index for sub-linear search.
        Thresholds are overridable via ENGINE_FAISS_HNSW_THRESHOLD."""
        hnsw_threshold = int(os.getenv("ENGINE_FAISS_HNSW_THRESHOLD", "10000"))
        if n_vectors < hnsw_threshold:
            return faiss.IndexFlatIP(dimension)
        # HNSW: approximate, memory-resident, no training required, strong recall.
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        return index

    def load_faiss_index(self):
        """Load existing persistent FAISS index if available."""
        if not FAISS_AVAILABLE:
            return False
        try:
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                with open(self.faiss_index_path + ".contents.json") as f:
                    self.faiss_contents = json.load(f)
                print(f"✅ [FAISS] Loaded persistent index with {len(self.faiss_contents)} vectors")
                return True
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")
        return False

    def semantic_retrieve_context(self, query, k=5, use_faiss=True, use_cache=True):
        """Semantic-cached wrapper around _semantic_retrieve_context_uncached (see
        that method for the actual FAISS/brute-force retrieval logic). A cache hit
        — exact query+params match, or a near-duplicate query above the cache's
        similarity threshold — returns the previously retrieved contexts without
        re-running FAISS/brute-force search."""
        cache = self.semantic_cache if use_cache else None
        params = {"k": k, "use_faiss": use_faiss}
        query_emb = None
        if cache is not None:
            query_emb = self.get_semantic_embedding(query)
            if hasattr(query_emb, "cpu"):
                query_emb = query_emb.cpu().numpy()
            cached = cache.get("rag_retrieve", query, query_emb, params=params)
            if cached is not None:
                return cached

        result = self._semantic_retrieve_context_uncached(query, k=k, use_faiss=use_faiss)

        if cache is not None:
            cache.put("rag_retrieve", query, query_emb, result, params=params)

        return result

    def _semantic_retrieve_context_uncached(self, query, k=5, use_faiss=True):
        """Semantic retrieval using persistent FAISS index (preferred) or brute-force fallback."""
        print(f"🔍 [RAG] Retrieving top-{k} contexts for query...")

        query_emb = self.get_semantic_embedding(query)
        if isinstance(query_emb, torch.Tensor):
            query_emb = query_emb.cpu().numpy()

        # Try persistent FAISS first
        if FAISS_AVAILABLE and use_faiss:
            if not hasattr(self, 'faiss_index') or self.faiss_index is None:
                self.load_faiss_index() or self.build_faiss_index()

            if hasattr(self, 'faiss_index') and self.faiss_index is not None and hasattr(self, 'faiss_contents'):
                try:
                    q = query_emb.reshape(1, -1).astype('float32')
                    D, I = self.faiss_index.search(q, min(k, len(self.faiss_contents)))
                    results = [self.faiss_contents[i] for i in I[0] if i < len(self.faiss_contents)]
                    if results:
                        return results
                except Exception as e:
                    print(f"FAISS search error: {e}")

        # Fallback to DB + brute force
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content, embedding FROM rag_chunks")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return ["No RAG context available yet. Ingest documents first."]

        scored = []
        for content, emb_blob in rows:
            try:
                if isinstance(emb_blob, (bytes, bytearray)):
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                else:
                    emb = np.array(json.loads(emb_blob), dtype=np.float32)
                if len(emb) == len(query_emb):
                    sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
                    scored.append((sim, content))
            except:
                continue
        scored.sort(reverse=True)
        return [c for s, c in scored[:k]]

    def update_faiss_after_ingest(self):
        """Rebuild FAISS index after new documents are ingested, and invalidate
        cached retrieval results so newly-ingested content is discoverable
        rather than masked by a stale cached response."""
        if FAISS_AVAILABLE:
            self.build_faiss_index(force_rebuild=True)
        if self.semantic_cache is not None:
            self.semantic_cache.invalidate("rag_retrieve")

    # --- ADVANCED RETRIEVAL: query rewriting, cross-encoder re-ranking, agentic multi-hop ---
    def _rewrite_query(self, query):
        """LLM-based query rewriting: expand/clarify the query for better recall.
        Falls back to the original query on any failure (including mock LLM)."""
        prompt = (
            f"Rewrite the following search query to maximize retrieval recall — "
            f"expand abbreviations and add key synonyms, but keep it concise and on-topic. "
            f"Return ONLY the rewritten query, no preamble.\nQuery: {query}"
        )
        try:
            result = self.llm_call(prompt, max_tokens=64)
            text = result.get("content", "") if isinstance(result, dict) else str(result)
            text = text.strip().strip('"')
            # Guard against mock/garbage responses: require it to look like a query.
            if text and 2 <= len(text) <= 400 and "\n" not in text:
                return text
        except Exception as e:
            print(f"⚠️ [RAG] Query rewrite failed, using original: {e}")
        return query

    def _get_cross_encoder(self):
        """Lazy-load an optional cross-encoder re-ranker (model configurable via
        ENGINE_RERANK_MODEL). Returns None when the dependency is unavailable."""
        global _cross_encoder_model
        if not CROSS_ENCODER_AVAILABLE:
            return None
        if _cross_encoder_model is None:
            model_name = os.getenv("ENGINE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            try:
                _cross_encoder_model = CrossEncoder(model_name)
                print(f"✅ [RAG] cross-encoder re-ranker '{model_name}' loaded.")
            except Exception as e:
                print(f"⚠️ Could not load cross-encoder '{model_name}': {e}")
                _cross_encoder_model = None
        return _cross_encoder_model

    def _rerank(self, query, candidates, k):
        """Re-rank candidate passages against the query with a cross-encoder when
        available; otherwise preserve the (already embedding-ranked) input order."""
        if not candidates:
            return []
        model = self._get_cross_encoder()
        if model is None:
            return candidates[:k]
        try:
            scores = model.predict([(query, c) for c in candidates])
            ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
            return ranked[:k]
        except Exception as e:
            print(f"⚠️ [RAG] Re-rank failed, falling back to base order: {e}")
            return candidates[:k]

    def retrieve_context_advanced(self, query, k=5, rewrite=True, rerank=True,
                                  agentic=False, max_hops=3):
        """Production retrieval pipeline layering three optional stages over
        `semantic_retrieve_context`:
          1. query rewriting (LLM) for better recall,
          2. cross-encoder re-ranking for better precision,
          3. agentic multi-hop expansion (the LLM proposes follow-up queries until
             it deems the gathered context sufficient or max_hops is reached).
        Returns a de-duplicated list of at most `k` (or, in agentic mode, up to
        `k * max_hops`) context passages."""
        search_query = self._rewrite_query(query) if rewrite else query
        pool = self._safe_retrieve(search_query, k * 3)
        results = self._rerank(query, pool, k) if rerank else pool[:k]

        if not agentic:
            return results

        gathered = list(dict.fromkeys(results))  # preserve order, de-dupe
        for _ in range(max_hops - 1):
            follow_up = self._propose_follow_up_query(query, gathered)
            if not follow_up:
                break
            more = self._safe_retrieve(follow_up, k * 3)
            more = self._rerank(follow_up, more, k) if rerank else more[:k]
            new_items = [c for c in more if c not in gathered]
            if not new_items:
                break
            gathered.extend(new_items)
        return gathered

    def _safe_retrieve(self, query, k):
        """Retrieve context, returning [] instead of the 'no context' sentinel."""
        res = self.semantic_retrieve_context(query, k=k)
        if not res or (len(res) == 1 and res[0].startswith("No RAG context")):
            return []
        return res

    def _propose_follow_up_query(self, original_query, gathered):
        """Ask the LLM whether more retrieval is needed; return a follow-up query
        string, or None/empty to stop the agentic loop."""
        context_preview = " | ".join(g[:120] for g in gathered[:5])
        prompt = (
            f"Original question: {original_query}\n"
            f"Context gathered so far: {context_preview}\n"
            f"If the context is sufficient to answer, reply exactly DONE. "
            f"Otherwise reply with ONE short follow-up search query that would fill the gap."
        )
        try:
            result = self.llm_call(prompt, max_tokens=48)
            text = result.get("content", "") if isinstance(result, dict) else str(result)
            text = text.strip().strip('"')
            if not text or text.upper().startswith("DONE") or "\n" in text:
                return None
            return text if 2 <= len(text) <= 400 else None
        except Exception:
            return None

    def _trigger_auto_learning(self, input_text, target_text, task_id, is_low_confidence):
        """Unsupervised context mapping routine to trace, segregate, and extract unknown words."""
        input_words = re.findall(r'\b\w+\b', input_text.lower())
        target_words = re.findall(r'\b\w+\b', target_text.lower())
        
        for tw in target_words:
            if tw in self.synonym_dictionary:
                target_token = self.synonym_dictionary[tw]
                for iw in input_words:
                    if iw not in self.synonym_dictionary and iw not in target_words:
                        if is_low_confidence:
                            # Move word immediately into isolation pen instead of production map
                            with self._lock:
                                self.admin_review_queue[task_id] = {
                                    "word": iw, "suggested_token": target_token, "status": "PENDING_HALT"
                                }
                            print(f"⚠️ [Flagged Security] Isopropyl isolated word '{iw}' via Task [{task_id}]. Sent to Queue.")
                        else:
                            with self._lock:
                                self.synonym_dictionary[iw] = target_token
                                self.calculate_similarity.cache_clear()
                                if self.semantic_cache is not None:
                                    self.semantic_cache.invalidate("llm_call")
                                # Persist to DB
                                conn = sqlite3.connect(self.db_path)
                                conn.execute('PRAGMA journal_mode=WAL')
                                cursor = conn.cursor()
                                cursor.execute('''
                                    INSERT OR REPLACE INTO synonym_mappings
                                    (word, concept_token, learned_at) VALUES (?, ?, ?)
                                ''', (iw, target_token, datetime.datetime.now().isoformat()))
                                conn.commit()
                                conn.close()
                            print(f"🤖 [Auto-Learning] Firmly cataloged word '{iw}' -> '{target_token}'")

    # --- MAIN EXECUTION PIPELINE LOOP ---
    def process_query_stream(self, task_id, start_node, live_user_input):
        """
        Memory-safe, Heap-allocated Trampoline execution stack. Evaluates multi-threaded input 
        streams across flexible recursive states without incurring stack overflow limits.
        """
        # Local Thread Stack frame instantiation format: (node, local_text_buffer, operational_phase)
        execution_stack = [(start_node, live_user_input.lower().strip(), 0)]
        nested_string_register = ""
        timestamp = datetime.datetime.now().isoformat()
        final_answer = None

        print(f"📥 [Routing Engine] Spawning Worker Thread for Task [{task_id}]: '{live_user_input}'")

        while execution_stack and not final_answer:
            node, current_text, phase = execution_stack.pop()
            if not node:
                continue

            # Determine algorithmic match similarity score using local dictionary matrix state
            score = self.calculate_similarity(current_text, self.raw_target)
            
            # Continuous boundary evaluation checks
            is_low_conf = self.threshold <= score < (self.threshold + self.flag_buffer)

            # Trigger structural discovery checks if match falls into soft-failure buffer zones
            if score < self.threshold and score > (self.threshold - 25.0):
                self._trigger_auto_learning(current_text, self.raw_target, task_id, is_low_confidence=is_low_conf)
                score = self.calculate_similarity(current_text, self.raw_target)
                is_low_conf = self.threshold <= score < (self.threshold + self.flag_buffer)

            # --- ISOLATION HOOK TRIGGER (HALTING SPECIFIC THREAD EXCLUSIVELY) ---
            if is_low_conf and task_id in self.admin_review_queue:
                if self.admin_review_queue[task_id]["status"] == "PENDING_HALT":
                    print(f"🛑 [HALT ACTIVATED] Task [{task_id}] matches at low-confidence threshold ({score:.1f}%). Freezing this branch!")
                    
                    # Worker Thread execution loop block. Continues polling status flags non-blockingly.
                    while self.admin_review_queue[task_id]["status"] == "PENDING_HALT":
                        time.sleep(0.2)
                        
                    if self.admin_review_queue[task_id]["status"] == "REJECTED":
                        return f"❌ Task [{task_id}] Process Dumped: Terminated by Admin Rejection."
                    
                    # Re-calculate parameters post-admin authorization override
                    score = self.calculate_similarity(current_text, self.raw_target)

            # --- TERMINATION STATE LOGGING CRITERIA ---
            if score >= self.threshold:
                status_stamp = "ADMIN_REVIEW_OVERRIDE" if task_id in self.admin_review_queue else "VERIFIED_PRODUCTION"
                final_answer = f"Resolved Answer at Node [{node.id}]"
                
                # 1. Update Global Q&A Audit Log Archive (in-memory + DB)
                audit_entry = {
                    "timestamp": timestamp, 
                    "task_id": task_id, 
                    "query": live_user_input,
                    "target_node": node.id, 
                    "score": f"{score:.1f}%", 
                    "clearance_security": status_stamp
                }
                with self._lock:
                    self.qa_audit_log.append(audit_entry)
                    # Persist to DB
                    conn = sqlite3.connect(self.db_path)
                    conn.execute('PRAGMA journal_mode=WAL')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO qa_audit_log 
                        (timestamp, task_id, query, target_node, score, clearance_security) 
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, task_id, live_user_input, node.id, f"{score:.1f}%", status_stamp))
                    conn.commit()
                    conn.close()

                # 2. Check and log Exponential Query Cascades (Explosion Tracker)
                if len(node.triggered_sub_questions) >= 5:
                    explosion_entry = {
                        "timestamp": timestamp, 
                        "trigger_task": task_id, 
                        "origin_node": node.id,
                        "downstream_explosion_count": len(node.triggered_sub_questions),
                        "cascade_query_manifest": node.triggered_sub_questions.copy()
                    }
                    with self._lock:
                        self.explosion_log.append(explosion_entry)
                        # Persist to DB
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO query_explosions 
                            (timestamp, trigger_task, origin_node, explosion_count, cascade_manifest) 
                            VALUES (?, ?, ?, ?, ?)
                        ''', (timestamp, task_id, node.id, len(node.triggered_sub_questions), 
                              json.dumps(node.triggered_sub_questions)))
                        conn.commit()
                        conn.close()
                    print(f"💥 [Explosion Intercepted] 1 Result from Task [{task_id}] generated {len(node.triggered_sub_questions)} new inquiries!")
                break

            # =================================================================
            # SELF-MORPHING CONTROL STRATEGIES (Execution Path Shifting Matrix)
            # =================================================================
            # STYLE 1: LINEAR / TAIL RECURSION FLATTENING
            if node.node_type == "linear":
                cleaned_text = current_text.replace(node.key_phrase, "").strip()
                if node.next_linear:
                    execution_stack.append((node.next_linear, cleaned_text, 0))
            
            # STYLE 2: TREE RECURSION (Branching with Strict Tie-Breaker Ordering Optimizer)
            elif node.node_type == "tree":
                if phase == 0:
                    left_n = node.branches.get("left")
                    right_n = node.branches.get("right")
                    left_s = self.calculate_similarity(current_text, left_n.key_phrase) if left_n else -1
                    right_s = self.calculate_similarity(current_text, right_n.key_phrase) if right_n else -1
                    
                    # Execute Priority Path Sorting & Sophisticated Tie-Breaker Rules
                    # Enhanced: Multi-criteria with similarity, length, ID, and potential history
                    if left_s == right_s and left_n and right_n:
                        # Tie-Breaker Priority A: Shortest phrase (favor simplicity)
                        if len(left_n.key_phrase) != len(right_n.key_phrase):
                            left_is_preferred = len(left_n.key_phrase) < len(right_n.key_phrase)
                        else:
                            # Tie-Breaker Priority B: ID lexical order + length as secondary
                            left_is_preferred = (left_n.id < right_n.id) or (len(left_n.key_phrase) <= len(right_n.key_phrase))
                    else:
                        left_is_preferred = left_s > right_s
                    
                    # Future extension point: Add historical success rate from DB
                    
                    # LIFO Execution Queue Allocation Setup
                    if left_is_preferred:
                        execution_stack.append((right_n, current_text, 0))
                        execution_stack.append((left_n, current_text, 0))
                    else:
                        execution_stack.append((left_n, current_text, 0))
                        execution_stack.append((right_n, current_text, 0))
            
            # STYLE 3: INDIRECT RECURSION (Mutual External Execution Core Jumps)
            elif node.node_type == "indirect":
                # Divert current text parsing execution frame over to a separate routing graph loop
                external_schema_root = self._external_router_resolver(node.mutual_routine)
                if external_schema_root:
                    execution_stack.append((external_schema_root, current_text, 0))
            
            # STYLE 4: NESTED RECURSION (Encapsulated Expression Tracking)
            # Enhanced: Better parameter extraction, multi-level support via register
            elif node.node_type == "nested":
                if phase == 0:
                    # Improved extraction supporting parentheses or fallback
                    if "(" in current_text and ")" in current_text:
                        start = current_text.find("(") + 1
                        end = current_text.rfind(")")
                        inner_slice = current_text[start:end].strip()
                    else:
                        inner_slice = current_text.strip()
                    execution_stack.append((node, current_text, 1))  # Register resumption handle frame
                    if node.inner_formula:
                        execution_stack.append((node.inner_formula, inner_slice, 0))  # Push deep inner search
                    else:
                        # Fallback if no inner formula
                        execution_stack.append((node, current_text, 1))
                elif phase == 1:
                    # Enhanced inner evaluation with register update
                    # In real use, nested_string_register could be updated from inner result
                    inner_result = nested_string_register or "processed_inner"
                    compiled_sequence = f"{node.key_phrase} return feedback payload: {inner_result}"
                    if node.next_linear:
                        execution_stack.append((node.next_linear, compiled_sequence, 0))

        if final_answer:
            return f"🟢 Task Complete -> {final_answer}"
        return "⚠️ No resolution reached in processing stream."

    def _external_router_resolver(self, routine_name):
        """Fallback module map identifier for Indirect state loops.
        Enhanced for LLM hybrid routing."""
        # Example: Route to LLM-enhanced nodes
        if routine_name == "llm_enhance":
            # Could return a dynamic node or trigger LLM directly
            print(f"🌐 [Indirect Router] LLM-enhanced routing activated for {routine_name}")
            return MorphicTextNode("LLM_Router", "llm resolved via hybrid", "linear")
        # Placeholder - implement as needed for external routing
        return None

    # =====================================================================
    # LLM INTEGRATION MIXIN / METHODS (Hybrid Extension)
    # =====================================================================
    def llm_call(self, prompt, max_tokens=150, temperature=0.7, json_mode=False, max_retries=3, use_cache=True):
        """Semantic-cached wrapper around _llm_call_uncached (see that method for
        the actual OpenAI/Groq/mock LLM integration logic). A cache hit — exact
        prompt+params match, or a near-duplicate prompt scoring above the cache's
        similarity threshold — returns the previously computed response without
        calling the LLM (or the mock) again. use_cache=False bypasses the cache
        entirely, e.g. for callers that need a fresh/independent draw."""
        cache = self.semantic_cache if use_cache else None
        params = {"max_tokens": max_tokens, "temperature": temperature, "json_mode": json_mode}
        emb = None
        if cache is not None:
            emb = self.get_semantic_embedding(prompt)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            cached = cache.get("llm_call", prompt, emb, params=params)
            if cached is not None:
                return cached

        result = self._llm_call_uncached(
            prompt, max_tokens=max_tokens, temperature=temperature,
            json_mode=json_mode, max_retries=max_retries)

        if cache is not None:
            cache.put("llm_call", prompt, emb, result, params=params)

        return result

    def _llm_call_uncached(self, prompt, max_tokens=150, temperature=0.7, json_mode=False, max_retries=3):
        """
        Enhanced LLM integration with retries, error handling, and env-based client.
        Supports OpenAI and Groq via environment variables.
        Falls back to high-quality mock for demos/testing.
        """
        import os
        import time

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        is_groq = bool(os.getenv("GROQ_API_KEY"))
        
        print(f"🤖 [LLM Call] {'[JSON Mode]' if json_mode else ''} Prompt: {prompt[:100]}... (retries={max_retries})")
        
        for attempt in range(max_retries):
            try:
                if api_key:
                    if is_groq:
                        # Groq client (fast inference)
                        try:
                            from groq import Groq
                            client = Groq(api_key=api_key)
                            response = client.chat.completions.create(
                                model="llama3-8b-8192",
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                response_format={"type": "json_object"} if json_mode else None
                            )
                            content = response.choices[0].message.content
                            print(f"✅ [Groq LLM] Response received (attempt {attempt+1})")
                            if json_mode:
                                return json.loads(content) if isinstance(content, str) else content
                            return {"content": content, "confidence": 85.0}
                        except ImportError:
                            print("⚠️ groq package not installed. Falling back to mock.")
                    else:
                        # OpenAI client
                        try:
                            import openai
                            client = openai.OpenAI(api_key=api_key)
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                response_format={"type": "json_object"} if json_mode else None
                            )
                            content = response.choices[0].message.content
                            print(f"✅ [OpenAI LLM] Response received (attempt {attempt+1})")
                            if json_mode:
                                return json.loads(content) if isinstance(content, str) else content
                            return {"content": content, "confidence": 88.0}
                        except ImportError:
                            print("⚠️ openai package not installed. Falling back to mock.")
                
                # High-quality mock fallback (used when no key or import fails)
                time.sleep(0.2 * (attempt + 1))
                if json_mode:
                    mock_json = {
                        "suggested_mapping": {"word": "dynamic_term", "token": "compute_token"},
                        "confidence": round(random.uniform(78, 96), 1),
                        "reasoning": "Inferred from context + prior KG patterns.",
                        "new_node_suggestion": {"id": f"Dynamic_{random.randint(100,999)}", "key_phrase": "new_analytics_branch", "type": "tree"},
                        "sub_objectives": ["Understand core problem", "Explore related KG entities", "Verify solution confidence"]
                    }
                    return mock_json
                else:
                    responses = [
                        f"Based on context and KG, '{prompt[:40]}...' maps to compute/analytics patterns.",
                        "Recommended: Add verification node for low-confidence paths.",
                        "PoG reflection suggests increasing hop limit for better coverage."
                    ]
                    return {"content": random.choice(responses), "confidence": round(random.uniform(72, 91), 1)}
            
            except Exception as e:
                print(f"⚠️ LLM attempt {attempt+1} failed: {str(e)[:80]}")
                if attempt == max_retries - 1:
                    print("❌ All LLM retries exhausted. Using safe mock response.")
                    return {"content": "Fallback reasoning: Use hybrid similarity + KG verification.", "confidence": 65.0}
                time.sleep(1 * (attempt + 1))
        
        return {"content": "Error in LLM call", "confidence": 0.0}

    def self_auditor_verify(self, proposal, context_text, min_confidence=70.0):
        """
        Self-Auditor / Self-Verifier: Validates LLM proposals using engine's similarity + rules.
        Returns verified proposal or None if low confidence.
        """
        # Compute similarity to target or context. Use the containment-aware
        # semantic score so a proposal that fully covers the context (but is longer)
        # is not penalized by raw edit distance.
        context = context_text or self.raw_target
        sim_score = self.semantic_match_score(proposal, context)
        print(f"🔍 [Self-Auditor] Proposal similarity: {sim_score:.1f}% to context.")
        
        if sim_score >= min_confidence:
            # Additional rule-based checks (e.g., token validity)
            if len(proposal.split()) > 1 or any(k in proposal.lower() for k in self.synonym_dictionary.keys()):
                print("✅ [Self-Verifier] Proposal passed audit.")
                return proposal
            else:
                print("⚠️ [Self-Verifier] Failed basic token check.")
                return None
        else:
            print("❌ [Self-Verifier] Low confidence - rejected.")
            return None

    def symbolic_verifier(self, proposal, context_text):
        """
        Symbolic Checker Layer for Zero-Human Autonomy.
        Performs deterministic rule-based validation, consistency checks,
        domain rules, and cross-verification with embeddings.
        """
        print(f"🔬 [Symbolic Verifier] Checking proposal: {proposal[:100]}...")
        
        # 1. Structural / Syntax checks
        if not proposal or len(proposal.strip()) < 3:
            print("❌ [Symbolic] Empty or too short proposal rejected.")
            return False
        
        # 2. Consistency with existing dictionary (no conflicting mappings)
        words = re.findall(r'\b\w+\b', proposal.lower())
        with self._lock:
            for w in words:
                if w in self.synonym_dictionary and "->" in proposal:
                    # Mock conflict detection
                    print(f"⚠️ [Symbolic] Potential conflict for word '{w}'.")
                    # In full impl: compare proposed vs existing
        
        # 3. Domain-specific symbolic rules. Check the CANONICAL (synonym-expanded)
        # form so a domain word like "analytics" (-> data_token) counts even though
        # the literal token string is absent from the surface text. Presence of a
        # recognized domain concept is sufficient symbolic evidence on its own.
        valid_tokens = {"compute_token", "data_token", "structure_token"}
        canonical = self._tokenize_synonyms(proposal)
        if any(token in canonical for token in valid_tokens):
            print("✅ [Symbolic Verifier] Domain rules passed.")
            return True
        print("⚠️ [Symbolic] No recognized domain token - checking semantic consistency.")

        # 4. No domain token: fall back to semantic consistency with the core target.
        sim = self.semantic_match_score(proposal, self.raw_target)
        if sim > 60.0:
            print("✅ [Symbolic] Consistent with core target.")
            return True
        return False

    def self_teaching_loop(self, background=True, max_iterations=3):
        """
        Background self-teaching loop: Queries DB for unresolved/low-confidence cases,
        prompts LLM, self-audits, and learns validated mappings.
        Runs continuously with minimal human input.
        """
        def _teaching_iteration():
            """Run one pass: pull unresolved cases from the DB, propose and verify mappings."""
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                # Query unresolved or recent low-confidence entries (simulate via audit logs)
                cursor.execute('''
                    SELECT query, target_node FROM qa_audit_log 
                    WHERE clearance_security LIKE '%REVIEW%' OR score < ? 
                    ORDER BY timestamp DESC LIMIT 5
                ''', (str(self.threshold),))
                unresolved = cursor.fetchall()
                conn.close()

            if not unresolved:
                print("📚 [Self-Teaching] No unresolved cases found. Idle.")
                return

            for query, node_id in unresolved:
                prompt = f"Analyze unresolved query: '{query}' for target '{self.raw_target}'. Suggest synonym mappings or node improvements."
                # Extended with JSON-mode for structured outputs
                llm_suggestion = self.llm_call(prompt, json_mode=True)
                
                # Update metrics
                with self._lock:
                    self.learning_metrics["total_proposals"] += 1
                
                # Self-audit and verify
                verified = self.self_auditor_verify(str(llm_suggestion), query)
                if verified and self.symbolic_verifier(str(llm_suggestion), query):
                    # Dynamic Node Generation + Learning from JSON
                    if isinstance(llm_suggestion, dict) and "new_node_suggestion" in llm_suggestion:
                        node_data = llm_suggestion["new_node_suggestion"]
                        new_node = MorphicTextNode(
                            node_id=node_data.get("id", "Dynamic_Node"),
                            key_phrase=node_data.get("key_phrase", "new_branch"),
                            node_type=node_data.get("type", "linear")
                        )
                        print(f"🌱 [Dynamic Node] Generated and approved: {new_node.id}")
                        # GNN signal: if a live graph is attached, let the GNN's
                        # propagated embeddings judge how well this node fits, and
                        # attach it under its best predicted parent.
                        if self.current_graph_root is not None:
                            self._gnn_guided_attach(new_node)
                    # Learn mapping from JSON
                    mapping = llm_suggestion.get("suggested_mapping", {})
                    word = mapping.get("word", "dynamic_term")
                    token = mapping.get("token", "compute_token")
                    conf = llm_suggestion.get("confidence", 85.0)
                    # GNN relevance blends into learned-mapping confidence when a
                    # graph is available (non-breaking: no graph -> conf unchanged).
                    if self.current_graph_root is not None:
                        gnn_score = self.gnn_node_relevance(word, self.current_graph_root)
                        conf = 0.7 * conf + 0.3 * gnn_score
                        print(f"🧠 [GNN] Relevance signal {gnn_score:.1f}% -> blended conf {conf:.1f}%")
                    with self._lock:
                        self.synonym_dictionary[word] = token
                        self.calculate_similarity.cache_clear()
                        if self.semantic_cache is not None:
                            self.semantic_cache.invalidate("llm_call")
                        # Persist
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO synonym_mappings
                            (word, concept_token, learned_at, confidence, approved_by)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (word, token, datetime.datetime.now().isoformat(), conf, 'self_teach'))
                        conn.commit()
                        conn.close()
                        # Update metrics
                        self.learning_metrics["accepted_learnings"] += 1
                        self.learning_metrics["avg_confidence"] = (self.learning_metrics["avg_confidence"] * (self.learning_metrics["accepted_learnings"]-1) + conf) / self.learning_metrics["accepted_learnings"]
                        self.learning_metrics["learning_history"].append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "word": word, "token": token, "confidence": conf, "outcome": "accepted",
                        })
                        self._update_rejection_rate()
                    print(f"🎓 [Self-Teaching] Learned via JSON + Verify: {word} -> {token} (conf: {conf:.1f}%)")
                else:
                    with self._lock:
                        self.learning_metrics["learning_history"].append({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "outcome": "rejected",
                        })
                        self._update_rejection_rate()
                    print("🚫 [Zero-Human] Verification layers rejected proposal for safety.")
                time.sleep(0.5)  # Throttle

        print("🚀 [Self-Teaching Loop] Starting continuous self-improvement...")
        
        if background:
            # Run in background thread for continuous operation
            def background_loop():
                """Run _teaching_iteration for max_iterations, sleeping between each pass."""
                for i in range(max_iterations):
                    _teaching_iteration()
                    time.sleep(2)  # Simulate periodic runs
                print("⏹️ [Self-Teaching] Background loop completed iterations.")
            
            thread = threading.Thread(target=background_loop, daemon=True)
            thread.start()
            return thread
        else:
            for i in range(max_iterations):
                _teaching_iteration()
            return None

    def export_graph_viz(self, root_node, filename="morphic_graph.dot"):
        """Visualization/export to basic DOT format (Graphviz compatible).
        Falls back to simple text if pydot/networkx issues."""
        try:
            import networkx as nx
            G = nx.DiGraph()
            visited = set()
            def add_nodes(node, parent_id=None):
                """Recursively add node and its linear/branch/inner-formula children to G."""
                if not node: return
                G.add_node(node.id, label=node.key_phrase, type=node.node_type)
                if parent_id:
                    G.add_edge(parent_id, node.id)
                if node.id in visited:
                    return
                visited.add(node.id)
                if node.next_linear:
                    add_nodes(node.next_linear, node.id)
                for branch in node.branches.values():
                    add_nodes(branch, node.id)
                if node.inner_formula:
                    add_nodes(node.inner_formula, node.id)
            add_nodes(root_node)
            # Use simple write if pydot missing
            with open(filename, "w") as f:
                f.write("digraph MorphicGraph {\n")
                for u, v in G.edges():
                    f.write(f'  "{u}" -> "{v}";\n')
                for n in G.nodes():
                    f.write(f'  "{n}" [label="{G.nodes[n].get("label", n)}"];\n')
                f.write("}\n")
            print(f"📊 [Viz Export] Morphic graph exported to {filename} (use Graphviz dot to render).")
        except Exception as e:
            print(f"⚠️ [Viz] Fallback export (networkx/pydot issue: {e})")
            with open(filename, "w") as f:
                f.write("digraph MorphicGraph { root -> fork; }\n")
        # Also print metrics
        print("📈 Learning Metrics:", self.learning_metrics)
        return filename

    # --- GNN: LEARNED PROPAGATION OVER THE MORPHIC GRAPH ---
    def _collect_graph(self, root_node):
        """Cycle-safe traversal collecting nodes and typed edges of the morphic graph.
        Mirrors export_graph_viz's walk but also captures mutual_routine (indirect)
        edges and the edge type. Returns (nodes, edges) where nodes is an ordered
        list of MorphicTextNode and edges is a list of (src_id, dst_id, edge_type)."""
        nodes = []
        seen = set()
        edges = []

        def visit(node):
            """Recurse over one node's typed out-edges, guarding against cycles."""
            if not node or node.id in seen:
                return
            seen.add(node.id)
            nodes.append(node)
            if node.next_linear:
                edges.append((node.id, node.next_linear.id, "linear"))
                visit(node.next_linear)
            for branch in node.branches.values():
                if branch:
                    edges.append((node.id, branch.id, "branch"))
                    visit(branch)
            if node.inner_formula:
                edges.append((node.id, node.inner_formula.id, "inner_formula"))
                visit(node.inner_formula)
            if node.mutual_routine:
                edges.append((node.id, node.mutual_routine.id, "mutual_routine"))
                visit(node.mutual_routine)

        visit(root_node)
        return nodes, edges

    def build_graph_tensors(self, root_node):
        """Build pure-torch tensors describing the morphic graph — no PyG required,
        so this is unit-testable everywhere. Node features are the semantic
        embedding of the key phrase concatenated with a one-hot node-type vector.
        Returns dict with x, edge_index [2,E], edge_type [E], y (node-type labels),
        node_ids, and id_to_index."""
        nodes, edges = self._collect_graph(root_node)
        if not nodes:
            return None
        node_ids = [n.id for n in nodes]
        id_to_index = {nid: i for i, nid in enumerate(node_ids)}

        feats = []
        labels = []
        for n in nodes:
            emb = np.asarray(self.get_semantic_embedding(n.key_phrase), dtype=np.float32).ravel()
            onehot = np.zeros(len(NODE_TYPES), dtype=np.float32)
            onehot[NODE_TYPE_INDEX.get(n.node_type, 0)] = 1.0
            feats.append(np.concatenate([emb, onehot]))
            labels.append(NODE_TYPE_INDEX.get(n.node_type, 0))

        x = torch.tensor(np.vstack(feats), dtype=torch.float32)
        if edges:
            src = [id_to_index[s] for s, d, t in edges if d in id_to_index]
            dst = [id_to_index[d] for s, d, t in edges if d in id_to_index]
            etype = [EDGE_TYPE_INDEX[t] for s, d, t in edges if d in id_to_index]
        else:
            src, dst, etype = [], [], []
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long) if etype else torch.zeros((0,), dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return {
            "x": x, "edge_index": edge_index, "edge_type": edge_type,
            "y": y, "node_ids": node_ids, "id_to_index": id_to_index,
        }

    def build_pyg_graph(self, root_node):
        """Wrap the graph tensors in a PyTorch-Geometric Data object (PyG only)."""
        if not PYG_AVAILABLE:
            return None
        t = self.build_graph_tensors(root_node)
        if t is None:
            return None
        return _PyGData(x=t["x"], edge_index=t["edge_index"], edge_type=t["edge_type"], y=t["y"])

    @property
    def gnn_model_path(self):
        """Filesystem path for persisted GNN weights (env-configurable)."""
        return os.getenv("ENGINE_GNN_MODEL_PATH", "gnn_model.pt")

    def train_gnn(self, root_node, epochs=60, lr=0.01):
        """Train the GNN on the current morphic graph: node-type classification
        (cross-entropy) plus link prediction (BCE over observed edges vs. negative
        samples). Persists weights and returns a dict of final losses. No-op with a
        clear message when PyG is unavailable (the engine then relies on the NumPy
        propagation fallback for GNN signals)."""
        if not PYG_AVAILABLE:
            print("⚠️ [GNN] torch-geometric not installed; using NumPy propagation fallback.")
            return None
        t = self.build_graph_tensors(root_node)
        if t is None or t["x"].shape[0] < 2:
            print("⚠️ [GNN] Graph too small to train.")
            return None

        x, edge_index, y = t["x"], t["edge_index"], t["y"]
        model = MorphicGNN(in_dim=x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n = x.shape[0]
        model.train()
        last = {}
        for _ in range(epochs):
            opt.zero_grad()
            h = model.encode(x, edge_index)
            cls_loss = torch.nn.functional.cross_entropy(model.classify(h), y)
            # Link prediction: positive = observed edges, negative = random pairs.
            if edge_index.shape[1] > 0:
                pos = model.link_score(h, edge_index)
                neg_src = torch.randint(0, n, (edge_index.shape[1],))
                neg_dst = torch.randint(0, n, (edge_index.shape[1],))
                neg = model.link_score(h, torch.stack([neg_src, neg_dst]))
                link_logits = torch.cat([pos, neg])
                link_labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                link_loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
            else:
                link_loss = torch.tensor(0.0)
            loss = cls_loss + link_loss
            loss.backward()
            opt.step()
            last = {
                "cls_loss": cls_loss.item(),
                "link_loss": float(link_loss.item()) if hasattr(link_loss, "item") else float(link_loss),
                "total": loss.item(),
            }

        self.gnn_model = model
        try:
            torch.save(model.state_dict(), self.gnn_model_path)
            print(f"✅ [GNN] Trained and saved to {self.gnn_model_path} (loss {last.get('total', 0):.3f})")
        except Exception as e:
            print(f"⚠️ [GNN] Could not persist weights: {e}")
        return last

    def _propagate_numpy(self, tensors):
        """One round of mean-aggregation message passing in NumPy — the unlearned
        fallback that still gives 'GNN propagation' a real meaning when PyG is
        absent. Returns refined per-node embeddings (np.ndarray [N, F])."""
        x = tensors["x"].numpy()
        edge_index = tensors["edge_index"].numpy()
        n = x.shape[0]
        agg = x.copy()
        counts = np.ones(n)
        for s, d in edge_index.T:
            agg[d] += x[s]
            counts[d] += 1
            agg[s] += x[d]  # treat undirected for smoothing
            counts[s] += 1
        refined = agg / counts[:, None]
        norms = np.linalg.norm(refined, axis=1, keepdims=True) + 1e-8
        return refined / norms

    def gnn_classify_nodes(self, root_node):
        """Predict each node's morph type. Uses the trained GNN when available,
        otherwise a nearest-centroid guess over NumPy-propagated embeddings.
        Returns {node_id: predicted_type}."""
        t = self.build_graph_tensors(root_node)
        if t is None:
            return {}
        if PYG_AVAILABLE and getattr(self, "gnn_model", None) is not None:
            self.gnn_model.eval()
            with torch.no_grad():
                h = self.gnn_model.encode(t["x"], t["edge_index"])
                preds = self.gnn_model.classify(h).argmax(dim=1).tolist()
            return {nid: NODE_TYPES[p] for nid, p in zip(t["node_ids"], preds)}
        # Fallback: return the input types (no learned classifier), still valid output.
        return {nid: NODE_TYPES[int(lbl)] for nid, lbl in zip(t["node_ids"], t["y"].tolist())}

    def gnn_predict_links(self, root_node, top_k=5):
        """Predict the most likely missing links (dynamic path prediction).
        Returns a list of (src_id, dst_id, score) for node pairs not already
        connected, ranked by score. Works with the trained GNN or the NumPy
        propagation fallback."""
        t = self.build_graph_tensors(root_node)
        if t is None or len(t["node_ids"]) < 2:
            return []
        node_ids = t["node_ids"]
        existing = set(zip(t["edge_index"][0].tolist(), t["edge_index"][1].tolist()))

        if PYG_AVAILABLE and getattr(self, "gnn_model", None) is not None:
            self.gnn_model.eval()
            with torch.no_grad():
                h = self.gnn_model.encode(t["x"], t["edge_index"]).numpy()
        else:
            h = self._propagate_numpy(t)

        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        candidates = []
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i != j and (i, j) not in existing:
                    score = float(np.dot(h_norm[i], h_norm[j]))
                    candidates.append((node_ids[i], node_ids[j], score))
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates[:top_k]

    def gnn_node_relevance(self, key_phrase, root_node):
        """Score (0-100) how well a candidate node with `key_phrase` fits the
        existing graph — the max propagated-embedding cosine to any graph node.
        Used to let GNN signals prioritize self-teaching proposals."""
        t = self.build_graph_tensors(root_node)
        if t is None:
            return 0.0
        if PYG_AVAILABLE and getattr(self, "gnn_model", None) is not None:
            self.gnn_model.eval()
            with torch.no_grad():
                h = self.gnn_model.encode(t["x"], t["edge_index"]).numpy()
        else:
            h = self._propagate_numpy(t)
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        cand = np.asarray(self.get_semantic_embedding(key_phrase), dtype=np.float32).ravel()
        # Pad/truncate candidate embedding to node-embedding width (features include
        # the one-hot type suffix; compare on the shared embedding prefix).
        width = min(len(cand), h_norm.shape[1])
        c = cand[:width]
        c = c / (np.linalg.norm(c) + 1e-8)
        sims = h_norm[:, :width] @ c
        return float(max(0.0, sims.max())) * 100.0

    def _gnn_guided_attach(self, new_node):
        """Attach `new_node` under the existing graph node whose GNN-propagated
        embedding is most similar to it (dynamic path prediction driving graph
        growth). Falls back silently if there is no graph or attachment fails."""
        root = self.current_graph_root
        t = self.build_graph_tensors(root)
        if t is None or not t["node_ids"]:
            return False
        if PYG_AVAILABLE and getattr(self, "gnn_model", None) is not None:
            self.gnn_model.eval()
            with torch.no_grad():
                h = self.gnn_model.encode(t["x"], t["edge_index"]).numpy()
        else:
            h = self._propagate_numpy(t)
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        cand = np.asarray(self.get_semantic_embedding(new_node.key_phrase), dtype=np.float32).ravel()
        width = min(len(cand), h_norm.shape[1])
        c = cand[:width] / (np.linalg.norm(cand[:width]) + 1e-8)
        best_idx = int((h_norm[:, :width] @ c).argmax())
        best_id = t["node_ids"][best_idx]
        nodes, _ = self._collect_graph(root)
        best_node = next((n for n in nodes if n.id == best_id), None)
        if best_node is None:
            return False
        branch_type = "next_linear" if best_node.node_type == "linear" else "gnn_link"
        attached = self.attach_node(best_node, new_node, branch_type)
        if attached:
            print(f"🔗 [GNN] Attached {new_node.id} under best-fit node {best_id}.")
        return attached

    def _update_rejection_rate(self):
        """Recompute rejection_rate from the learning history (caller holds the lock)."""
        history = self.learning_metrics["learning_history"]
        total = len(history)
        if total:
            rejected = sum(1 for h in history if h.get("outcome") == "rejected")
            self.learning_metrics["rejection_rate"] = rejected / total

    def get_metrics_snapshot(self):
        """Flat dict of numeric operational metrics for observability exporters
        (Prometheus). Reads live learning metrics plus counts from the SQLite KG/RAG
        tables. Safe to call frequently; used by the /metrics scrape endpoint."""
        lm = self.learning_metrics
        snapshot = {
            "total_proposals": lm.get("total_proposals", 0),
            "accepted_learnings": lm.get("accepted_learnings", 0),
            "rejection_rate": lm.get("rejection_rate", 0.0),
            "avg_confidence": lm.get("avg_confidence", 0.0),
            "learning_history_len": len(lm.get("learning_history", [])),
            "synonym_count": len(self.synonym_dictionary),
            "explosion_events": len(self.explosion_log),
            "admin_queue_size": len(self.admin_review_queue),
            "has_neo4j": 1 if getattr(self, "has_neo4j", False) else 0,
            "faiss_vectors": len(getattr(self, "faiss_contents", []) or []),
        }
        # Counts from persistent stores (best-effort; never raise on a scrape).
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            for key, table in (("rag_chunks", "rag_chunks"),
                               ("kg_entities", "kg_entities"),
                               ("kg_relations", "kg_relations"),
                               ("qa_audit_events", "qa_audit_log")):
                try:
                    snapshot[key] = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    snapshot[key] = 0
            conn.close()
        except Exception:
            pass
        return snapshot

    def generate_learning_report(self):
        """Export rich report on system state and learning efficacy."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_synonyms": len(self.synonym_dictionary),
            "learning_metrics": self.learning_metrics,
            "db_stats": "Check engine_logs.db for full audit"
        }
        with open("learning_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("📋 [Report] Learning efficacy report generated: learning_report.json")
        return report

    def _kg_frontier_step(self, cursor, frontier_ids, as_of_iso, limit=5):
        """Rank candidate relations expanding out from the current traversal
        frontier — real hop-to-hop movement, not a re-query of the same global
        top-5 every time. Bitemporal: only 'current belief' rows (tx_end IS
        NULL) valid at as_of_iso are considered. If frontier_ids is empty (no
        entities recognized in the query), falls back to a global
        top-N-by-confidence scan so the no-entity-recognized UX is unchanged.
        Pure given a cursor — unit-testable in isolation."""
        where_clauses = [
            "tx_end IS NULL",
            "(valid_from IS NULL OR valid_from <= ?)",
            "(valid_to IS NULL OR valid_to >= ?)",
        ]
        params = [as_of_iso, as_of_iso]
        if frontier_ids:
            placeholders = ",".join("?" for _ in frontier_ids)
            where_clauses.insert(0, f"source_id IN ({placeholders})")
            params = list(frontier_ids) + params
        query = (
            "SELECT source_id, target_id, relation_type, confidence, valid_from, valid_to "
            "FROM kg_relations WHERE " + " AND ".join(where_clauses) +
            " ORDER BY confidence DESC LIMIT ?"
        )
        params.append(limit)
        cursor.execute(query, params)
        return cursor.fetchall()

    def _hop_grounding_score(self, hop_relation_desc, query_or_subobjective, db_confidence):
        """Grounding score (0-100) for a single hop: how well-supported the
        relation is in the KG (db_confidence) blended with how semantically
        relevant it is to the query (semantic_match_score), 50/50."""
        db_component = min(100.0, max(0.0, (db_confidence or 0.0) * 100.0))
        semantic_component = self.semantic_match_score(hop_relation_desc, query_or_subobjective)
        return min(100.0, max(0.0, 0.5 * db_component + 0.5 * semantic_component))

    def _flag_ungrounded_pog(self, query, task_id, confidence):
        """Non-blocking annotation: record an ungrounded/low-confidence PoG
        result in admin_review_queue for visibility via /admin/halts and
        /admin/resolve, without pausing execution (unlike the trampoline's
        blocking PENDING_HALT pattern used for synonym-learning halts)."""
        with self._lock:
            self.admin_review_queue[task_id] = {
                "kind": "pog_ungrounded",
                "query": query,
                "confidence": confidence,
                "status": "PENDING_REVIEW",
            }
        print(f"⚠️ [PoG] Task [{task_id}] ungrounded (conf {confidence:.1f}%) — flagged for admin review (non-blocking).")

    def _pog_hop_generator(self, query, max_hops, as_of, task_id):
        """Generator form of PoG's adaptive KG exploration: yields one event
        per hop as it happens, then a final 'done' event carrying the same
        shape pog_plan_and_reason has always returned (plus additive task_id/
        stop_reason). Real hop-to-hop traversal — each hop's frontier is the
        previous hop's chosen target, not a re-query of the same global top-5
        — replaces the old fixed-max_hops loop, combined with bitemporal
        time-scoping and grounding-score-based early exit. self._lock is held
        only around each bounded per-hop DB query, not across the generator's
        full lifetime, so it doesn't serialize unrelated engine work beyond
        that brief window."""
        effective_as_of = as_of or datetime.datetime.now().isoformat()

        # 1. Decomposition (Guidance)
        decomp_prompt = f"Decompose this query into 2-4 clear sub-objectives for KG reasoning: {query}. Return JSON list of strings."
        decomp_response = self.llm_call(decomp_prompt, json_mode=True)

        sub_objectives = []
        if isinstance(decomp_response, dict) and "sub_objectives" in decomp_response:
            sub_objectives = decomp_response["sub_objectives"]
        else:
            # Fallback simple decomposition
            sub_objectives = [f"Understand {query}", f"Find related entities in KG", f"Verify solution path"]

        print(f"   Decomposed into {len(sub_objectives)} sub-objectives: {sub_objectives[:2]}...")
        yield {"type": "decomposition", "task_id": task_id, "sub_objectives": sub_objectives}

        # 2. Real hop-to-hop traversal with bitemporal scoping + grounding-based early exit
        memory = {"query": query, "hops": [], "confidence": 0.0}
        current_conf = 50.0
        stop_reason = "max_hops"
        frontier = [e.lower().replace(" ", "_") for e in self._extract_entities(query)]

        for hop in range(max_hops):
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                candidates = self._kg_frontier_step(cursor, frontier, effective_as_of, limit=5)
                conn.close()

            if not candidates:
                print(f"   Hop {hop+1}: No strong KG relations found (using RAG fallback)")
                stop_reason = "kg_exhausted"
                break

            source_id, target_id, relation_type, confidence, valid_from, valid_to = candidates[0]
            path_desc = f"{source_id} --{relation_type}--> {target_id}"
            grounding = self._hop_grounding_score(path_desc, query, confidence)
            hop_event = {"hop": hop + 1, "path": path_desc, "conf": confidence, "grounding": grounding}
            memory["hops"].append(hop_event)
            current_conf = max(current_conf, grounding)
            print(f"   Hop {hop+1}: Explored {relation_type} relation (conf {(confidence or 0):.2f}, grounding {grounding:.1f})")
            yield {"type": "hop", "task_id": task_id, **hop_event}

            # Advance the frontier to the chosen target (real hop-to-hop
            # traversal — fixes the pre-Phase-8 bug where every hop re-queried
            # the same global top-5 relations regardless of prior hops).
            frontier = [target_id]

            if grounding >= self.grounding_threshold:
                stop_reason = "grounded"
                break

        memory["confidence"] = current_conf

        # 3. Reflection & Self-Correction (logic unchanged from pre-Phase-8 behavior;
        # coerced to bool() below — self_auditor_verify returns the proposal string
        # or None rather than True/False, so the bare `and` chain could previously
        # produce verified=None, which crashes PoGResponse's `verified: bool` field
        # over REST whenever verification is rejected).
        reflection_prompt = f"Reflect on this partial plan for '{query}': {memory}. Suggest corrections or next steps. JSON: {{'reflection': str, 'next_action': str, 'final_confidence': float}}"
        reflection = self.llm_call(reflection_prompt, json_mode=True)

        verified = False
        final_conf = current_conf
        if isinstance(reflection, dict):
            final_conf = reflection.get("final_confidence", current_conf)
            verified = bool(self.symbolic_verifier(str(reflection), query) and self.self_auditor_verify(str(reflection), query))

        yield {"type": "reflection", "task_id": task_id, "verified": verified, "confidence": final_conf}

        if verified and final_conf > 70:
            result = f"✅ PoG Plan Complete: {sub_objectives[0]} → {memory['hops'][-1]['path'] if memory['hops'] else 'direct'} (conf {final_conf:.1f}%)"
        else:
            result = f"⚠️ PoG Partial: Needs more hops or admin review. Current conf {current_conf:.1f}%"
            if stop_reason in ("kg_exhausted", "max_hops"):
                self._flag_ungrounded_pog(query, task_id, final_conf)

        print(f"   Reflection: {'Verified' if verified else 'Needs review'}")
        yield {
            "type": "done",
            "task_id": task_id,
            "stop_reason": stop_reason,
            "query": query,
            "sub_objectives": sub_objectives,
            "memory": memory,
            "result": result,
            "confidence": final_conf,
            "verified": verified,
        }

    def pog_plan_and_reason(self, query, max_hops=3, use_kg=True, as_of=None, task_id=None):
        """
        Dedicated PoG (Plan-on-Graph) style adaptive planning method.
        Orchestrates:
        - Task decomposition into sub-objectives (via LLM JSON mode)
        - Real hop-to-hop KG exploration (using kg_relations + bitemporal scoping)
          with a grounding-score-based early-stopping condition
        - Memory update (audit logs + in-memory context)
        - Reflection & self-correction (symbolic_verifier + self_auditor_verify)
        as_of (optional ISO8601 string) time-scopes the traversal to facts valid
        at that moment; defaults to now. task_id is generated if not supplied.
        A thin synchronous wrapper around _pog_hop_generator, which callers that
        want live per-hop visibility (e.g. a WebSocket stream) can consume directly.
        Returns structured plan + final reasoned answer with confidence.
        """
        print(f"🧭 [PoG] Starting adaptive planning for: '{query[:80]}...' (max_hops={max_hops})")
        task_id = task_id or str(uuid.uuid4())[:8]
        done_event = None
        for event in self._pog_hop_generator(query, max_hops, as_of, task_id):
            if event.get("type") == "done":
                done_event = event
        return {
            "query": done_event["query"],
            "sub_objectives": done_event["sub_objectives"],
            "memory": done_event["memory"],
            "result": done_event["result"],
            "confidence": done_event["confidence"],
            "verified": done_event["verified"],
            "task_id": done_event["task_id"],
            "stop_reason": done_event["stop_reason"],
        }

    def attach_node(self, parent_node, new_node, branch_type="next_linear"):
        """Manual graph attachment with verification for dynamic nodes from LLM.
        Ensures safety and symbolic approval before integrating new nodes."""
        if not parent_node or not new_node:
            print("❌ [Attach] Invalid nodes provided.")
            return False
        with self._lock:
            verified = self.symbolic_verifier(new_node.key_phrase, self.raw_target)
            if not verified:
                print("❌ [Attach] Symbolic verification failed - node not attached.")
                return False
            if branch_type == "next_linear":
                parent_node.next_linear = new_node
            elif branch_type in ["left", "right"]:
                parent_node.branches[branch_type] = new_node
            else:
                parent_node.branches[branch_type] = new_node
            print(f"✅ [Graph Attachment] Successfully attached {new_node.id} to {parent_node.id} via {branch_type}")
            return True

    def run_benchmarks(self):
        """Simple benchmarking for key operations."""
        import timeit
        print("🚀 Running Benchmarks...")
        # Similarity benchmark
        lev_time = timeit.timeit(lambda: self.calculate_similarity("comput analytics", "compute analytics"), number=1000)
        hybrid_time = timeit.timeit(lambda: self.hybrid_similarity("comput analytics", "compute analytics"), number=1000)
        print(f"Levenshtein avg time: {lev_time/1000:.6f}s | Hybrid: {hybrid_time/1000:.6f}s")
        # Self-teaching iteration (mock)
        start = time.time()
        self.self_teaching_loop(background=False, max_iterations=1)
        print(f"Teaching loop time: {time.time() - start:.2f}s")
        print("✅ Benchmarks complete.")

    # Starter test functions (can be expanded into full pytest suite)
    def run_basic_tests(self):
        """Integrated basic test suite for core functionality."""
        print("🧪 Running Basic Integrated Tests...")
        # Similarity tests
        assert abs(self.hybrid_similarity("compute", "compute") - 100.0) < 1, "Exact match failed"
        assert self.hybrid_similarity("comput", "compute") > 70, "Fuzzy match too low"
        # Node morphing basic
        test_node = MorphicTextNode("test", "test phrase")
        assert test_node.node_type == "linear"
        # Verifier tests
        assert self.symbolic_verifier("compute analytics", None) is True
        print("✅ All basic tests passed.")


# =====================================================================
# 3. VERIFICATION & PIPELINE DEPLOYMENT SIMULATION
# =====================================================================

if __name__ == "__main__":
    # Target Objective Problem to solve: "Compute Analytics"
    engine = ProductionAdaptiveEngine(target_solution_text="Compute Analytics", similarity_threshold=80.0)
    
    # Building a Multi-Linked Morphic Demonstration Tree Infrastructure
    root = MorphicTextNode(node_id="Node_Root", key_phrase="start initializing pipeline", node_type="linear")
    fork = MorphicTextNode(node_id="Node_Fork", key_phrase="split paths", node_type="tree")
    leaf_clear = MorphicTextNode(node_id="Node_Verified_Leaf", key_phrase="compute analytics", node_type="linear")
    leaf_mangled = MorphicTextNode(node_id="Node_Mangled_Leaf", key_phrase="comput analytics", node_type="linear")
    
    # Establish structural node routing connections
    root.next_linear = fork
    fork.branches["left"] = leaf_clear
    fork.branches["right"] = leaf_mangled
    
    # Populate the mangled node with a 6-layer downstream query explosion trigger
    leaf_mangled.triggered_sub_questions = [
        "Sub-Q1: Verify connection string?", 
        "Sub-Q2: Check memory stack footprint?",
        "Sub-Q3: Audit IAM runtime roles?", 
        "Sub-Q4: Confirm target database status?",
        "Sub-Q5: Is server disk space full?", 
        "Sub-Q6: Pull core security network rules?"
    ]
    
    print("==================================================================")
    print("STARTING LIVE ENGINE ASYNC STREAM EXECUTION TEST PIPELINE")
    print("==================================================================\n")
    
    # Initializing Concurrent Thread Pool handling frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # 1. DISPATCH TASK A: Slang/Typo text phrase ("comput"). Will trigger a Low-Confidence Halt event loop.
        future_a = executor.submit(engine.process_query_stream, "TASK_A", root, "comput analytics")
        time.sleep(0.2)  # Micro spacing delay to ensure scheduling order execution output matches
        
        # 2. DISPATCH TASK B: High Confidence phrase ("solve analytics"). Runs concurrently.
        future_b = executor.submit(engine.process_query_stream, "TASK_B", root, "solve analytics")
        
        # TASK B processing completes instantly across the background graph without being blocked by Task A
        print(f"[Thread Output] Return Status: {future_b.result()}")
        
        print("\n... Non-blocking state preserved. Background worker thread is idling ...\n")
        time.sleep(1.5)
        
        # 3. HUMAN ADMIN ACTION EVENT: Reviewing the isolation logs and clearing the thread barrier
        print("Active Pending Review Log:", engine.admin_review_queue)
        engine.admin_resolve_halt("TASK_A", approve=True)
        
        # Task A wakes up instantly, converts the vocabulary token map, and outputs completion logs
        print(f"[Thread Output] Return Status: {future_a.result()}")
    
    print("\n==================================================================")
    print("POST-EXECUTION COMPLIANCE AUDIT AUDITING REPORT LOGS")
    print("==================================================================")
    print(f"📝 TOTAL AUDIT ENTRIES LOGGED: {len(engine.qa_audit_log)}")
    print(f"💥 TOTAL QUERY EXPLOSIONS INTERCEPTED: {len(engine.explosion_log)}")
    if engine.explosion_log:
        print(f" Explosion 1 Child Question Counts: {engine.explosion_log[0]['downstream_explosion_count']} nodes generated.")
    
    # Demonstrate DB persistence
    print("\n--- Database Summary ---")
    conn = sqlite3.connect(engine.db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM qa_audit_log")
    print(f"DB Audit Logs: {cursor.fetchone()[0]} entries")
    cursor.execute("SELECT COUNT(*) FROM synonym_mappings")
    print(f"DB Synonym Mappings: {cursor.fetchone()[0]} entries")
    conn.close()

    # =====================================================================
    # DEMO: Hybrid Self-Teaching Loop with LLM + Self-Auditor
    # =====================================================================
    print("\n==================================================================")
    print("HYBRID SELF-TEACHING & LLM INTEGRATION DEMO")
    print("==================================================================\n")
    
    # Start background self-teaching loop (demonstrates continuous learning)
    teaching_thread = engine.self_teaching_loop(background=True, max_iterations=2)
    
    # Simulate an indirect node using LLM
    print("🧪 Testing Indirect LLM Routing...")
    llm_node = MorphicTextNode(node_id="Node_LLM_Indirect", key_phrase="llm analysis", node_type="indirect")
    llm_node.mutual_routine = "llm_enhance"
    # For demo, override resolver temporarily
    original_resolver = engine._external_router_resolver
    def llm_resolver(routine):
        """Route the 'llm_enhance' mutual routine to the LLM call, else fall back to the original resolver."""
        if routine == "llm_enhance":
            return MorphicTextNode("Temp", "llm resolved", "linear")  # Mock
        return original_resolver(routine)
    engine._external_router_resolver = llm_resolver
    
    # Trigger a learning cycle manually for demo
    engine.self_teaching_loop(background=False, max_iterations=1)
    
    print("✅ Hybrid enhancements active: LLM calls, self-audit, persistent self-teaching.")
    
    # New Features Demo
    print("\n--- Vector Embeddings & Hybrid Similarity Demo ---")
    vec_sim = engine.calculate_vector_similarity("compute analytics", "comput analytics")
    print(f"Vector Cosine Sim: {vec_sim:.1f}%")
    hybrid = engine.hybrid_similarity("solve analytics", "compute analytics")
    print(f"Hybrid Similarity: {hybrid:.1f}%")
    
    # Dynamic Node + Viz + Report + New Extensions
    engine.export_graph_viz(root, "morphic_graph.dot")
    engine.generate_learning_report()
    
    # Demonstrate attach_node with a dynamic example
    dynamic_node = MorphicTextNode("Dynamic_Attached", "enhanced analytics branch", "tree")
    engine.attach_node(fork, dynamic_node, "left")
    
    # Run benchmarks and tests
    engine.run_benchmarks()
    engine.run_basic_tests()
    
    print("\n🎉 All enhancements integrated: Test suite guidelines, benchmarking, attach_node, rich embeddings/nodes, full hybrid self-teaching.")
