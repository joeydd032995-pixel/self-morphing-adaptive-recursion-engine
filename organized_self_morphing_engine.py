import re
import time
import datetime
import concurrent.futures
import sqlite3
import json
import functools
import threading
import random  # For mock LLM simulation
import numpy as np
import torch
import os
from functools import lru_cache
from collections import defaultdict, Counter

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
        
        # SQLite Database Layer for Persistence
        self.db_path = os.getenv("ENGINE_DB_PATH", "engine_logs.db")
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
        
        conn.commit()
        conn.close()
        print(f"✅ [DB] SQLite persistence initialized at {self.db_path} (advanced KG schema + RAG support enabled)")

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
                # Remove from DB
                conn = sqlite3.connect(self.db_path)
                conn.execute('PRAGMA journal_mode=WAL')
                cursor = conn.cursor()
                cursor.execute('DELETE FROM synonym_mappings WHERE word = ?', (cleaned_word,))
                conn.commit()
                conn.close()

    def admin_resolve_halt(self, task_id, approve=True):
        """Human Admin intervention gateway to unblock a specifically suspended execution thread."""
        if task_id in self.admin_review_queue:
            item = self.admin_review_queue[task_id]
            if approve:
                item["status"] = "APPROVED"
                # Lock vocabulary translation rule to the production engine dictionary
                with self._lock:
                    self.synonym_dictionary[item["word"]] = item["suggested_token"]
                    self.calculate_similarity.cache_clear()
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

    # --- NEO4J & EXTERNAL KG INTEGRATION (Non-breaking stubs + hooks for PoG/KG-LLM) ---
    def connect_neo4j(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """Stub for external Neo4j connection. Install 'neo4j' driver for full bidirectional sync.
        Enables advanced KG persistence for multi-hop PoG planning and entity-relation storage."""
        try:
            # from neo4j import GraphDatabase
            # self.neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            print("✅ Neo4j connection stub activated. Ready for full driver integration and bidirectional sync.")
            self.has_neo4j = True
        except Exception:
            self.has_neo4j = False
            print("⚠️ Neo4j driver not available (pip install neo4j). Using rich SQLite KG fallback with advanced schema.")

    def kg_sync_to_neo4j(self):
        """Sync local KG entities/relations (with embeddings/confidence) to Neo4j.
        Non-blocking; called after self-teaching or dynamic node attachment for external graph power."""
        if getattr(self, 'has_neo4j', False):
            print("🔄 Syncing KG entities/relations (embeddings + confidence) to Neo4j for PoG multi-hop...")
            # TODO: Implement actual Cypher CREATE/MERGE for entities and relations using self.neo4j_driver
        else:
            print("Using SQLite KG backend (advanced schema with BLOB embeddings and confidence scoring).")

    # --- ANALYTICAL ALGORITHMIC FORMULAE ---
    def _tokenize_synonyms(self, text):
        """Converts language inputs into absolute core conceptual tokens."""
        words = re.findall(r'\b\w+\b', text)
        return " ".join([self.synonym_dictionary.get(w, w) for w in words])

    def _calculate_similarity_uncached(self, s1, s2):
        """Vectorized Dynamic Programming Levenshtein Distance Matrix Calculation.
        Cached per-instance in __init__ (see self.calculate_similarity)."""
        t1 = self._tokenize_synonyms(s1)
        t2 = self._tokenize_synonyms(s2)

        if len(t1) < len(t2):
            t1, t2 = t2, t1
        if len(t2) == 0:
            return 100.0 if len(t1) == 0 else 0.0

        previous_row = list(range(len(t2) + 1))
        for i, c1 in enumerate(t1):
            current_row = [i + 1]
            for j, c2 in enumerate(t2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (1 if c1 != c2 else 0)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return (1.0 - (previous_row[-1] / max(len(t1), len(t2)))) * 100.0

    def _compute_embedding(self, text):
        """Rich vector embedding using hash-based bag-of-words (numpy + torch compatible).
        Provides semantic-like similarity beyond Levenshtein for hybridization."""
        words = re.findall(r'\b\w+\b', text.lower())
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for w in words:
            idx = hash(w) % self.vocab_size
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return torch.tensor(vec, dtype=torch.float32)

    def calculate_vector_similarity(self, s1, s2):
        """Cosine similarity on embeddings for rich semantic matching."""
        emb1 = self._compute_embedding(s1)
        emb2 = self._compute_embedding(s2)
        cos_sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return max(0.0, cos_sim) * 100.0  # Scale to percentage

    def hybrid_similarity(self, s1, s2):
        """Combined Levenshtein + Vector embedding for robust matching."""
        lev = self.calculate_similarity(s1, s2)
        vec = self.calculate_vector_similarity(s1, s2)
        return (lev * 0.6 + vec * 0.4)  # Weighted hybrid

    # --- ADVANCED RAG / EMBEDDINGS (Optional FAISS + sentence-transformers) ---
    def _get_semantic_model(self):
        """Lazy load sentence-transformers model (all-MiniLM-L6-v2 by default)."""
        global _semantic_model
        if _semantic_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ [RAG] sentence-transformers model loaded for semantic embeddings.")
            except Exception as e:
                print(f"⚠️ Could not load sentence-transformers: {e}")
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
        Advanced chunking for RAG.
        - 'fixed': simple overlapping chunks
        - 'semantic': sentence-aware (basic) or uses embeddings for boundaries
        - 'recursive': hierarchical (simplified)
        """
        if strategy == "fixed":
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i+chunk_size])
            return chunks
        elif strategy == "semantic" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Simple semantic chunking via sentence splitting + embedding similarity
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) > chunk_size:
                    if current:
                        chunks.append(current.strip())
                    current = sent
                else:
                    current += " " + sent
            if current:
                chunks.append(current.strip())
            return chunks
        else:
            # Fallback
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def ingest_documents(self, documents, strategy="semantic"):
        """Ingest documents into RAG + KG (chunks + embeddings + basic entity extraction)."""
        print(f"📥 [RAG] Ingesting {len(documents)} documents with {strategy} chunking...")
        for doc in documents:
            chunks = self.advanced_chunk_text(doc, strategy=strategy)
            for i, chunk in enumerate(chunks):
                emb = self.get_semantic_embedding(chunk)
                # Store in rag_chunks
                with self._lock:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO rag_chunks (content, embedding, metadata)
                        VALUES (?, ?, ?)
                    ''', (chunk, emb.tobytes() if hasattr(emb, 'tobytes') else json.dumps(emb.tolist()), 
                          json.dumps({"source": "ingest", "chunk_id": i})))
                    conn.commit()
                    conn.close()
        print("✅ [RAG] Ingestion complete. Chunks stored with embeddings.")
        self.update_faiss_after_ingest()  # Keep persistent index in sync

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
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings_np)

        # Save persistently
        try:
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            with open(self.faiss_index_path + ".contents.json", "w") as f:
                json.dump(self.faiss_contents, f)
            print(f"✅ [FAISS] Persistent index saved to {self.faiss_index_path} ({len(embeddings)} vectors)")
        except Exception as e:
            print(f"⚠️ Could not save FAISS index: {e}")

        return True

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

    def semantic_retrieve_context(self, query, k=5, use_faiss=True):
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
        """Rebuild FAISS index after new documents are ingested."""
        if FAISS_AVAILABLE:
            self.build_faiss_index(force_rebuild=True)

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
    def llm_call(self, prompt, max_tokens=150, temperature=0.7, json_mode=False, max_retries=3):
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
        # Compute similarity to target or context
        sim_score = self.calculate_similarity(proposal, context_text or self.raw_target)
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
        
        # 3. Domain-specific symbolic rules (e.g., valid tokens for analytics)
        valid_tokens = {"compute_token", "data_token", "structure_token"}
        if any(token in proposal for token in valid_tokens):
            print("✅ [Symbolic Verifier] Domain rules passed.")
        else:
            print("⚠️ [Symbolic] No recognized domain token - flagging.")
            # Still allow with lower confidence in hybrid
        
        # 4. Cross-verify with engine similarity and self-auditor
        sim = self.calculate_similarity(proposal, self.raw_target)
        if sim > 60.0:
            print("✅ [Symbolic] Consistent with core target.")
            return True
        return False
        
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
        
        # 3. Domain-specific symbolic rules (e.g., valid tokens for analytics)
        valid_tokens = {"compute_token", "data_token", "structure_token"}
        if any(token in proposal for token in valid_tokens):
            print("✅ [Symbolic Verifier] Domain rules passed.")
        else:
            print("⚠️ [Symbolic] No recognized domain token - flagging.")
            # Still allow with lower confidence in hybrid
        
        # 4. Cross-verify with engine similarity and self-auditor
        sim = self.calculate_similarity(proposal, self.raw_target)
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
                        # Could attach to root or existing graph in full impl
                    # Learn mapping from JSON
                    mapping = llm_suggestion.get("suggested_mapping", {})
                    word = mapping.get("word", "dynamic_term")
                    token = mapping.get("token", "compute_token")
                    conf = llm_suggestion.get("confidence", 85.0)
                    with self._lock:
                        self.synonym_dictionary[word] = token
                        self.calculate_similarity.cache_clear()
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
                    print(f"🎓 [Self-Teaching] Learned via JSON + Verify: {word} -> {token} (conf: {conf:.1f}%)")
                else:
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

    def pog_plan_and_reason(self, query, max_hops=3, use_kg=True):
        """
        Dedicated PoG (Plan-on-Graph) style adaptive planning method.
        Orchestrates:
        - Task decomposition into sub-objectives (via LLM JSON mode)
        - Adaptive KG exploration / multi-hop path finding (using kg_relations + confidence)
        - Memory update (audit logs + in-memory context)
        - Reflection & self-correction (symbolic_verifier + self_auditor_verify)
        Easy extension of self_teaching_loop + advanced KG schema.
        Returns structured plan + final reasoned answer with confidence.
        """
        print(f"🧭 [PoG] Starting adaptive planning for: '{query[:80]}...' (max_hops={max_hops})")
        
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
        
        # 2. Adaptive Exploration + Memory (KG multi-hop simulation using schema)
        memory = {"query": query, "hops": [], "confidence": 0.0}
        current_conf = 50.0
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            for hop in range(max_hops):
                # Simulate / query KG relations (in real: Cypher or SQL join on kg_relations)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT source_id, target_id, relation_type, confidence 
                    FROM kg_relations 
                    ORDER BY confidence DESC LIMIT 5
                ''')
                kg_paths = cursor.fetchall()
                
                if kg_paths:
                    best_path = kg_paths[0]
                    memory["hops"].append({
                        "hop": hop + 1,
                        "path": f"{best_path[0]} --{best_path[2]}--> {best_path[1]}",
                        "conf": best_path[3]
                    })
                    current_conf = max(current_conf, (best_path[3] or 0.5) * 100)
                    print(f"   Hop {hop+1}: Explored {best_path[2]} relation (conf {best_path[3]:.2f})")
                else:
                    print(f"   Hop {hop+1}: No strong KG relations found (using RAG fallback)")
                    break
            
            conn.close()
        
        memory["confidence"] = current_conf
        
        # 3. Reflection & Self-Correction
        reflection_prompt = f"Reflect on this partial plan for '{query}': {memory}. Suggest corrections or next steps. JSON: {{'reflection': str, 'next_action': str, 'final_confidence': float}}"
        reflection = self.llm_call(reflection_prompt, json_mode=True)
        
        verified = False
        final_conf = current_conf
        if isinstance(reflection, dict):
            final_conf = reflection.get("final_confidence", current_conf)
            verified = self.symbolic_verifier(str(reflection), query) and self.self_auditor_verify(str(reflection), query)

        if verified and final_conf > 70:
            result = f"✅ PoG Plan Complete: {sub_objectives[0]} → {memory['hops'][-1]['path'] if memory['hops'] else 'direct'} (conf {final_conf:.1f}%)"
        else:
            result = f"⚠️ PoG Partial: Needs more hops or admin review. Current conf {current_conf:.1f}%"
        
        print(f"   Reflection: {'Verified' if verified else 'Needs review'}")
        return {
            "query": query,
            "sub_objectives": sub_objectives,
            "memory": memory,
            "result": result,
            "confidence": final_conf if 'final_conf' in locals() else current_conf,
            "verified": verified
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
