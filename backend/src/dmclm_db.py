"""
DMCLM Derived Database Layer
Manages the local derived SQLite database at `data/dmclm.db`.
NEVER touches the Collector database.
Stores:
- `memory_records`: Normalized episodic memory records with calculated features and lifecycle decisions.
- `long_term_memories`: Abstracted consolidated knowledge units.
- `retrieval_events`: DMCLM local retrieval audit trail.
- `duplicate_screenshots`: Derived screenshot deduplication registry.
- `preprocessing_reports`: Execution logs and dataset metrics.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DMCLM_DB_PATH = DATA_DIR / "dmclm.db"


def get_dmclm_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Connect to the derived DMCLM SQLite database."""
    target_path = db_path or DMCLM_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_dmclm_database(db_path: Optional[Path] = None) -> None:
    """Initialize schema for derived DMCLM storage."""
    conn = get_dmclm_connection(db_path)
    try:
        cur = conn.cursor()

        # 1. Normalized episodic memory records
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_records (
                memory_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_event_ids TEXT,          -- JSON list of original raw event IDs
                session_ids TEXT,               -- JSON list of original session IDs
                application TEXT,
                content TEXT,
                domain TEXT,
                first_seen TEXT,
                last_seen TEXT,
                total_duration INTEGER DEFAULT 0,
                occurrence_count INTEGER DEFAULT 1,
                interaction_count INTEGER DEFAULT 0,
                screenshot_count INTEGER DEFAULT 0,
                retrieval_count INTEGER DEFAULT 0,
                useful_retrieval_count INTEGER DEFAULT 0,
                recency_score REAL,
                frequency_score REAL,
                retrieval_history_score REAL,
                semantic_relevance REAL,
                relationship_strength REAL,
                importance_score REAL,
                lifecycle_state TEXT,
                created_time TEXT,
                updated_time TEXT
            )
        """)

        # 2. Consolidated Long-Term Memory (LTM) nodes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                ltm_id TEXT PRIMARY KEY,
                topic TEXT,
                application TEXT,
                session_id TEXT,
                summary TEXT,
                unique_windows_accessed TEXT,    -- JSON list
                source_count INTEGER DEFAULT 0,
                source_memory_ids TEXT,          -- JSON list
                first_observed TEXT,
                last_reinforced TEXT,
                total_duration_ms INTEGER DEFAULT 0,
                total_interactions INTEGER DEFAULT 0,
                stability_score REAL DEFAULT 0.5,
                lifecycle_origin TEXT DEFAULT 'COMPRESS',
                created_at TEXT
            )
        """)

        # 3. Phase 6 Consolidated Memories
        cur.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_memories (
                consolidated_memory_id TEXT PRIMARY KEY,
                source_count INTEGER DEFAULT 0,
                source_memory_ids TEXT,          -- JSON list
                source_episode_ids TEXT,         -- JSON list
                primary_application TEXT,
                applications TEXT,               -- JSON list
                domains TEXT,                    -- JSON list
                contexts TEXT,                   -- JSON list
                start_time TEXT,
                end_time TEXT,
                duration_ms INTEGER DEFAULT 0,
                span_duration_ms INTEGER DEFAULT 0,
                semantic_representation TEXT,
                interaction_statistics TEXT,     -- JSON object
                baseline_information TEXT,       -- JSON object
                consolidation_evidence TEXT,     -- JSON object
                consolidation_metadata TEXT,     -- JSON object
                created_at TEXT
            )
        """)

        # 4. Phase 7 Adaptive Retention Evaluated Memories
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retention_evaluated_memories (
                consolidated_memory_id TEXT PRIMARY KEY,
                source_count INTEGER DEFAULT 0,
                source_memory_ids TEXT,          -- JSON list
                source_episode_ids TEXT,         -- JSON list
                primary_application TEXT,
                baseline_score REAL,
                recency REAL,
                frequency REAL,
                retrieval_history REAL,
                semantic_relevance REAL,
                relationship_strength REAL,
                stability_score REAL,
                age_days REAL,
                decay_factor REAL,
                effective_decay_rate REAL,
                adaptive_resilience REAL,
                adaptive_retention_factor REAL,
                final_retention_strength REAL,
                retention_state TEXT,
                is_protected INTEGER DEFAULT 0,
                forgetting_eligible INTEGER DEFAULT 0,
                protection_reason TEXT,
                retention_explanation TEXT,
                evaluation_timestamp TEXT,
                created_at TEXT
            )
        """)

        # 5. Phase 8 Memory Relationships
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                relationship_strength REAL NOT NULL,
                evidence TEXT,                  -- JSON object
                source_memory_source_count INTEGER DEFAULT 1,
                target_memory_source_count INTEGER DEFAULT 1,
                source_episode_ids_count INTEGER DEFAULT 0,
                target_episode_ids_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        # 6. Relationship Graph Summary Statistics
        cur.execute("""
            CREATE TABLE IF NOT EXISTS relationship_graph_stats (
                stat_id TEXT PRIMARY KEY,
                total_nodes INTEGER DEFAULT 0,
                total_edges INTEGER DEFAULT 0,
                density REAL DEFAULT 0.0,
                average_degree REAL DEFAULT 0.0,
                average_strength REAL DEFAULT 0.0,
                isolated_nodes_count INTEGER DEFAULT 0,
                connected_components_count INTEGER DEFAULT 0,
                type_distribution TEXT,         -- JSON object
                top_relationships TEXT,         -- JSON list
                created_at TEXT
            )
        """)

        # 7. Local DMCLM Retrieval audit trail
        cur.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_events (
                retrieval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                retrieval_timestamp TEXT NOT NULL,
                retrieval_type TEXT DEFAULT 'user_query',
                is_useful INTEGER DEFAULT 1,
                FOREIGN KEY (memory_id) REFERENCES memory_records (memory_id)
            )
        """)

        # 8. Screenshot deduplication metadata
        cur.execute("""
            CREATE TABLE IF NOT EXISTS duplicate_screenshots (
                sha256 TEXT PRIMARY KEY,
                canonical_screenshot_id TEXT NOT NULL,
                duplicate_count INTEGER DEFAULT 1,
                screenshot_ids TEXT,             -- JSON list of all matching screenshot IDs
                application_name TEXT,
                window_title TEXT,
                file_size INTEGER
            )
        """)

        # 9. Preprocessing execution logs & reports
        cur.execute("""
            CREATE TABLE IF NOT EXISTS preprocessing_reports (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
        """)

        # 10. Phase 9 Training Dataset Manifests
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_manifests (
                version_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                total_source_candidates INTEGER,
                total_valid_examples INTEGER,
                rejected_records_count INTEGER,
                ml_input_features_count INTEGER,
                primary_categorical_target TEXT,
                primary_continuous_target TEXT,
                train_count INTEGER,
                val_count INTEGER,
                test_count INTEGER,
                data_hash_sha256 TEXT,
                manifest_json TEXT NOT NULL
            )
        """)

        # 11. Phase 10 ML Model Training Runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_training_runs (
                run_id TEXT PRIMARY KEY,
                training_timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                feature_names TEXT NOT NULL,          -- JSON list
                hyperparameters TEXT NOT NULL,        -- JSON dict
                train_samples INTEGER,
                val_samples INTEGER,
                test_samples INTEGER,
                val_accuracy REAL,
                val_f1_macro REAL,
                test_accuracy REAL,
                test_f1_macro REAL,
                metadata_json TEXT NOT NULL
            )
        """)

        # 12. Phase 11 ML vs Baseline Comparison Runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS baseline_ml_comparisons (
                comparison_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                split TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                baseline_accuracy REAL,
                baseline_f1_macro REAL,
                ml_accuracy REAL,
                ml_f1_macro REAL,
                accuracy_delta REAL,
                f1_macro_delta REAL,
                agreement_rate REAL,
                cohen_kappa REAL,
                mcnemar_pvalue REAL,
                wilcoxon_pvalue REAL,
                summary_json TEXT NOT NULL
            )
        """)

        # 13. Phase 12 Experiment Suite Runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                experiment_id TEXT PRIMARY KEY,
                suite_id TEXT DEFAULT '',
                experiment_name TEXT NOT NULL,
                experiment_type TEXT DEFAULT '',
                category TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0,
                execution_time_sec REAL DEFAULT 0.0,
                status TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                params_json TEXT DEFAULT '{}',
                parameters_json TEXT DEFAULT '{}',
                summary_text TEXT DEFAULT ''
            )
        """)

        # Defensive migration for experiment_runs columns if table was created in earlier version
        for col, col_type in [
            ("suite_id", "TEXT DEFAULT ''"),
            ("experiment_type", "TEXT DEFAULT ''"),
            ("category", "TEXT DEFAULT ''"),
            ("sample_count", "INTEGER DEFAULT 0"),
            ("params_json", "TEXT DEFAULT '{}'"),
            ("parameters_json", "TEXT DEFAULT '{}'"),
            ("summary_text", "TEXT DEFAULT ''"),
        ]:
            try:
                cur.execute(f"ALTER TABLE experiment_runs ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass

        # 14. Phase 13 Resource Monitoring Logs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS resource_monitoring_logs (
                snapshot_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                stage TEXT DEFAULT '',
                cpu_percent REAL NOT NULL,
                cpu_count INTEGER NOT NULL,
                memory_total_mb REAL NOT NULL,
                memory_available_mb REAL NOT NULL,
                memory_percent REAL NOT NULL,
                process_rss_mb REAL NOT NULL,
                process_vms_mb REAL NOT NULL,
                process_cpu_percent REAL NOT NULL,
                profile_assigned TEXT DEFAULT ''
            )
        """)

        # 15. Phase 13 Resource-Aware Execution Runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS resource_execution_runs (
                execution_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                profile_used TEXT NOT NULL,
                stage TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                total_candidates INTEGER NOT NULL,
                processed_count INTEGER NOT NULL,
                deferred_count INTEGER NOT NULL,
                cache_hits INTEGER NOT NULL,
                cache_misses INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
        """)

        # 16. Phase 13 Deferred Memory Queue
        cur.execute("""
            CREATE TABLE IF NOT EXISTS deferred_memory_queue (
                queue_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                priority_score REAL NOT NULL,
                deferred_reason TEXT NOT NULL,
                queued_at TEXT NOT NULL,
                processing_stage TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT DEFAULT '{}'
            )
        """)

        # Create indices for fast querying
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_lifecycle ON memory_records(lifecycle_state)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_app ON memory_records(application)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_last_seen ON memory_records(last_seen)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_memory_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_memory_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON memory_relationships(relationship_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_strength ON memory_relationships(relationship_strength)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_res_log_ts ON resource_monitoring_logs(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_res_run_ts ON resource_execution_runs(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_def_queue_status ON deferred_memory_queue(status)")

        conn.commit()
    finally:
        conn.close()


def insert_or_replace_memory_records(records: List[Dict[str, Any]]) -> None:
    """Store batch of normalized memory records into SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        cur = conn.cursor()
        for r in records:
            src_ids = r.get("source_event_ids")
            if isinstance(src_ids, list):
                src_ids = json.dumps(src_ids)
            sess_ids = r.get("session_ids")
            if isinstance(sess_ids, list):
                sess_ids = json.dumps(sess_ids)

            cur.execute("""
                INSERT OR REPLACE INTO memory_records (
                    memory_id, source_type, source_event_ids, session_ids,
                    application, content, domain, first_seen, last_seen,
                    total_duration, occurrence_count, interaction_count,
                    screenshot_count, retrieval_count, useful_retrieval_count,
                    recency_score, frequency_score, retrieval_history_score,
                    semantic_relevance, relationship_strength, importance_score,
                    lifecycle_state, created_time, updated_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(r["memory_id"]),
                r.get("source_type", "activity"),
                src_ids,
                sess_ids,
                r.get("application"),
                r.get("content"),
                r.get("domain"),
                r.get("first_seen") or r.get("timestamp"),
                r.get("last_seen") or r.get("timestamp"),
                int(r.get("total_duration") or r.get("duration_ms") or 0),
                int(r.get("occurrence_count") or 1),
                int(r.get("interaction_count") or 0),
                int(r.get("screenshot_count") or 0),
                int(r.get("retrieval_count") or 0),
                int(r.get("useful_retrieval_count") or 0),
                r.get("recency_score") if r.get("recency_score") is not None else r.get("recency"),
                r.get("frequency_score") if r.get("frequency_score") is not None else r.get("frequency"),
                r.get("retrieval_history_score") if r.get("retrieval_history_score") is not None else r.get("retrieval_history"),
                r.get("semantic_relevance"),
                r.get("relationship_strength"),
                r.get("importance_score"),
                r.get("lifecycle_state") or r.get("lifecycle_prediction"),
                r.get("created_time") or datetime.now().isoformat(),
                r.get("updated_time") or datetime.now().isoformat(),
            ))
        conn.commit()
    finally:
        conn.close()


def query_memory_records(
    lifecycle: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    app_filter: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Query normalized memory records from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        where_clauses = []
        params: List[Any] = []

        if lifecycle:
            where_clauses.append("lifecycle_state = ?")
            params.append(lifecycle)
        if source_type:
            where_clauses.append("source_type = ?")
            params.append(source_type)
        if app_filter:
            where_clauses.append("application = ?")
            params.append(app_filter)
        if search:
            where_clauses.append("(content LIKE ? OR application LIKE ? OR memory_id LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"""
            SELECT * FROM memory_records
            {where_sql}
            ORDER BY last_seen DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            # Rehydrate JSON lists
            for field in ["source_event_ids", "session_ids"]:
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except Exception:
                        pass
            result.append(d)
        return result
    finally:
        conn.close()


def record_retrieval_in_db(
    memory_id: str,
    retrieval_type: str = "user_inspection",
    is_useful: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Record retrieval for an existing memory and update counters atomically.

    Validates the identifier against ``memory_records`` before inserting an
    event. The event insert and counter update share one SQLite transaction;
    any failure rolls back the transaction. ``db_path`` is optional to allow
    isolated tests to use a temporary database.
    """
    str_mid = str(memory_id).strip()
    if not str_mid:
        raise ValueError("memory_id must not be empty.")

    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        # Foreign-key enforcement is connection-specific in SQLite.
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()

        # Validate before inserting anything into the audit trail.
        cur.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ? LIMIT 1",
            (str_mid,),
        )
        if cur.fetchone() is None:
            raise ValueError(
                f"Memory record {str_mid} not found in SQLite memory_records."
            )

        cur.execute(
            """
            INSERT INTO retrieval_events (
                memory_id, retrieval_timestamp, retrieval_type, is_useful
            ) VALUES (?, ?, ?, ?)
            """,
            (str_mid, now_iso, retrieval_type, 1 if is_useful else 0),
        )

        cur.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(is_useful), 0)
            FROM retrieval_events
            WHERE memory_id = ?
            """,
            (str_mid,),
        )
        total_retrievals, useful_retrievals = cur.fetchone()

        cur.execute(
            """
            UPDATE memory_records
            SET retrieval_count = ?,
                useful_retrieval_count = ?,
                updated_time = ?
            WHERE memory_id = ?
            """,
            (total_retrievals, useful_retrievals, now_iso, str_mid),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"Expected to update one memory record for {str_mid}; "
                f"updated {cur.rowcount}."
            )

        conn.commit()
        return {
            "memory_id": str_mid,
            "retrieval_timestamp": now_iso,
            "total_retrievals": total_retrievals,
            "useful_retrievals": useful_retrievals,
            "retrieval_ratio": (
                useful_retrievals / total_retrievals
                if total_retrievals > 0
                else 0.0
            ),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_or_replace_consolidated_memories(nodes: List[Dict[str, Any]]) -> None:
    """Store batch of consolidated memory nodes into SQLite tables."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        cur = conn.cursor()
        for node in nodes:
            cid = str(node.get("consolidated_memory_id") or node.get("ltm_id"))
            src_mids = json.dumps(node.get("source_memory_ids", []))
            src_eids = json.dumps(node.get("source_episode_ids", []))
            apps = json.dumps(node.get("applications", []))
            doms = json.dumps(node.get("domains", []))
            ctxs = json.dumps(node.get("contexts", []) or node.get("unique_windows_accessed", []))
            istats = json.dumps(node.get("interaction_statistics", {}))
            binfo = json.dumps(node.get("baseline_information", {}))
            cevidence = json.dumps(node.get("consolidation_evidence", {}))
            cmetadata = json.dumps(node.get("consolidation_metadata", {}))

            cur.execute("""
                INSERT OR REPLACE INTO consolidated_memories (
                    consolidated_memory_id, source_count, source_memory_ids, source_episode_ids,
                    primary_application, applications, domains, contexts,
                    start_time, end_time, duration_ms, span_duration_ms,
                    semantic_representation, interaction_statistics, baseline_information,
                    consolidation_evidence, consolidation_metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                int(node.get("source_count", 0)),
                src_mids,
                src_eids,
                str(node.get("primary_application") or node.get("application", "")),
                apps,
                doms,
                ctxs,
                str(node.get("start_time") or node.get("first_observed", "")),
                str(node.get("end_time") or node.get("last_reinforced", "")),
                int(node.get("duration_ms") or node.get("total_duration_ms", 0)),
                int(node.get("span_duration_ms") or 0),
                str(node.get("semantic_representation") or node.get("summary", "")),
                istats,
                binfo,
                cevidence,
                cmetadata,
                str(node.get("created_at") or datetime.now().isoformat())
            ))

            # Synchronize to long_term_memories table for dashboard compatibility
            cur.execute("""
                INSERT OR REPLACE INTO long_term_memories (
                    ltm_id, topic, application, session_id, summary,
                    unique_windows_accessed, source_count, source_memory_ids,
                    first_observed, last_reinforced, total_duration_ms,
                    total_interactions, stability_score, lifecycle_origin, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                str(node.get("topic") or f"{node.get('primary_application')}: consolidated"),
                str(node.get("primary_application") or node.get("application", "")),
                str(node.get("session_id") or node.get("consolidation_metadata", {}).get("primary_session_id", "")),
                str(node.get("semantic_representation") or node.get("summary", "")),
                ctxs,
                int(node.get("source_count", 0)),
                src_mids,
                str(node.get("start_time") or node.get("first_observed", "")),
                str(node.get("end_time") or node.get("last_reinforced", "")),
                int(node.get("duration_ms") or node.get("total_duration_ms", 0)),
                int(node.get("total_interactions") or node.get("interaction_statistics", {}).get("total_interactions", 0)),
                float(node.get("stability_score") or node.get("baseline_information", {}).get("stability_score", 0.5)),
                "COMPRESS",
                str(node.get("created_at") or datetime.now().isoformat())
            ))

        conn.commit()
    finally:
        conn.close()


def query_consolidated_memories(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Query consolidated memories from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        rows = conn.execute("""
            SELECT * FROM consolidated_memories
            ORDER BY source_count DESC, duration_ms DESC
            LIMIT ? OFFSET ?
        """, (limit, offset)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for jf in ["source_memory_ids", "source_episode_ids", "applications", "domains", "contexts"]:
                if d.get(jf):
                    try:
                        d[jf] = json.loads(d[jf])
                    except Exception:
                        pass
            for of in ["interaction_statistics", "baseline_information", "consolidation_evidence", "consolidation_metadata"]:
                if d.get(of):
                    try:
                        d[of] = json.loads(d[of])
                    except Exception:
                        pass
            result.append(d)
        return result
    finally:
        conn.close()


def insert_or_replace_retention_memories(memories: List[Dict[str, Any]]) -> None:
    """Store batch of Phase 7 retention evaluated memories into SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        cur = conn.cursor()
        for m in memories:
            cid = str(m.get("consolidated_memory_id") or m.get("ltm_id"))
            src_mids = json.dumps(m.get("source_memory_ids", []))
            src_eids = json.dumps(m.get("source_episode_ids", []))

            cur.execute("""
                INSERT OR REPLACE INTO retention_evaluated_memories (
                    consolidated_memory_id, source_count, source_memory_ids, source_episode_ids,
                    primary_application, baseline_score, recency, frequency,
                    retrieval_history, semantic_relevance, relationship_strength, stability_score,
                    age_days, decay_factor, effective_decay_rate, adaptive_resilience,
                    adaptive_retention_factor, final_retention_strength, retention_state,
                    is_protected, forgetting_eligible, protection_reason, retention_explanation,
                    evaluation_timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cid,
                int(m.get("source_count", 0)),
                src_mids,
                src_eids,
                str(m.get("primary_application", "")),
                float(m.get("baseline_score", 0.0)),
                float(m.get("recency", 0.0)),
                float(m.get("frequency", 0.0)),
                float(m.get("retrieval_history", 0.0)),
                float(m.get("semantic_relevance", 0.0)),
                float(m.get("relationship_strength", 0.0)),
                float(m.get("stability_score", 0.0)),
                float(m.get("age_days", 0.0)),
                float(m.get("decay_factor", 0.0)),
                float(m.get("effective_decay_rate", 0.0)),
                float(m.get("adaptive_resilience", 0.0)),
                float(m.get("adaptive_retention_factor", 0.0)),
                float(m.get("final_retention_strength", 0.0)),
                str(m.get("retention_state", "RETAINED")),
                1 if m.get("is_protected") else 0,
                1 if m.get("forgetting_eligible") else 0,
                m.get("protection_reason"),
                str(m.get("retention_explanation", "")),
                str(m.get("evaluation_timestamp", "")),
                str(m.get("created_at") or datetime.now().isoformat())
            ))
        conn.commit()
    finally:
        conn.close()


def query_retention_memories(
    state_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Query Phase 7 retention evaluated memories from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        where_sql = "WHERE retention_state = ?" if state_filter else ""
        params: List[Any] = [state_filter] if state_filter else []
        params.extend([limit, offset])

        query = f"""
            SELECT * FROM retention_evaluated_memories
            {where_sql}
            ORDER BY final_retention_strength DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for jf in ["source_memory_ids", "source_episode_ids"]:
                if d.get(jf):
                    try:
                        d[jf] = json.loads(d[jf])
                    except Exception:
                        pass
            d["is_protected"] = bool(d.get("is_protected", 0))
            d["forgetting_eligible"] = bool(d.get("forgetting_eligible", 0))
            result.append(d)
        return result
    finally:
        conn.close()


def insert_or_replace_relationships(relationships: List[Dict[str, Any]]) -> None:
    """Store batch of Phase 8 memory relationships into SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        cur = conn.cursor()
        for rel in relationships:
            rid = str(rel.get("relationship_id"))
            evidence_json = json.dumps(rel.get("evidence", {}))

            cur.execute("""
                INSERT OR REPLACE INTO memory_relationships (
                    relationship_id, source_memory_id, target_memory_id,
                    relationship_type, relationship_strength, evidence,
                    source_memory_source_count, target_memory_source_count,
                    source_episode_ids_count, target_episode_ids_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rid,
                str(rel.get("source_memory_id")),
                str(rel.get("target_memory_id")),
                str(rel.get("relationship_type")),
                float(rel.get("relationship_strength", 0.0)),
                evidence_json,
                int(rel.get("source_memory_source_count", 1)),
                int(rel.get("target_memory_source_count", 1)),
                int(rel.get("source_episode_ids_count", 0)),
                int(rel.get("target_episode_ids_count", 0)),
                str(rel.get("created_at") or datetime.now().isoformat())
            ))
        conn.commit()
    finally:
        conn.close()


def insert_or_replace_graph_stats(stats: Dict[str, Any], stat_id: str = "latest") -> None:
    """Store Phase 8 relationship graph summary metrics into SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        cur = conn.cursor()
        type_dist_json = json.dumps(stats.get("type_distribution", {}))
        top_rels_json = json.dumps(stats.get("top_relationships", []))

        cur.execute("""
            INSERT OR REPLACE INTO relationship_graph_stats (
                stat_id, total_nodes, total_edges, density, average_degree,
                average_strength, isolated_nodes_count, connected_components_count,
                type_distribution, top_relationships, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stat_id,
            int(stats.get("total_nodes", 0)),
            int(stats.get("total_edges", 0)),
            float(stats.get("density", 0.0)),
            float(stats.get("average_degree", 0.0)),
            float(stats.get("average_strength", 0.0)),
            int(stats.get("isolated_nodes_count", 0)),
            int(stats.get("connected_components_count", 0)),
            type_dist_json,
            top_rels_json,
            str(stats.get("created_at") or datetime.now().isoformat())
        ))
        conn.commit()
    finally:
        conn.close()


def query_relationships(
    memory_id: Optional[str] = None,
    rel_type: Optional[str] = None,
    min_strength: float = 0.0,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Query Phase 8 memory relationships from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        where_clauses = ["relationship_strength >= ?"]
        params: List[Any] = [min_strength]

        if memory_id:
            where_clauses.append("(source_memory_id = ? OR target_memory_id = ?)")
            params.extend([memory_id, memory_id])

        if rel_type:
            where_clauses.append("relationship_type = ?")
            params.append(rel_type)

        where_sql = f"WHERE {' AND '.join(where_clauses)}"
        params.extend([limit, offset])

        query = f"""
            SELECT * FROM memory_relationships
            {where_sql}
            ORDER BY relationship_strength DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("evidence"):
                try:
                    d["evidence"] = json.loads(d["evidence"])
                except Exception:
                    pass
            result.append(d)
        return result
    finally:
        conn.close()


def get_relationship_graph_stats(stat_id: str = "latest") -> Optional[Dict[str, Any]]:
    """Retrieve Phase 8 relationship graph summary metrics from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        row = conn.execute("SELECT * FROM relationship_graph_stats WHERE stat_id = ?", (stat_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for jf in ["type_distribution", "top_relationships"]:
            if d.get(jf):
                try:
                    d[jf] = json.loads(d[jf])
                except Exception:
                    pass
        return d
    finally:
        conn.close()


def insert_or_replace_dataset_manifest(manifest: Dict[str, Any], version_id: Optional[str] = None) -> None:
    """Store or update Phase 9 dataset manifest in SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        vid = version_id or manifest.get("dataset_version", "v1.0")
        conn.execute("""
            INSERT OR REPLACE INTO dataset_manifests (
                version_id, created_at, total_source_candidates, total_valid_examples,
                rejected_records_count, ml_input_features_count, primary_categorical_target,
                primary_continuous_target, train_count, val_count, test_count,
                data_hash_sha256, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vid,
            manifest.get("created_at"),
            manifest.get("total_source_candidates"),
            manifest.get("total_valid_examples"),
            manifest.get("rejected_records_count"),
            manifest.get("ml_input_features_count"),
            manifest.get("primary_categorical_target"),
            manifest.get("primary_continuous_target"),
            manifest.get("train_count"),
            manifest.get("val_count"),
            manifest.get("test_count"),
            manifest.get("data_hash_sha256"),
            json.dumps(manifest),
        ))
        conn.commit()
    finally:
        conn.close()


def get_dataset_manifest(version_id: str = "latest") -> Optional[Dict[str, Any]]:
    """Retrieve Phase 9 dataset manifest from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        if version_id == "latest":
            row = conn.execute("SELECT * FROM dataset_manifests ORDER BY created_at DESC LIMIT 1").fetchone()
        else:
            row = conn.execute("SELECT * FROM dataset_manifests WHERE version_id = ?", (version_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("manifest_json"):
            try:
                return json.loads(d["manifest_json"])
            except Exception:
                pass
        return d
    finally:
        conn.close()


def insert_or_replace_model_run(metadata: Dict[str, Any], run_id: Optional[str] = None) -> None:
    """Store or update Phase 10 ML model training run in SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        rid = run_id or f"run_{metadata.get('model_version', 'v1.0')}_{int(datetime.now().timestamp())}"
        val_m = metadata.get("val_metrics", {})
        test_m = metadata.get("test_metrics", {})

        conn.execute("""
            INSERT OR REPLACE INTO model_training_runs (
                run_id, training_timestamp, model_version, algorithm, feature_names,
                hyperparameters, train_samples, val_samples, test_samples,
                val_accuracy, val_f1_macro, test_accuracy, test_f1_macro, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rid,
            metadata.get("training_timestamp"),
            metadata.get("model_version", "v1.0"),
            metadata.get("algorithm", "RandomForestClassifier"),
            json.dumps(metadata.get("feature_names", [])),
            json.dumps(metadata.get("hyperparameters", {})),
            metadata.get("train_samples", 0),
            metadata.get("val_samples", 0),
            metadata.get("test_samples", 0),
            val_m.get("accuracy"),
            val_m.get("f1_macro"),
            test_m.get("accuracy"),
            test_m.get("f1_macro"),
            json.dumps(metadata),
        ))
        conn.commit()
    finally:
        conn.close()


def get_latest_model_run() -> Optional[Dict[str, Any]]:
    """Retrieve the latest Phase 10 model training run metadata from SQLite."""
    init_dmclm_database()
    conn = get_dmclm_connection()
    try:
        row = conn.execute("SELECT * FROM model_training_runs ORDER BY training_timestamp DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("metadata_json"):
            try:
                return json.loads(d["metadata_json"])
            except Exception:
                pass
        return d
    finally:
        conn.close()


def insert_or_replace_comparison_run(summary: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """Inserts or updates Phase 11 baseline vs ML comparison run summary in SQLite."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        cid = summary.get("comparison_id") or f"comp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        b_class = summary.get("baseline_classification_metrics", {})
        m_class = summary.get("ml_classification_metrics", {})
        class_diff = summary.get("classification_differences", {})
        agree = summary.get("agreement_analysis", {})
        stats = summary.get("statistical_tests", {})

        mcnemar_p = stats.get("mcnemar_test", {}).get("p_value")
        wilcoxon_p = stats.get("wilcoxon_test", {}).get("p_value")

        conn.execute("""
            INSERT OR REPLACE INTO baseline_ml_comparisons (
                comparison_id, timestamp, split, sample_count,
                baseline_accuracy, baseline_f1_macro,
                ml_accuracy, ml_f1_macro,
                accuracy_delta, f1_macro_delta,
                agreement_rate, cohen_kappa,
                mcnemar_pvalue, wilcoxon_pvalue,
                summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cid,
            summary.get("timestamp"),
            summary.get("evaluation_split", "test"),
            summary.get("sample_count", 0),
            b_class.get("accuracy"),
            b_class.get("f1_macro"),
            m_class.get("accuracy"),
            m_class.get("f1_macro"),
            class_diff.get("accuracy_delta"),
            class_diff.get("f1_macro_delta"),
            agree.get("agreement_rate"),
            agree.get("cohen_kappa"),
            mcnemar_p,
            wilcoxon_p,
            json.dumps(summary),
        ))
        conn.commit()
    finally:
        conn.close()


def get_latest_comparison_run(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve the latest Phase 11 baseline vs ML comparison run summary from SQLite."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM baseline_ml_comparisons ORDER BY timestamp DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("summary_json"):
            try:
                return json.loads(d["summary_json"])
            except Exception:
                pass
        return d
    finally:
        conn.close()


def insert_or_replace_experiment_run(record: Dict[str, Any], db_path: Optional[Path] = None) -> None:
    """Inserts or updates an individual Phase 12 experiment run record in SQLite."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        eid = record.get("experiment_id") or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        suite_id = record.get("suite_id") or record.get("parameters", {}).get("suite_id", "dmclm_phase12_suite")
        name = record.get("experiment_name", "unnamed_experiment")
        etype = record.get("experiment_type", "EVALUATION")
        ts = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
        samples = record.get("sample_count", 0)
        exec_sec = float(record.get("execution_time_seconds", 0.0))
        status = record.get("status", "SUCCESS")
        metrics_json = json.dumps(record.get("metrics", {}))
        params_json = json.dumps(record.get("parameters", {}))

        category = record.get("category") or etype
        summary_text = record.get("summary_text", "")

        conn.execute("""
            INSERT OR REPLACE INTO experiment_runs (
                experiment_id, suite_id, experiment_name, experiment_type,
                category, timestamp, sample_count, execution_time_sec, status,
                metrics_json, params_json, parameters_json, summary_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eid, suite_id, name, etype,
            category, ts, samples, exec_sec, status,
            metrics_json, params_json, params_json, summary_text
        ))
        conn.commit()
    finally:
        conn.close()


def get_latest_experiment_runs(suite_id: Optional[str] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve Phase 12 experiment runs from SQLite, optionally filtered by suite_id."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        if suite_id:
            rows = conn.execute(
                "SELECT * FROM experiment_runs WHERE suite_id = ? ORDER BY timestamp ASC",
                (suite_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiment_runs ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            if d.get("metrics_json"):
                try:
                    d["metrics"] = json.loads(d["metrics_json"])
                except Exception:
                    pass
            if d.get("params_json"):
                try:
                    d["parameters"] = json.loads(d["params_json"])
                except Exception:
                    pass
            results.append(d)
        return results
    finally:
        conn.close()


# =============================================================================
# PHASE 13: RESOURCE-AWARE REASONING DATABASE OPERATIONS
# =============================================================================

def insert_resource_monitoring_log(
    snapshot: Dict[str, Any],
    profile_assigned: str = "",
    db_path: Optional[Path] = None
) -> None:
    """Records an empirical technical system resource measurement snapshot."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        snap_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        ts = snapshot.get("timestamp") or datetime.now(timezone.utc).isoformat()
        stage = snapshot.get("stage", "")
        cpu_p = float(snapshot.get("cpu_percent", 0.0))
        cpu_c = int(snapshot.get("cpu_count", 1))
        mem_tot = float(snapshot.get("memory_total_mb", 0.0))
        mem_avail = float(snapshot.get("memory_available_mb", 0.0))
        mem_p = float(snapshot.get("memory_percent", 0.0))
        rss = float(snapshot.get("process_rss_mb", 0.0))
        vms = float(snapshot.get("process_vms_mb", 0.0))
        p_cpu = float(snapshot.get("process_cpu_percent", 0.0))

        conn.execute("""
            INSERT OR REPLACE INTO resource_monitoring_logs (
                snapshot_id, timestamp, stage, cpu_percent, cpu_count,
                memory_total_mb, memory_available_mb, memory_percent,
                process_rss_mb, process_vms_mb, process_cpu_percent,
                profile_assigned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snap_id, ts, stage, cpu_p, cpu_c, mem_tot, mem_avail,
            mem_p, rss, vms, p_cpu, profile_assigned
        ))
        conn.commit()
    finally:
        conn.close()


def insert_resource_execution_run(
    run_result: Dict[str, Any],
    db_path: Optional[Path] = None
) -> None:
    """Stores the empirical execution summary of a resource-aware processing cycle."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        eid = run_result.get("execution_id") or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        ts = run_result.get("timestamp") or datetime.now(timezone.utc).isoformat()
        prof = run_result.get("profile_used", "NORMAL")
        stage = run_result.get("stage", "general")
        duration = float(run_result.get("duration_seconds", 0.0))
        total_cand = int(run_result.get("total_candidates", 0))
        proc_cnt = int(run_result.get("processed_count", 0))
        def_cnt = int(run_result.get("deferred_count", 0))
        c_hits = int(run_result.get("cache_hits", 0))
        c_miss = int(run_result.get("cache_misses", 0))
        status = run_result.get("status", "SUCCESS")
        summary_json = json.dumps(run_result)

        conn.execute("""
            INSERT OR REPLACE INTO resource_execution_runs (
                execution_id, timestamp, profile_used, stage, duration_seconds,
                total_candidates, processed_count, deferred_count, cache_hits,
                cache_misses, status, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            eid, ts, prof, stage, duration, total_cand, proc_cnt, def_cnt,
            c_hits, c_miss, status, summary_json
        ))
        conn.commit()
    finally:
        conn.close()


def enqueue_deferred_memory(
    record: Dict[str, Any],
    db_path: Optional[Path] = None
) -> None:
    """Inserts a memory into the deferred queue during resource-constrained execution."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        qid = record.get("queue_id") or f"q_{hashlib.sha256(str(record.get('memory_id', '')).encode('utf-8')).hexdigest()[:16]}"
        mid = str(record.get("memory_id", ""))
        p_score = float(record.get("priority_score", 0.0))
        reason = str(record.get("deferred_reason", "resource_pressure"))
        q_at = record.get("queued_at") or datetime.now(timezone.utc).isoformat()
        stage = str(record.get("processing_stage", "consolidation"))
        status = str(record.get("status", "PENDING"))
        payload_json = json.dumps(record.get("payload", {}))

        conn.execute("""
            INSERT OR REPLACE INTO deferred_memory_queue (
                queue_id, memory_id, priority_score, deferred_reason,
                queued_at, processing_stage, status, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (qid, mid, p_score, reason, q_at, stage, status, payload_json))
        conn.commit()
    finally:
        conn.close()


def get_pending_deferred_memories(
    limit: int = 100,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Retrieves pending deferred memories ordered by priority score descending."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT * FROM deferred_memory_queue
            WHERE status = 'PENDING'
            ORDER BY priority_score DESC, queued_at ASC
            LIMIT ?
        """, (limit,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("payload_json"):
                try:
                    d["payload"] = json.loads(d["payload_json"])
                except Exception:
                    d["payload"] = {}
            results.append(d)
        return results
    finally:
        conn.close()


def update_deferred_memory_status(
    queue_id: str,
    status: str,
    db_path: Optional[Path] = None
) -> None:
    """Updates status of a queued deferred memory (e.g. to 'PROCESSED')."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        conn.execute(
            "UPDATE deferred_memory_queue SET status = ? WHERE queue_id = ?",
            (status, queue_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_resource_runs(
    limit: int = 50,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Returns recent Phase 13 resource execution runs."""
    init_dmclm_database(db_path)
    conn = get_dmclm_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT * FROM resource_execution_runs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("summary_json"):
                try:
                    d["summary"] = json.loads(d["summary_json"])
                except Exception:
                    pass
            results.append(d)
        return results
    finally:
        conn.close()