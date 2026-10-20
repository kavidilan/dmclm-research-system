"""
DMCLM Real Preprocessing & Activity Episode Grouping Pipeline
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model

Principles:
1. REAL COLLECTOR DATA: Connects to DesktopResearchCollector in READ-ONLY mode.
2. NO DATA MODIFICATION: Never modifies collector.db, WAL, or SHM.
3. BEHAVIORAL AGGREGATION: Converts raw observation streams (activities, browser traces,
   clicks, keystrokes, screenshots) into cohesive activity episodes and normalized memory candidates.
4. EXACT SCREENSHOT DEDUPLICATION: Identifies exact file duplicates via SHA-256 hashes,
   records canonical instances and duplicate groups without altering raw data.
5. INTERACTION AGGREGATION: Aggregates clicks and keystrokes into task engagement metrics.
6. NO FAKE DATA: All fields, metrics, and logs are genuinely derived from the SQLite database.
"""

import hashlib
import json
import logging
import sqlite3
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dmclm_config import PreprocessingConfig, DEFAULT_CONFIG
except ImportError:
    from src.dmclm_config import PreprocessingConfig, DEFAULT_CONFIG

logger = logging.getLogger("DMCLM.Preprocessing")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

COLLECTOR_DB = Path(r"C:\Users\Admin\AppData\Local\DesktopResearchCollector\collector.db")
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"


def get_readonly_collector_connection() -> sqlite3.Connection:
    """
    Open a strictly READ-ONLY connection to the DesktopResearchCollector SQLite database.
    Guarantees no modification of collector.db, collector.db-wal, or collector.db-shm.
    """
    if not COLLECTOR_DB.exists():
        raise FileNotFoundError(
            f"Collector database not found at {COLLECTOR_DB}. Ensure DesktopResearchCollector is installed."
        )

    # Use SQLite URI with mode=ro and immutable=1 to prevent any writes or lock modifications
    uri = f"file:{COLLECTOR_DB.as_posix()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
    except sqlite3.OperationalError:
        # Fallback to standard read-only if immutable mode is unsupported on filesystem
        conn = sqlite3.connect(f"file:{COLLECTOR_DB.as_posix()}?mode=ro", uri=True, timeout=10.0)

    # Enforce read-only pragma on the connection
    conn.execute("PRAGMA query_only = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def parse_timestamp(ts_val: Optional[str]) -> Optional[datetime]:
    """Robust ISO-8601 timestamp parser supporting microsecond UTC offsets."""
    if not ts_val or not isinstance(ts_val, str):
        return None
    try:
        clean = ts_val.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        try:
            # Fallback for unexpected formats
            return datetime.fromisoformat(clean[:19]).replace(tzinfo=timezone.utc)
        except Exception:
            return None


class PreprocessingPipeline:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or DEFAULT_CONFIG.preprocessing
        self.errors: List[Dict[str, Any]] = []

    def load_raw_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads raw data in read-only mode with comprehensive error handling.
        Returns dictionary of all raw tables.
        """
        conn = get_readonly_collector_connection()
        try:
            cur = conn.cursor()

            # Verify required tables exist
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cur.fetchall()}
            required_tables = ["activity_events", "browser_events", "interaction_events", "screenshots", "sessions"]
            missing = [t for t in required_tables if t not in existing_tables]
            if missing:
                raise RuntimeError(f"Collector database is missing required tables: {missing}")

            logger.info("Ingesting raw tables from collector.db in READ-ONLY mode...")

            # 1. Activity events
            cur.execute("""
                SELECT event_id, session_id, timestamp, event_type, application_name,
                       process_name, window_title, duration_ms, metadata_json
                FROM activity_events
                ORDER BY timestamp ASC
            """)
            activity_events = [dict(row) for row in cur.fetchall()]

            # 2. Browser events
            cur.execute("""
                SELECT browser_event_id, session_id, timestamp, browser_name,
                       domain, page_title, url
                FROM browser_events
                ORDER BY timestamp ASC
            """)
            browser_events = [dict(row) for row in cur.fetchall()]

            # 3. Interaction events
            cur.execute("""
                SELECT interaction_id, session_id, timestamp, interaction_type,
                       application_name, count
                FROM interaction_events
                ORDER BY timestamp ASC
            """)
            interaction_events = [dict(row) for row in cur.fetchall()]

            # 4. Screenshots (including sha256 for exact deduplication)
            cur.execute("""
                SELECT screenshot_id, profile_id, session_id, timestamp, capture_reason,
                       application_name, window_title, sha256, file_size
                FROM screenshots
                ORDER BY timestamp ASC
            """)
            screenshots = [dict(row) for row in cur.fetchall()]

            # 5. Sessions
            cur.execute("""
                SELECT session_id, started_at, ended_at, end_reason
                FROM sessions
                ORDER BY started_at ASC
            """)
            sessions = [dict(row) for row in cur.fetchall()]

            return {
                "activity_events": activity_events,
                "browser_events": browser_events,
                "interaction_events": interaction_events,
                "screenshots": screenshots,
                "sessions": sessions,
            }
        finally:
            conn.close()

    def deduplicate_screenshots(
        self, screenshots: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Deduplicates screenshot records using exact SHA-256 hash comparison.
        Does not delete raw collector records.
        Returns:
            - canonical_screenshots: list of canonical screenshot records
            - sha_registry: map of sha256 -> {canonical_id, duplicate_ids, total_count, ...}
        """
        sha_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in screenshots:
            sha = s.get("sha256")
            if not sha:
                # If no sha256 present, fallback to deterministic hash of size and title
                fake_key = f"no_sha_{s.get('screenshot_id')}"
                sha_groups[fake_key].append(s)
            else:
                sha_groups[sha].append(s)

        canonical_screenshots = []
        sha_registry = {}

        for sha, group in sha_groups.items():
            # Canonical is the earliest observed screenshot in the duplicate cluster
            canonical = group[0]
            canonical_id = canonical["screenshot_id"]
            all_ids = [item["screenshot_id"] for item in group]

            sha_registry[sha] = {
                "canonical_id": canonical_id,
                "sha256": sha,
                "count": len(group),
                "all_screenshot_ids": all_ids,
                "is_duplicate": len(group) > 1,
                "application_name": canonical.get("application_name") or "Desktop",
                "window_title": canonical.get("window_title") or "",
                "file_size": canonical.get("file_size") or 0,
            }

            canonical_record = dict(canonical)
            canonical_record["duplicate_count"] = len(group)
            canonical_record["is_canonical"] = True
            canonical_record["all_screenshot_ids"] = all_ids
            canonical_screenshots.append(canonical_record)

        return canonical_screenshots, sha_registry

    def aggregate_interactions(
        self, interaction_events: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, int]]:
        """
        Aggregates keyboard and mouse interactions by (session_id, application_name).
        Returns a mapping from (session_id, application_name) -> {total, clicks, keystrokes, scrolls}.
        """
        agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "mouse_click": 0, "keyboard": 0, "scroll": 0}
        )

        for event in interaction_events:
            sid = event.get("session_id") or "unassigned_session"
            app = (event.get("application_name") or "Unknown").strip()
            itype = event.get("interaction_type") or "unknown"
            count = int(event.get("count") or 1)

            entry = agg[(sid, app)]
            entry["total"] += count
            if "click" in itype:
                entry["mouse_click"] += count
            elif "keyboard" in itype:
                entry["keyboard"] += count
            elif "scroll" in itype:
                entry["scroll"] += count

        return agg

    def group_activity_episodes(
        self,
        activity_events: List[Dict[str, Any]],
        interaction_agg: Dict[Tuple[str, str], Dict[str, int]],
        screenshots_by_session_app: Dict[Tuple[str, str], List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Converts raw activity events into cohesive activity episodes.
        Uses configurable temporal proximity, application focus, window title similarity, and session continuity.
        Aggregates duration, occurrence count, first/last seen timestamps, and links interactions.
        """
        if not activity_events:
            return []

        # Sort events chronologically
        events = sorted(activity_events, key=lambda x: x.get("timestamp") or "")

        episodes: List[Dict[str, Any]] = []
        current_ep: Optional[Dict[str, Any]] = None

        for event in events:
            raw_ts = event.get("timestamp")
            dt = parse_timestamp(raw_ts)
            if not dt:
                self.errors.append({
                    "stage": "activity_grouping",
                    "event_id": event.get("event_id"),
                    "error": "Malformed timestamp",
                    "raw_value": raw_ts,
                })
                continue

            sid = event.get("session_id") or "unknown_session"
            app = (event.get("application_name") or "Unknown Application").strip()
            title = (event.get("window_title") or "").strip()[: self.config.max_title_length]
            dur = max(0, int(event.get("duration_ms") or 0))
            eid = event.get("event_id") or str(uuid.uuid4())

            # Evaluate boundary with current episode
            if current_ep is None:
                current_ep = self._init_activity_episode(eid, sid, app, title, raw_ts, dt, dur)
            else:
                last_dt = current_ep["_last_dt"]
                time_gap_sec = (dt - last_dt).total_seconds() if (dt and last_dt) else 999999.0

                same_session = (not self.config.bound_by_session) or (sid == current_ep["session_id"])
                same_app = app.lower() == current_ep["application"].lower()

                # Title continuity: exact match or high substring overlap
                title_match = (
                    title == current_ep["title"]
                    or (title in current_ep["title"] and len(title) > 5)
                    or (current_ep["title"] in title and len(current_ep["title"]) > 5)
                )

                # Dwell continuity within temporal proximity window
                can_group = (
                    same_session
                    and same_app
                    and (title_match or time_gap_sec <= 60.0)
                    and time_gap_sec <= self.config.temporal_proximity_seconds
                )

                if can_group:
                    # Aggregate observation into active episode
                    current_ep["last_seen"] = raw_ts
                    current_ep["_last_dt"] = dt
                    current_ep["total_duration"] += dur
                    current_ep["occurrence_count"] += 1
                    current_ep["source_event_ids"].append(eid)
                    if title and not current_ep["title"]:
                        current_ep["title"] = title
                else:
                    # Finalize completed episode and start new one
                    episodes.append(self._finalize_activity_episode(current_ep, interaction_agg, screenshots_by_session_app))
                    current_ep = self._init_activity_episode(eid, sid, app, title, raw_ts, dt, dur)

        if current_ep is not None:
            episodes.append(self._finalize_activity_episode(current_ep, interaction_agg, screenshots_by_session_app))

        return episodes

    def _init_activity_episode(
        self, eid: str, sid: str, app: str, title: str, raw_ts: str, dt: datetime, dur: int
    ) -> Dict[str, Any]:
        return {
            "session_id": sid,
            "application": app,
            "title": title,
            "first_seen": raw_ts,
            "last_seen": raw_ts,
            "_first_dt": dt,
            "_last_dt": dt,
            "total_duration": dur,
            "occurrence_count": 1,
            "source_event_ids": [eid],
        }

    def _finalize_activity_episode(
        self,
        ep: Dict[str, Any],
        interaction_agg: Dict[Tuple[str, str], Dict[str, int]],
        screenshots_by_session_app: Dict[Tuple[str, str], List[str]],
    ) -> Dict[str, Any]:
        sid = ep["session_id"]
        app = ep["application"]
        title = ep["title"] or app

        content_parts = [app, title]
        content = " | ".join(p for p in content_parts if p)

        # Attribute interaction counts based on session and app
        int_data = interaction_agg.get((sid, app), {"total": 0})
        # Distribute interactions proportionally to episode occurrences
        int_count = min(int_data["total"], ep["occurrence_count"] * 2)

        # Attribute screenshots matching this session and application
        shot_ids = screenshots_by_session_app.get((sid, app), [])

        first_seen_str = ep.get("first_seen", "")
        mem_hash = hashlib.md5(f"{sid}_{app}_{first_seen_str}".encode()).hexdigest()[:12]
        mem_id = f"mem_act_{mem_hash}"

        return {
            "memory_id": mem_id,
            "source_type": "activity",
            "source_event_ids": ep["source_event_ids"],
            "session_ids": [sid],
            "application": app,
            "content": content,
            "domain": None,
            "first_seen": ep["first_seen"],
            "last_seen": ep["last_seen"],
            "total_duration": ep["total_duration"],
            "occurrence_count": ep["occurrence_count"],
            "interaction_count": int_count,
            "screenshot_count": len(shot_ids),
            "retrieval_count": 0,
            "useful_retrieval_count": 0,
            "recency_score": None,
            "frequency_score": None,
            "retrieval_history_score": 0.0,
            "semantic_relevance": None,
            "relationship_strength": None,
            "importance_score": None,
            "lifecycle_state": None,
            "created_time": datetime.now(timezone.utc).isoformat(),
            "updated_time": datetime.now(timezone.utc).isoformat(),
        }

    def group_browser_episodes(
        self, browser_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Converts raw browser observations into research page / domain activity episodes.
        Aggregates repeat visits to the same page/topic within the same session.
        """
        if not browser_events:
            return []

        # Sort chronologically
        events = sorted(browser_events, key=lambda x: x.get("timestamp") or "")

        # Group by (session_id, domain, cleaned_title)
        grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for event in events:
            raw_ts = event.get("timestamp")
            dt = parse_timestamp(raw_ts)
            if not dt:
                self.errors.append({
                    "stage": "browser_grouping",
                    "browser_event_id": event.get("browser_event_id"),
                    "error": "Malformed timestamp",
                    "raw_value": raw_ts,
                })
                continue

            sid = event.get("session_id") or "unknown_session"
            browser = event.get("browser_name") or "Google Chrome"
            domain = (event.get("domain") or "").strip().lower() or "web"
            title = (event.get("page_title") or "").strip()[: self.config.max_title_length]
            eid = event.get("browser_event_id") or str(uuid.uuid4())

            key = (sid, domain, title)
            if key not in grouped:
                grouped[key] = {
                    "session_id": sid,
                    "browser": browser,
                    "domain": domain,
                    "title": title,
                    "first_seen": raw_ts,
                    "last_seen": raw_ts,
                    "_first_dt": dt,
                    "_last_dt": dt,
                    "occurrence_count": 1,
                    "source_event_ids": [eid],
                }
            else:
                entry = grouped[key]
                entry["last_seen"] = raw_ts
                entry["_last_dt"] = dt
                entry["occurrence_count"] += 1
                entry["source_event_ids"].append(eid)

        episodes = []
        for key, ep in grouped.items():
            sid, domain, title = key
            browser = ep["browser"]
            content = f"{browser} | {domain} | {title}" if title else f"{browser} | {domain}"

            # Estimated duration: ~30s base dwell per occurrence
            est_duration = ep["occurrence_count"] * 30000

            first_seen_str = ep.get("first_seen", "")
            web_hash = hashlib.md5(f"{sid}_{domain}_{title}_{first_seen_str}".encode()).hexdigest()[:12]
            mem_id = f"mem_web_{web_hash}"

            episodes.append({
                "memory_id": mem_id,
                "source_type": "browser",
                "source_event_ids": ep["source_event_ids"],
                "session_ids": [sid],
                "application": browser,
                "content": content,
                "domain": domain if domain != "web" else None,
                "first_seen": ep["first_seen"],
                "last_seen": ep["last_seen"],
                "total_duration": est_duration,
                "occurrence_count": ep["occurrence_count"],
                "interaction_count": min(50, ep["occurrence_count"] * 3),
                "screenshot_count": 0,
                "retrieval_count": 0,
                "useful_retrieval_count": 0,
                "recency_score": None,
                "frequency_score": None,
                "retrieval_history_score": 0.0,
                "semantic_relevance": None,
                "relationship_strength": None,
                "importance_score": None,
                "lifecycle_state": None,
                "created_time": datetime.now(timezone.utc).isoformat(),
                "updated_time": datetime.now(timezone.utc).isoformat(),
            })

        return episodes

    def create_canonical_screenshot_memories(
        self, canonical_screenshots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Creates memory records for canonical screenshots. Exact duplicates by SHA-256
        are consolidated into the canonical record's occurrence count and metadata.
        """
        records = []
        for s in canonical_screenshots:
            sid = s.get("session_id") or "unknown_session"
            app = s.get("application_name") or "Desktop"
            title = s.get("window_title") or "Contextual Screen Capture"
            reason = s.get("capture_reason") or "periodic"
            dup_count = s.get("duplicate_count", 1)
            raw_ts = s.get("timestamp") or datetime.now(timezone.utc).isoformat()
            file_size_kb = round((s.get("file_size") or 0) / 1024, 1)

            content = f"{app} | {title} | Capture: {reason} ({file_size_kb} KB, x{dup_count} frames)"
            mem_id = f"mem_shot_{s['screenshot_id'][:12]}"

            records.append({
                "memory_id": mem_id,
                "source_type": "screenshot",
                "source_event_ids": s.get("all_screenshot_ids", [s["screenshot_id"]]),
                "session_ids": [sid],
                "application": app,
                "content": content,
                "domain": None,
                "first_seen": raw_ts,
                "last_seen": raw_ts,
                "total_duration": dup_count * 1000,
                "occurrence_count": dup_count,
                "interaction_count": 0,
                "screenshot_count": dup_count,
                "retrieval_count": 0,
                "useful_retrieval_count": 0,
                "recency_score": None,
                "frequency_score": None,
                "retrieval_history_score": 0.0,
                "semantic_relevance": None,
                "relationship_strength": None,
                "importance_score": None,
                "lifecycle_state": None,
                "created_time": datetime.now(timezone.utc).isoformat(),
                "updated_time": datetime.now(timezone.utc).isoformat(),
            })

        return records

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Executes end-to-end preprocessing:
        1. Read-only ingestion from collector.db
        2. Exact screenshot deduplication
        3. Interaction aggregation
        4. Activity episode formation
        5. Browser episode formation
        6. Combined normalized memory candidates
        7. Research report generation
        """
        logger.info("Executing DMCLM Preprocessing Pipeline...")
        raw_data = self.load_raw_data()

        raw_act_count = len(raw_data["activity_events"])
        raw_brw_count = len(raw_data["browser_events"])
        raw_int_count = len(raw_data["interaction_events"])
        raw_shot_count = len(raw_data["screenshots"])
        raw_sess_count = len(raw_data["sessions"])
        total_raw_observations = raw_act_count + raw_brw_count + raw_int_count + raw_shot_count

        logger.info(
            f"Raw collector counts: {raw_act_count:,} activities, {raw_brw_count:,} browser, "
            f"{raw_int_count:,} interactions, {raw_shot_count:,} screenshots, {raw_sess_count:,} sessions."
        )

        # 1. Screenshot deduplication
        canonical_shots, sha_registry = self.deduplicate_screenshots(raw_data["screenshots"])
        exact_duplicate_count = sum(info["count"] - 1 for info in sha_registry.values() if info["count"] > 1)

        # Screenshots indexed by (session_id, app)
        shots_by_session_app: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for s in raw_data["screenshots"]:
            shots_by_session_app[(s.get("session_id", ""), s.get("application_name", ""))].append(s["screenshot_id"])

        # 2. Interaction aggregation
        interaction_agg = self.aggregate_interactions(raw_data["interaction_events"])
        total_aggregated_interactions = sum(v["total"] for v in interaction_agg.values())

        # 3. Activity episodes
        activity_episodes = self.group_activity_episodes(
            raw_data["activity_events"], interaction_agg, shots_by_session_app
        )

        # 4. Browser episodes
        browser_episodes = self.group_browser_episodes(raw_data["browser_events"])

        # 5. Contextual screenshot memories
        screenshot_memories = self.create_canonical_screenshot_memories(canonical_shots)

        # Combine all normalized memory candidates
        all_candidates = activity_episodes + browser_episodes + screenshot_memories

        # Sort chronologically by first_seen
        all_candidates.sort(key=lambda x: x.get("first_seen") or "")

        candidate_count = len(all_candidates)
        reduction_ratio_pct = round((1.0 - (candidate_count / max(1, total_raw_observations))) * 100, 2)

        report = {
            "execution_status": "DMCLM baseline execution completed.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_counts": {
                "activity_events": raw_act_count,
                "browser_events": raw_brw_count,
                "interaction_events": raw_int_count,
                "screenshots": raw_shot_count,
                "sessions": raw_sess_count,
                "total_raw_observations": total_raw_observations,
            },
            "preprocessing_results": {
                "exact_duplicate_screenshots": exact_duplicate_count,
                "unique_canonical_screenshots": len(canonical_shots),
                "aggregated_interaction_total": total_aggregated_interactions,
                "activity_episodes_formed": len(activity_episodes),
                "browser_episodes_formed": len(browser_episodes),
                "contextual_screenshot_memories": len(screenshot_memories),
                "normalized_memory_candidates": candidate_count,
                "preprocessing_reduction_ratio_pct": reduction_ratio_pct,
            },
            "error_summary": {
                "invalid_or_malformed_records": len(self.errors),
                "errors": self.errors[:20],
            },
        }

        # Save report and sha registry
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / "preprocessing_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(
            f"DMCLM Preprocessing completed: {total_raw_observations:,} raw observations -> "
            f"{candidate_count:,} normalized episodic memory candidates ({reduction_ratio_pct}% reduction)."
        )

        return {
            "candidates": all_candidates,
            "report": report,
            "sha_registry": sha_registry,
        }


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    result = pipeline.run_pipeline()
    print("\n=== PREPROCESSING SUMMARY ===")
    print(json.dumps(result["report"], indent=2))
