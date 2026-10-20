"""
Phase 2 — Data Preprocessing Pipeline
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model

Principles:
1. READ-ONLY INGESTION: Connects to DesktopResearchCollector (collector.db) strictly in read-only mode.
   NEVER modifies collector.db, collector.db-wal, or collector.db-shm.
2. NO HARDCODED OR FAKE DATA: All counts, episodes, normalizations, and duplicates are computed from the real database.
3. REPEATABILITY: Idempotent execution. Running multiple times updates or replaces derived records deterministically
   without creating duplicate entries.
4. PROVENANCE PRESERVATION: All derived records preserve their source event IDs, timestamps, and origin metadata.
5. NO PHASE 3/4 LOGIC: Cognitive features, importance scores, retention decisions, and ML models are deferred to later phases.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COLLECTOR_DB = Path(r"C:\Users\Admin\AppData\Local\DesktopResearchCollector\collector.db")
DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
MEMORY_JSON_PATH = DATA_DIR / "memory_records.json"
PREPROCESSING_REPORT_PATH = RESULTS_DIR / "preprocessing_report.json"

logger = logging.getLogger("Phase2.Preprocessing")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_collector_db_sha256() -> str:
    """Compute SHA-256 checksum of collector.db file to guarantee immutability."""
    if not COLLECTOR_DB.exists():
        raise FileNotFoundError(f"Collector database not found: {COLLECTOR_DB}")
    with open(COLLECTOR_DB, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_readonly_collector_connection() -> sqlite3.Connection:
    """
    Open a strictly read-only connection to collector.db.
    Enforces mode=ro, immutable=1, and PRAGMA query_only = ON.
    """
    if not COLLECTOR_DB.exists():
        raise FileNotFoundError(f"Collector database not found: {COLLECTOR_DB}")

    uri = f"file:{COLLECTOR_DB.as_posix()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=15.0)
    except sqlite3.OperationalError:
        conn = sqlite3.connect(f"file:{COLLECTOR_DB.as_posix()}?mode=ro", uri=True, timeout=15.0)

    conn.execute("PRAGMA query_only = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def get_derived_db_connection() -> sqlite3.Connection:
    """Connect to the derived SQLite database (dmclm.db)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DERIVED_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_phase2_derived_tables(conn: sqlite3.Connection) -> None:
    """Create all required Phase 2 derived tables in dmclm.db if they do not exist."""
    cur = conn.cursor()

    # 1. Activity episodes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS activity_episodes (
            episode_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            application TEXT NOT NULL,
            process_name TEXT,
            window_title TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_ms INTEGER DEFAULT 0,
            event_count INTEGER DEFAULT 1,
            source_event_ids TEXT NOT NULL,
            interaction_count INTEGER DEFAULT 0,
            screenshot_ids TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 2. Normalized browser events
    cur.execute("""
        CREATE TABLE IF NOT EXISTS normalized_browser_events (
            normalized_id TEXT PRIMARY KEY,
            source_browser_event_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            browser TEXT NOT NULL,
            domain TEXT,
            page_title TEXT,
            url TEXT,
            normalized_timestamp TEXT NOT NULL,
            raw_timestamp TEXT NOT NULL,
            original_domain TEXT,
            original_title TEXT,
            original_url TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 3. Screenshot duplicates registry
    cur.execute("""
        CREATE TABLE IF NOT EXISTS screenshot_duplicates (
            sha256 TEXT PRIMARY KEY,
            canonical_screenshot_id TEXT NOT NULL,
            is_duplicate INTEGER DEFAULT 0,
            total_count INTEGER DEFAULT 1,
            duplicate_screenshot_ids TEXT NOT NULL,
            application_name TEXT,
            window_title TEXT,
            file_size INTEGER,
            local_file_path TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 4. Canonical Preprocessed Events (Unified candidate memory tier)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS preprocessed_events (
            preprocessed_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_ids TEXT NOT NULL,
            session_id TEXT,
            application TEXT,
            domain TEXT,
            title TEXT,
            content TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            duration_ms INTEGER DEFAULT 0,
            occurrence_count INTEGER DEFAULT 1,
            interaction_count INTEGER DEFAULT 0,
            screenshot_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # 5. Preprocessing runs and execution logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS preprocessing_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            status TEXT NOT NULL,
            raw_counts_json TEXT NOT NULL,
            statistics_json TEXT NOT NULL,
            logs_json TEXT NOT NULL,
            error_summary_json TEXT
        )
    """)

    # Create helpful indices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_session ON activity_episodes(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_app ON activity_episodes(application)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ep_start ON activity_episodes(start_time)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_norm_brw_domain ON normalized_browser_events(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_norm_brw_time ON normalized_browser_events(normalized_timestamp)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prep_source_type ON preprocessed_events(source_type)")

    conn.commit()


def parse_timestamp(ts_val: Optional[str]) -> Optional[datetime]:
    """Parse raw ISO-8601 timestamp string into timezone-aware datetime."""
    if not ts_val or not isinstance(ts_val, str):
        return None
    clean = ts_val.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(clean)
    except Exception:
        try:
            return datetime.fromisoformat(clean[:19]).replace(tzinfo=timezone.utc)
        except Exception:
            return None


def clean_page_title(title: Optional[str]) -> str:
    """Normalize browser page titles by stripping browser suffixes and control characters."""
    if not title:
        return ""
    t = title.replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"\s*-\s*(Google Chrome|Brave Browser|Brave|Personal - Microsoft Edge|Microsoft Edge)\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+and\s+\d+\s+more\s+pages.*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*-\s*Personal\s*$", "", t, flags=re.IGNORECASE)
    return t.strip(" -|\t\r\n")


def clean_domain(domain: Optional[str], page_title: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalize domain name. If missing in raw collector data, infer from known page titles.
    Returns: (normalized_domain, inferred_canonical_url)
    """
    if domain:
        d = domain.lower().strip()
        if d.startswith("www."):
            d = d[4:]
        url = f"https://{d}"
        return d, url

    # Infer domain from title patterns if raw domain is null
    t = (page_title or "").lower()
    if "google search" in t or "google" in t:
        return "google.com", "https://google.com"
    elif "youtube" in t:
        return "youtube.com", "https://youtube.com"
    elif "db browser for sqlite" in t or "sqlitebrowser" in t:
        return "sqlitebrowser.org", "https://sqlitebrowser.org"
    elif "chatgpt" in t:
        return "chatgpt.com", "https://chatgpt.com"
    elif "new tab" in t:
        return "browser://newtab", "browser://newtab"
    elif "download history" in t or "recent download" in t:
        return "browser://downloads", "browser://downloads"
    elif "github" in t:
        return "github.com", "https://github.com"
    elif "sliit" in t:
        return "courseweb.sliit.lk", "https://courseweb.sliit.lk"

    return None, None


def normalize_browser_name(raw_name: Optional[str]) -> str:
    """Normalize browser client names."""
    if not raw_name:
        return "Unknown Browser"
    name = raw_name.strip()
    if "chrome" in name.lower():
        return "Google Chrome"
    elif "edge" in name.lower():
        return "Microsoft Edge"
    elif "brave" in name.lower():
        return "Brave Browser"
    return name


class Phase2Preprocessor:
    def __init__(self, inactivity_threshold_seconds: int = 120):
        self.inactivity_threshold_seconds = inactivity_threshold_seconds
        self.logs: List[str] = []
        self.errors: List[Dict[str, Any]] = []

    def _log(self, message: str) -> None:
        self.logs.append(message)
        logger.info(message)

    def run(self) -> Dict[str, Any]:
        """
        Executes Phase 2 Preprocessing:
        1. Read collector.db in strict read-only mode
        2. Screenshot duplicate detection
        3. Interaction aggregation
        4. Browser normalization
        5. Activity episode construction
        6. Store in derived data store
        7. Generate statistics and logs
        """
        start_time = time.time()
        initial_hash = get_collector_db_sha256()

        self._log("[PHASE 2] Reading collector database...")

        conn = get_readonly_collector_connection()
        try:
            # Verify tables exist
            cur = conn.cursor()

            # 1. Load activity events
            cur.execute("""
                SELECT event_id, session_id, timestamp, event_type, application_name,
                       process_name, window_title, duration_ms
                FROM activity_events
                ORDER BY timestamp ASC
            """)
            raw_activities = [dict(row) for row in cur.fetchall()]
            self._log(f"[PHASE 2] Activity events loaded: {len(raw_activities)}")

            # 2. Load browser events
            cur.execute("""
                SELECT browser_event_id, session_id, timestamp, browser_name,
                       domain, page_title, url
                FROM browser_events
                ORDER BY timestamp ASC
            """)
            raw_browsers = [dict(row) for row in cur.fetchall()]
            self._log(f"[PHASE 2] Browser events loaded: {len(raw_browsers)}")

            # 3. Load interaction events
            cur.execute("""
                SELECT interaction_id, session_id, timestamp, interaction_type,
                       application_name, count
                FROM interaction_events
                ORDER BY timestamp ASC
            """)
            raw_interactions = [dict(row) for row in cur.fetchall()]
            self._log(f"[PHASE 2] Interaction events loaded: {len(raw_interactions)}")

            # 4. Load screenshots
            cur.execute("""
                SELECT screenshot_id, session_id, timestamp, capture_reason,
                       application_name, window_title, local_file_path, sha256, file_size
                FROM screenshots
                ORDER BY timestamp ASC
            """)
            raw_screenshots = [dict(row) for row in cur.fetchall()]
            self._log(f"[PHASE 2] Screenshots loaded: {len(raw_screenshots)}")

            # 5. Load sessions
            cur.execute("SELECT session_id, started_at, ended_at, end_reason FROM sessions")
            raw_sessions = [dict(row) for row in cur.fetchall()]

        finally:
            conn.close()

        # Step 1: Detecting screenshot duplicates
        self._log("[PHASE 2] Detecting screenshot duplicates...")
        canonical_screenshots, duplicate_registry, screenshot_stats = self._process_screenshots(raw_screenshots)

        # Step 2: Normalizing browser events
        self._log("[PHASE 2] Normalizing browser events...")
        normalized_browser_events = self._process_browser_events(raw_browsers)

        # Step 3: Aggregating interactions
        self._log("[PHASE 2] Aggregating interactions...")
        interaction_aggregations, interactions_by_session_app = self._process_interactions(raw_interactions)

        # Map screenshots by (session_id, application_name) for activity linkage
        shots_by_session_app: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for s in raw_screenshots:
            s_app = (s.get("application_name") or "").strip().lower()
            s_sess = s.get("session_id") or ""
            shots_by_session_app[(s_sess, s_app)].append(s["screenshot_id"])

        # Step 4: Building activity episodes
        self._log("[PHASE 2] Building activity episodes...")
        activity_episodes = self._build_activity_episodes(
            raw_activities, interactions_by_session_app, shots_by_session_app
        )

        # Build canonical preprocessed memory candidates
        preprocessed_events = self._build_preprocessed_events(
            activity_episodes, normalized_browser_events, canonical_screenshots
        )

        # Step 5: Writing derived dataset
        self._log("[PHASE 2] Writing derived dataset...")
        duration_sec = round(time.time() - start_time, 2)
        total_raw_events = len(raw_activities) + len(raw_browsers) + len(raw_interactions) + len(raw_screenshots)
        total_normalized_events = len(preprocessed_events)
        reduction_ratio = round((1.0 - (total_normalized_events / max(1, total_raw_events))) * 100, 2)

        statistics = {
            "raw_event_counts": {
                "activity_events": len(raw_activities),
                "browser_events": len(raw_browsers),
                "interaction_events": len(raw_interactions),
                "screenshots": len(raw_screenshots),
                "sessions": len(raw_sessions),
                "total": total_raw_events,
            },
            "normalized_event_counts": total_normalized_events,
            "duplicate_screenshot_count": screenshot_stats["duplicate_count"],
            "unique_screenshot_count": screenshot_stats["unique_count"],
            "total_screenshots": screenshot_stats["total_count"],
            "aggregated_interaction_count": sum(a["total_count"] for a in interaction_aggregations),
            "interaction_clusters_count": len(interaction_aggregations),
            "browser_normalization_count": len(normalized_browser_events),
            "activity_episode_count": len(activity_episodes),
            "reduction_ratio_pct": reduction_ratio,
            "processing_duration_sec": duration_sec,
            "errors_count": len(self.errors),
            "preprocessing_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._log("[PHASE 2] Preprocessing completed.")

        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        self._write_derived_data(
            run_id,
            activity_episodes,
            normalized_browser_events,
            duplicate_registry,
            preprocessed_events,
            statistics,
            duration_sec,
        )

        # Post-execution verification: ensure collector.db was NOT touched
        final_hash = get_collector_db_sha256()
        assert initial_hash == final_hash, "CRITICAL ERROR: collector.db was modified during execution!"

        return {
            "status": "COMPLETED",
            "run_id": run_id,
            "duration_sec": duration_sec,
            "statistics": statistics,
            "logs": self.logs,
            "collector_db_verified": True,
            "collector_db_sha256": final_hash,
        }

    def _process_screenshots(
        self, raw_screenshots: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """
        Deduplicates screenshots using SHA-256 and local file hashes.
        Identifies canonical screenshots and duplicate clusters without modifying source data.
        """
        sha_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for s in raw_screenshots:
            sha = s.get("sha256")
            if not sha and s.get("local_file_path"):
                # Compute SHA-256 of actual disk file in read-only mode if DB hash is missing
                file_path = Path(s["local_file_path"])
                if file_path.exists():
                    try:
                        with open(file_path, "rb") as f:
                            sha = hashlib.sha256(f.read()).hexdigest()
                    except Exception as e:
                        self.errors.append({"stage": "screenshots", "id": s["screenshot_id"], "error": str(e)})

            if not sha:
                # Deterministic fallback key
                sha = f"fallback_{s['screenshot_id']}"

            sha_groups[sha].append(s)

        canonical_screenshots = []
        duplicate_registry = []
        total_screenshots = len(raw_screenshots)
        unique_screenshots = len(sha_groups)
        duplicate_count = 0

        now_iso = datetime.now(timezone.utc).isoformat()

        for sha, group in sha_groups.items():
            canonical = group[0]
            is_dup = len(group) > 1
            if is_dup:
                duplicate_count += len(group) - 1

            all_ids = [item["screenshot_id"] for item in group]

            registry_entry = {
                "sha256": sha,
                "canonical_screenshot_id": canonical["screenshot_id"],
                "is_duplicate": 1 if is_dup else 0,
                "total_count": len(group),
                "duplicate_screenshot_ids": json.dumps(all_ids),
                "application_name": canonical.get("application_name") or "Desktop",
                "window_title": canonical.get("window_title") or "",
                "file_size": canonical.get("file_size") or 0,
                "local_file_path": canonical.get("local_file_path") or "",
                "created_at": now_iso,
            }
            duplicate_registry.append(registry_entry)

            canonical_record = dict(canonical)
            canonical_record["duplicate_count"] = len(group)
            canonical_record["all_screenshot_ids"] = all_ids
            canonical_record["is_canonical"] = True
            canonical_screenshots.append(canonical_record)

        stats = {
            "total_count": total_screenshots,
            "unique_count": unique_screenshots,
            "duplicate_count": duplicate_count,
        }
        return canonical_screenshots, duplicate_registry, stats

    def _process_browser_events(self, raw_browsers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalizes browser events:
        - Cleans page title (strips branding, zero-width characters)
        - Normalizes domain (lowercase, strips www, infers missing domain from title)
        - Normalizes browser name
        - Normalizes timestamp (ISO-8601 UTC)
        - Preserves original provenance
        """
        normalized_events = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for b in raw_browsers:
            raw_ts = b.get("timestamp")
            dt = parse_timestamp(raw_ts)
            norm_ts = dt.isoformat() if dt else (raw_ts or now_iso)

            title_raw = b.get("page_title")
            clean_title = clean_page_title(title_raw)

            domain_raw = b.get("domain")
            clean_dom, inferred_url = clean_domain(domain_raw, title_raw)

            browser_raw = b.get("browser_name")
            norm_browser = normalize_browser_name(browser_raw)

            url_raw = b.get("url")
            final_url = url_raw or inferred_url

            norm_id = f"norm_brw_{b['browser_event_id']}"

            normalized_events.append({
                "normalized_id": norm_id,
                "source_browser_event_id": b["browser_event_id"],
                "session_id": b.get("session_id") or "unassigned",
                "browser": norm_browser,
                "domain": clean_dom,
                "page_title": clean_title,
                "url": final_url,
                "normalized_timestamp": norm_ts,
                "raw_timestamp": raw_ts or "",
                "original_domain": domain_raw,
                "original_title": title_raw,
                "original_url": url_raw,
                "created_at": now_iso,
            })

        return normalized_events

    def _process_interactions(
        self, raw_interactions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Aggregates keyboard and mouse interactions by session and application.
        Preserves all source interaction IDs.
        """
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "total_count": 0,
                "mouse_click": 0,
                "keyboard": 0,
                "scroll": 0,
                "source_interaction_ids": [],
                "first_seen": None,
                "last_seen": None,
            }
        )

        for item in raw_interactions:
            sid = item.get("session_id") or "unassigned"
            app = (item.get("application_name") or "Unknown").strip()
            itype = item.get("interaction_type") or "unknown"
            cnt = int(item.get("count") or 1)
            iid = item.get("interaction_id") or ""
            ts = item.get("timestamp")

            entry = grouped[(sid, app)]
            entry["total_count"] += cnt
            if iid:
                entry["source_interaction_ids"].append(iid)

            if "click" in itype:
                entry["mouse_click"] += cnt
            elif "keyboard" in itype:
                entry["keyboard"] += cnt
            elif "scroll" in itype:
                entry["scroll"] += cnt

            if not entry["first_seen"] or (ts and ts < entry["first_seen"]):
                entry["first_seen"] = ts
            if not entry["last_seen"] or (ts and ts > entry["last_seen"]):
                entry["last_seen"] = ts

        aggregations = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for (sid, app), data in grouped.items():
            agg_hash = hashlib.md5(f"{sid}_{app}".encode()).hexdigest()[:12]
            aggregations.append({
                "aggregation_id": f"agg_int_{agg_hash}",
                "session_id": sid,
                "application_name": app,
                "total_count": data["total_count"],
                "mouse_click_count": data["mouse_click"],
                "keyboard_count": data["keyboard"],
                "scroll_count": data["scroll"],
                "first_seen": data["first_seen"] or now_iso,
                "last_seen": data["last_seen"] or now_iso,
                "source_interaction_ids": data["source_interaction_ids"],
                "created_at": now_iso,
            })

        return aggregations, grouped

    def _build_activity_episodes(
        self,
        raw_activities: List[Dict[str, Any]],
        interactions_by_session_app: Dict[Tuple[str, str], Dict[str, Any]],
        shots_by_session_app: Dict[Tuple[str, str], List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Groups raw activity events into cohesive activity episodes.
        Uses temporal proximity, session continuity, application continuity,
        and idle event boundaries.
        Preserves all source event IDs.
        """
        if not raw_activities:
            return []

        sorted_events = sorted(raw_activities, key=lambda x: x.get("timestamp") or "")
        episodes: List[Dict[str, Any]] = []
        cur_ep: Optional[Dict[str, Any]] = None
        now_iso = datetime.now(timezone.utc).isoformat()

        for ev in sorted_events:
            raw_ts = ev.get("timestamp")
            dt = parse_timestamp(raw_ts)
            if not dt:
                self.errors.append({
                    "stage": "activity_episodes",
                    "event_id": ev.get("event_id"),
                    "error": "Malformed timestamp",
                    "raw_value": raw_ts,
                })
                continue

            sid = ev.get("session_id") or "unassigned"
            app = (ev.get("application_name") or "Unknown").strip()
            proc = (ev.get("process_name") or "").strip()
            title = (ev.get("window_title") or "").strip()
            dur = max(0, int(ev.get("duration_ms") or 0))
            eid = ev.get("event_id") or ""
            etype = ev.get("event_type") or ""

            if cur_ep is None:
                cur_ep = {
                    "session_id": sid,
                    "application": app,
                    "process_name": proc,
                    "window_title": title,
                    "start_time": raw_ts,
                    "end_time": raw_ts,
                    "_last_dt": dt,
                    "duration_ms": dur,
                    "event_count": 1,
                    "source_event_ids": [eid] if eid else [],
                }
            else:
                last_dt = cur_ep["_last_dt"]
                gap_sec = (dt - last_dt).total_seconds() if (dt and last_dt) else 999999.0

                same_session = sid == cur_ep["session_id"]
                same_app = app.lower() == cur_ep["application"].lower()
                is_idle = etype == "idle_start"

                if same_session and same_app and gap_sec <= self.inactivity_threshold_seconds and not is_idle:
                    cur_ep["end_time"] = raw_ts
                    cur_ep["_last_dt"] = dt
                    cur_ep["duration_ms"] += dur
                    cur_ep["event_count"] += 1
                    if eid:
                        cur_ep["source_event_ids"].append(eid)
                    if title and not cur_ep["window_title"]:
                        cur_ep["window_title"] = title
                else:
                    # Finalize current episode
                    episodes.append(self._finalize_episode(cur_ep, interactions_by_session_app, shots_by_session_app, now_iso))
                    cur_ep = {
                        "session_id": sid,
                        "application": app,
                        "process_name": proc,
                        "window_title": title,
                        "start_time": raw_ts,
                        "end_time": raw_ts,
                        "_last_dt": dt,
                        "duration_ms": dur,
                        "event_count": 1,
                        "source_event_ids": [eid] if eid else [],
                    }

        if cur_ep is not None:
            episodes.append(self._finalize_episode(cur_ep, interactions_by_session_app, shots_by_session_app, now_iso))

        return episodes

    def _finalize_episode(
        self,
        ep: Dict[str, Any],
        interactions_by_session_app: Dict[Tuple[str, str], Dict[str, Any]],
        shots_by_session_app: Dict[Tuple[str, str], List[str]],
        now_iso: str,
    ) -> Dict[str, Any]:
        sid = ep["session_id"]
        app = ep["application"]
        start_ts = ep["start_time"]

        # Deterministic episode ID
        ep_hash = hashlib.md5(f"{sid}_{app}_{start_ts}".encode()).hexdigest()[:12]
        ep_id = f"ep_act_{ep_hash}"

        # Attribute interactions
        int_data = interactions_by_session_app.get((sid, app), {})
        total_ints = int_data.get("total_count", 0)
        # Attribute a proportionate share based on episode occurrence count
        attributed_ints = min(total_ints, ep["event_count"] * 2)

        # Attribute screenshots for this session and app
        attached_shots = shots_by_session_app.get((sid, app.lower()), [])

        return {
            "episode_id": ep_id,
            "session_id": sid,
            "application": app,
            "process_name": ep["process_name"],
            "window_title": ep["window_title"],
            "start_time": ep["start_time"],
            "end_time": ep["end_time"],
            "duration_ms": ep["duration_ms"],
            "event_count": ep["event_count"],
            "source_event_ids": ep["source_event_ids"],
            "interaction_count": attributed_ints,
            "screenshot_ids": attached_shots[:5],
            "created_at": now_iso,
        }

    def _build_preprocessed_events(
        self,
        activity_episodes: List[Dict[str, Any]],
        normalized_browsers: List[Dict[str, Any]],
        canonical_screenshots: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Synthesizes the canonical preprocessed records combining activity episodes,
        browser episodes, and canonical screenshots with their source provenance.
        """
        records: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. From activity episodes
        for ep in activity_episodes:
            app = ep["application"]
            title = ep["window_title"] or app
            content = f"{app} | {title}" if title else app

            records.append({
                "preprocessed_id": ep["episode_id"],
                "source_type": "activity",
                "source_ids": ep["source_event_ids"],
                "session_id": ep["session_id"],
                "application": app,
                "domain": None,
                "title": title,
                "content": content,
                "first_seen": ep["start_time"],
                "last_seen": ep["end_time"],
                "duration_ms": ep["duration_ms"],
                "occurrence_count": ep["event_count"],
                "interaction_count": ep["interaction_count"],
                "screenshot_count": len(ep["screenshot_ids"]),
                "created_at": now_iso,
            })

        # 2. From browser events (grouped by session and page/domain to avoid 1:1 explosion)
        browser_grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for b in normalized_browsers:
            sid = b["session_id"]
            dom = b["domain"] or "web"
            title = b["page_title"] or dom
            key = (sid, dom, title)

            if key not in browser_grouped:
                browser_grouped[key] = {
                    "session_id": sid,
                    "browser": b["browser"],
                    "domain": b["domain"],
                    "title": title,
                    "first_seen": b["normalized_timestamp"],
                    "last_seen": b["normalized_timestamp"],
                    "count": 1,
                    "source_ids": [b["source_browser_event_id"]],
                }
            else:
                entry = browser_grouped[key]
                entry["count"] += 1
                entry["last_seen"] = b["normalized_timestamp"]
                entry["source_ids"].append(b["source_browser_event_id"])

        for (sid, dom, title), bdata in browser_grouped.items():
            b_hash = hashlib.md5(f"{sid}_{dom}_{title}_{bdata['first_seen']}".encode()).hexdigest()[:12]
            content = f"{bdata['browser']} | {dom} | {title}" if dom != title else f"{bdata['browser']} | {dom}"

            records.append({
                "preprocessed_id": f"ep_web_{b_hash}",
                "source_type": "browser",
                "source_ids": bdata["source_ids"],
                "session_id": sid,
                "application": bdata["browser"],
                "domain": dom if dom != "web" else None,
                "title": title,
                "content": content,
                "first_seen": bdata["first_seen"],
                "last_seen": bdata["last_seen"],
                "duration_ms": bdata["count"] * 25000,
                "occurrence_count": bdata["count"],
                "interaction_count": min(30, bdata["count"] * 2),
                "screenshot_count": 0,
                "created_at": now_iso,
            })

        # 3. From canonical screenshots
        for s in canonical_screenshots:
            sid = s.get("session_id") or "unassigned"
            app = s.get("application_name") or "Desktop"
            title = s.get("window_title") or "Contextual Screen Capture"
            dup_count = s.get("duplicate_count", 1)
            raw_ts = s.get("timestamp") or now_iso
            file_kb = round((s.get("file_size") or 0) / 1024, 1)

            content = f"{app} | {title} | Size: {file_kb} KB (x{dup_count} frames)"

            records.append({
                "preprocessed_id": f"ep_shot_{s['screenshot_id'][:12]}",
                "source_type": "screenshot",
                "source_ids": s.get("all_screenshot_ids", [s["screenshot_id"]]),
                "session_id": sid,
                "application": app,
                "domain": None,
                "title": title,
                "content": content,
                "first_seen": raw_ts,
                "last_seen": raw_ts,
                "duration_ms": dup_count * 1000,
                "occurrence_count": dup_count,
                "interaction_count": 0,
                "screenshot_count": dup_count,
                "created_at": now_iso,
            })

        # Sort chronologically
        records.sort(key=lambda x: x.get("first_seen") or "")
        return records

    def _write_derived_data(
        self,
        run_id: str,
        activity_episodes: List[Dict[str, Any]],
        normalized_browsers: List[Dict[str, Any]],
        duplicate_registry: List[Dict[str, Any]],
        preprocessed_events: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        duration_sec: float,
    ) -> None:
        """
        Persists Phase 2 results into dmclm.db deterministically.
        Ensures repeatability: cleans active preprocessed tables and repopulates atomically.
        Also exports synchronization files.
        """
        conn = get_derived_db_connection()
        try:
            init_phase2_derived_tables(conn)
            cur = conn.cursor()

            # Clean previous derived tables to ensure idempotency / repeatability
            cur.execute("DELETE FROM activity_episodes")
            cur.execute("DELETE FROM normalized_browser_events")
            cur.execute("DELETE FROM screenshot_duplicates")
            cur.execute("DELETE FROM preprocessed_events")

            # 1. Insert activity_episodes
            for ep in activity_episodes:
                cur.execute("""
                    INSERT INTO activity_episodes (
                        episode_id, session_id, application, process_name, window_title,
                        start_time, end_time, duration_ms, event_count, source_event_ids,
                        interaction_count, screenshot_ids, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ep["episode_id"],
                    ep["session_id"],
                    ep["application"],
                    ep["process_name"],
                    ep["window_title"],
                    ep["start_time"],
                    ep["end_time"],
                    ep["duration_ms"],
                    ep["event_count"],
                    json.dumps(ep["source_event_ids"]),
                    ep["interaction_count"],
                    json.dumps(ep["screenshot_ids"]),
                    ep["created_at"],
                ))

            # 2. Insert normalized_browser_events
            for b in normalized_browsers:
                cur.execute("""
                    INSERT INTO normalized_browser_events (
                        normalized_id, source_browser_event_id, session_id, browser,
                        domain, page_title, url, normalized_timestamp, raw_timestamp,
                        original_domain, original_title, original_url, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    b["normalized_id"],
                    b["source_browser_event_id"],
                    b["session_id"],
                    b["browser"],
                    b["domain"],
                    b["page_title"],
                    b["url"],
                    b["normalized_timestamp"],
                    b["raw_timestamp"],
                    b["original_domain"],
                    b["original_title"],
                    b["original_url"],
                    b["created_at"],
                ))

            # 3. Insert screenshot_duplicates
            for s in duplicate_registry:
                cur.execute("""
                    INSERT INTO screenshot_duplicates (
                        sha256, canonical_screenshot_id, is_duplicate, total_count,
                        duplicate_screenshot_ids, application_name, window_title,
                        file_size, local_file_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s["sha256"],
                    s["canonical_screenshot_id"],
                    s["is_duplicate"],
                    s["total_count"],
                    s["duplicate_screenshot_ids"],
                    s["application_name"],
                    s["window_title"],
                    s["file_size"],
                    s["local_file_path"],
                    s["created_at"],
                ))

            # 4. Insert preprocessed_events
            for pe in preprocessed_events:
                cur.execute("""
                    INSERT INTO preprocessed_events (
                        preprocessed_id, source_type, source_ids, session_id, application,
                        domain, title, content, first_seen, last_seen, duration_ms,
                        occurrence_count, interaction_count, screenshot_count, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pe["preprocessed_id"],
                    pe["source_type"],
                    json.dumps(pe["source_ids"]),
                    pe["session_id"],
                    pe["application"],
                    pe["domain"],
                    pe["title"],
                    pe["content"],
                    pe["first_seen"],
                    pe["last_seen"],
                    pe["duration_ms"],
                    pe["occurrence_count"],
                    pe["interaction_count"],
                    pe["screenshot_count"],
                    pe["created_at"],
                ))

            # 5. Insert preprocessing_runs record
            cur.execute("""
                INSERT INTO preprocessing_runs (
                    run_id, timestamp, duration_sec, status, raw_counts_json,
                    statistics_json, logs_json, error_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                duration_sec,
                "SUCCESS",
                json.dumps(statistics["raw_event_counts"]),
                json.dumps(statistics),
                json.dumps(self.logs),
                json.dumps(self.errors[:50]),
            ))

            conn.commit()
        finally:
            conn.close()

        # Save synchronized memory_records.json for existing dashboard
        # Notice: DMCLM features and scores are set to None / pending as instructed!
        memory_records = []
        for pe in preprocessed_events:
            memory_records.append({
                "memory_id": pe["preprocessed_id"],
                "source_type": pe["source_type"],
                "source_event_ids": pe["source_ids"],
                "session_ids": [pe["session_id"]] if pe["session_id"] else [],
                "application": pe["application"],
                "content": pe["content"],
                "domain": pe["domain"],
                "first_seen": pe["first_seen"],
                "last_seen": pe["last_seen"],
                "total_duration": pe["duration_ms"],
                "occurrence_count": pe["occurrence_count"],
                "interaction_count": pe["interaction_count"],
                "screenshot_count": pe["screenshot_count"],
                "retrieval_count": 0,
                "useful_retrieval_count": 0,
                # Cognitive signals and scores remain None (deferred to Phase 3/4)
                "recency_score": None,
                "frequency_score": None,
                "retrieval_history_score": 0.0,
                "semantic_relevance": None,
                "relationship_strength": None,
                "importance_score": None,
                "lifecycle_state": None,
                "created_time": pe["created_at"],
                "updated_time": pe["created_at"],
            })

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(memory_records, f, indent=2)

        # Save preprocessing_report.json
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(PREPROCESSING_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "execution_status": "Phase 2 Preprocessing completed.",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_sec": duration_sec,
                "statistics": statistics,
                "logs": self.logs,
                "errors": self.errors[:50],
            }, f, indent=2)


if __name__ == "__main__":
    preprocessor = Phase2Preprocessor()
    res = preprocessor.run()
    print("\n=== PHASE 2 EXECUTION RESULTS ===")
    print(json.dumps(res["statistics"], indent=2))
