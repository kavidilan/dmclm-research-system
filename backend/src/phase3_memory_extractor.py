"""
Phase 3 — Memory Extraction Pipeline
Member 4 — Adaptive Long-Term Memory Consolidation / Dynamic Memory Consolidation Learning Model (DMCLM)

Principles:
1. DERIVED DATA INGESTION: Ingests real Phase 2 activity episodes, normalized browser events,
   and screenshot deduplication metadata from data/dmclm.db.
2. MEANINGFUL ACTIVITY EPISODES: Determines what desktop activity episodes are meaningful enough
   to become structured memory candidates, filtering out transient system noise and fleeting zero-dwell events.
3. CONTEXT & PROVENANCE: Every memory candidate maintains full traceability back to its Phase 2 source episode(s).
4. DYNAMIC LOCAL SUMMARY: Generates semantic activity descriptions dynamically from actual fields
   (application, window context, domains, duration, interactions) without external LLMs or third-party APIs.
5. DETERMINISTIC & REPEATABLE: Candidate IDs are deterministic hashes. Running the pipeline multiple times
   yields identical candidates with zero duplicate growth.
6. NO PHASE 4 LOGIC: Cognitive features (R, F, H, S, Rel), importance scores, consolidation, forgetting,
   and ML training are deferred to later phases.
"""

import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.dmclm_config import MemoryExtractionConfig, DEFAULT_CONFIG
    from src.models import StructuredMemoryCandidate
except ImportError:
    from dmclm_config import MemoryExtractionConfig, DEFAULT_CONFIG
    from models import StructuredMemoryCandidate

DERIVED_DB_PATH = DATA_DIR / "dmclm.db"
CANDIDATES_JSON_PATH = DATA_DIR / "memory_candidates.json"
MEMORY_RECORDS_JSON_PATH = DATA_DIR / "memory_records.json"
EXTRACTION_REPORT_PATH = RESULTS_DIR / "memory_extraction_report.json"

logger = logging.getLogger("Phase3.MemoryExtraction")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_timestamp(ts_val: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamp string into timezone-aware datetime."""
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


def format_duration_human(duration_ms: int) -> str:
    """Format millisecond duration into a clean human-readable string."""
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    sec = duration_ms // 1000
    if sec < 60:
        return f"{sec}s"
    minutes = sec // 60
    rem_sec = sec % 60
    if minutes < 60:
        return f"{minutes}m {rem_sec}s" if rem_sec > 0 else f"{minutes}m"
    hours = minutes // 60
    rem_min = minutes % 60
    return f"{hours}h {rem_min}m"


def categorize_application_activity(app_name: str) -> str:
    """Determine semantic action category from application name."""
    lower = app_name.lower()
    if any(k in lower for k in ["sqlite", "dbeaver", "database", "pgadmin", "mysql", "sql"]):
        return "Database management"
    elif any(k in lower for k in ["code", "pycharm", "antigravity", "visual studio", "sublime", "git", "terminal"]):
        return "Software development"
    elif any(k in lower for k in ["chrome", "edge", "brave", "firefox", "browser", "safari"]):
        return "Web research"
    elif any(k in lower for k in ["explorer", "folder", "directory", "file"]):
        return "File and workspace organization"
    elif any(k in lower for k in ["notepad", "notes", "word", "document", "pdf", "photos", "reader"]):
        return "Document and media review"
    elif any(k in lower for k in ["whatsapp", "slack", "teams", "discord", "mail", "outlook"]):
        return "Communication session"
    return "Desktop activity"


def generate_dynamic_activity_summary(
    primary_app: str,
    window_title: str,
    browser_domains: List[str],
    duration_ms: int,
    interaction_count: int,
    screenshot_count: int,
    max_length: int = 350
) -> str:
    """
    Dynamically generates a semantic text description from actual episode attributes.
    Zero external APIs or LLMs used; completely local and traceable.
    """
    category = categorize_application_activity(primary_app)
    dur_str = format_duration_human(duration_ms)
    
    clean_title = window_title.strip() if window_title else ""
    # Avoid repetitive title when title equals app name
    title_part = f'focused on "{clean_title}"' if clean_title and clean_title.lower() != primary_app.lower() else ""

    domains_part = ""
    if browser_domains:
        top_domains = browser_domains[:3]
        domains_part = f"(referencing: {', '.join(top_domains)})"

    engagement_parts = [f"dwell: {dur_str}"]
    if interaction_count > 0:
        engagement_parts.append(f"{interaction_count} user actions")
    if screenshot_count > 0:
        engagement_parts.append(f"{screenshot_count} screen context{'s' if screenshot_count > 1 else ''}")

    engagement_str = f"[{', '.join(engagement_parts)}]"

    parts = [f"{category} in {primary_app}"]
    if title_part:
        parts.append(title_part)
    if domains_part:
        parts.append(domains_part)
    parts.append(engagement_str)

    summary = " ".join(p for p in parts if p).strip()
    if len(summary) > max_length:
        summary = summary[:max_length - 3] + "..."
    return summary


class Phase3MemoryExtractor:
    """
    Phase 3 Engine: Identifies meaningful Phase 2 activity episodes and transforms
    them into structured memory candidates.
    """
    def __init__(
        self,
        config: Optional[MemoryExtractionConfig] = None,
        db_path: Optional[Path] = None
    ):
        self.config = config or DEFAULT_CONFIG.memory_extraction
        self.db_path = db_path or DERIVED_DB_PATH
        self.logs: List[str] = []
        self.errors: List[Dict[str, Any]] = []
        self.skipped_records: Dict[str, int] = {
            "transient_system_noise": 0,
            "insufficient_dwell_or_interactions": 0,
            "malformed_or_missing_timestamp": 0,
            "missing_application": 0,
        }

    def _log(self, message: str) -> None:
        self.logs.append(message)
        logger.info(message)

    def load_phase2_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads Phase 2 derived tables from dmclm.db.
        Raises descriptive error if database does not exist or has empty episodes.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Phase 2 derived database not found at {self.db_path}. "
                "Ensure Phase 2 preprocessing has been executed."
            )

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            
            # Check table existence
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
            if "activity_episodes" not in tables:
                raise RuntimeError("Phase 2 activity_episodes table does not exist in derived database.")

            # 1. Load activity episodes
            cur.execute("""
                SELECT episode_id, session_id, application, process_name, window_title,
                       start_time, end_time, duration_ms, event_count, source_event_ids,
                       interaction_count, screenshot_ids, created_at
                FROM activity_episodes
                ORDER BY start_time ASC
            """)
            episodes = [dict(row) for row in cur.fetchall()]

            # 2. Load normalized browser events (if present)
            browser_events = []
            if "normalized_browser_events" in tables:
                cur.execute("""
                    SELECT normalized_id, session_id, browser, domain, page_title, url, normalized_timestamp
                    FROM normalized_browser_events
                    ORDER BY normalized_timestamp ASC
                """)
                browser_events = [dict(row) for row in cur.fetchall()]

            # 3. Load screenshot deduplication info (if present)
            screenshots = []
            if "screenshot_duplicates" in tables:
                cur.execute("""
                    SELECT sha256, canonical_screenshot_id, is_duplicate, total_count,
                           application_name, window_title, file_size
                    FROM screenshot_duplicates
                """)
                screenshots = [dict(row) for row in cur.fetchall()]

            return {
                "activity_episodes": episodes,
                "browser_events": browser_events,
                "screenshots": screenshots,
            }
        finally:
            conn.close()

    def evaluate_episode_meaningfulness(
        self, episode: Dict[str, Any]
    ) -> Tuple[bool, List[str], Optional[str]]:
        """
        Evaluates whether an activity episode is meaningful enough to become a memory candidate.
        Returns: (is_meaningful, criteria_matched, skip_reason)
        """
        app = (episode.get("application") or "").strip()
        if not app or app.lower() == "unknown":
            self.skipped_records["missing_application"] += 1
            return False, [], "missing_application"

        st = episode.get("start_time")
        et = episode.get("end_time")
        if not st or not et:
            self.skipped_records["malformed_or_missing_timestamp"] += 1
            return False, [], "malformed_or_missing_timestamp"

        duration_ms = max(0, int(episode.get("duration_ms") or 0))
        interaction_count = max(0, int(episode.get("interaction_count") or 0))
        
        # Check attached screenshots
        shots = episode.get("screenshot_ids")
        shot_list = json.loads(shots) if isinstance(shots, str) and shots.startswith("[") else (shots or [])
        has_screenshot = len(shot_list) > 0

        # Check window title
        title = (episode.get("window_title") or "").strip()
        has_substantive_title = len(title) > 3 and title.lower() != app.lower()

        # 1. Filter known transient system processes that lack user engagement
        clean_app = app.lower().replace(".exe", "").strip()
        system_noise_clean = tuple(n.lower().replace(".exe", "").strip() for n in self.config.system_noise_applications)
        if self.config.filter_transient_system_noise and (app in self.config.system_noise_applications or clean_app in system_noise_clean):
            if interaction_count == 0 and duration_ms < self.config.min_meaningful_duration_ms:
                self.skipped_records["transient_system_noise"] += 1
                return False, [], "transient_system_noise"

        # 2. Meaningfulness Criteria Evaluation
        criteria: List[str] = []
        if duration_ms >= self.config.min_meaningful_duration_ms:
            criteria.append("duration_threshold")
            criteria.append("sustained_duration")
        if interaction_count >= self.config.min_interactions_for_short_burst:
            criteria.append("user_interactions")
            criteria.append("active_interactions")
        if has_screenshot:
            criteria.append("screenshot_context")

        # Episode qualifies if at least one meaningful engagement criterion is met
        if criteria:
            if has_substantive_title:
                criteria.append("distinct_task_focus")
            return True, criteria, None
        else:
            self.skipped_records["insufficient_dwell_or_interactions"] += 1
            return False, [], "insufficient_dwell_or_interactions"

    def extract_memory_candidates(
        self,
        episodes: Optional[List[Dict[str, Any]]] = None,
        browser_events: Optional[List[Dict[str, Any]]] = None
    ) -> List[StructuredMemoryCandidate]:
        """
        Transforms validated meaningful activity episodes into rich structured memory candidates.
        Correlates concurrent browser domains and interactions.
        """
        if episodes is None or browser_events is None:
            phase2_data = self.load_phase2_data()
            if episodes is None:
                episodes = phase2_data["activity_episodes"]
            if browser_events is None:
                browser_events = phase2_data["browser_events"]
        # Pre-index browser events by session_id for fast temporal lookups
        browsers_by_session: Dict[str, List[Dict[str, Any]]] = {}
        for b in browser_events:
            sid = b.get("session_id") or ""
            browsers_by_session.setdefault(sid, []).append(b)

        now_iso = datetime.now(timezone.utc).isoformat()
        candidates: List[StructuredMemoryCandidate] = []
        seen_candidate_ids: Set[str] = set()

        for ep in episodes:
            eid = ep.get("episode_id")
            sid = ep.get("session_id") or "unassigned_session"
            app = ep.get("application") or "Desktop"
            proc = ep.get("process_name") or ""
            title = ep.get("window_title") or ""
            st = ep.get("start_time") or now_iso
            et = ep.get("end_time") or now_iso
            dur = max(0, int(ep.get("duration_ms") or 0))
            ints = max(0, int(ep.get("interaction_count") or 0))

            # Parse screenshot IDs
            shots_raw = ep.get("screenshot_ids")
            if isinstance(shots_raw, str):
                try:
                    shots = json.loads(shots_raw)
                except Exception:
                    shots = []
            elif isinstance(shots_raw, list):
                shots = shots_raw
            else:
                shots = []

            # Evaluate meaningfulness
            is_meaningful, criteria, skip_reason = self.evaluate_episode_meaningfulness(ep)
            if not is_meaningful:
                continue

            # Correlate browser events in the same session within temporal window
            sess_browsers = browsers_by_session.get(sid, [])
            correlated_domains: List[str] = []
            correlated_browser_events = 0
            correlated_apps = [app]

            st_dt = parse_timestamp(st)
            et_dt = parse_timestamp(et)

            for b in sess_browsers:
                b_ts = b.get("normalized_timestamp")
                b_dt = parse_timestamp(b_ts)
                if st_dt and et_dt and b_dt:
                    # Within episode window extended by configurable margin
                    gap_before = (b_dt - st_dt).total_seconds()
                    gap_after = (et_dt - b_dt).total_seconds()
                    margin = self.config.browser_correlation_window_sec
                    if -margin <= gap_before and -margin <= gap_after:
                        dom = b.get("domain")
                        if dom and dom not in correlated_domains:
                            correlated_domains.append(dom)
                        correlated_browser_events += 1
                        b_name = b.get("browser")
                        if b_name and b_name not in correlated_apps:
                            correlated_apps.append(b_name)

            if correlated_domains:
                criteria.append("correlated_web_research")

            # Generate dynamic activity summary
            summary = generate_dynamic_activity_summary(
                primary_app=app,
                window_title=title,
                browser_domains=correlated_domains,
                duration_ms=dur,
                interaction_count=ints,
                screenshot_count=len(shots),
                max_length=self.config.max_summary_length
            )

            # Deterministic Candidate ID based on episode properties
            id_seed = f"{sid}_{app}_{st}_{eid}"
            cand_hash = hashlib.sha256(id_seed.encode("utf-8")).hexdigest()[:16]
            candidate_id = f"mem_cand_{cand_hash}"

            # Prevent duplicate candidate IDs if multiple episodes share identical identity
            if candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)

            cand = StructuredMemoryCandidate(
                candidate_id=candidate_id,
                session_id=sid,
                source_episode_ids=[eid],
                primary_application=app,
                applications_involved=correlated_apps,
                browser_domains=correlated_domains,
                window_context=title,
                start_time=st,
                end_time=et,
                duration_ms=dur,
                interaction_count=ints,
                browser_event_count=correlated_browser_events,
                screenshot_ids=shots,
                activity_summary=summary,
                meaningful_criteria=criteria,
                created_at=now_iso,
            )
            candidates.append(cand)

        return candidates

    def write_candidates_to_store(
        self,
        candidates: List[StructuredMemoryCandidate],
        statistics: Dict[str, Any]
    ) -> None:
        """
        Saves structured memory candidates to data/dmclm.db (memory_candidates table),
        synchronizes memory_records and memory_records.json, and outputs extraction report.
        Repeatable / idempotent.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.cursor()

            # 1. Create memory_candidates table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_episode_ids TEXT NOT NULL,
                    primary_application TEXT NOT NULL,
                    applications_involved TEXT NOT NULL,
                    browser_domains TEXT NOT NULL,
                    window_context TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    interaction_count INTEGER DEFAULT 0,
                    browser_event_count INTEGER DEFAULT 0,
                    screenshot_ids TEXT,
                    activity_summary TEXT NOT NULL,
                    meaningful_criteria TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cand_session ON memory_candidates(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cand_app ON memory_candidates(primary_application)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cand_start ON memory_candidates(start_time)")

            # Clear previous candidates to guarantee repeatability
            cur.execute("DELETE FROM memory_candidates")

            for c in candidates:
                cur.execute("""
                    INSERT INTO memory_candidates (
                        candidate_id, session_id, source_episode_ids, primary_application,
                        applications_involved, browser_domains, window_context, start_time,
                        end_time, duration_ms, interaction_count, browser_event_count,
                        screenshot_ids, activity_summary, meaningful_criteria, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.candidate_id,
                    c.session_id,
                    json.dumps(c.source_episode_ids),
                    c.primary_application,
                    json.dumps(c.applications_involved),
                    json.dumps(c.browser_domains),
                    c.window_context,
                    c.start_time,
                    c.end_time,
                    c.duration_ms,
                    c.interaction_count,
                    c.browser_event_count,
                    json.dumps(c.screenshot_ids),
                    c.activity_summary,
                    json.dumps(c.meaningful_criteria),
                    c.created_at,
                ))

            conn.commit()
        finally:
            conn.close()

        # 2. Export candidates JSON
        candidates_dict = [c.to_dict() for c in candidates]
        with open(CANDIDATES_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(candidates_dict, f, indent=2)

        # 3. Synchronize memory_records.json for dashboard visibility
        # Note: 5 cognitive features and importance scores remain None!
        memory_records = []
        for c in candidates:
            memory_records.append({
                "memory_id": c.candidate_id,
                "source_type": "activity_episode",
                "source_event_ids": c.source_episode_ids,
                "session_ids": [c.session_id],
                "application": c.primary_application,
                "content": c.activity_summary,
                "domain": c.browser_domains[0] if c.browser_domains else None,
                "first_seen": c.start_time,
                "last_seen": c.end_time,
                "total_duration": c.duration_ms,
                "occurrence_count": 1,
                "interaction_count": c.interaction_count,
                "screenshot_count": len(c.screenshot_ids),
                "retrieval_count": 0,
                "useful_retrieval_count": 0,
                # Phase 4 features remain pending
                "recency_score": None,
                "frequency_score": None,
                "retrieval_history_score": 0.0,
                "semantic_relevance": None,
                "relationship_strength": None,
                "importance_score": None,
                "lifecycle_state": None,
                "created_time": c.created_at,
                "updated_time": c.created_at,
            })

        with open(MEMORY_RECORDS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(memory_records, f, indent=2)

        # 4. Save comprehensive Phase 3 report
        report = {
            "execution_status": "Phase 3 Memory Extraction completed.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": statistics,
            "logs": self.logs,
            "sample_candidates": candidates_dict[:5],
            "output_locations": {
                "sqlite_database": str(self.db_path),
                "sqlite_table": "memory_candidates",
                "candidates_json": str(CANDIDATES_JSON_PATH),
                "memory_records_json": str(MEMORY_RECORDS_JSON_PATH),
            }
        }
        with open(EXTRACTION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """
        Executes end-to-end Phase 3 Memory Extraction:
        1. Loads Phase 2 derived data
        2. Evaluates episode meaningfulness
        3. Extracts structured memory candidates
        4. Writes candidates to derived store
        5. Logs metrics and generates report
        """
        start_time = time.time()
        self._log("[PHASE 3] Starting Memory Extraction from Phase 2 derived data...")

        # 1. Load data
        phase2_data = self.load_phase2_data()
        episodes = phase2_data["activity_episodes"]
        browsers = phase2_data["browser_events"]
        screenshots = phase2_data["screenshots"]

        self._log(f"[PHASE 3] Loaded Phase 2 activity episodes: {len(episodes):,}")
        self._log(f"[PHASE 3] Loaded Phase 2 browser events: {len(browsers):,}")
        self._log(f"[PHASE 3] Loaded Phase 2 screenshots: {len(screenshots):,}")

        # 2. Extract structured memory candidates
        self._log("[PHASE 3] Evaluating episode meaningfulness and validity...")
        candidates = self.extract_memory_candidates(episodes, browsers)

        candidates_count = len(candidates)
        skipped_count = sum(self.skipped_records.values())
        valid_episodes = len(episodes) - self.skipped_records["malformed_or_missing_timestamp"] - self.skipped_records["missing_application"]

        with_browser_context = sum(1 for c in candidates if c.browser_domains or c.browser_event_count > 0)
        with_interactions = sum(1 for c in candidates if c.interaction_count > 0)
        with_screenshots = sum(1 for c in candidates if c.screenshot_ids)

        self._log(
            f"[PHASE 3] Meaningful activity episodes identified: {candidates_count:,} "
            f"(skipped {skipped_count:,} transient noise/invalid records)"
        )
        self._log("[PHASE 3] Extracting structured memory candidates...")
        self._log(f"[PHASE 3] Memory candidates extracted: {candidates_count:,}")
        self._log(f"[PHASE 3] Candidates containing browser context: {with_browser_context:,}")
        self._log(f"[PHASE 3] Candidates containing interaction telemetry: {with_interactions:,}")
        self._log(f"[PHASE 3] Candidates containing attached screenshots: {with_screenshots:,}")

        # 3. Write candidates to storage
        self._log("[PHASE 3] Writing structured memory candidates to derived storage...")
        duration_sec = round(time.time() - start_time, 2)

        statistics = {
            "phase2_episodes_loaded": len(episodes),
            "valid_episodes": valid_episodes,
            "meaningful_episodes_identified": candidates_count,
            "memory_candidates_extracted": candidates_count,
            "candidates_with_browser_context": with_browser_context,
            "candidates_with_interaction_telemetry": with_interactions,
            "candidates_with_screenshots": with_screenshots,
            "skipped_records_count": skipped_count,
            "skipped_records_breakdown": self.skipped_records,
            "processing_duration_sec": duration_sec,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.write_candidates_to_store(candidates, statistics)

        self._log(
            f"[PHASE 3] Memory extraction completed in {duration_sec}s. "
            f"Output: data/dmclm.db (memory_candidates) & data/memory_candidates.json"
        )

        return {
            "status": "COMPLETED",
            "total_episodes_loaded": len(episodes),
            "valid_episodes": valid_episodes,
            "meaningful_episodes": candidates_count,
            "candidates_extracted": candidates_count,
            "candidates_count": candidates_count,
            "skipped_records": self.skipped_records,
            "duration_sec": duration_sec,
            "statistics": statistics,
            "logs": self.logs,
            "candidates": candidates,
        }


if __name__ == "__main__":
    extractor = Phase3MemoryExtractor()
    res = extractor.run()
    print("\n=== PHASE 3 MEMORY EXTRACTION RESULTS ===")
    print(json.dumps(res["statistics"], indent=2))
    print("\n=== SAMPLE CANDIDATES ===")
    for c in res["candidates"][:3]:
        print(f"[{c.candidate_id}] {c.activity_summary}")
