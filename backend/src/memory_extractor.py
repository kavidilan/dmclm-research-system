"""
DMCLM Memory Extractor
Transforms raw multi-modal DesktopResearchCollector events into normalized episodic memory records.
Ingests:
- activity_events (app switches, window titles, duration)
- browser_events (web domains, page titles, URLs)
- interaction_events (keyboard/mouse interactions)
- screenshots (contextual screen capture metadata without raw binaries)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collector_reader import load_collector_data

# Output file for normalized DMCLM memory records
OUTPUT_FILE = ROOT_DIR / "data" / "memory_records.json"


def create_activity_memory(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an activity event into a normalized memory record."""
    application = event.get("application_name") or "Unknown Application"
    window_title = event.get("window_title") or ""
    event_type = event.get("event_type") or "unknown"

    content_parts = [application, window_title, event_type]
    content = " | ".join(part for part in content_parts if part)

    return {
        "memory_id": event["event_id"],
        "source_type": "activity",
        "timestamp": event["timestamp"],
        "session_id": event["session_id"],
        "application": application,
        "content": content,
        "duration_ms": event.get("duration_ms") or 0,
        "domain": None,
        "interaction_count": 0,
        # DMCLM features (populated during Feature Engineering)
        "recency": 0.0,
        "frequency": 0.0,
        "retrieval_history": 0.0,
        "semantic_relevance": 0.0,
        "relationship_strength": 0.0,
    }


def create_browser_memory(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a browser event into a normalized memory record."""
    browser = event.get("browser_name") or "Unknown Browser"
    domain = event.get("domain") or ""
    page_title = event.get("page_title") or ""
    url = event.get("url") or ""

    content_parts = [browser, domain, page_title, url]
    content = " | ".join(part for part in content_parts if part)

    return {
        "memory_id": event["browser_event_id"],
        "source_type": "browser",
        "timestamp": event["timestamp"],
        "session_id": event["session_id"],
        "application": browser,
        "content": content,
        "duration_ms": 0,
        "domain": domain or None,
        "interaction_count": 0,
        # DMCLM features (populated during Feature Engineering)
        "recency": 0.0,
        "frequency": 0.0,
        "retrieval_history": 0.0,
        "semantic_relevance": 0.0,
        "relationship_strength": 0.0,
    }


def create_interaction_memory(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an interaction event into a normalized memory record."""
    application = event.get("application_name") or "Unknown Application"
    interaction_type = event.get("interaction_type") or "unknown"
    count = event.get("count") or 0

    content = f"{application} | {interaction_type} (x{count})"

    return {
        "memory_id": event["interaction_id"],
        "source_type": "interaction",
        "timestamp": event["timestamp"],
        "session_id": event["session_id"],
        "application": application,
        "content": content,
        "duration_ms": 0,
        "domain": None,
        "interaction_count": count,
        # DMCLM features (populated during Feature Engineering)
        "recency": 0.0,
        "frequency": 0.0,
        "retrieval_history": 0.0,
        "semantic_relevance": 0.0,
        "relationship_strength": 0.0,
    }


def create_screenshot_memory(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a screenshot event metadata into a normalized memory record."""
    application = event.get("application_name") or "System"
    window_title = event.get("window_title") or ""
    capture_reason = event.get("capture_reason") or "periodic_capture"
    file_size_kb = round((event.get("file_size") or 0) / 1024, 1)

    content_parts = [application, window_title, f"Capture: {capture_reason} ({file_size_kb} KB)"]
    content = " | ".join(part for part in content_parts if part)

    return {
        "memory_id": event["screenshot_id"],
        "source_type": "screenshot",
        "timestamp": event["timestamp"],
        "session_id": event["session_id"],
        "application": application,
        "content": content,
        "duration_ms": 0,
        "domain": None,
        "interaction_count": 0,
        # DMCLM features (populated during Feature Engineering)
        "recency": 0.0,
        "frequency": 0.0,
        "retrieval_history": 0.0,
        "semantic_relevance": 0.0,
        "relationship_strength": 0.0,
    }


def extract_memory_records(limit: int = 1500) -> List[Dict[str, Any]]:
    """
    Read real DesktopResearchCollector data and convert it into DMCLM memory records.
    Ingests up to `limit` records per table across activity, browser, interaction, and screenshots.
    """
    print(f"Reading real DesktopResearchCollector data (limit={limit} per modality)...")
    data = load_collector_data(limit=limit, include_screenshots=True)

    memories: List[Dict[str, Any]] = []

    # Activity events
    for event in data.get("activity_events", []):
        memories.append(create_activity_memory(event))

    # Browser events
    for event in data.get("browser_events", []):
        memories.append(create_browser_memory(event))

    # Interaction events
    for event in data.get("interaction_events", []):
        memories.append(create_interaction_memory(event))

    # Screenshot metadata
    for event in data.get("screenshots", []):
        memories.append(create_screenshot_memory(event))

    # Sort all records chronologically
    memories.sort(key=lambda x: x.get("timestamp") or "")

    return memories


def save_memory_records(memories: List[Dict[str, Any]]) -> None:
    """Save normalized memory records as JSON."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(memories, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(memories)} normalized memory records to: {OUTPUT_FILE}")


if __name__ == "__main__":
    print("\n=== DMCLM REAL DATA MEMORY EXTRACTION ===")
    memories = extract_memory_records(limit=1500)
    print(f"Extracted {len(memories)} real multi-modal records.")
    
    # Counts by source type
    source_counts: Dict[str, int] = {}
    for m in memories:
        st = m.get("source_type", "unknown")
        source_counts[st] = source_counts.get(st, 0) + 1
    
    print("Breakdown by source modality:")
    for st, count in sorted(source_counts.items()):
        print(f"  - {st:15}: {count} records")

    save_memory_records(memories)