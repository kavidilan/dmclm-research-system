import sqlite3
from pathlib import Path
from typing import Any


COLLECTOR_DB = Path(
    r"C:\Users\Admin\AppData\Local\DesktopResearchCollector\collector.db"
)


def get_connection():
    """Create a read-only connection to the Collector database."""
    if not COLLECTOR_DB.exists():
        raise FileNotFoundError(
            f"Collector database not found: {COLLECTOR_DB}"
        )

    # SQLite read-only mode
    uri = f"file:{COLLECTOR_DB.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_table_count(connection, table_name: str) -> int:
    """Return the number of records in a table."""
    query = f"SELECT COUNT(*) FROM {table_name}"
    return connection.execute(query).fetchone()[0]


def load_activity_events(
    connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Load application/activity events."""
    rows = connection.execute(
        """
        SELECT
            event_id,
            session_id,
            timestamp,
            event_type,
            application_name,
            process_name,
            window_title,
            duration_ms,
            metadata_json
        FROM activity_events
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    columns = [
        "event_id",
        "session_id",
        "timestamp",
        "event_type",
        "application_name",
        "process_name",
        "window_title",
        "duration_ms",
        "metadata_json",
    ]

    return [dict(zip(columns, row)) for row in rows]


def load_browser_events(
    connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Load browser/research activity."""
    rows = connection.execute(
        """
        SELECT
            browser_event_id,
            session_id,
            timestamp,
            browser_name,
            domain,
            page_title,
            url
        FROM browser_events
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    columns = [
        "browser_event_id",
        "session_id",
        "timestamp",
        "browser_name",
        "domain",
        "page_title",
        "url",
    ]

    return [dict(zip(columns, row)) for row in rows]


def load_interaction_events(
    connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Load keyboard/mouse interaction activity."""
    rows = connection.execute(
        """
        SELECT
            interaction_id,
            session_id,
            timestamp,
            interaction_type,
            application_name,
            count
        FROM interaction_events
        ORDER BY timestamp ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    columns = [
        "interaction_id",
        "session_id",
        "timestamp",
        "interaction_type",
        "application_name",
        "count",
    ]

    return [dict(zip(columns, row)) for row in rows]


def load_sessions(
    connection, limit: int = 1000
) -> list[dict[str, Any]]:
    """Load collection sessions."""
    rows = connection.execute(
        """
        SELECT
            session_id,
            started_at,
            ended_at,
            end_reason
        FROM sessions
        ORDER BY started_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    columns = [
        "session_id",
        "started_at",
        "ended_at",
        "end_reason",
    ]

    return [dict(zip(columns, row)) for row in rows]


def load_screenshots(
    connection, limit: int = 1000, offset: int = 0
) -> list[dict[str, Any]]:
    """Load screenshot events metadata (excluding raw image binary data)."""
    rows = connection.execute(
        """
        SELECT
            screenshot_id,
            session_id,
            timestamp,
            capture_reason,
            application_name,
            window_title,
            file_size
        FROM screenshots
        ORDER BY timestamp ASC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()

    columns = [
        "screenshot_id",
        "session_id",
        "timestamp",
        "capture_reason",
        "application_name",
        "window_title",
        "file_size",
    ]

    return [dict(zip(columns, row)) for row in rows]


def load_collector_data(
    limit: int = 1000,
    offset: int = 0,
    include_screenshots: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Load real data from DesktopResearchCollector."""
    connection = get_connection()

    try:
        data = {
            "activity_events": load_activity_events(
                connection, limit
            ),
            "browser_events": load_browser_events(
                connection, limit
            ),
            "interaction_events": load_interaction_events(
                connection, limit
            ),
            "sessions": load_sessions(
                connection, limit
            ),
        }
        if include_screenshots:
            data["screenshots"] = load_screenshots(
                connection, limit, offset
            )

        return data

    finally:
        connection.close()


def print_database_summary():
    """Print a summary of the real Collector database."""
    connection = get_connection()

    try:
        tables = [
            "activity_events",
            "browser_events",
            "interaction_events",
            "screenshots",
            "sessions",
            "collection_policies",
            "upload_batches",
        ]

        print("\n=== DesktopResearchCollector Database ===")
        print(f"Database: {COLLECTOR_DB}\n")

        for table in tables:
            count = get_table_count(connection, table)
            print(f"{table:25} {count:>8} records")

    finally:
        connection.close()


if __name__ == "__main__":
    print_database_summary()

    data = load_collector_data(limit=5)

    print("\n=== Sample Activity Events ===")
    for item in data["activity_events"]:
        print(item)

    print("\n=== Sample Browser Events ===")
    for item in data["browser_events"]:
        print(item)

    print("\n=== Sample Interaction Events ===")
    for item in data["interaction_events"]:
        print(item)

    print("\n=== Sample Sessions ===")
    for item in data["sessions"]:
        print(item)