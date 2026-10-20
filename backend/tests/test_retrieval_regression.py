
import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import src.adaptive_manager as manager


@pytest.fixture
def isolated_retrieval_store(tmp_path, monkeypatch):
    """Use temporary JSON stores and prevent live database writes."""
    memory_path = tmp_path / "memory_records.json"
    history_path = tmp_path / "retrieval_history.json"

    memory_path.write_text(
        json.dumps([
            {
                "memory_id": "mem_test_valid",
                "recency_score": 0.5,
                "frequency_score": 0.6,
                "retrieval_history_score": 0.0,
                "semantic_relevance": 0.7,
                "relationship_strength": 0.5,
                "lifecycle_state": "ARCHIVE",
            }
        ]),
        encoding="utf-8",
    )

    history_path.write_text(
        json.dumps({
            "description": "Test retrieval history",
            "total_retrievals": 0,
            "retrievals": [],
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(manager, "MEMORY_JSON_PATH", memory_path)
    monkeypatch.setattr(manager, "RETRIEVAL_HISTORY_PATH", history_path)
    monkeypatch.setattr(
        manager, "record_retrieval_in_db", lambda *a, **k: None
    )
    monkeypatch.setattr(
        manager, "insert_or_replace_memory_records", lambda *a, **k: None
    )

    return memory_path, history_path


def test_invalid_memory_does_not_write_retrieval_history(
    isolated_retrieval_store,
):
    memory_path, history_path = isolated_retrieval_store
    before = json.loads(history_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not found in memory store"):
        manager.record_memory_retrieval("mem_cand_invalid_test")

    after = json.loads(history_path.read_text(encoding="utf-8"))

    assert after == before
    assert json.loads(
        memory_path.read_text(encoding="utf-8")
    )[0]["memory_id"] == "mem_test_valid"


def test_valid_memory_updates_retrieval_features(isolated_retrieval_store):
    memory_path, history_path = isolated_retrieval_store

    result = manager.record_memory_retrieval("mem_test_valid")

    records = json.loads(memory_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    memory = records[0]

    assert result["memory_id"] == "mem_test_valid"
    assert result["retrieval_count"] == 1
    assert memory["recency_score"] == 1.0
    assert memory["retrieval_count"] == 1
    assert memory["useful_retrieval_count"] == 1
    assert len(history["retrievals"]) == 1
    assert history["retrievals"][0]["memory_id"] == "mem_test_valid"


def test_candidate_id_is_rejected_without_audit_entry(
    isolated_retrieval_store,
):
    _, history_path = isolated_retrieval_store
    before = json.loads(history_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not found in memory store"):
        manager.record_memory_retrieval("mem_cand_0010349e242fad1d")

    assert json.loads(history_path.read_text(encoding="utf-8")) == before


def test_database_retrieval_failure_is_not_silently_ignored(
    isolated_retrieval_store, monkeypatch
):
    _, history_path = isolated_retrieval_store

    def fail_database_write(*args, **kwargs):
        raise RuntimeError("simulated SQLite retrieval failure")

    monkeypatch.setattr(
        manager, "record_retrieval_in_db", fail_database_write
    )

    with pytest.raises(
        RuntimeError, match="simulated SQLite retrieval failure"
    ):
        manager.record_memory_retrieval("mem_test_valid")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["total_retrievals"] == 0


def test_database_memory_update_failure_is_not_silently_ignored(
    isolated_retrieval_store, monkeypatch
):
    memory_path, _ = isolated_retrieval_store
    before = json.loads(memory_path.read_text(encoding="utf-8"))

    def fail_database_write(*args, **kwargs):
        raise RuntimeError("simulated SQLite memory update failure")

    monkeypatch.setattr(
        manager, "insert_or_replace_memory_records", fail_database_write
    )

    with pytest.raises(
        RuntimeError, match="simulated SQLite memory update failure"
    ):
        manager.record_memory_retrieval("mem_test_valid")

    after = json.loads(memory_path.read_text(encoding="utf-8"))
    assert after == before
