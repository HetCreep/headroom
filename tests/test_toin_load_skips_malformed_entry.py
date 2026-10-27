"""A malformed pattern entry must not abort the TOIN load or shrink the file on the next save."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from headroom.telemetry.models import ToolSignature
from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork


def _record(toin: ToolIntelligenceNetwork, index: int) -> None:
    toin.record_compression(
        tool_signature=ToolSignature.from_items([{f"field_{index}": "x"}]),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
    )


def _new(path: Path) -> ToolIntelligenceNetwork:
    return ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))


def test_malformed_entry_is_skipped_and_file_keeps_valid_entries(tmp_path: Path, caplog) -> None:
    path = tmp_path / "toin.json"
    toin = _new(path)
    for i in range(5):
        _record(toin, i)
    toin.save()

    data = json.loads(path.read_text())
    assert len(data["patterns"]) == 5
    del data["patterns"][list(data["patterns"])[1]]["tool_signature_hash"]
    corrupted_text = json.dumps(data)
    path.write_text(corrupted_text, newline="\n")

    with caplog.at_level(logging.WARNING):
        reloaded = _new(path)
    assert len(reloaded._patterns) == 4
    skipped_logs = [
        r for r in caplog.records if getattr(r, "event", "") == "toin_import_entries_skipped"
    ]
    assert len(skipped_logs) == 1 and skipped_logs[0].skipped == 1

    _record(reloaded, 99)
    reloaded.save()

    assert len(json.loads(path.read_text())["patterns"]) == 5  # 4 valid + 1 new
    # The pre-save file is kept once, never overwritten by a later save.
    bad = tmp_path / "toin.json.bad"
    assert bad.read_text() == corrupted_text
    _record(reloaded, 100)
    reloaded.save()
    assert bad.read_text() == corrupted_text


def test_clean_load_leaves_no_copy(tmp_path: Path) -> None:
    path = tmp_path / "toin.json"
    toin = _new(path)
    _record(toin, 0)
    toin.save()
    reloaded = _new(path)
    _record(reloaded, 1)
    reloaded.save()
    assert not (tmp_path / "toin.json.bad").exists()


def _corrupt(path: Path, field: str, value: object) -> str:
    """Save 3 patterns, set ``field`` of the first one to ``value``; return the new text."""
    toin = _new(path)
    for i in range(3):
        _record(toin, i)
    toin.save()
    data = json.loads(path.read_text())
    data["patterns"][list(data["patterns"])[0]][field] = value
    text = json.dumps(data)
    path.write_text(text, newline="\n")
    return text


@pytest.mark.parametrize("value", ["oops", None, True, [1]])
def test_wrong_typed_numeric_field_counts_as_malformed(tmp_path: Path, value: object) -> None:
    path = tmp_path / "toin.json"
    _corrupt(path, "total_compressions", value)
    reloaded = _new(path)
    assert len(reloaded._patterns) == 2
    assert reloaded.get_stats()["total_compressions"] == 2


def test_failed_copy_skips_save_and_keeps_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    path = tmp_path / "toin.json"
    text = _corrupt(path, "total_compressions", "oops")
    reloaded = _new(path)
    _record(reloaded, 99)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", boom)
    with caplog.at_level(logging.WARNING):
        reloaded.save()
        reloaded.save()
    assert path.read_text() == text  # original never overwritten without a copy
    assert not (tmp_path / "toin.json.bad").exists()
    assert len([r for r in caplog.records if "Could not keep a copy" in r.getMessage()]) == 1

    monkeypatch.undo()  # copy works again: the next save keeps the copy, then writes
    reloaded.save()
    assert (tmp_path / "toin.json.bad").read_text() == text
    assert len(json.loads(path.read_text())["patterns"]) == 3  # 2 valid + 1 new


def test_existing_bad_copy_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "toin.json"
    _corrupt(path, "total_compressions", "oops")
    (tmp_path / "toin.json.bad").write_text("older copy", newline="\n")
    reloaded = _new(path)
    reloaded.save()
    assert (tmp_path / "toin.json.bad").read_text() == "older copy"
    assert len(json.loads(path.read_text())["patterns"]) == 2


def test_store_removed_after_load_still_saves(tmp_path: Path) -> None:
    path = tmp_path / "toin.json"
    _corrupt(path, "total_compressions", "oops")
    reloaded = _new(path)
    path.unlink()
    reloaded.save()
    assert len(json.loads(path.read_text())["patterns"]) == 2
    assert not (tmp_path / "toin.json.bad").exists()


def test_non_file_backend_saves_without_copy() -> None:
    class MemBackend:
        saved: dict | None = None

        def load(self) -> dict:
            return {}

        def save(self, data: dict) -> None:
            self.saved = data

    backend = MemBackend()
    toin = ToolIntelligenceNetwork(TOINConfig(auto_save_interval=0), backend=backend)  # type: ignore[arg-type]
    toin._load_skipped_entries = True
    _record(toin, 0)
    toin.save()
    assert backend.saved is not None and len(backend.saved["patterns"]) == 1


def test_disabled_toin_import_skips_nothing(tmp_path: Path) -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(tmp_path / "t.json"), enabled=False))
    assert toin.import_patterns({"patterns": {"bad": {}}}) == 0


def _foreign_export(path: Path, field: str, value: object) -> str:
    """Like ``_corrupt`` but the file came from another instance, so tracking runs on import."""
    _corrupt(path, field, value)
    data = json.loads(path.read_text())
    data["instance_id"] = "foreign-instance"
    text = json.dumps(data)
    path.write_text(text, newline="\n")
    return text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seen_instance_hashes", "bad"),
        ("seen_instance_hashes", {"a": 1}),
        ("seen_instance_hashes", [1]),
        ("seen_instance_hashes", 5),
        ("commonly_retrieved_fields", "bad"),
        ("common_query_patterns", {"a": 1}),
        ("preserve_fields", [["x"]]),
        ("field_retrieval_frequency", {"a": "x"}),
        ("query_pattern_frequency", [1]),
        ("strategy_success_rates", {"a": None}),
        ("optimal_strategy", ["s"]),
        ("tool_signature_hash", 5),
    ],
)
def test_malformed_collection_field_skips_only_that_entry(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "toin.json"
    text = _foreign_export(path, field, value)
    reloaded = _new(path)
    assert len(reloaded._patterns) == 2  # the two healthy entries survive
    assert reloaded._load_skipped_entries is True
    for p in reloaded._patterns.values():  # tracked once each, none half-added
        assert p.user_count == 2 and p._seen_instance_hashes.count("foreign-instance") == 1
    reloaded.save()
    assert (tmp_path / "toin.json.bad").read_bytes() == text.encode()
    assert len(json.loads(path.read_text())["patterns"]) == 2


def test_failed_merge_leaves_existing_pattern_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _new(tmp_path / "a.json")
    _record(source, 0)
    exported = source.export_patterns()
    target = _new(tmp_path / "b.json")
    _record(target, 0)  # same signature: the import merges into this pattern
    before = [p.to_dict() for p in target._patterns.values()]

    def boom(*args, **kwargs):
        raise RuntimeError("merge failed late")

    monkeypatch.setattr(target, "_update_recommendations", boom)
    assert target.import_patterns(exported) == 1
    assert [p.to_dict() for p in target._patterns.values()] == before
    monkeypatch.undo()
    assert target.import_patterns(exported) == 0
    assert next(iter(target._patterns.values())).total_compressions == 2


def test_foreign_import_tracks_source_once_and_caps_at_100(tmp_path: Path) -> None:
    source = _new(tmp_path / "a.json")
    for i in range(2):
        _record(source, i)
    exported = source.export_patterns()
    exported["instance_id"] = "src"
    first, second = list(exported["patterns"].values())
    first["seen_instance_hashes"] = ["src"]  # already counted: not added again
    second["seen_instance_hashes"] = [f"h{i}" for i in range(100)]  # list at its cap
    target = _new(tmp_path / "b.json")
    assert target.import_patterns(exported) == 0
    counts = sorted(len(p._seen_instance_hashes) for p in target._patterns.values())
    assert counts == [1, 100]
    assert sorted(p.user_count for p in target._patterns.values()) == [1, 2]  # cap still counts
