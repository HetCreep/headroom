"""A malformed pattern entry must not abort the TOIN load or shrink the file on the next save."""

from __future__ import annotations

import json
import logging
from pathlib import Path

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
    path.write_text(corrupted_text)

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
