"""A negative compression-savings total must survive a restart, not reload as 0."""

from __future__ import annotations

import pytest

from headroom.proxy.persistent_metrics import PersistentMetricsState
from headroom.proxy.savings_tracker import SavingsTracker


def _losing_request(tracker: SavingsTracker) -> None:
    tracker.record_request(
        model="claude-sonnet-4",
        tokens_saved=100,
        input_tokens=1000,
        estimated_savings_usd={
            "compression": -0.5,
            "tool_schema": 0.0,
            "compression_list": -0.5,
            "tool_schema_list": 0.0,
            "basis": "cache_aware",
        },
    )
    tracker.flush()


def test_negative_savings_survive_reload(tmp_path) -> None:
    path = str(tmp_path / "savings.json")
    live = SavingsTracker(path=path)
    _losing_request(live)
    snap = live.snapshot()
    assert snap["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert snap["display_session"]["compression_savings_usd"] == pytest.approx(-0.5)

    reloaded = SavingsTracker(path=path).snapshot()
    assert reloaded["lifetime"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["display_session"]["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["history"][-1]["compression_savings_usd"] == pytest.approx(-0.5)


def test_lifetime_metrics_keep_negative_compression_savings() -> None:
    state = PersistentMetricsState()
    state.record_request(
        provider="anthropic",
        model="claude-sonnet-4",
        stack=None,
        compression_savings_usd=-0.5,
    )
    cost = state.to_dict()["cost"]
    assert cost["compression_savings_usd"] == pytest.approx(-0.5)
    assert cost["compression_savings_list_usd"] == pytest.approx(-0.5)

    reloaded = PersistentMetricsState(state.to_dict()).to_dict()["cost"]
    assert reloaded["compression_savings_usd"] == pytest.approx(-0.5)
    assert reloaded["compression_savings_list_usd"] == pytest.approx(-0.5)
