"""A client-supplied ``model`` string must not grow per-model state without bound."""

from __future__ import annotations

import json

from headroom.proxy.cost import CostTracker
from headroom.proxy.persistent_metrics import (
    MAX_LABEL_LENGTH,
    MAX_TRACKED_MODELS,
)
from headroom.proxy.savings_tracker import SavingsTracker

_NAMES = MAX_TRACKED_MODELS + 50


def test_cost_tracker_per_model_maps_are_bounded():
    tracker = CostTracker()
    for i in range(_NAMES):
        tracker.record_tokens(f"m-{i}-" + "x" * 2000, tokens_saved=1, tokens_sent=10)

    assert len(tracker._requests_by_model) == MAX_TRACKED_MODELS + 1
    assert len(tracker._tokens_sent_by_model) == MAX_TRACKED_MODELS + 1
    assert tracker._requests_by_model["other"] == _NAMES - MAX_TRACKED_MODELS
    assert all(len(k) <= MAX_LABEL_LENGTH for k in tracker._requests_by_model)
    # Totals stay exact: over-cap requests are bucketed, not dropped.
    assert sum(tracker._tokens_sent_by_model.values()) == 10 * _NAMES


def test_estimated_basis_warning_set_is_bounded():
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_estimated_basis_models.clear()
    tracker = CostTracker()
    for i in range(_NAMES):
        tracker.record_tokens(f"w-{i}", tokens_saved=0, tokens_sent=10)
    assert len(cost_mod._warned_estimated_basis_models) <= MAX_TRACKED_MODELS + 1
    cost_mod._warned_estimated_basis_models.clear()


def test_savings_tracker_by_model_is_bounded_and_persisted_small(tmp_path):
    path = tmp_path / "proxy_savings.json"
    tracker = SavingsTracker(path=str(path), save_flush_every=1)
    for i in range(_NAMES):
        tracker.record_request(model=f"s-{i}-" + "y" * 2000, input_tokens=100, tokens_saved=10)

    by_model = json.loads(path.read_text(encoding="utf-8"))["by_model"]
    assert len(by_model) == MAX_TRACKED_MODELS + 1
    assert all(len(k) <= MAX_LABEL_LENGTH for k in by_model)
    assert by_model["other"]["requests"] == _NAMES - MAX_TRACKED_MODELS
    assert sum(e["tokens_saved"] for e in by_model.values()) == 10 * _NAMES
