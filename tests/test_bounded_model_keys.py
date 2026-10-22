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


def test_overflowing_passthrough_names_stay_out_of_processed_totals():
    tracker = CostTracker()
    for i in range(MAX_TRACKED_MODELS):
        tracker.record_tokens(f"passthrough:batch:m{i}", tokens_saved=1, tokens_sent=1)
    tracker.record_tokens("passthrough:batch:next", tokens_saved=10, tokens_sent=90)

    stats = tracker.stats()
    assert stats["total_tokens_saved"] == 0
    assert stats["total_input_tokens"] == 0
    assert "other" not in tracker._requests_by_model
    assert tracker._requests_by_model["passthrough:other"] == 1


def test_estimated_basis_warning_set_is_capped_across_resets():
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_estimated_basis_models.clear()
    tracker = CostTracker()
    for i in range(3):
        for j in range(MAX_TRACKED_MODELS):
            tracker.record_tokens(f"r{i}-{j}", tokens_saved=0, tokens_sent=10)
        tracker.reset_runtime()
    assert len(cost_mod._warned_estimated_basis_models) <= MAX_TRACKED_MODELS
    cost_mod._warned_estimated_basis_models.clear()


def test_savings_history_checkpoints_clip_model_names(tmp_path):
    tracker = SavingsTracker(path=str(tmp_path / "proxy_savings.json"), save_flush_every=1)
    tracker.record_request(model="h-" + "z" * 2000, input_tokens=100, tokens_saved=10)

    history = json.loads((tmp_path / "proxy_savings.json").read_text(encoding="utf-8"))["history"]
    assert history
    assert all(len(point["model"]) <= MAX_LABEL_LENGTH for point in history)
