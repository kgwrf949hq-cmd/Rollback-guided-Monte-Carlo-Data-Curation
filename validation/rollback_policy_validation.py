#!/usr/bin/env python3
"""Deterministic validation for the pure rollback-point policy layer."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from pick_place_failure import FailureDecision, FailureStatus
from rollback_policy import (
    RollbackCheckpoint,
    StageAwareRollbackPolicy,
    TemporalRollbackPolicy,
)


def _decision(
    reason: str,
    stage: str,
    *,
    step: int = 20,
    warning_onset_step: Optional[int] = 18,
) -> FailureDecision:
    return FailureDecision(
        step=step,
        status=FailureStatus.RECOVERABLE_FAILURE,
        risk_score=1.0,
        reason=reason,
        current_stage=stage,
        consecutive_steps=2,
        hard_failure=True,
        warning_onset_step=warning_onset_step,
        evidence={"fixture": True},
    )


def _checkpoint(
    checkpoint_id: str,
    step: int,
    *,
    anchor: Optional[str] = None,
    labels: Tuple[str, ...] = (),
    env_done: bool = False,
    semantic_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> RollbackCheckpoint:
    checkpoint_metadata = dict(metadata or {})
    checkpoint_labels = list(labels)
    if anchor is not None:
        checkpoint_metadata.update(
            {
                "checkpoint_source": "subevent_completion",
                "last_completed_event": anchor,
                "subevent_transition": {
                    "event_name": anchor,
                    "confidence": 0.9,
                },
            }
        )
        checkpoint_labels.extend(("subevent_completed", anchor))
    return RollbackCheckpoint(
        checkpoint_id=checkpoint_id,
        owner_branch_id="branch_fixture",
        step=step,
        env_done=env_done,
        semantic_score=semantic_score,
        labels=tuple(checkpoint_labels),
        metadata=checkpoint_metadata,
    )


def _validate_temporal_strict_cutoff() -> None:
    selection = TemporalRollbackPolicy().select(
        _decision("fixture", "grasp_completed", warning_onset_step=12),
        (
            _checkpoint("cp_8", 8),
            _checkpoint("cp_9", 9),
            _checkpoint("cp_10", 10),
            _checkpoint("cp_terminal", 7, env_done=True),
        ),
        safety_margin_steps=2,
    )
    assert selection.cutoff_step == 10
    assert selection.selected_checkpoint_id == "cp_9"
    candidates = {item.checkpoint_id: item for item in selection.candidates}
    assert candidates["cp_9"].eligible
    assert candidates["cp_10"].rejection_reason == "not_strictly_before_cutoff"
    assert candidates["cp_terminal"].rejection_reason == "terminal_checkpoint"


def _validate_failure_specific_anchors() -> None:
    policy = StageAwareRollbackPolicy()
    common = (
        _checkpoint("episode_start", 0),
        _checkpoint("approach_anchor", 3, anchor="approach_object"),
        _checkpoint("grasp_anchor", 6, anchor="grasp_completed"),
        _checkpoint("lift_anchor", 9, anchor="object_lifted"),
        _checkpoint("place_anchor", 12, anchor="place_completed"),
        _checkpoint(
            "ordinary_late_checkpoint",
            16,
            metadata={
                "completed_event_count": 4,
                "last_completed_event": "place_completed",
                "event_detector_state": {
                    "completed_events": [
                        "approach_object",
                        "grasp_completed",
                        "object_lifted",
                        "place_completed",
                    ]
                },
            },
        ),
    )

    missed = policy.select(
        _decision("missed_grasp", "grasp_completed"),
        common,
        safety_margin_steps=1,
    )
    assert missed.selected_checkpoint_id == "approach_anchor"
    assert missed.selected_anchor == "approach_object"
    assert not missed.fallback_used
    missed_candidates = {
        item.checkpoint_id: item for item in missed.candidates
    }
    assert not missed_candidates["ordinary_late_checkpoint"].eligible
    assert (
        missed_candidates["ordinary_late_checkpoint"].rejection_reason
        == "no_requested_anchor"
    )

    lift = policy.select(
        _decision("lift_no_progress", "object_lifted"),
        common,
        safety_margin_steps=1,
    )
    assert lift.selected_checkpoint_id == "grasp_anchor"
    assert lift.selected_anchor == "grasp_completed"

    dropped = policy.select(
        _decision("object_dropped", "place_completed"),
        common,
        safety_margin_steps=1,
    )
    assert dropped.selected_checkpoint_id == "lift_anchor"
    assert dropped.selected_anchor == "object_lifted"
    dropped_candidates = {
        item.checkpoint_id: item for item in dropped.candidates
    }
    assert dropped_candidates["grasp_anchor"].matched_anchor == "grasp_completed"
    assert (
        dropped_candidates["grasp_anchor"].rejection_reason
        == "lower_priority_anchor"
    )

    dropped_without_lift = policy.select(
        _decision("object_dropped", "place_completed"),
        tuple(item for item in common if item.checkpoint_id != "lift_anchor"),
        safety_margin_steps=1,
    )
    assert dropped_without_lift.selected_checkpoint_id == "grasp_anchor"
    assert dropped_without_lift.selected_anchor == "grasp_completed"

    release = policy.select(
        _decision("release_timeout", "release_completed"),
        common,
        safety_margin_steps=1,
    )
    assert release.selected_checkpoint_id == "place_anchor"
    assert release.selected_anchor == "place_completed"


def _validate_cutoff_and_fallbacks() -> None:
    policy = StageAwareRollbackPolicy()
    anchor_after_cutoff = policy.select(
        _decision(
            "missed_grasp",
            "grasp_completed",
            warning_onset_step=10,
        ),
        (
            _checkpoint("temporal_7", 7),
            _checkpoint("late_approach_anchor", 9, anchor="approach_object"),
        ),
        safety_margin_steps=1,
    )
    assert anchor_after_cutoff.cutoff_step == 9
    assert anchor_after_cutoff.fallback_used
    assert (
        anchor_after_cutoff.fallback_reason
        == "no_matching_anchor_before_cutoff"
    )
    assert anchor_after_cutoff.selected_checkpoint_id == "temporal_7"
    late = {
        item.checkpoint_id: item for item in anchor_after_cutoff.candidates
    }["late_approach_anchor"]
    assert not late.eligible
    assert late.rejection_reason == "not_strictly_before_cutoff"

    missing_anchor = policy.select(
        _decision("release_timeout", "release_completed"),
        (_checkpoint("temporal_11", 11), _checkpoint("temporal_15", 15)),
        safety_margin_steps=1,
    )
    assert missing_anchor.fallback_used
    assert missing_anchor.selected_checkpoint_id == "temporal_15"
    assert missing_anchor.strategy == "latest_safe"

    stage_predecessor = policy.select(
        _decision("new_unconfigured_failure", "place_completed"),
        (
            _checkpoint("grasp", 5, anchor="grasp_completed"),
            _checkpoint("lift", 10, anchor="object_lifted"),
        ),
        safety_margin_steps=1,
    )
    assert stage_predecessor.requested_anchors == (
        "object_lifted",
        "grasp_completed",
    )
    assert stage_predecessor.selected_checkpoint_id == "lift"
    assert not stage_predecessor.fallback_used


def main() -> None:
    _validate_temporal_strict_cutoff()
    _validate_failure_specific_anchors()
    _validate_cutoff_and_fallbacks()

    serializable = StageAwareRollbackPolicy().select(
        _decision("missed_grasp", "grasp_completed"),
        (_checkpoint("approach", 4, anchor="approach_object"),),
        safety_margin_steps=1,
    ).to_summary()
    json.dumps(serializable, sort_keys=True, allow_nan=False)

    print(
        json.dumps(
            {
                "checks": {
                    "anchor_after_cutoff_rejected": True,
                    "failure_specific_anchor_priority": True,
                    "ordinary_checkpoint_not_event_anchor": True,
                    "selection_json_serializable": True,
                    "stage_predecessor_fallback": True,
                    "temporal_fallback": True,
                    "temporal_strict_cutoff": True,
                },
                "policy": "stage_aware_with_temporal_fallback",
                "validation_passed": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
