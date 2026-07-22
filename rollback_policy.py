#!/usr/bin/env python3
"""Pluggable rollback-point policies for Monte Carlo curation.

Policies in this module are pure: they inspect a failure decision and an
already branch-filtered checkpoint view, then return an auditable selection.
They never restore an environment, mutate checkpoint registries, create
branches, or write data.

``TemporalRollbackPolicy`` preserves the original strict time baseline.
``StageAwareRollbackPolicy`` first searches automatically generated pick-place
event anchors and falls back to that temporal baseline when an anchor is not
configured or not present in the active lineage.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pick_place_failure import FailureDecision


@dataclasses.dataclass(frozen=True)
class RollbackCheckpoint:
    """Policy-facing, simulator-independent checkpoint description."""

    checkpoint_id: str
    owner_branch_id: str
    step: int
    env_done: bool
    semantic_score: float = 0.0
    labels: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RollbackCandidate:
    checkpoint_id: str
    owner_branch_id: str
    step: int
    eligible: bool
    rejection_reason: Optional[str]
    matched_anchor: Optional[str] = None
    policy_score: Optional[float] = None
    score_components: Mapping[str, float] = dataclasses.field(default_factory=dict)
    rank: Optional[int] = None

    def to_summary(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["score_components"] = {
            str(key): float(value) for key, value in self.score_components.items()
        }
        return result


@dataclasses.dataclass
class RollbackSelection:
    failure_step: int
    warning_onset_step: Optional[int]
    safety_margin_steps: int
    strategy: str
    selection_basis: str
    cutoff_step: int
    candidates: List[RollbackCandidate]
    selected_checkpoint_id: Optional[str]
    selected_step: Optional[int]
    policy_name: str = "temporal"
    requested_anchors: Tuple[str, ...] = ()
    selected_anchor: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    ranked_checkpoint_ids: Tuple[str, ...] = ()

    def to_summary(self) -> Dict[str, Any]:
        return {
            "failure_step": int(self.failure_step),
            "warning_onset_step": self.warning_onset_step,
            "safety_margin_steps": int(self.safety_margin_steps),
            "strategy": self.strategy,
            "selection_basis": self.selection_basis,
            "cutoff_step": int(self.cutoff_step),
            "policy_name": self.policy_name,
            "requested_anchors": list(self.requested_anchors),
            "selected_anchor": self.selected_anchor,
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "ranked_checkpoint_ids": list(self.ranked_checkpoint_ids),
            "candidates": [item.to_summary() for item in self.candidates],
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "selected_step": self.selected_step,
        }


class RollbackPolicy:
    """Pure interface for selecting one checkpoint from a lineage view."""

    name = "rollback_policy"

    def select(
        self,
        decision: FailureDecision,
        checkpoints: Sequence[RollbackCheckpoint],
        *,
        safety_margin_steps: int,
    ) -> RollbackSelection:
        raise NotImplementedError


def _temporal_cutoff(
    decision: FailureDecision, safety_margin_steps: int
) -> Tuple[str, str, int]:
    if int(safety_margin_steps) < 0:
        raise ValueError("safety_margin_steps must be non-negative")
    if decision.warning_onset_step is not None:
        return (
            "latest_safe",
            "warning_onset_temporal_margin",
            int(decision.warning_onset_step) - int(safety_margin_steps),
        )
    return "latest_before_failure", "temporal_only", int(decision.step)


def _event_anchors(checkpoint: RollbackCheckpoint) -> Tuple[str, ...]:
    """Extract automatically generated event anchors from checkpoint metadata."""

    anchors = []
    metadata = checkpoint.metadata
    known_events = {
        "approach_object",
        "grasp_completed",
        "object_lifted",
        "place_completed",
        "release_completed",
    }
    anchors.extend(str(label) for label in checkpoint.labels if label in known_events)
    transition = metadata.get("subevent_transition")
    if isinstance(transition, Mapping) and transition.get("event_name"):
        anchors.append(str(transition["event_name"]))
    if metadata.get("checkpoint_source") == "subevent_completion":
        last_completed = metadata.get("last_completed_event")
        if last_completed:
            anchors.append(str(last_completed))
    return tuple(dict.fromkeys(anchors))


def _semantic_confidence(checkpoint: RollbackCheckpoint) -> float:
    transition = checkpoint.metadata.get("subevent_transition")
    if isinstance(transition, Mapping):
        try:
            return float(transition.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _completed_event_count(checkpoint: RollbackCheckpoint) -> float:
    try:
        return float(checkpoint.metadata.get("completed_event_count", 0.0))
    except (TypeError, ValueError):
        return 0.0


class TemporalRollbackPolicy(RollbackPolicy):
    """Select the latest non-terminal checkpoint before a strict cutoff."""

    name = "temporal"

    def select(
        self,
        decision: FailureDecision,
        checkpoints: Sequence[RollbackCheckpoint],
        *,
        safety_margin_steps: int,
    ) -> RollbackSelection:
        strategy, basis, cutoff = _temporal_cutoff(
            decision, safety_margin_steps
        )
        evaluated = []
        eligible = []
        for order, checkpoint in enumerate(checkpoints):
            reason = None
            if checkpoint.env_done:
                reason = "terminal_checkpoint"
            elif int(checkpoint.step) >= cutoff:
                reason = "not_strictly_before_cutoff"
            score = float(checkpoint.step) if reason is None else None
            candidate = RollbackCandidate(
                checkpoint_id=checkpoint.checkpoint_id,
                owner_branch_id=checkpoint.owner_branch_id,
                step=int(checkpoint.step),
                eligible=reason is None,
                rejection_reason=reason,
                policy_score=score,
                score_components=(
                    {"recency_step": float(checkpoint.step)}
                    if reason is None
                    else {}
                ),
            )
            evaluated.append(candidate)
            if reason is None:
                eligible.append((int(checkpoint.step), order, checkpoint))

        ranked = sorted(eligible, key=lambda item: (item[0], item[1]), reverse=True)
        ranks = {
            item[2].checkpoint_id: rank for rank, item in enumerate(ranked, start=1)
        }
        evaluated = [
            dataclasses.replace(
                candidate, rank=ranks.get(candidate.checkpoint_id)
            )
            for candidate in evaluated
        ]
        selected = ranked[0][2] if ranked else None
        return RollbackSelection(
            failure_step=int(decision.step),
            warning_onset_step=decision.warning_onset_step,
            safety_margin_steps=int(safety_margin_steps),
            strategy=strategy,
            selection_basis=basis,
            cutoff_step=int(cutoff),
            candidates=evaluated,
            selected_checkpoint_id=(selected.checkpoint_id if selected else None),
            selected_step=(int(selected.step) if selected else None),
            policy_name=self.name,
            ranked_checkpoint_ids=tuple(item[2].checkpoint_id for item in ranked),
        )


def _default_reason_anchors() -> Dict[str, Tuple[str, ...]]:
    return {
        "missed_grasp": ("approach_object",),
        "grasp_timeout": ("approach_object",),
        "grasp_lost_before_lift": ("approach_object",),
        "lift_no_progress": ("grasp_completed", "approach_object"),
        "lift_timeout": ("grasp_completed", "approach_object"),
        "object_dropped": ("object_lifted", "grasp_completed"),
        "object_fell_below_initial_height": (
            "grasp_completed",
            "approach_object",
        ),
        "transport_or_place_stalled": (
            "object_lifted",
            "grasp_completed",
        ),
        "placement_timeout": ("object_lifted", "grasp_completed"),
        "released_outside_goal": ("place_completed", "object_lifted"),
        "release_timeout": ("place_completed", "object_lifted"),
    }


@dataclasses.dataclass(frozen=True)
class StageAwareRollbackConfig:
    """Explainable pick-place failure-to-anchor mapping and score weights."""

    reason_anchors: Mapping[str, Tuple[str, ...]] = dataclasses.field(
        default_factory=_default_reason_anchors
    )
    use_stage_fallback: bool = True
    semantic_score_weight: float = 0.05
    event_confidence_weight: float = 1.0
    completed_event_weight: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "semantic_score_weight",
            "event_confidence_weight",
            "completed_event_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


_STAGE_PREDECESSOR_ANCHORS: Mapping[str, Tuple[str, ...]] = {
    "grasp_completed": ("approach_object",),
    "object_lifted": ("grasp_completed", "approach_object"),
    "place_completed": ("object_lifted", "grasp_completed"),
    "release_completed": ("place_completed", "object_lifted"),
    "task_complete": ("place_completed", "object_lifted"),
}


class StageAwareRollbackPolicy(RollbackPolicy):
    """Prefer failure-specific semantic anchors, then use temporal fallback."""

    name = "stage_aware"

    def __init__(
        self,
        config: Optional[StageAwareRollbackConfig] = None,
        temporal_fallback: Optional[TemporalRollbackPolicy] = None,
    ) -> None:
        self.config = config or StageAwareRollbackConfig()
        self.temporal_fallback = temporal_fallback or TemporalRollbackPolicy()

    def _requested_anchors(self, decision: FailureDecision) -> Tuple[str, ...]:
        reason = decision.reason or ""
        configured = self.config.reason_anchors.get(reason)
        if configured:
            return tuple(configured)
        if self.config.use_stage_fallback:
            return tuple(
                _STAGE_PREDECESSOR_ANCHORS.get(decision.current_stage, ())
            )
        return ()

    def _fallback(
        self,
        decision: FailureDecision,
        checkpoints: Sequence[RollbackCheckpoint],
        safety_margin_steps: int,
        requested_anchors: Tuple[str, ...],
        reason: str,
    ) -> RollbackSelection:
        selection = self.temporal_fallback.select(
            decision,
            checkpoints,
            safety_margin_steps=safety_margin_steps,
        )
        selection.policy_name = self.name
        selection.requested_anchors = requested_anchors
        selection.fallback_used = True
        selection.fallback_reason = reason
        return selection

    def select(
        self,
        decision: FailureDecision,
        checkpoints: Sequence[RollbackCheckpoint],
        *,
        safety_margin_steps: int,
    ) -> RollbackSelection:
        requested_anchors = self._requested_anchors(decision)
        if not requested_anchors:
            return self._fallback(
                decision,
                checkpoints,
                safety_margin_steps,
                requested_anchors,
                "no_stage_anchor_configured",
            )

        temporal = self.temporal_fallback.select(
            decision,
            checkpoints,
            safety_margin_steps=safety_margin_steps,
        )
        hard_eligible_ids = {
            candidate.checkpoint_id
            for candidate in temporal.candidates
            if candidate.eligible
        }
        selected_anchor = None
        anchor_matches: List[Tuple[float, int, RollbackCheckpoint]] = []
        score_by_id: Dict[str, Tuple[float, Dict[str, float]]] = {}

        for requested_anchor in requested_anchors:
            current_matches = []
            for order, checkpoint in enumerate(checkpoints):
                if checkpoint.checkpoint_id not in hard_eligible_ids:
                    continue
                if requested_anchor not in _event_anchors(checkpoint):
                    continue
                components = {
                    "recency_step": float(checkpoint.step),
                    "semantic_score": float(checkpoint.semantic_score),
                    "event_confidence": _semantic_confidence(checkpoint),
                    "completed_event_count": _completed_event_count(checkpoint),
                }
                score = (
                    components["recency_step"]
                    + self.config.semantic_score_weight
                    * components["semantic_score"]
                    + self.config.event_confidence_weight
                    * components["event_confidence"]
                    + self.config.completed_event_weight
                    * components["completed_event_count"]
                )
                current_matches.append((score, order, checkpoint))
                score_by_id[checkpoint.checkpoint_id] = (score, components)
            if current_matches:
                selected_anchor = requested_anchor
                anchor_matches = current_matches
                break

        if not anchor_matches:
            return self._fallback(
                decision,
                checkpoints,
                safety_margin_steps,
                requested_anchors,
                "no_matching_anchor_before_cutoff",
            )

        ranked = sorted(
            anchor_matches, key=lambda item: (item[0], item[1]), reverse=True
        )
        ranks = {
            item[2].checkpoint_id: rank for rank, item in enumerate(ranked, start=1)
        }
        temporal_by_id = {
            candidate.checkpoint_id: candidate for candidate in temporal.candidates
        }
        evaluated = []
        for checkpoint in checkpoints:
            base = temporal_by_id[checkpoint.checkpoint_id]
            score_data = score_by_id.get(checkpoint.checkpoint_id)
            checkpoint_anchors = _event_anchors(checkpoint)
            matched_anchor = next(
                (
                    anchor
                    for anchor in requested_anchors
                    if anchor in checkpoint_anchors
                ),
                None,
            )
            if not base.eligible:
                eligible = False
                rejection_reason = base.rejection_reason
            elif matched_anchor == selected_anchor:
                eligible = True
                rejection_reason = None
            elif matched_anchor is not None:
                eligible = False
                rejection_reason = "lower_priority_anchor"
            else:
                eligible = False
                rejection_reason = "no_requested_anchor"
            evaluated.append(
                dataclasses.replace(
                    base,
                    eligible=eligible,
                    rejection_reason=rejection_reason,
                    matched_anchor=matched_anchor,
                    policy_score=(score_data[0] if score_data else None),
                    score_components=(score_data[1] if score_data else {}),
                    rank=ranks.get(checkpoint.checkpoint_id),
                )
            )

        selected = ranked[0][2]
        return RollbackSelection(
            failure_step=int(decision.step),
            warning_onset_step=decision.warning_onset_step,
            safety_margin_steps=int(safety_margin_steps),
            strategy="stage_aware",
            selection_basis=f"failure_anchor:{selected_anchor}",
            cutoff_step=temporal.cutoff_step,
            candidates=evaluated,
            selected_checkpoint_id=selected.checkpoint_id,
            selected_step=int(selected.step),
            policy_name=self.name,
            requested_anchors=requested_anchors,
            selected_anchor=selected_anchor,
            fallback_used=False,
            ranked_checkpoint_ids=tuple(
                item[2].checkpoint_id for item in ranked
            ),
        )


__all__ = [
    "RollbackCandidate",
    "RollbackCheckpoint",
    "RollbackPolicy",
    "RollbackSelection",
    "StageAwareRollbackConfig",
    "StageAwareRollbackPolicy",
    "TemporalRollbackPolicy",
]
