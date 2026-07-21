#!/usr/bin/env python3
"""Stage-aware rule failure detection for pick-and-place rollouts.

This module is intentionally detection-only.  It does not capture or select
checkpoints, restore simulator state, replay actions, or call a policy.  Each
update returns an explainable risk decision that an outer coordinator may use
later.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pick_place_checkpoint import PICK_PLACE_EVENTS, PickPlaceTaskSpec


class FailureStatus(str, enum.Enum):
    NORMAL = "normal"
    WARNING = "warning"
    RECOVERABLE_FAILURE = "recoverable_failure"
    TERMINAL_FAILURE = "terminal_failure"


@dataclasses.dataclass(frozen=True)
class FailureDecision:
    step: int
    status: FailureStatus
    risk_score: float
    reason: Optional[str]
    current_stage: str
    consecutive_steps: int
    hard_failure: bool
    evidence: Dict[str, Any]

    @property
    def failed(self) -> bool:
        return self.status in {
            FailureStatus.RECOVERABLE_FAILURE,
            FailureStatus.TERMINAL_FAILURE,
        }

    def to_summary(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["status"] = self.status.value
        return result


@dataclasses.dataclass(frozen=True)
class PickPlaceFailureConfig:
    """Thresholds for the first explainable rule baseline."""

    history_window: int = 12
    confirmation_steps: int = 3
    warning_threshold: float = 0.45
    failure_threshold: float = 0.75
    min_approach_progress: float = 0.006
    min_goal_progress: float = 0.008
    min_lift_progress: float = 0.015
    approach_timeout: int = 100
    grasp_timeout: int = 25
    lift_timeout: int = 30
    place_timeout: int = 120
    release_timeout: int = 25
    missed_grasp_distance_factor: float = 1.5
    lost_grasp_distance_factor: float = 2.5
    drop_height: float = 0.035
    below_initial_height_margin: float = 0.035
    wrong_place_distance_factor: float = 1.5
    workspace_abs_xy: float = 2.0
    workspace_min_z: float = 0.70
    workspace_max_z: float = 2.0

    def __post_init__(self) -> None:
        if self.history_window < 2:
            raise ValueError("history_window must be >= 2")
        if self.confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive")
        if not 0.0 <= self.warning_threshold <= self.failure_threshold <= 1.0:
            raise ValueError("risk thresholds must satisfy 0 <= warning <= failure <= 1")
        for name in (
            "min_approach_progress",
            "min_goal_progress",
            "min_lift_progress",
            "missed_grasp_distance_factor",
            "lost_grasp_distance_factor",
            "drop_height",
            "below_initial_height_margin",
            "wrong_place_distance_factor",
            "workspace_abs_xy",
            "workspace_min_z",
            "workspace_max_z",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "approach_timeout",
            "grasp_timeout",
            "lift_timeout",
            "place_timeout",
            "release_timeout",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclasses.dataclass(frozen=True)
class FailureFrame:
    step: int
    stage: str
    eef_target_distance: float
    target_goal_distance: float
    target_height: float
    gripper_aperture: float
    target_pos: np.ndarray
    eef_pos: np.ndarray
    action_norm: float
    reward: float


@dataclasses.dataclass(frozen=True)
class _FailureCandidate:
    reason: str
    risk: float
    hard: bool
    terminal: bool
    evidence: Dict[str, Any]


def _finite_vector(
    obs: Mapping[str, Any], key: str, size: int
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if key not in obs:
        return None, f"missing:{key}"
    try:
        value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    except Exception as exc:
        return None, f"invalid:{key}:{exc!r}"
    if value.size < size:
        return None, f"shape:{key}:{value.shape}"
    value = value[:size]
    if not np.all(np.isfinite(value)):
        return None, f"non_finite:{key}"
    return value.copy(), None


def _info_flag(info: Optional[Mapping[str, Any]], names: Sequence[str]) -> bool:
    if not isinstance(info, Mapping):
        return False
    return any(bool(info.get(name, False)) for name in names)


def stage_from_completed_events(completed_events: Sequence[str]) -> str:
    """Return the event currently being attempted."""

    completed = set(completed_events)
    for event_name in PICK_PLACE_EVENTS:
        if event_name not in completed:
            return event_name
    return "task_complete"


class PickPlaceRuleFailureDetector:
    """Detect stage-specific failures from observations and a short history."""

    def __init__(
        self,
        task_spec: PickPlaceTaskSpec,
        config: Optional[PickPlaceFailureConfig] = None,
    ) -> None:
        self.task_spec = task_spec
        self.config = config or PickPlaceFailureConfig()
        self.history: Deque[FailureFrame] = collections.deque(
            maxlen=self.config.history_window
        )
        self.decisions: List[FailureDecision] = []
        self._initial_target_height: Optional[float] = None
        self._maximum_target_height: Optional[float] = None
        self._current_stage: Optional[str] = None
        self._stage_start_step = 0
        self._active_reason: Optional[str] = None
        self._active_reason_steps = 0

    def reset(self) -> None:
        self.history.clear()
        self.decisions.clear()
        self._initial_target_height = None
        self._maximum_target_height = None
        self._current_stage = None
        self._stage_start_step = 0
        self._active_reason = None
        self._active_reason_steps = 0

    def _terminal_decision(
        self,
        *,
        step: int,
        stage: str,
        reason: str,
        evidence: Dict[str, Any],
    ) -> FailureDecision:
        if self._active_reason == reason:
            self._active_reason_steps += 1
        else:
            self._active_reason = reason
            self._active_reason_steps = 1
        return FailureDecision(
            step=int(step),
            status=FailureStatus.TERMINAL_FAILURE,
            risk_score=1.0,
            reason=reason,
            current_stage=stage,
            consecutive_steps=self._active_reason_steps,
            hard_failure=True,
            evidence=evidence,
        )

    def _extract_frame(
        self,
        obs: Mapping[str, Any],
        *,
        step: int,
        stage: str,
        action: Optional[np.ndarray],
        reward: float,
    ) -> Tuple[Optional[FailureFrame], Optional[str]]:
        target, error = _finite_vector(
            obs, f"{self.task_spec.target_object}_pos", 3
        )
        if error:
            return None, error
        goal, error = _finite_vector(obs, f"{self.task_spec.goal_object}_pos", 3)
        if error:
            return None, error
        eef, error = _finite_vector(obs, "robot0_eef_pos", 3)
        if error:
            return None, error
        gripper, error = _finite_vector(obs, "robot0_gripper_qpos", 2)
        if error:
            return None, error

        relative, _ = _finite_vector(
            obs, f"{self.task_spec.target_object}_to_robot0_eef_pos", 3
        )
        eef_target_distance = float(
            np.linalg.norm(relative)
            if relative is not None
            else np.linalg.norm(eef - target)
        )
        action_norm = 0.0
        if action is not None:
            try:
                action_array = np.asarray(action, dtype=np.float64).reshape(-1)
                if not np.all(np.isfinite(action_array)):
                    return None, "non_finite:action"
                action_norm = float(np.linalg.norm(action_array))
            except Exception as exc:
                return None, f"invalid:action:{exc!r}"
        return (
            FailureFrame(
                step=int(step),
                stage=stage,
                eef_target_distance=eef_target_distance,
                target_goal_distance=float(np.linalg.norm(target - goal)),
                target_height=float(target[2]),
                gripper_aperture=float(abs(gripper[0] - gripper[1])),
                target_pos=target,
                eef_pos=eef,
                action_norm=action_norm,
                reward=float(reward),
            ),
            None,
        )

    def _history_progress(
        self, field: str, current: FailureFrame
    ) -> Optional[float]:
        same_stage = [frame for frame in self.history if frame.stage == current.stage]
        if len(same_stage) < self.config.history_window - 1:
            return None
        oldest = same_stage[0]
        # Distance progress is positive when the distance becomes smaller.
        return float(getattr(oldest, field) - getattr(current, field))

    def _stage_candidates(
        self,
        frame: FailureFrame,
        completed_events: Sequence[str],
    ) -> List[_FailureCandidate]:
        config = self.config
        spec = self.task_spec
        candidates: List[_FailureCandidate] = []
        stage_steps = frame.step - self._stage_start_step

        if frame.stage == "approach_object":
            progress = self._history_progress("eef_target_distance", frame)
            if progress is not None and progress < config.min_approach_progress:
                deficit = config.min_approach_progress - progress
                candidates.append(
                    _FailureCandidate(
                        "approach_stalled",
                        min(0.9, 0.55 + deficit / max(config.min_approach_progress, 1e-12)),
                        False,
                        False,
                        {
                            "distance_progress": progress,
                            "required_progress": config.min_approach_progress,
                            "window": config.history_window,
                        },
                    )
                )
            if stage_steps >= config.approach_timeout:
                candidates.append(
                    _FailureCandidate(
                        "approach_timeout",
                        1.0,
                        True,
                        False,
                        {"stage_steps": stage_steps, "timeout": config.approach_timeout},
                    )
                )

        elif frame.stage == "grasp_completed":
            closed = frame.gripper_aperture <= spec.gripper_closed_aperture
            too_far = (
                frame.eef_target_distance
                > spec.grasp_distance * config.missed_grasp_distance_factor
            )
            if closed and too_far:
                candidates.append(
                    _FailureCandidate(
                        "missed_grasp",
                        0.92,
                        False,
                        False,
                        {
                            "gripper_aperture": frame.gripper_aperture,
                            "eef_target_distance": frame.eef_target_distance,
                            "maximum_grasp_distance": (
                                spec.grasp_distance
                                * config.missed_grasp_distance_factor
                            ),
                        },
                    )
                )
            if stage_steps >= config.grasp_timeout:
                candidates.append(
                    _FailureCandidate(
                        "grasp_timeout",
                        1.0,
                        True,
                        False,
                        {"stage_steps": stage_steps, "timeout": config.grasp_timeout},
                    )
                )

        elif frame.stage == "object_lifted":
            height_progress = (
                frame.target_height - self._initial_target_height
                if self._initial_target_height is not None
                else 0.0
            )
            lost_distance = (
                frame.eef_target_distance
                > spec.grasp_distance * config.lost_grasp_distance_factor
            )
            if lost_distance:
                candidates.append(
                    _FailureCandidate(
                        "grasp_lost_before_lift",
                        0.9,
                        False,
                        False,
                        {
                            "eef_target_distance": frame.eef_target_distance,
                            "maximum_attached_distance": (
                                spec.grasp_distance * config.lost_grasp_distance_factor
                            ),
                        },
                    )
                )
            if stage_steps >= config.history_window and height_progress < config.min_lift_progress:
                candidates.append(
                    _FailureCandidate(
                        "lift_no_progress",
                        0.88,
                        False,
                        False,
                        {
                            "height_progress": height_progress,
                            "required_progress": config.min_lift_progress,
                            "stage_steps": stage_steps,
                        },
                    )
                )
            if stage_steps >= config.lift_timeout:
                candidates.append(
                    _FailureCandidate(
                        "lift_timeout",
                        1.0,
                        True,
                        False,
                        {"stage_steps": stage_steps, "timeout": config.lift_timeout},
                    )
                )

        elif frame.stage == "place_completed":
            goal_progress = self._history_progress("target_goal_distance", frame)
            if goal_progress is not None and goal_progress < config.min_goal_progress:
                deficit = config.min_goal_progress - goal_progress
                candidates.append(
                    _FailureCandidate(
                        "transport_or_place_stalled",
                        min(0.9, 0.5 + deficit / max(config.min_goal_progress, 1e-12)),
                        False,
                        False,
                        {
                            "goal_progress": goal_progress,
                            "required_progress": config.min_goal_progress,
                            "window": config.history_window,
                        },
                    )
                )
            if stage_steps >= config.place_timeout:
                candidates.append(
                    _FailureCandidate(
                        "placement_timeout",
                        1.0,
                        True,
                        False,
                        {"stage_steps": stage_steps, "timeout": config.place_timeout},
                    )
                )

        elif frame.stage == "release_completed":
            gripper_open = frame.gripper_aperture >= spec.gripper_open_aperture
            outside_goal = (
                frame.target_goal_distance
                > spec.place_distance * config.wrong_place_distance_factor
            )
            if gripper_open and outside_goal:
                candidates.append(
                    _FailureCandidate(
                        "released_outside_goal",
                        0.95,
                        False,
                        False,
                        {
                            "target_goal_distance": frame.target_goal_distance,
                            "maximum_release_distance": (
                                spec.place_distance * config.wrong_place_distance_factor
                            ),
                        },
                    )
                )
            if stage_steps >= config.release_timeout:
                candidates.append(
                    _FailureCandidate(
                        "release_timeout",
                        1.0,
                        True,
                        False,
                        {"stage_steps": stage_steps, "timeout": config.release_timeout},
                    )
                )

        if "object_lifted" in set(completed_events):
            maximum_height = self._maximum_target_height or frame.target_height
            height_drop = maximum_height - frame.target_height
            detached = (
                frame.eef_target_distance
                > spec.grasp_distance * config.lost_grasp_distance_factor
            )
            safely_at_goal = (
                "place_completed" in set(completed_events)
                and frame.target_goal_distance
                <= spec.place_distance * config.wrong_place_distance_factor
            )
            if height_drop >= config.drop_height and detached and not safely_at_goal:
                candidates.append(
                    _FailureCandidate(
                        "object_dropped",
                        1.0,
                        True,
                        False,
                        {
                            "maximum_target_height": maximum_height,
                            "target_height": frame.target_height,
                            "height_drop": height_drop,
                            "eef_target_distance": frame.eef_target_distance,
                            "detached": detached,
                            "safely_at_goal": safely_at_goal,
                        },
                    )
                )
        return candidates

    def _global_candidates(self, frame: FailureFrame) -> List[_FailureCandidate]:
        config = self.config
        candidates: List[_FailureCandidate] = []
        x, y, z = frame.target_pos
        if abs(x) > config.workspace_abs_xy or abs(y) > config.workspace_abs_xy:
            candidates.append(
                _FailureCandidate(
                    "object_out_of_workspace",
                    1.0,
                    True,
                    False,
                    {"target_pos": frame.target_pos.tolist()},
                )
            )
        if z < config.workspace_min_z or z > config.workspace_max_z:
            candidates.append(
                _FailureCandidate(
                    "object_height_out_of_workspace",
                    1.0,
                    True,
                    False,
                    {"target_height": float(z)},
                )
            )
        if (
            self._initial_target_height is not None
            and z < self._initial_target_height - config.below_initial_height_margin
        ):
            candidates.append(
                _FailureCandidate(
                    "object_fell_below_initial_height",
                    1.0,
                    True,
                    False,
                    {
                        "initial_target_height": self._initial_target_height,
                        "target_height": float(z),
                    },
                )
            )
        return candidates

    def _finalize_candidate(
        self,
        candidate: Optional[_FailureCandidate],
        *,
        step: int,
        stage: str,
    ) -> FailureDecision:
        if candidate is None:
            self._active_reason = None
            self._active_reason_steps = 0
            return FailureDecision(
                step=int(step),
                status=FailureStatus.NORMAL,
                risk_score=0.0,
                reason=None,
                current_stage=stage,
                consecutive_steps=0,
                hard_failure=False,
                evidence={},
            )

        if candidate.reason == self._active_reason:
            self._active_reason_steps += 1
        else:
            self._active_reason = candidate.reason
            self._active_reason_steps = 1

        risk = float(np.clip(candidate.risk, 0.0, 1.0))
        if candidate.terminal:
            status = FailureStatus.TERMINAL_FAILURE
        elif candidate.hard:
            status = FailureStatus.RECOVERABLE_FAILURE
        elif (
            risk >= self.config.failure_threshold
            and self._active_reason_steps >= self.config.confirmation_steps
        ):
            status = FailureStatus.RECOVERABLE_FAILURE
        elif risk >= self.config.warning_threshold:
            status = FailureStatus.WARNING
        else:
            status = FailureStatus.NORMAL
        return FailureDecision(
            step=int(step),
            status=status,
            risk_score=risk,
            reason=candidate.reason,
            current_stage=stage,
            consecutive_steps=self._active_reason_steps,
            hard_failure=candidate.hard,
            evidence=dict(candidate.evidence),
        )

    def update(
        self,
        obs: Mapping[str, Any],
        *,
        step: int,
        completed_events: Sequence[str] = (),
        action: Optional[np.ndarray] = None,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> FailureDecision:
        """Return a failure decision without mutating any environment state."""

        stage = stage_from_completed_events(completed_events)
        if stage != self._current_stage:
            self._current_stage = stage
            self._stage_start_step = int(step)
            self._active_reason = None
            self._active_reason_steps = 0
            self.history.clear()

        if _info_flag(info, ("sim_error", "simulation_error", "fatal_error")):
            decision = self._terminal_decision(
                step=step,
                stage=stage,
                reason="simulation_error",
                evidence={"info": dict(info or {})},
            )
            self.decisions.append(decision)
            return decision

        explicit_terminal = _info_flag(info, ("terminal_failure",))
        explicit_recoverable = _info_flag(
            info, ("failure", "recoverable_failure", "collision")
        )
        success = _info_flag(info, ("success", "is_success", "task_success"))
        if done and not success:
            explicit_terminal = True

        frame, extraction_error = self._extract_frame(
            obs,
            step=step,
            stage=stage,
            action=action,
            reward=reward,
        )
        if extraction_error is not None or frame is None:
            decision = self._terminal_decision(
                step=step,
                stage=stage,
                reason="invalid_observation",
                evidence={"error": extraction_error},
            )
            self.decisions.append(decision)
            return decision

        if self._initial_target_height is None:
            self._initial_target_height = frame.target_height
        self._maximum_target_height = max(
            frame.target_height,
            self._maximum_target_height
            if self._maximum_target_height is not None
            else frame.target_height,
        )

        candidates = self._global_candidates(frame)
        candidates.extend(self._stage_candidates(frame, completed_events))
        if explicit_terminal:
            candidates.append(
                _FailureCandidate(
                    "episode_terminated_without_success" if done else "explicit_terminal_failure",
                    1.0,
                    True,
                    True,
                    {"done": bool(done), "info": dict(info or {})},
                )
            )
        elif explicit_recoverable:
            candidates.append(
                _FailureCandidate(
                    str((info or {}).get("failure_reason", "explicit_recoverable_failure")),
                    1.0,
                    True,
                    False,
                    {"info": dict(info or {})},
                )
            )

        candidate = (
            max(
                candidates,
                key=lambda item: (item.risk, item.terminal, item.hard),
            )
            if candidates
            else None
        )
        decision = self._finalize_candidate(
            candidate,
            step=step,
            stage=stage,
        )
        self.history.append(frame)
        self.decisions.append(decision)
        return decision

    def summary(self) -> Dict[str, Any]:
        counts = {status.value: 0 for status in FailureStatus}
        for decision in self.decisions:
            counts[decision.status.value] += 1
        return {
            "task_spec": dataclasses.asdict(self.task_spec),
            "config": dataclasses.asdict(self.config),
            "decision_counts": counts,
            "last_decision": (
                self.decisions[-1].to_summary() if self.decisions else None
            ),
        }


__all__ = [
    "FailureDecision",
    "FailureStatus",
    "PickPlaceFailureConfig",
    "PickPlaceRuleFailureDetector",
    "stage_from_completed_events",
]
