#!/usr/bin/env python3
"""Pick-and-place sub-events and risk-aware checkpoint selection.

Humans define the event vocabulary, ordering, object bindings, and edge risk
priors.  The detector evaluates those definitions from LIBERO observations;
the checkpoint manager then makes event checkpoints mandatory and gates the
existing heuristic selector according to the active event edge.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from libero_rollback_tool import EnvRollbackTool, get_observation
from semantic_checkpoint import (
    SemanticKeyframe,
    SemanticKeyframeConfig,
    SemanticKeyframeSelector,
)


PICK_PLACE_EVENTS: Tuple[str, ...] = (
    "approach_object",
    "grasp_completed",
    "object_lifted",
    "place_completed",
    "release_completed",
)


@dataclasses.dataclass(frozen=True)
class PickPlaceTaskSpec:
    """Manual semantic definition for one pick-and-place task instance."""

    target_object: str
    goal_object: str
    approach_distance: float = 0.08
    grasp_distance: float = 0.06
    gripper_closed_aperture: float = 0.018
    gripper_open_aperture: float = 0.032
    lift_height: float = 0.04
    place_distance: float = 0.10
    place_height_tolerance: float = 0.08
    stable_object_motion: float = 0.008
    confirmation_steps: int = 2

    def __post_init__(self) -> None:
        if not self.target_object or not self.goal_object:
            raise ValueError("target_object and goal_object are required")
        if self.confirmation_steps <= 0:
            raise ValueError("confirmation_steps must be positive")
        for name in (
            "approach_distance",
            "grasp_distance",
            "gripper_closed_aperture",
            "gripper_open_aperture",
            "lift_height",
            "place_distance",
            "place_height_tolerance",
            "stable_object_motion",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class EventEdgePolicy:
    """Checkpoint policy for the interval leading to one sub-event."""

    from_event: Optional[str]
    to_event: str
    risk_coefficient: float
    heuristic_enabled: bool
    min_interval: int
    max_interval: int

    @property
    def name(self) -> str:
        return f"{self.from_event or 'episode_start'}->{self.to_event}"

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_coefficient <= 1.0:
            raise ValueError("risk_coefficient must be within [0, 1]")
        if self.min_interval < 0 or self.max_interval <= 0:
            raise ValueError("event edge intervals must be non-negative/positive")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")


def default_pick_place_edge_policies() -> Tuple[EventEdgePolicy, ...]:
    """Risk prior used before a learned failure predictor is available."""

    return (
        # Free-space approach is long but usually recoverable: keep it sparse.
        EventEdgePolicy(None, "approach_object", 0.15, False, 8, 30),
        # Closing the gripper is contact-rich and failure-sensitive.
        EventEdgePolicy("approach_object", "grasp_completed", 1.00, True, 2, 6),
        # Early lifting can reveal an unstable or missed grasp.
        EventEdgePolicy("grasp_completed", "object_lifted", 0.90, True, 2, 7),
        # Transport is normally safer; retain moderate coverage.
        EventEdgePolicy("object_lifted", "place_completed", 0.45, True, 5, 15),
        # Placement and release are precise contact transitions.
        EventEdgePolicy("place_completed", "release_completed", 1.00, True, 2, 5),
    )


def load_pick_place_config(
    path: Path,
) -> Tuple[PickPlaceTaskSpec, Tuple[EventEdgePolicy, ...]]:
    """Load a human-authored task binding and edge-risk policy from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("task"), dict):
        raise ValueError("pick-place config must contain a 'task' object")
    task_spec = PickPlaceTaskSpec(**data["task"])
    raw_policies = data.get("edge_policies")
    if raw_policies is None:
        policies = default_pick_place_edge_policies()
    else:
        if not isinstance(raw_policies, list):
            raise ValueError("edge_policies must be a list")
        policies = tuple(EventEdgePolicy(**item) for item in raw_policies)
    if tuple(policy.to_event for policy in policies) != PICK_PLACE_EVENTS:
        raise ValueError("edge_policies must follow PICK_PLACE_EVENTS order")
    return task_spec, policies


@dataclasses.dataclass(frozen=True)
class PickPlaceFeatures:
    eef_target_distance: float
    target_goal_distance: float
    target_height: float
    goal_height: float
    lifted_height: float
    gripper_aperture: float
    object_motion: float

    def as_dict(self) -> Dict[str, float]:
        return {
            field.name: float(getattr(self, field.name))
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class SubEventTransition:
    event_name: str
    event_index: int
    step: int
    confidence: float
    features: Dict[str, float]
    source: str

    def to_summary(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _vector(obs: Mapping[str, Any], key: str, size: int) -> Optional[np.ndarray]:
    if key not in obs:
        return None
    try:
        value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if value.size < size or not np.all(np.isfinite(value[:size])):
        return None
    return value[:size].copy()


def _event_override(info: Optional[Mapping[str, Any]], event_name: str) -> bool:
    if not isinstance(info, Mapping):
        return False
    for key in (event_name, f"{event_name}_completed", f"subevent/{event_name}"):
        if key in info and bool(info[key]):
            return True
    return False


class PickPlaceSubEventDetector:
    """Monotonic state machine for manually defined pick-and-place events."""

    def __init__(self, spec: PickPlaceTaskSpec) -> None:
        self.spec = spec
        self.completed_events: List[str] = []
        self.transitions: List[SubEventTransition] = []
        self._confirmation_count = 0
        self._initial_target_height: Optional[float] = None
        self._previous_target_pos: Optional[np.ndarray] = None
        self._last_features: Optional[PickPlaceFeatures] = None

    @property
    def current_event_index(self) -> int:
        return len(self.completed_events)

    @property
    def next_event(self) -> Optional[str]:
        if self.current_event_index >= len(PICK_PLACE_EVENTS):
            return None
        return PICK_PLACE_EVENTS[self.current_event_index]

    @property
    def last_completed_event(self) -> Optional[str]:
        return self.completed_events[-1] if self.completed_events else None

    def reset(self) -> None:
        self.completed_events.clear()
        self.transitions.clear()
        self._confirmation_count = 0
        self._initial_target_height = None
        self._previous_target_pos = None
        self._last_features = None

    def restore_progress(
        self,
        completed_event_count: int,
        obs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Restore event progress recorded in checkpoint metadata."""

        count = max(0, min(int(completed_event_count), len(PICK_PLACE_EVENTS)))
        self.completed_events = list(PICK_PLACE_EVENTS[:count])
        self.transitions.clear()
        self._confirmation_count = 0
        self._previous_target_pos = None
        self._last_features = None
        if isinstance(obs, Mapping):
            target = _vector(obs, f"{self.spec.target_object}_pos", 3)
            if target is not None:
                self._previous_target_pos = target
                if self._initial_target_height is None:
                    self._initial_target_height = float(target[2])

    def extract_features(self, obs: Mapping[str, Any]) -> PickPlaceFeatures:
        target = _vector(obs, f"{self.spec.target_object}_pos", 3)
        goal = _vector(obs, f"{self.spec.goal_object}_pos", 3)
        eef = _vector(obs, "robot0_eef_pos", 3)
        gripper = _vector(obs, "robot0_gripper_qpos", 2)
        if target is None:
            raise KeyError(f"Missing observation {self.spec.target_object!r}_pos")
        if goal is None:
            raise KeyError(f"Missing observation {self.spec.goal_object!r}_pos")
        if eef is None or gripper is None:
            raise KeyError("Missing robot0_eef_pos or robot0_gripper_qpos")

        relative = _vector(
            obs, f"{self.spec.target_object}_to_robot0_eef_pos", 3
        )
        eef_target_distance = float(
            np.linalg.norm(relative) if relative is not None else np.linalg.norm(eef - target)
        )
        object_motion = (
            float(np.linalg.norm(target - self._previous_target_pos))
            if self._previous_target_pos is not None
            else 0.0
        )
        if self._initial_target_height is None:
            self._initial_target_height = float(target[2])
        lifted_height = float(target[2] - self._initial_target_height)
        # Panda fingers use opposite signs, so their separation is the useful
        # aperture signal rather than either joint in isolation.
        gripper_aperture = float(abs(gripper[0] - gripper[1]))
        return PickPlaceFeatures(
            eef_target_distance=eef_target_distance,
            target_goal_distance=float(np.linalg.norm(target - goal)),
            target_height=float(target[2]),
            goal_height=float(goal[2]),
            lifted_height=lifted_height,
            gripper_aperture=gripper_aperture,
            object_motion=object_motion,
        )

    def _predicate(self, event_name: str, features: PickPlaceFeatures) -> Tuple[bool, float]:
        spec = self.spec
        if event_name == "approach_object":
            ratio = features.eef_target_distance / max(spec.approach_distance, 1e-12)
            return ratio <= 1.0, float(np.clip(1.0 - ratio, 0.0, 1.0))
        if event_name == "grasp_completed":
            distance_ratio = features.eef_target_distance / max(spec.grasp_distance, 1e-12)
            aperture_ratio = features.gripper_aperture / max(
                spec.gripper_closed_aperture, 1e-12
            )
            passed = distance_ratio <= 1.0 and aperture_ratio <= 1.0
            confidence = 1.0 - max(distance_ratio, aperture_ratio)
            return passed, float(np.clip(confidence, 0.0, 1.0))
        if event_name == "object_lifted":
            ratio = features.lifted_height / max(spec.lift_height, 1e-12)
            return ratio >= 1.0, float(np.clip(ratio, 0.0, 1.0))
        if event_name == "place_completed":
            distance_ratio = features.target_goal_distance / max(spec.place_distance, 1e-12)
            height_error = abs(features.target_height - features.goal_height)
            height_ratio = height_error / max(spec.place_height_tolerance, 1e-12)
            stable = features.object_motion <= spec.stable_object_motion
            passed = distance_ratio <= 1.0 and height_ratio <= 1.0 and stable
            confidence = 1.0 - max(distance_ratio, height_ratio)
            return passed, float(np.clip(confidence, 0.0, 1.0))
        if event_name == "release_completed":
            distance_ratio = features.target_goal_distance / max(spec.place_distance, 1e-12)
            aperture_ratio = features.gripper_aperture / max(
                spec.gripper_open_aperture, 1e-12
            )
            stable = features.object_motion <= spec.stable_object_motion
            passed = distance_ratio <= 1.0 and aperture_ratio >= 1.0 and stable
            confidence = min(1.0 - distance_ratio, aperture_ratio - 1.0)
            return passed, float(np.clip(confidence, 0.0, 1.0))
        raise KeyError(f"Unknown pick-and-place event: {event_name}")

    def update(
        self,
        obs: Mapping[str, Any],
        *,
        step: int,
        info: Optional[Mapping[str, Any]] = None,
    ) -> Optional[SubEventTransition]:
        """Evaluate the next event and emit at most one ordered transition."""

        features = self.extract_features(obs)
        event_name = self.next_event
        if event_name is None:
            self._previous_target_pos = _vector(
                obs, f"{self.spec.target_object}_pos", 3
            )
            self._last_features = features
            return None

        overridden = _event_override(info, event_name)
        passed, confidence = self._predicate(event_name, features)
        if overridden:
            passed, confidence = True, 1.0
        self._confirmation_count = self._confirmation_count + 1 if passed else 0
        transition = None
        if overridden or self._confirmation_count >= self.spec.confirmation_steps:
            self.completed_events.append(event_name)
            transition = SubEventTransition(
                event_name=event_name,
                event_index=len(self.completed_events) - 1,
                step=int(step),
                confidence=float(confidence),
                features=features.as_dict(),
                source="explicit_info" if overridden else "observation_predicate",
            )
            self.transitions.append(transition)
            self._confirmation_count = 0

        self._previous_target_pos = _vector(
            obs, f"{self.spec.target_object}_pos", 3
        )
        self._last_features = features
        return transition

    def summary(self) -> Dict[str, Any]:
        return {
            "task_spec": dataclasses.asdict(self.spec),
            "completed_events": list(self.completed_events),
            "next_event": self.next_event,
            "transitions": [transition.to_summary() for transition in self.transitions],
            "last_features": (
                self._last_features.as_dict() if self._last_features is not None else None
            ),
        }


@dataclasses.dataclass
class RiskAwareObservation:
    step: int
    edge_policy: EventEdgePolicy
    transition: Optional[SubEventTransition]
    keyframe: Optional[SemanticKeyframe]


class PickPlaceRiskAwareCheckpointManager:
    """Combine mandatory sub-events with edge-specific heuristic density."""

    def __init__(
        self,
        task_spec: PickPlaceTaskSpec,
        *,
        tool: Optional[EnvRollbackTool] = None,
        selector_config: Optional[SemanticKeyframeConfig] = None,
        edge_policies: Optional[Sequence[EventEdgePolicy]] = None,
    ) -> None:
        self.detector = PickPlaceSubEventDetector(task_spec)
        self.selector = SemanticKeyframeSelector(
            tool=tool,
            config=selector_config or SemanticKeyframeConfig(),
        )
        self.base_selector_config = self.selector.config
        policies = tuple(edge_policies or default_pick_place_edge_policies())
        if tuple(policy.to_event for policy in policies) != PICK_PLACE_EVENTS:
            raise ValueError("edge policies must follow PICK_PLACE_EVENTS order")
        self.edge_policies = policies

    @property
    def current_edge_policy(self) -> EventEdgePolicy:
        index = min(self.detector.current_event_index, len(self.edge_policies) - 1)
        return self.edge_policies[index]

    def _effective_selector_config(
        self, policy: EventEdgePolicy
    ) -> SemanticKeyframeConfig:
        risk = max(policy.risk_coefficient, 1e-6)
        if policy.heuristic_enabled:
            score_threshold = self.base_selector_config.score_threshold / risk
            force_score = self.base_selector_config.force_score / risk
        else:
            score_threshold = float("inf")
            force_score = float("inf")
        return dataclasses.replace(
            self.base_selector_config,
            min_interval=policy.min_interval,
            max_interval=policy.max_interval,
            score_threshold=score_threshold,
            force_score=force_score,
        )

    def observe(
        self,
        env: Any,
        step: int,
        obs: Mapping[str, Any],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
        *,
        action: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        rollout_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RiskAwareObservation:
        """Update the event graph and maybe capture one checkpoint."""

        policy_before_transition = self.current_edge_policy
        transition = self.detector.update(obs, step=step, info=info)
        common_metadata: Dict[str, Any] = dict(metadata or {})
        common_metadata.update(
            {
                "task_type": "pick_and_place",
                "target_object": self.detector.spec.target_object,
                "goal_object": self.detector.spec.goal_object,
                "event_edge": policy_before_transition.name,
                "edge_risk_coefficient": policy_before_transition.risk_coefficient,
                "edge_heuristic_enabled": policy_before_transition.heuristic_enabled,
                "completed_event_count": self.detector.current_event_index,
                "last_completed_event": self.detector.last_completed_event,
                "next_event": self.detector.next_event,
            }
        )

        if transition is not None:
            decision = self.selector.observe_only(
                env,
                step,
                obs,
                reward,
                done,
                info,
            )
            features = dict(decision.features)
            features.update(transition.features)
            features["edge_risk_coefficient"] = policy_before_transition.risk_coefficient
            common_metadata.update(
                {
                    "checkpoint_source": "subevent_completion",
                    "subevent_transition": transition.to_summary(),
                }
            )
            keyframe = self.selector.force_capture(
                env,
                step,
                obs,
                labels=("subevent_completed", transition.event_name),
                score=max(decision.score, 5.0 * policy_before_transition.risk_coefficient),
                features=features,
                reward=reward,
                done=done,
                action=action,
                rng=rng,
                rollout_state=rollout_state,
                metadata=common_metadata,
            )
            return RiskAwareObservation(
                step=int(step),
                edge_policy=policy_before_transition,
                transition=transition,
                keyframe=keyframe,
            )

        effective_config = self._effective_selector_config(policy_before_transition)
        previous_config = self.selector.config
        self.selector.config = effective_config
        try:
            common_metadata["checkpoint_source"] = (
                "risk_heuristic"
                if policy_before_transition.heuristic_enabled
                else "sparse_max_interval"
            )
            keyframe = self.selector.maybe_capture(
                env,
                step,
                obs,
                reward,
                done,
                info,
                action=action,
                rng=rng,
                rollout_state=rollout_state,
                metadata=common_metadata,
            )
        finally:
            self.selector.config = previous_config
        return RiskAwareObservation(
            step=int(step),
            edge_policy=policy_before_transition,
            transition=None,
            keyframe=keyframe,
        )

    def restore_for_failure(
        self,
        env: Any,
        failure_step: int,
        *,
        strategy: str = "latest_safe",
        reload_xml: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[SemanticKeyframe, Dict[str, Any]]:
        keyframe, report = self.selector.restore_for_failure(
            env,
            failure_step,
            strategy=strategy,
            reload_xml=reload_xml,
            rng=rng,
        )
        restored_obs = get_observation(env, force_update=False)
        completed_count = int(
            keyframe.checkpoint.metadata.get("completed_event_count", 0)
        )
        self.detector.restore_progress(completed_count, restored_obs)
        self.selector.reset_history(restored_obs)
        return keyframe, report

    def summary(self) -> Dict[str, Any]:
        return {
            "event_detector": self.detector.summary(),
            "edge_policies": [dataclasses.asdict(policy) for policy in self.edge_policies],
            "checkpoint_selector": self.selector.summary(),
        }


__all__ = [
    "EventEdgePolicy",
    "PICK_PLACE_EVENTS",
    "PickPlaceFeatures",
    "PickPlaceRiskAwareCheckpointManager",
    "PickPlaceSubEventDetector",
    "PickPlaceTaskSpec",
    "RiskAwareObservation",
    "SubEventTransition",
    "default_pick_place_edge_policies",
    "load_pick_place_config",
]
