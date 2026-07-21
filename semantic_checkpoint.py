#!/usr/bin/env python3
"""Semantic keyframe selection on top of :mod:`libero_rollback_tool`.

The low-level rollback tool answers *how* to restore a simulator.  This module
answers the next question: *which states are worth saving?*  A keyframe is
accepted when one of the following occurs:

* the episode starts, succeeds, or terminates;
* a meaningful motion, gripper, object, reward, or direction-change event is
  detected;
* the maximum spacing since the previous keyframe is reached.

The detector is deliberately observation-driven and task-agnostic.  It uses
common LIBERO names when available and falls back to small numeric observation
vectors, so task-specific learned failure detectors can be added later without
changing the checkpoint format.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from libero_rollback_tool import Checkpoint, EnvRollbackTool, copy_obs


@dataclasses.dataclass(frozen=True)
class SemanticKeyframeConfig:
    """Thresholds controlling semantic keyframe density.

    ``min_interval`` prevents bursts of nearly identical checkpoints.
    ``max_interval`` is a hard upper bound: even a quiet episode gets a
    periodic recovery point.  Thresholds are expressed in observation units
    and can be tuned per benchmark without changing the detector logic.
    """

    min_interval: int = 3
    max_interval: int = 15
    score_threshold: float = 1.0
    force_score: float = 3.0
    motion_threshold: float = 0.025
    object_motion_threshold: float = 0.01
    gripper_threshold: float = 0.02
    reward_delta_threshold: float = 0.25
    direction_turn_cosine: float = 0.25
    max_keyframes: int = 64

    def __post_init__(self) -> None:
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if self.max_interval <= 0:
            raise ValueError("max_interval must be positive")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval")
        if self.score_threshold < 0 or self.force_score < 0:
            raise ValueError("score thresholds must be non-negative")
        if self.max_keyframes <= 0:
            raise ValueError("max_keyframes must be positive")


@dataclasses.dataclass(frozen=True)
class KeyframeDecision:
    """Detector output for one environment state."""

    step: int
    score: float
    labels: Tuple[str, ...]
    features: Mapping[str, float]
    should_capture: bool
    forced: bool
    blocked_by_min_interval: bool


@dataclasses.dataclass
class SemanticKeyframe:
    """A captured checkpoint plus the semantic reason it was retained."""

    checkpoint: Checkpoint
    step: int
    score: float
    labels: Tuple[str, ...]
    features: Dict[str, float]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "score": float(self.score),
            "labels": list(self.labels),
            "features": {key: float(value) for key, value in self.features.items()},
            "checkpoint_metadata": dict(self.checkpoint.metadata),
        }


def _as_vector(value: Any, *, max_size: int = 128) -> Optional[np.ndarray]:
    """Return a finite, small numeric vector or ``None``."""

    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if not np.issubdtype(array.dtype, np.number) or array.size == 0 or array.size > max_size:
        return None
    array = array.astype(np.float64, copy=False).reshape(-1)
    if not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _first_vector(
    obs: Optional[Mapping[str, Any]], names: Iterable[str], *, max_size: int = 128
) -> Optional[np.ndarray]:
    if not isinstance(obs, Mapping):
        return None
    for name in names:
        vector = _as_vector(obs.get(name), max_size=max_size)
        if vector is not None:
            return vector
    return None


def _find_vector(
    obs: Optional[Mapping[str, Any]], *, include: Sequence[str], exclude: Sequence[str] = ()
) -> Optional[np.ndarray]:
    """Find the first compact numeric observation matching semantic names."""

    if not isinstance(obs, Mapping):
        return None
    for key, value in obs.items():
        name = str(key).lower()
        if any(token not in name for token in include) or any(
            token in name for token in exclude
        ):
            continue
        vector = _as_vector(value)
        if vector is not None:
            return vector
    return None


def _observation_vector(obs: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    """Build a compact fallback vector while ignoring camera observations."""

    if not isinstance(obs, Mapping):
        return None
    pieces: List[np.ndarray] = []
    for key in sorted(obs, key=str):
        name = str(key).lower()
        if "image" in name or "depth" in name or "segmentation" in name:
            continue
        vector = _as_vector(obs[key], max_size=64)
        if vector is not None:
            pieces.append(vector)
    if not pieces:
        return None
    return np.concatenate(pieces)[:256]


def _scalar_from_mapping(mapping: Optional[Mapping[str, Any]], names: Iterable[str]) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        if isinstance(value, (bool, int, float, np.number)):
            value = float(value)
            if np.isfinite(value):
                return value
    return None


class SemanticKeyframeSelector:
    """Detect and store semantically meaningful simulator checkpoints.

    The selector is stateful only with respect to previous observations.  It
    can therefore be used in a rollout loop as follows::

        selector.maybe_capture(env, 0, obs)
        for step in range(1, horizon + 1):
            obs, reward, done, info = step_env(env, action)
            selector.maybe_capture(env, step, obs, reward, done, info, action=action)

    Checkpoints are captured after the state represented by ``step`` is
    produced.  ``select_for_rollback(failure_step)`` only returns states with
    ``checkpoint.step < failure_step`` so the failed state itself is never used
    as a recovery point.
    """

    def __init__(
        self,
        tool: Optional[EnvRollbackTool] = None,
        config: Optional[SemanticKeyframeConfig] = None,
    ) -> None:
        self.tool = tool or EnvRollbackTool()
        self.config = config or SemanticKeyframeConfig()
        self.keyframes: List[SemanticKeyframe] = []
        self._previous_obs: Optional[Dict[str, Any]] = None
        self._previous_reward: Optional[float] = None
        self._previous_eef_delta: Optional[np.ndarray] = None
        self._previous_progress: Optional[float] = None
        self._last_decision: Optional[KeyframeDecision] = None

    @property
    def last_decision(self) -> Optional[KeyframeDecision]:
        return self._last_decision

    @property
    def last_keyframe(self) -> Optional[SemanticKeyframe]:
        return self.keyframes[-1] if self.keyframes else None

    def reset(self) -> None:
        """Clear detector history before starting another episode."""

        self.keyframes.clear()
        self._previous_obs = None
        self._previous_reward = None
        self._previous_eef_delta = None
        self._previous_progress = None
        self._last_decision = None

    def reset_history(
        self, obs: Optional[Mapping[str, Any]] = None, reward: Optional[float] = None
    ) -> None:
        """Re-seed feature history after a rollback without deleting keyframes.

        A restored state may be several steps behind the last observed failed
        state.  Resetting the temporal detector prevents that discontinuity
        from being mistaken for a new semantic event on the first replay step.
        """

        self._previous_obs = copy_obs(obs)
        self._previous_reward = float(reward) if reward is not None else None
        self._previous_eef_delta = None
        self._previous_progress = None
        self._last_decision = None

    def observe_only(
        self,
        env: Any,
        step: int,
        obs: Optional[Mapping[str, Any]],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> KeyframeDecision:
        """Advance feature history without allowing a heuristic capture.

        Risk-aware policies use this on low-risk event edges and then reserve
        checkpoint creation for explicit sub-event transitions or max-interval
        fallbacks.
        """

        decision = self.evaluate(
            env, step=step, obs=obs, reward=reward, done=done, info=info
        )
        self._previous_obs = copy_obs(obs)
        self._previous_reward = float(reward)
        return decision

    def _success_signal(self, env: Any, info: Optional[Mapping[str, Any]]) -> bool:
        success = _scalar_from_mapping(info, ("success", "is_success", "task_success"))
        if success is not None and bool(success):
            return True
        # check_success is a read-only robosuite/LIBERO convention.  It is used
        # only when info does not already provide an explicit signal.
        for owner in (env, getattr(env, "env", None), getattr(env, "_env", None)):
            method = getattr(owner, "check_success", None)
            if callable(method):
                try:
                    return bool(method())
                except Exception:
                    continue
        return False

    def evaluate(
        self,
        env: Any,
        *,
        step: int,
        obs: Optional[Mapping[str, Any]],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> KeyframeDecision:
        """Score the current state without capturing it."""

        current = dict(obs) if isinstance(obs, Mapping) else None
        labels: List[str] = []
        features: Dict[str, float] = {}
        score = 0.0

        first_state = not self.keyframes and self._previous_obs is None
        if first_state or int(step) == 0:
            labels.append("episode_start")
            score += 4.0

        success = self._success_signal(env, info)
        if success:
            labels.append("success")
            score += 5.0
        if done:
            labels.append("terminal")
            score += 5.0

        eef = _first_vector(
            current,
            ("robot0_eef_pos", "eef_pos", "end_effector_pos", "robot_eef_pos"),
            max_size=8,
        )
        if eef is None:
            eef = _find_vector(current, include=("eef", "pos"))
        previous_eef = _first_vector(
            self._previous_obs,
            ("robot0_eef_pos", "eef_pos", "end_effector_pos", "robot_eef_pos"),
            max_size=8,
        )
        if previous_eef is None:
            previous_eef = _find_vector(self._previous_obs, include=("eef", "pos"))
        if eef is not None and previous_eef is not None and eef.shape == previous_eef.shape:
            eef_delta = eef - previous_eef
            motion = float(np.linalg.norm(eef_delta))
            features["eef_motion"] = motion
            if motion >= self.config.motion_threshold:
                labels.append("eef_motion")
                score += min(2.0, motion / self.config.motion_threshold)
            if self._previous_eef_delta is not None:
                previous_norm = float(np.linalg.norm(self._previous_eef_delta))
                current_norm = float(np.linalg.norm(eef_delta))
                # Ignore numerical jitter: a direction change is semantic only
                # when both adjacent motions have a meaningful magnitude.
                turn_motion_floor = 0.25 * self.config.motion_threshold
                if previous_norm >= turn_motion_floor and current_norm >= turn_motion_floor:
                    cosine = float(
                        np.dot(self._previous_eef_delta, eef_delta)
                        / (previous_norm * current_norm)
                    )
                    features["eef_turn_cosine"] = cosine
                    if cosine <= self.config.direction_turn_cosine:
                        labels.append("motion_turn")
                        score += 1.25
            self._previous_eef_delta = eef_delta.copy()

        gripper = _first_vector(
            current,
            ("robot0_gripper_qpos", "gripper_qpos", "gripper_pos"),
            max_size=16,
        )
        previous_gripper = _first_vector(
            self._previous_obs,
            ("robot0_gripper_qpos", "gripper_qpos", "gripper_pos"),
            max_size=16,
        )
        if gripper is not None and previous_gripper is not None and gripper.shape == previous_gripper.shape:
            gripper_delta = float(np.linalg.norm(gripper - previous_gripper))
            features["gripper_delta"] = gripper_delta
            if gripper_delta >= self.config.gripper_threshold:
                labels.append("gripper_transition")
                score += 2.0

        object_now = _find_vector(current, include=("object",), exclude=("image",))
        object_before = _find_vector(self._previous_obs, include=("object",), exclude=("image",))
        if object_now is not None and object_before is not None and object_now.shape == object_before.shape:
            object_delta = float(np.linalg.norm(object_now - object_before))
            features["object_motion"] = object_delta
            if object_delta >= self.config.object_motion_threshold:
                labels.append("object_motion")
                score += 2.0

        if self._previous_reward is not None:
            reward_delta = abs(float(reward) - self._previous_reward)
            features["reward_delta"] = reward_delta
            if reward_delta >= self.config.reward_delta_threshold:
                labels.append("reward_change")
                score += 1.5
        progress = _scalar_from_mapping(info, ("progress", "task_progress", "normalized_progress"))
        if progress is not None:
            features["progress"] = progress
            previous_progress = getattr(self, "_previous_progress", None)
            if previous_progress is not None and abs(progress - previous_progress) >= 0.05:
                labels.append("progress_change")
                score += 1.5
            self._previous_progress = progress

        if self._previous_obs is not None and not features:
            before_vector = _observation_vector(self._previous_obs)
            current_vector = _observation_vector(current)
            if before_vector is not None and current_vector is not None and before_vector.shape == current_vector.shape:
                generic_delta = float(np.linalg.norm(current_vector - before_vector))
                features["observation_delta"] = generic_delta
                # The fallback is intentionally low weight; max_interval still
                # bounds checkpoint spacing when no semantic field is exposed.
                if generic_delta > 1e-6:
                    labels.append("observation_change")
                    score += 0.5

        last_step = self.last_keyframe.step if self.last_keyframe is not None else None
        interval = int(step) - int(last_step) if last_step is not None else int(step)
        features["steps_since_keyframe"] = float(max(0, interval))
        high_score = score >= self.config.force_score
        if high_score and "high_score_event" not in labels:
            labels.append("high_score_event")
        forced = (
            first_state
            or success
            or done
            or high_score
            or interval >= self.config.max_interval
        )
        if interval >= self.config.max_interval:
            labels.append("max_interval")
            score += 2.0

        semantic_candidate = score >= self.config.score_threshold
        can_capture = last_step is None or interval >= self.config.min_interval
        blocked = bool(semantic_candidate and not can_capture and not forced)
        should_capture = bool(forced or (semantic_candidate and can_capture))
        if blocked:
            labels.append("min_interval_suppressed")

        decision = KeyframeDecision(
            step=int(step),
            score=float(score),
            labels=tuple(dict.fromkeys(labels)),
            features=features,
            should_capture=should_capture,
            forced=bool(forced),
            blocked_by_min_interval=blocked,
        )
        self._last_decision = decision
        return decision

    def maybe_capture(
        self,
        env: Any,
        step: int,
        obs: Optional[Mapping[str, Any]],
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
        *,
        action: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        rollout_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SemanticKeyframe]:
        """Evaluate and, if selected, capture the current environment state."""

        decision = self.evaluate(
            env, step=step, obs=obs, reward=reward, done=done, info=info
        )
        if decision.should_capture:
            capture_metadata: Dict[str, Any] = dict(metadata or {})
            capture_metadata.update(
                {
                    "checkpoint_kind": "semantic_keyframe",
                    "semantic_score": decision.score,
                    "semantic_labels": list(decision.labels),
                    "semantic_features": dict(decision.features),
                    "reward_at_capture": float(reward),
                    "done_at_capture": bool(done),
                }
            )
            if action is not None:
                capture_metadata["action_at_capture"] = np.asarray(action).tolist()
            checkpoint = self.tool.capture(
                env,
                step=int(step),
                obs=dict(obs) if isinstance(obs, Mapping) else None,
                metadata=capture_metadata,
                rng=rng,
                rollout_state=rollout_state,
            )
            keyframe = SemanticKeyframe(
                checkpoint=checkpoint,
                step=int(step),
                score=decision.score,
                labels=decision.labels,
                features=dict(decision.features),
            )
            self.keyframes.append(keyframe)
            self._enforce_memory_budget()

        self._previous_obs = copy_obs(obs)
        self._previous_reward = float(reward)
        return self.last_keyframe if self.last_keyframe and self.last_keyframe.step == int(step) else None

    def force_capture(
        self,
        env: Any,
        step: int,
        obs: Optional[Mapping[str, Any]],
        *,
        labels: Sequence[str],
        score: float = 0.0,
        features: Optional[Mapping[str, float]] = None,
        reward: float = 0.0,
        done: bool = False,
        action: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        rollout_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SemanticKeyframe:
        """Capture a mandatory event checkpoint regardless of score/cooldown."""

        capture_metadata: Dict[str, Any] = dict(metadata or {})
        capture_metadata.update(
            {
                "checkpoint_kind": "semantic_event_keyframe",
                "semantic_score": float(score),
                "semantic_labels": list(labels),
                "semantic_features": dict(features or {}),
                "reward_at_capture": float(reward),
                "done_at_capture": bool(done),
            }
        )
        if action is not None:
            capture_metadata["action_at_capture"] = np.asarray(action).tolist()
        checkpoint = self.tool.capture(
            env,
            step=int(step),
            obs=dict(obs) if isinstance(obs, Mapping) else None,
            metadata=capture_metadata,
            rng=rng,
            rollout_state=rollout_state,
        )
        keyframe = SemanticKeyframe(
            checkpoint=checkpoint,
            step=int(step),
            score=float(score),
            labels=tuple(dict.fromkeys(labels)),
            features={key: float(value) for key, value in (features or {}).items()},
        )
        self.keyframes.append(keyframe)
        self._enforce_memory_budget()
        self._previous_obs = copy_obs(obs)
        self._previous_reward = float(reward)
        return keyframe

    def _enforce_memory_budget(self) -> None:
        """Bound full-state memory while retaining episode boundaries."""

        while len(self.keyframes) > self.config.max_keyframes:
            removable = next(
                (
                    index
                    for index, keyframe in enumerate(self.keyframes)
                    if index != len(self.keyframes) - 1
                    and not set(keyframe.labels) & {"episode_start", "success", "terminal"}
                ),
                None,
            )
            if removable is None:
                # This is only possible with a very small user-supplied budget;
                # keep the most recent protected boundary.
                removable = 0
            del self.keyframes[removable]

    def select_for_rollback(
        self, failure_step: int, *, strategy: str = "latest_safe"
    ) -> SemanticKeyframe:
        """Select a checkpoint strictly before a detected failure.

        ``latest_safe`` is the default because it minimizes lost progress.
        ``best_score`` is useful for experiments that prefer a high-confidence
        semantic transition over temporal proximity.
        """

        eligible = [
            keyframe
            for keyframe in self.keyframes
            if keyframe.step < int(failure_step) and not keyframe.checkpoint.env_done
        ]
        if not eligible:
            eligible = [keyframe for keyframe in self.keyframes if keyframe.step < int(failure_step)]
        if not eligible:
            raise LookupError(
                "No checkpoint exists strictly before failure step "
                f"{failure_step}; capture an episode-start keyframe first."
            )
        if strategy == "latest_safe":
            return max(eligible, key=lambda keyframe: keyframe.step)
        if strategy == "best_score":
            return max(eligible, key=lambda keyframe: (keyframe.score, keyframe.step))
        raise ValueError("strategy must be 'latest_safe' or 'best_score'")

    def restore_for_failure(
        self,
        env: Any,
        failure_step: int,
        *,
        strategy: str = "latest_safe",
        reload_xml: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[SemanticKeyframe, Dict[str, Any]]:
        """Restore the selected semantic keyframe and return its report."""

        keyframe = self.select_for_rollback(failure_step, strategy=strategy)
        report = self.tool.restore(
            env,
            keyframe.checkpoint,
            reload_xml=reload_xml,
            rng=rng,
        )
        return keyframe, report

    def summary(self) -> Dict[str, Any]:
        return {
            "config": dataclasses.asdict(self.config),
            "keyframe_count": len(self.keyframes),
            "keyframes": [keyframe.to_summary() for keyframe in self.keyframes],
        }


__all__ = [
    "KeyframeDecision",
    "SemanticKeyframe",
    "SemanticKeyframeConfig",
    "SemanticKeyframeSelector",
]
