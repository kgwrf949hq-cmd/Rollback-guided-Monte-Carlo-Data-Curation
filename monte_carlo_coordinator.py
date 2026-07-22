#!/usr/bin/env python3
"""Budgeted breadth-first Monte Carlo curation for pick-and-place rollouts.

The coordinator is the only layer in this repository that owns rollout side
effects.  Failure detection, semantic checkpoint selection, simulator restore,
and action sampling remain separate components.  Version one deliberately:

* runs one environment sequentially with a FIFO breadth-first frontier;
* executes every branch until verified success, confirmed failure, ``done``,
  or the full episode horizon;
* keeps every executed branch, including failures and quarantined runs;
* varies actions, never environment state, between checkpoint siblings; and
* stores full observations in memory while exposing deterministic JSON
  summaries.  A durable dataset writer is intentionally out of scope.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import hashlib
import json
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from libero_rollback_tool import (
    EnvRollbackTool,
    copy_obs,
    get_observation,
    make_branch_rng,
    step_env,
)
from pick_place_checkpoint import (
    EventEdgePolicy,
    PickPlaceRiskAwareCheckpointManager,
    PickPlaceTaskSpec,
    PickPlaceTaskTruth,
    extract_pick_place_task_truth,
)
from pick_place_failure import (
    FailureDecision,
    FailureStatus,
    PickPlaceFailureConfig,
    PickPlaceRuleFailureDetector,
)
from rollback_policy import (
    RollbackCandidate,
    RollbackCheckpoint,
    RollbackPolicy,
    RollbackSelection,
    StageAwareRollbackPolicy,
)
from semantic_checkpoint import SemanticKeyframe, SemanticKeyframeConfig


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy/dataclass values to deterministic JSON values."""

    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, collections.deque)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(
        _json_safe(parts), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def _stable_uint32(*parts: Any) -> int:
    payload = json.dumps(
        _json_safe(parts), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _observation_summary(obs: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(obs, Mapping):
        return None
    summary: Dict[str, Any] = {}
    for key in sorted(obs):
        value = obs[key]
        try:
            array = np.asarray(value)
        except Exception:
            summary[str(key)] = {"type": type(value).__name__}
            continue
        item: Dict[str, Any] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        if array.size and np.issubdtype(array.dtype, np.number):
            item["sha256"] = _array_digest(array)
        summary[str(key)] = item
    return summary


class BranchStatus(str, enum.Enum):
    ACTIVE = "active"
    SUCCESS = "success"
    RECOVERABLE_FAILURE = "recoverable_failure"
    TERMINAL_FAILURE = "terminal_failure"
    HORIZON_EXHAUSTED = "horizon_exhausted"
    QUARANTINED = "quarantined"
    RESTORE_FAILED = "restore_failed"


@dataclasses.dataclass(frozen=True)
class MonteCarloConfig:
    """Budgets and stopping rules for curation-only breadth-first search."""

    episode_horizon: int = 180
    max_total_branches: int = 8
    max_children_per_fork: int = 3
    max_attempts_per_checkpoint: int = 6
    max_depth: int = 5
    safety_margin_steps: int = 1
    duplicate_failure_limit: int = 2
    failure_signature_quantization: float = 0.01
    expand_terminal_failures: bool = True
    trusted_success_sources: Tuple[str, ...] = ("env.check_success",)
    quarantine_failure_reasons: Tuple[str, ...] = (
        "invalid_observation",
        "simulation_error",
    )

    def __post_init__(self) -> None:
        for name in (
            "episode_horizon",
            "max_total_branches",
            "max_children_per_fork",
            "max_attempts_per_checkpoint",
            "max_depth",
            "duplicate_failure_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.safety_margin_steps < 0:
            raise ValueError("safety_margin_steps must be non-negative")
        if self.failure_signature_quantization <= 0:
            raise ValueError("failure_signature_quantization must be positive")


@dataclasses.dataclass(frozen=True)
class StructuredActionSamplerConfig:
    """Action-only sibling diversity for an action-chunk policy callback.

    The callback receives ``(observation, sampling_seed)`` and returns a
    sequence of actions.  With ``native_sampling=True``, every sibling uses a
    branch-specific policy seed and no artificial noise.  Otherwise siblings
    share a policy seed at the same fork and attempts 1/2 add low/medium
    temporally correlated noise to motion dimensions only.
    """

    replan_steps: int = 5
    native_sampling: bool = False
    low_noise_std: float = 0.02
    medium_noise_std: float = 0.05
    temporal_correlation: float = 0.75
    motion_dimensions: int = 6
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if self.low_noise_std < 0 or self.medium_noise_std < 0:
            raise ValueError("action noise standard deviations must be non-negative")
        if not 0.0 <= self.temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must be in [0, 1)")
        if self.motion_dimensions < 0:
            raise ValueError("motion_dimensions must be non-negative")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be smaller than action_high")


@dataclasses.dataclass
class ActionSample:
    policy_action: np.ndarray
    executed_action: np.ndarray
    source: str
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "policy_action": np.asarray(self.policy_action).tolist(),
            "executed_action": np.asarray(self.executed_action).tolist(),
            "source": self.source,
            "metadata": _json_safe(self.metadata),
        }


class BranchActionSampler:
    """Small interface implemented by policy/action proposal adapters."""

    def branch_variant(self, attempt_index: int, *, is_root: bool = False) -> str:
        return "policy" if is_root else "policy_replan"

    def start_branch(self, branch: "BranchRecord", rng: np.random.Generator) -> Any:
        return None

    def next_action(
        self,
        obs: Mapping[str, Any],
        branch: "BranchRecord",
        branch_step: int,
        rng: np.random.Generator,
        state: Any,
    ) -> ActionSample:
        raise NotImplementedError

    def checkpoint_state(self, state: Any) -> Optional[Dict[str, Any]]:
        return None


class StructuredActionChunkSampler(BranchActionSampler):
    """Replan at every branch and optionally perturb only policy actions."""

    def __init__(
        self,
        policy_infer: Callable[[Mapping[str, Any], int], Sequence[np.ndarray]],
        config: Optional[StructuredActionSamplerConfig] = None,
    ) -> None:
        self.policy_infer = policy_infer
        self.config = config or StructuredActionSamplerConfig()

    def branch_variant(self, attempt_index: int, *, is_root: bool = False) -> str:
        if is_root:
            return "policy"
        if self.config.native_sampling:
            return "native_policy_sample"
        if int(attempt_index) == 0:
            return "policy_replan"
        if int(attempt_index) == 1:
            return "action_noise_low"
        return "action_noise_medium"

    def start_branch(
        self, branch: "BranchRecord", rng: np.random.Generator
    ) -> Dict[str, Any]:
        del rng
        return {
            "pending": collections.deque(),
            "query_index": 0,
            "noise_state": None,
            "last_sampling_seed": None,
            "last_chunk_sha256": None,
        }

    def _sampling_seed(self, branch: "BranchRecord", query_index: int) -> int:
        if self.config.native_sampling:
            identity = branch.branch_id
        else:
            # Siblings at the same fork receive the same base policy seed, so
            # artificial action noise remains the only varied factor.
            identity = branch.fork_checkpoint_id or branch.branch_id
        return _stable_uint32(
            branch.root_seed,
            branch.episode_id,
            identity,
            int(query_index),
        )

    def _refill(
        self,
        obs: Mapping[str, Any],
        branch: "BranchRecord",
        state: Dict[str, Any],
    ) -> None:
        query_index = int(state["query_index"])
        sampling_seed = self._sampling_seed(branch, query_index)
        raw_chunk = self.policy_infer(obs, sampling_seed)
        chunk = [
            np.asarray(action, dtype=np.float32).reshape(-1) for action in raw_chunk
        ]
        if len(chunk) < self.config.replan_steps:
            raise RuntimeError(
                "policy returned %d actions, fewer than replan_steps=%d"
                % (len(chunk), self.config.replan_steps)
            )
        for action in chunk[: self.config.replan_steps]:
            if action.size == 0 or not np.all(np.isfinite(action)):
                raise ValueError("policy returned an empty or non-finite action")
        selected = chunk[: self.config.replan_steps]
        state["pending"].extend(selected)
        state["query_index"] = query_index + 1
        state["last_sampling_seed"] = int(sampling_seed)
        joined = np.concatenate(selected)
        state["last_chunk_sha256"] = _array_digest(joined)

    def next_action(
        self,
        obs: Mapping[str, Any],
        branch: "BranchRecord",
        branch_step: int,
        rng: np.random.Generator,
        state: Dict[str, Any],
    ) -> ActionSample:
        del branch_step
        if not state["pending"]:
            self._refill(obs, branch, state)
        policy_action = np.asarray(state["pending"].popleft(), dtype=np.float32)
        executed = policy_action.astype(np.float64, copy=True)

        if branch.action_variant == "action_noise_low":
            noise_std = self.config.low_noise_std
        elif branch.action_variant == "action_noise_medium":
            noise_std = self.config.medium_noise_std
        else:
            noise_std = 0.0
        dimensions = min(self.config.motion_dimensions, executed.size)
        if dimensions and noise_std > 0.0:
            previous = state.get("noise_state")
            if previous is None or np.asarray(previous).shape != (dimensions,):
                previous = np.zeros(dimensions, dtype=np.float64)
            rho = self.config.temporal_correlation
            innovation = rng.normal(size=dimensions)
            noise = (
                rho * np.asarray(previous)
                + np.sqrt(1.0 - rho * rho) * noise_std * innovation
            )
            executed[:dimensions] += noise
            state["noise_state"] = noise.copy()
        executed = np.clip(
            executed, self.config.action_low, self.config.action_high
        ).astype(np.float32)
        return ActionSample(
            policy_action=policy_action.copy(),
            executed_action=executed,
            source=branch.action_variant,
            metadata={
                "policy_sampling_seed": state["last_sampling_seed"],
                "policy_chunk_sha256": state["last_chunk_sha256"],
                "native_sampling_requested": self.config.native_sampling,
                "action_noise_std": float(noise_std),
                "environment_perturbed": False,
            },
        )

    def checkpoint_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "pending_action_plan": [
                np.asarray(action, dtype=np.float32).copy()
                for action in state.get("pending", ())
            ],
            "query_index": int(state.get("query_index", 0)),
            "last_sampling_seed": state.get("last_sampling_seed"),
            "last_chunk_sha256": state.get("last_chunk_sha256"),
            # Children intentionally do not restore this state; every child
            # replans from its fork checkpoint.
            "restore_on_child": False,
        }


@dataclasses.dataclass
class TransitionRecord:
    branch_step: int
    env_step: int
    observation: Optional[Dict[str, Any]]
    next_observation: Optional[Dict[str, Any]]
    policy_action: np.ndarray
    executed_action: np.ndarray
    action_source: str
    action_metadata: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    task_truth: Dict[str, Any]
    completed_events: Tuple[str, ...]
    event_stage: str
    failure_decision: FailureDecision
    checkpoint_id: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "branch_step": int(self.branch_step),
            "env_step": int(self.env_step),
            "observation": _observation_summary(self.observation),
            "next_observation": _observation_summary(self.next_observation),
            "policy_action": np.asarray(self.policy_action).tolist(),
            "executed_action": np.asarray(self.executed_action).tolist(),
            "action_source": self.action_source,
            "action_metadata": _json_safe(self.action_metadata),
            "reward": float(self.reward),
            "done": bool(self.done),
            "info": _json_safe(self.info),
            "task_truth": _json_safe(self.task_truth),
            "completed_events": list(self.completed_events),
            "event_stage": self.event_stage,
            "failure_decision": self.failure_decision.to_summary(),
            "checkpoint_id": self.checkpoint_id,
        }


@dataclasses.dataclass
class CheckpointRecord:
    checkpoint_id: str
    episode_id: str
    owner_branch_id: str
    step: int
    ordinal: int
    keyframe: SemanticKeyframe
    event_detector_state: Dict[str, Any]
    selector_state: Dict[str, Any]
    failure_detector_state: Dict[str, Any]

    @property
    def checkpoint(self) -> Any:
        return self.keyframe.checkpoint

    def to_summary(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "episode_id": self.episode_id,
            "owner_branch_id": self.owner_branch_id,
            "step": int(self.step),
            "ordinal": int(self.ordinal),
            "score": float(self.keyframe.score),
            "labels": list(self.keyframe.labels),
            "env_done": bool(self.checkpoint.env_done),
            "logical_state": {
                "event_detector": True,
                "semantic_selector": True,
                "failure_detector": True,
            },
        }


@dataclasses.dataclass
class BranchRecord:
    episode_id: str
    branch_id: str
    parent_branch_id: Optional[str]
    fork_checkpoint_id: Optional[str]
    fork_step: int
    depth: int
    attempt_index: int
    root_seed: int
    action_variant: str
    execution_index: int
    status: BranchStatus = BranchStatus.ACTIVE
    outcome: str = "active"
    expansion_outcome: Optional[str] = None
    transitions: List[TransitionRecord] = dataclasses.field(default_factory=list)
    active_checkpoint_ids: List[str] = dataclasses.field(default_factory=list)
    failure_decision: Optional[FailureDecision] = None
    rollback_selection: Optional[RollbackSelection] = None
    restore_report: Optional[Dict[str, Any]] = None
    failure_signature: Optional[str] = None
    task_success_source: Optional[str] = None
    error: Optional[str] = None
    children: List[str] = dataclasses.field(default_factory=list)

    @property
    def final_env_step(self) -> int:
        if self.transitions:
            return int(self.transitions[-1].env_step)
        return int(self.fork_step)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "fork_checkpoint_id": self.fork_checkpoint_id,
            "fork_step": int(self.fork_step),
            "depth": int(self.depth),
            "attempt_index": int(self.attempt_index),
            "root_seed": int(self.root_seed),
            "action_variant": self.action_variant,
            "execution_index": int(self.execution_index),
            "status": self.status.value,
            "outcome": self.outcome,
            "expansion_outcome": self.expansion_outcome,
            "transition_count": len(self.transitions),
            "transitions": [item.to_summary() for item in self.transitions],
            "active_checkpoint_ids": list(self.active_checkpoint_ids),
            "failure_decision": (
                self.failure_decision.to_summary()
                if self.failure_decision is not None
                else None
            ),
            "rollback_selection": (
                self.rollback_selection.to_summary()
                if self.rollback_selection is not None
                else None
            ),
            "restore_report": _json_safe(self.restore_report),
            "failure_signature": self.failure_signature,
            "task_success_source": self.task_success_source,
            "final_env_step": self.final_env_step,
            "error": self.error,
            "children": list(self.children),
        }


@dataclasses.dataclass(frozen=True)
class _BranchJob:
    branch_id: str
    parent_branch_id: Optional[str]
    fork_checkpoint_id: Optional[str]
    fork_step: int
    depth: int
    attempt_index: int
    action_variant: str
    inherited_checkpoint_ids: Tuple[str, ...]


@dataclasses.dataclass
class MonteCarloResult:
    episode_id: str
    root_seed: int
    config: MonteCarloConfig
    branches: Dict[str, BranchRecord]
    checkpoints: Dict[str, CheckpointRecord]
    execution_order: List[str]

    def to_summary(self) -> Dict[str, Any]:
        counts = {
            status.value: 0 for status in BranchStatus if status != BranchStatus.ACTIVE
        }
        for branch in self.branches.values():
            if branch.status != BranchStatus.ACTIVE:
                counts[branch.status.value] += 1
        return {
            "schema_version": 1,
            "mode": "curation",
            "scheduler": "bounded_fifo_bfs",
            "episode_id": self.episode_id,
            "root_seed": int(self.root_seed),
            "config": _json_safe(self.config),
            "branch_count": len(self.branches),
            "checkpoint_count": len(self.checkpoints),
            "status_counts": counts,
            "execution_order": list(self.execution_order),
            "branches": [
                self.branches[branch_id].to_summary()
                for branch_id in self.execution_order
            ],
            "checkpoints": [
                record.to_summary() for record in self.checkpoints.values()
            ],
        }


class MonteCarloCoordinator:
    """Run curation-only, full-horizon branches through a FIFO frontier."""

    def __init__(
        self,
        task_spec: PickPlaceTaskSpec,
        action_sampler: BranchActionSampler,
        *,
        config: Optional[MonteCarloConfig] = None,
        rollback_tool: Optional[EnvRollbackTool] = None,
        selector_config: Optional[SemanticKeyframeConfig] = None,
        edge_policies: Optional[Sequence[EventEdgePolicy]] = None,
        failure_config: Optional[PickPlaceFailureConfig] = None,
        rollback_policy: Optional[RollbackPolicy] = None,
        step_fn: Callable[
            [Any, np.ndarray],
            Tuple[Optional[Dict[str, Any]], float, bool, Dict[str, Any]],
        ] = step_env,
        checkpoint_manager_factory: Optional[
            Callable[[], PickPlaceRiskAwareCheckpointManager]
        ] = None,
        failure_detector_factory: Optional[
            Callable[[], PickPlaceRuleFailureDetector]
        ] = None,
    ) -> None:
        self.task_spec = task_spec
        self.action_sampler = action_sampler
        self.config = config or MonteCarloConfig()
        self.rollback_tool = rollback_tool or EnvRollbackTool()
        self.selector_config = selector_config
        self.edge_policies = tuple(edge_policies) if edge_policies is not None else None
        self.failure_config = failure_config
        self.rollback_policy = rollback_policy or StageAwareRollbackPolicy()
        self.step_fn = step_fn
        self._checkpoint_manager_factory = checkpoint_manager_factory
        self._failure_detector_factory = failure_detector_factory
        self._reset_state()

    def _reset_state(self) -> None:
        self._episode_id = ""
        self._root_seed = 0
        self._branches: Dict[str, BranchRecord] = {}
        self._checkpoints: Dict[str, CheckpointRecord] = {}
        self._execution_order: List[str] = []
        self._frontier: Deque[_BranchJob] = collections.deque()
        self._checkpoint_ordinals: Dict[str, int] = collections.defaultdict(int)
        self._checkpoint_attempts: Dict[str, int] = collections.defaultdict(int)
        self._next_attempt: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        self._failure_signature_counts: Dict[str, int] = collections.defaultdict(int)
        self._initial_obs: Optional[Dict[str, Any]] = None
        self._initial_info: Dict[str, Any] = {}
        self._initial_step = 0

    def _new_manager(self) -> PickPlaceRiskAwareCheckpointManager:
        if self._checkpoint_manager_factory is not None:
            return self._checkpoint_manager_factory()
        return PickPlaceRiskAwareCheckpointManager(
            self.task_spec,
            tool=self.rollback_tool,
            selector_config=self.selector_config,
            edge_policies=self.edge_policies,
        )

    def _new_failure_detector(self) -> PickPlaceRuleFailureDetector:
        if self._failure_detector_factory is not None:
            return self._failure_detector_factory()
        return PickPlaceRuleFailureDetector(self.task_spec, self.failure_config)

    def run(
        self,
        env: Any,
        initial_obs: Mapping[str, Any],
        *,
        episode_id: str,
        root_seed: int,
        initial_step: int = 0,
        initial_info: Optional[Mapping[str, Any]] = None,
    ) -> MonteCarloResult:
        """Execute all budgeted branches; success never cancels siblings."""

        if not episode_id:
            raise ValueError("episode_id is required for stable branch identities")
        if int(initial_step) < 0 or int(initial_step) >= self.config.episode_horizon:
            raise ValueError("initial_step must be within the episode horizon")
        self._reset_state()
        self._episode_id = str(episode_id)
        self._root_seed = int(root_seed)
        self._initial_obs = copy_obs(initial_obs)
        self._initial_info = dict(initial_info or {})
        self._initial_step = int(initial_step)

        root_id = _stable_id("branch", self._episode_id, self._root_seed, "root")
        self._frontier.append(
            _BranchJob(
                branch_id=root_id,
                parent_branch_id=None,
                fork_checkpoint_id=None,
                fork_step=self._initial_step,
                depth=0,
                attempt_index=0,
                action_variant=self.action_sampler.branch_variant(0, is_root=True),
                inherited_checkpoint_ids=(),
            )
        )

        while self._frontier and len(self._branches) < self.config.max_total_branches:
            job = self._frontier.popleft()
            branch = self._execute_job(env, job)
            self._branches[branch.branch_id] = branch
            self._execution_order.append(branch.branch_id)
            self._maybe_expand(branch)

        return MonteCarloResult(
            episode_id=self._episode_id,
            root_seed=self._root_seed,
            config=self.config,
            branches=dict(self._branches),
            checkpoints=dict(self._checkpoints),
            execution_order=list(self._execution_order),
        )

    def _make_branch(self, job: _BranchJob) -> BranchRecord:
        return BranchRecord(
            episode_id=self._episode_id,
            branch_id=job.branch_id,
            parent_branch_id=job.parent_branch_id,
            fork_checkpoint_id=job.fork_checkpoint_id,
            fork_step=job.fork_step,
            depth=job.depth,
            attempt_index=job.attempt_index,
            root_seed=self._root_seed,
            action_variant=job.action_variant,
            execution_index=len(self._execution_order),
            active_checkpoint_ids=list(job.inherited_checkpoint_ids),
        )

    def _restore_child(
        self,
        env: Any,
        branch: BranchRecord,
        manager: PickPlaceRiskAwareCheckpointManager,
        detector: PickPlaceRuleFailureDetector,
        rng: np.random.Generator,
    ) -> Optional[Dict[str, Any]]:
        if branch.fork_checkpoint_id is None:
            return copy_obs(self._initial_obs)
        record = self._checkpoints[branch.fork_checkpoint_id]
        try:
            report = self.rollback_tool.restore(
                env,
                record.checkpoint,
                rng=rng,
                restore_rng=False,
            )
        except Exception as exc:
            branch.status = BranchStatus.RESTORE_FAILED
            branch.outcome = "restore_exception"
            branch.error = repr(exc)
            branch.restore_report = {"success": False, "error": repr(exc)}
            return None
        branch.restore_report = copy.deepcopy(report)
        if not report.get("success"):
            branch.status = BranchStatus.RESTORE_FAILED
            branch.outcome = "restore_verification_failed"
            return None
        restored_obs = get_observation(env, force_update=False)
        if restored_obs is None:
            branch.status = BranchStatus.RESTORE_FAILED
            branch.outcome = "restore_observation_unavailable"
            return None

        manager.selector.keyframes = [
            self._checkpoints[checkpoint_id].keyframe
            for checkpoint_id in branch.active_checkpoint_ids
        ]
        try:
            manager.detector.restore_state(record.event_detector_state, restored_obs)
            manager.selector.restore_state(record.selector_state, restored_obs)
            detector.restore_state(record.failure_detector_state)
        except Exception as exc:
            branch.status = BranchStatus.RESTORE_FAILED
            branch.outcome = "logical_state_restore_failed"
            branch.error = repr(exc)
            return None
        return copy_obs(restored_obs)

    def _bootstrap_root(
        self,
        env: Any,
        branch: BranchRecord,
        manager: PickPlaceRiskAwareCheckpointManager,
        detector: PickPlaceRuleFailureDetector,
        rng: np.random.Generator,
        obs: Mapping[str, Any],
    ) -> Tuple[FailureDecision, PickPlaceTaskTruth]:
        truth = extract_pick_place_task_truth(env, self.task_spec, self._initial_info)
        observed = manager.observe(
            env,
            self._initial_step,
            obs,
            0.0,
            False,
            self._initial_info,
            rng=rng,
            metadata={
                "episode_id": self._episode_id,
                "branch_id": branch.branch_id,
                "phase": "monte_carlo_root",
            },
        )
        decision = detector.update(
            obs,
            step=self._initial_step,
            completed_events=manager.detector.completed_events,
            reward=0.0,
            done=False,
            info=self._initial_info,
            task_truth=truth,
        )
        if observed.keyframe is not None:
            self._register_keyframe(branch, observed.keyframe, manager, detector)
        return decision, truth

    def _execute_job(self, env: Any, job: _BranchJob) -> BranchRecord:
        branch = self._make_branch(job)
        manager = self._new_manager()
        detector = self._new_failure_detector()
        rng = make_branch_rng(self._root_seed, branch.branch_id)
        obs = self._restore_child(env, branch, manager, detector, rng)
        if obs is None:
            branch.expansion_outcome = "not_expanded_after_restore_failure"
            return branch

        if branch.parent_branch_id is None:
            try:
                initial_decision, initial_truth = self._bootstrap_root(
                    env, branch, manager, detector, rng, obs
                )
            except Exception as exc:
                branch.status = BranchStatus.QUARANTINED
                branch.outcome = "initial_state_processing_error"
                branch.error = repr(exc)
                return branch
            if initial_truth.task_success is True:
                self._finish_success(branch, initial_truth, at_initial_state=True)
                return branch
            if initial_decision.failed:
                self._finish_failure(branch, initial_decision)
                return branch

        action_state = self.action_sampler.start_branch(branch, rng)
        env_step = int(branch.fork_step)
        branch_step = 0
        while env_step < self.config.episode_horizon:
            try:
                action_observation = copy_obs(obs)
                action_sample = self.action_sampler.next_action(
                    obs, branch, branch_step, rng, action_state
                )
                next_obs, reward, done, info = self.step_fn(
                    env, action_sample.executed_action
                )
                env_step += 1
                branch_step += 1
                if not isinstance(next_obs, Mapping):
                    raise ValueError("environment returned a non-mapping observation")
                obs = copy_obs(next_obs)
                truth = extract_pick_place_task_truth(env, self.task_spec, info)
                rollout_state = self.action_sampler.checkpoint_state(action_state)
                observed = manager.observe(
                    env,
                    env_step,
                    obs,
                    reward,
                    done,
                    info,
                    action=action_sample.executed_action,
                    rng=rng,
                    rollout_state=rollout_state,
                    metadata={
                        "episode_id": self._episode_id,
                        "branch_id": branch.branch_id,
                        "parent_branch_id": branch.parent_branch_id,
                        "fork_checkpoint_id": branch.fork_checkpoint_id,
                        "action_variant": branch.action_variant,
                        "environment_perturbed": False,
                    },
                )
                decision = detector.update(
                    obs,
                    step=env_step,
                    completed_events=manager.detector.completed_events,
                    action=action_sample.executed_action,
                    reward=reward,
                    done=done,
                    info=info,
                    task_truth=truth,
                )
                transition = TransitionRecord(
                    branch_step=branch_step,
                    env_step=env_step,
                    observation=action_observation,
                    next_observation=copy_obs(obs),
                    policy_action=action_sample.policy_action.copy(),
                    executed_action=action_sample.executed_action.copy(),
                    action_source=action_sample.source,
                    action_metadata=copy.deepcopy(action_sample.metadata),
                    reward=float(reward),
                    done=bool(done),
                    info=copy.deepcopy(dict(info)),
                    task_truth=truth.as_dict(),
                    completed_events=tuple(manager.detector.completed_events),
                    event_stage=decision.current_stage,
                    failure_decision=decision,
                )
                if observed.keyframe is not None:
                    transition.checkpoint_id = self._register_keyframe(
                        branch, observed.keyframe, manager, detector
                    )
                branch.transitions.append(transition)
            except Exception as exc:
                branch.status = BranchStatus.QUARANTINED
                branch.outcome = "runner_or_observation_processing_error"
                branch.error = repr(exc)
                branch.expansion_outcome = "quarantined_not_expanded"
                return branch

            if truth.task_success is True:
                self._finish_success(branch, truth, at_initial_state=False)
                return branch
            if decision.failed:
                self._finish_failure(branch, decision)
                return branch
            if done:
                # Defensive fallback: the detector should already classify a
                # done-without-success state as terminal failure.
                fallback = FailureDecision(
                    step=env_step,
                    status=FailureStatus.TERMINAL_FAILURE,
                    risk_score=1.0,
                    reason="episode_terminated_without_verified_success",
                    current_stage=decision.current_stage,
                    consecutive_steps=1,
                    hard_failure=True,
                    warning_onset_step=env_step,
                    evidence={"done": True, "task_success": truth.task_success},
                )
                self._finish_failure(branch, fallback)
                return branch

        branch.status = BranchStatus.HORIZON_EXHAUSTED
        branch.outcome = "full_episode_horizon_without_verified_outcome"
        branch.expansion_outcome = "horizon_not_expanded_without_failure_decision"
        return branch

    def _finish_success(
        self,
        branch: BranchRecord,
        truth: PickPlaceTaskTruth,
        *,
        at_initial_state: bool,
    ) -> None:
        source = truth.sources.get("task_success", "unavailable")
        branch.task_success_source = source
        if source not in self.config.trusted_success_sources:
            branch.status = BranchStatus.QUARANTINED
            branch.outcome = "untrusted_success_signal"
            branch.expansion_outcome = "quarantined_not_expanded"
            return
        branch.status = BranchStatus.SUCCESS
        branch.outcome = (
            "verified_task_success_at_initial_state"
            if at_initial_state
            else "verified_task_success"
        )
        branch.expansion_outcome = "success_branch_complete_siblings_preserved"

    def _finish_failure(self, branch: BranchRecord, decision: FailureDecision) -> None:
        branch.failure_decision = decision
        if decision.reason in self.config.quarantine_failure_reasons:
            branch.status = BranchStatus.QUARANTINED
            branch.outcome = f"quarantined_{decision.reason}"
            branch.expansion_outcome = "quarantined_not_expanded"
        elif decision.status == FailureStatus.RECOVERABLE_FAILURE:
            branch.status = BranchStatus.RECOVERABLE_FAILURE
            branch.outcome = decision.reason or "recoverable_failure"
        else:
            branch.status = BranchStatus.TERMINAL_FAILURE
            branch.outcome = decision.reason or "terminal_failure"

    def _register_keyframe(
        self,
        branch: BranchRecord,
        keyframe: SemanticKeyframe,
        manager: PickPlaceRiskAwareCheckpointManager,
        detector: PickPlaceRuleFailureDetector,
    ) -> str:
        ordinal = self._checkpoint_ordinals[branch.branch_id]
        self._checkpoint_ordinals[branch.branch_id] += 1
        checkpoint_id = _stable_id(
            "checkpoint",
            self._episode_id,
            branch.branch_id,
            int(keyframe.step),
            int(ordinal),
        )
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            episode_id=self._episode_id,
            owner_branch_id=branch.branch_id,
            step=int(keyframe.step),
            ordinal=int(ordinal),
            keyframe=keyframe,
            event_detector_state=copy.deepcopy(manager.detector.snapshot_state()),
            selector_state=copy.deepcopy(manager.selector.snapshot_state()),
            failure_detector_state=copy.deepcopy(detector.snapshot_state()),
        )
        self._checkpoints[checkpoint_id] = record
        keyframe.checkpoint.metadata["monte_carlo_checkpoint_id"] = checkpoint_id
        keyframe.checkpoint.metadata["failure_detector_state"] = copy.deepcopy(
            record.failure_detector_state
        )

        active_ids = []
        for active_keyframe in manager.selector.keyframes:
            active_id = active_keyframe.checkpoint.metadata.get(
                "monte_carlo_checkpoint_id"
            )
            if active_id in self._checkpoints:
                active_ids.append(str(active_id))
        branch.active_checkpoint_ids = active_ids
        return checkpoint_id

    def select_rollback(
        self, branch: BranchRecord, decision: FailureDecision
    ) -> RollbackSelection:
        """Delegate pure selection over the branch-local lineage view."""

        checkpoints = []
        for checkpoint_id in branch.active_checkpoint_ids:
            record = self._checkpoints[checkpoint_id]
            checkpoints.append(
                RollbackCheckpoint(
                    checkpoint_id=checkpoint_id,
                    owner_branch_id=record.owner_branch_id,
                    step=int(record.step),
                    env_done=bool(record.checkpoint.env_done),
                    semantic_score=float(record.keyframe.score),
                    labels=tuple(record.keyframe.labels),
                    metadata=copy.deepcopy(record.checkpoint.metadata),
                )
            )
        return self.rollback_policy.select(
            decision,
            checkpoints,
            safety_margin_steps=self.config.safety_margin_steps,
        )

    def _failure_signature(
        self,
        branch: BranchRecord,
        selection: RollbackSelection,
        decision: FailureDecision,
    ) -> str:
        obs = branch.transitions[-1].next_observation if branch.transitions else None
        quantization = self.config.failure_signature_quantization

        def quantized(key: str) -> Optional[List[int]]:
            if not isinstance(obs, Mapping) or key not in obs:
                return None
            try:
                value = np.asarray(obs[key], dtype=np.float64).reshape(-1)[:3]
            except Exception:
                return None
            if value.size != 3 or not np.all(np.isfinite(value)):
                return None
            return np.rint(value / quantization).astype(np.int64).tolist()

        payload = {
            "fork_checkpoint_id": selection.selected_checkpoint_id,
            "event_stage": decision.current_stage,
            "failure_reason": decision.reason,
            "target_position": quantized(f"{self.task_spec.target_object}_pos"),
            "eef_position": quantized("robot0_eef_pos"),
        }
        return _stable_id("failure", payload)

    def _maybe_expand(self, branch: BranchRecord) -> None:
        if branch.status not in {
            BranchStatus.RECOVERABLE_FAILURE,
            BranchStatus.TERMINAL_FAILURE,
        }:
            return
        if (
            branch.status == BranchStatus.TERMINAL_FAILURE
            and not self.config.expand_terminal_failures
        ):
            branch.expansion_outcome = "terminal_expansion_disabled"
            return
        if branch.failure_decision is None:
            branch.expansion_outcome = "missing_failure_decision"
            return

        selection = self.select_rollback(branch, branch.failure_decision)
        branch.rollback_selection = selection
        checkpoint_id = selection.selected_checkpoint_id
        if checkpoint_id is None:
            branch.expansion_outcome = "no_eligible_checkpoint"
            return
        if branch.depth >= self.config.max_depth:
            branch.expansion_outcome = "stopped_max_depth"
            return

        signature = self._failure_signature(branch, selection, branch.failure_decision)
        branch.failure_signature = signature
        self._failure_signature_counts[signature] += 1
        if (
            self._failure_signature_counts[signature]
            > self.config.duplicate_failure_limit
        ):
            branch.expansion_outcome = "stopped_duplicate_failure"
            return

        available_global = (
            self.config.max_total_branches - len(self._branches) - len(self._frontier)
        )
        available_checkpoint = (
            self.config.max_attempts_per_checkpoint
            - self._checkpoint_attempts[checkpoint_id]
        )
        child_count = min(
            self.config.max_children_per_fork,
            available_global,
            available_checkpoint,
        )
        if child_count <= 0:
            branch.expansion_outcome = "stopped_branch_budget"
            return

        selected_index = branch.active_checkpoint_ids.index(checkpoint_id)
        inherited = tuple(branch.active_checkpoint_ids[: selected_index + 1])
        parent_fork_key = (branch.branch_id, checkpoint_id)
        for _ in range(child_count):
            attempt = self._next_attempt[parent_fork_key]
            self._next_attempt[parent_fork_key] += 1
            child_id = _stable_id(
                "branch",
                self._episode_id,
                branch.branch_id,
                checkpoint_id,
                int(attempt),
            )
            variant = self.action_sampler.branch_variant(attempt, is_root=False)
            self._frontier.append(
                _BranchJob(
                    branch_id=child_id,
                    parent_branch_id=branch.branch_id,
                    fork_checkpoint_id=checkpoint_id,
                    fork_step=int(selection.selected_step),
                    depth=branch.depth + 1,
                    attempt_index=int(attempt),
                    action_variant=variant,
                    inherited_checkpoint_ids=inherited,
                )
            )
            branch.children.append(child_id)
            self._checkpoint_attempts[checkpoint_id] += 1
        branch.expansion_outcome = "children_enqueued"


__all__ = [
    "ActionSample",
    "BranchActionSampler",
    "BranchRecord",
    "BranchStatus",
    "CheckpointRecord",
    "MonteCarloConfig",
    "MonteCarloCoordinator",
    "MonteCarloResult",
    "RollbackCandidate",
    "RollbackSelection",
    "StructuredActionChunkSampler",
    "StructuredActionSamplerConfig",
    "TransitionRecord",
]
