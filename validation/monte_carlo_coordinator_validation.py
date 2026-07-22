#!/usr/bin/env python3
"""Deterministic validation for the in-memory Monte Carlo coordinator.

This test intentionally uses no LIBERO installation and no policy server.  A
small stateful environment exercises full-horizon BFS, rollback lineage,
curation-after-success, detector state restoration, restore failure isolation,
and deterministic summaries.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from monte_carlo_coordinator import (
    ActionSample,
    BranchActionSampler,
    BranchStatus,
    MonteCarloConfig,
    MonteCarloCoordinator,
    StructuredActionChunkSampler,
    StructuredActionSamplerConfig,
)
from libero_rollback_tool import make_branch_rng
from pick_place_checkpoint import (
    PickPlaceTaskSpec,
    default_pick_place_edge_policies,
)
from pick_place_failure import (
    FailureDecision,
    FailureStatus,
    PickPlaceFailureConfig,
    PickPlaceRuleFailureDetector,
)
from semantic_checkpoint import SemanticKeyframeConfig


@dataclasses.dataclass
class _FakeCheckpoint:
    step: int
    env_done: bool
    obs: Dict[str, Any]
    metadata: Dict[str, Any]
    state: Dict[str, Any]


class _FakeRollbackTool:
    def __init__(self, fail_restore_calls: Sequence[int] = ()) -> None:
        self.fail_restore_calls = set(int(item) for item in fail_restore_calls)
        self.restore_calls = 0
        self.restored_states = []

    def capture(
        self,
        env,
        *,
        step,
        obs=None,
        metadata=None,
        rng=None,
        rollout_state=None,
    ):
        del rng, rollout_state
        return _FakeCheckpoint(
            step=int(step),
            env_done=bool(env.done),
            obs=copy.deepcopy(dict(obs or {})),
            metadata=dict(metadata or {}),
            state=env.snapshot(),
        )

    def restore(
        self,
        env,
        checkpoint,
        *,
        reload_xml=False,
        rng=None,
        restore_rng=True,
    ):
        del reload_xml, rng
        self.restore_calls += 1
        if self.restore_calls in self.fail_restore_calls:
            return {
                "success": False,
                "restore_call": self.restore_calls,
                "restore_rng": bool(restore_rng),
            }
        env.restore(checkpoint.state)
        self.restored_states.append(env.snapshot())
        return {
            "success": True,
            "restore_call": self.restore_calls,
            "restore_rng": bool(restore_rng),
            "restored_step": int(checkpoint.step),
        }


class _FakePickPlaceEnv:
    def __init__(
        self,
        *,
        terminal_step: Optional[int] = None,
        trusted_truth: bool = True,
    ) -> None:
        self.eef_x = 0.0
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self.terminal_step = terminal_step
        self.trusted_truth = bool(trusted_truth)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "eef_x": float(self.eef_x),
            "timestep": int(self.timestep),
            "cur_time": float(self.cur_time),
            "done": bool(self.done),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.eef_x = float(state["eef_x"])
        self.timestep = int(state["timestep"])
        self.cur_time = float(state["cur_time"])
        self.done = bool(state["done"])

    def _get_observations(self, force_update=False):
        del force_update
        target = np.asarray([0.5, 0.0, 1.0], dtype=np.float64)
        eef = np.asarray([self.eef_x, 0.0, 1.0], dtype=np.float64)
        return {
            "target_pos": target,
            "goal_pos": np.asarray([0.8, 0.0, 1.0], dtype=np.float64),
            "robot0_eef_pos": eef,
            "robot0_gripper_qpos": np.asarray([0.02, -0.02]),
            "target_to_robot0_eef_pos": target - eef,
        }

    def check_success(self) -> bool:
        if not self.trusted_truth:
            raise RuntimeError("fixture disables the trusted environment truth API")
        return bool(self.eef_x >= 0.70)

    def step(self, action):
        value = np.asarray(action, dtype=np.float64).reshape(-1)
        self.eef_x += float(value[0])
        self.timestep += 1
        self.cur_time += 0.05
        success = bool(self.eef_x >= 0.70)
        forced_terminal = (
            self.terminal_step is not None and self.timestep >= self.terminal_step
        )
        self.done = bool(success or forced_terminal)
        info = {"task_success": True} if success else {}
        return self._get_observations(), float(success), self.done, info


class _ScriptedBranchSampler(BranchActionSampler):
    """Root and attempt 1 stall; attempts 0/2 make task progress."""

    def branch_variant(self, attempt_index: int, *, is_root: bool = False) -> str:
        if is_root:
            return "root_stall"
        return ("replan_progress", "replan_stall", "replan_progress_alt")[
            min(int(attempt_index), 2)
        ]

    def start_branch(self, branch, rng):
        del branch, rng
        return {"step": 0}

    def next_action(self, obs, branch, branch_step, rng, state):
        del obs, branch_step
        state["step"] += 1
        rng_probe = float(rng.uniform(0.0, 1.0))
        if (
            branch.action_variant == "root_stall"
            or branch.action_variant == "replan_stall"
        ):
            delta = 0.0
        elif branch.action_variant == "replan_progress":
            delta = 0.25 + 0.001 * rng_probe
        else:
            delta = 0.30 + 0.001 * rng_probe
        action = np.zeros(7, dtype=np.float32)
        action[0] = delta
        action[-1] = -1.0
        return ActionSample(
            policy_action=action.copy(),
            executed_action=action.copy(),
            source=branch.action_variant,
            metadata={
                "rng_probe": rng_probe,
                "environment_perturbed": False,
            },
        )


class _HorizonSampler(BranchActionSampler):
    def next_action(self, obs, branch, branch_step, rng, state):
        del obs, branch, branch_step, rng, state
        action = np.zeros(7, dtype=np.float32)
        action[0] = 0.02
        action[-1] = -1.0
        return ActionSample(
            policy_action=action.copy(),
            executed_action=action.copy(),
            source="steady_progress",
            metadata={"environment_perturbed": False},
        )


TASK_SPEC = PickPlaceTaskSpec(
    target_object="target",
    goal_object="goal",
    confirmation_steps=1,
    require_grasp_truth=False,
    require_place_truth=False,
)


def _failure_config() -> PickPlaceFailureConfig:
    return PickPlaceFailureConfig(
        history_window=3,
        confirmation_steps=2,
        min_approach_progress=0.01,
        approach_timeout=20,
        grasp_timeout=20,
        lift_timeout=20,
        place_timeout=20,
        release_timeout=20,
        below_initial_height_margin=0.50,
    )


def _dense_edge_policies():
    return tuple(
        dataclasses.replace(policy, min_interval=0, max_interval=1)
        for policy in default_pick_place_edge_policies()
    )


def _run(
    *,
    seed=19,
    config=None,
    fail_restore_calls=(),
    terminal_step=None,
    trusted_truth=True,
    action_sampler=None,
):
    env = _FakePickPlaceEnv(
        terminal_step=terminal_step,
        trusted_truth=trusted_truth,
    )
    tool = _FakeRollbackTool(fail_restore_calls)
    coordinator = MonteCarloCoordinator(
        TASK_SPEC,
        action_sampler or _ScriptedBranchSampler(),
        config=config
        or MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=7,
            max_children_per_fork=3,
            max_attempts_per_checkpoint=6,
            max_depth=5,
            safety_margin_steps=1,
            duplicate_failure_limit=2,
        ),
        rollback_tool=tool,
        selector_config=SemanticKeyframeConfig(
            min_interval=0,
            max_interval=1,
            max_keyframes=32,
        ),
        edge_policies=_dense_edge_policies(),
        failure_config=_failure_config(),
    )
    result = coordinator.run(
        env,
        env._get_observations(),
        episode_id="fixture-episode-0",
        root_seed=seed,
    )
    return coordinator, result, tool


def _validate_detector_snapshot() -> Dict[str, Any]:
    detector = PickPlaceRuleFailureDetector(TASK_SPEC, _failure_config())
    env = _FakePickPlaceEnv()
    decisions = []
    for step in range(3):
        decisions.append(detector.update(env._get_observations(), step=step))
    assert decisions[-1].status == FailureStatus.WARNING
    snapshot = detector.snapshot_state()
    json.dumps(snapshot, allow_nan=False)

    expected = detector.update(env._get_observations(), step=3)
    restored = PickPlaceRuleFailureDetector(TASK_SPEC, _failure_config())
    restored.restore_state(snapshot)
    actual = restored.update(env._get_observations(), step=3)
    assert actual.to_summary() == expected.to_summary()
    assert len(restored.decisions) == 1
    return {
        "warning_onset_step": actual.warning_onset_step,
        "consecutive_steps": actual.consecutive_steps,
        "status": actual.status.value,
    }


def _validate_structured_action_sampling(branches, obs) -> Dict[str, Any]:
    policy_calls = []

    def policy_infer(_obs, sampling_seed):
        policy_calls.append(int(sampling_seed))
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -0.75
        return [action.copy(), action.copy()]

    sampler = StructuredActionChunkSampler(
        policy_infer,
        StructuredActionSamplerConfig(
            replan_steps=2,
            native_sampling=False,
            low_noise_std=0.03,
            medium_noise_std=0.06,
        ),
    )
    baseline = dataclasses.replace(
        branches[0], action_variant=sampler.branch_variant(0)
    )
    perturbed = dataclasses.replace(
        branches[1], action_variant=sampler.branch_variant(1)
    )
    baseline_rng = make_branch_rng(baseline.root_seed, baseline.branch_id)
    perturbed_rng = make_branch_rng(perturbed.root_seed, perturbed.branch_id)
    baseline_state = sampler.start_branch(baseline, baseline_rng)
    perturbed_state = sampler.start_branch(perturbed, perturbed_rng)
    baseline_action = sampler.next_action(
        obs, baseline, 0, baseline_rng, baseline_state
    )
    perturbed_action = sampler.next_action(
        obs, perturbed, 0, perturbed_rng, perturbed_state
    )
    assert (
        baseline_action.metadata["policy_sampling_seed"]
        == perturbed_action.metadata["policy_sampling_seed"]
    )
    assert np.array_equal(baseline_action.policy_action, perturbed_action.policy_action)
    assert np.array_equal(
        baseline_action.executed_action, baseline_action.policy_action
    )
    assert not np.array_equal(
        perturbed_action.executed_action[:6], perturbed_action.policy_action[:6]
    )
    assert perturbed_action.executed_action[-1] == perturbed_action.policy_action[-1]

    repeat_rng = make_branch_rng(perturbed.root_seed, perturbed.branch_id)
    repeat_state = sampler.start_branch(perturbed, repeat_rng)
    repeated_action = sampler.next_action(obs, perturbed, 0, repeat_rng, repeat_state)
    assert repeated_action.to_summary() == perturbed_action.to_summary()

    native_sampler = StructuredActionChunkSampler(
        policy_infer,
        StructuredActionSamplerConfig(replan_steps=2, native_sampling=True),
    )
    native_a = dataclasses.replace(
        branches[0], action_variant=native_sampler.branch_variant(0)
    )
    native_b = dataclasses.replace(
        branches[1], action_variant=native_sampler.branch_variant(1)
    )
    native_a_rng = make_branch_rng(native_a.root_seed, native_a.branch_id)
    native_b_rng = make_branch_rng(native_b.root_seed, native_b.branch_id)
    native_action_a = native_sampler.next_action(
        obs,
        native_a,
        0,
        native_a_rng,
        native_sampler.start_branch(native_a, native_a_rng),
    )
    native_action_b = native_sampler.next_action(
        obs,
        native_b,
        0,
        native_b_rng,
        native_sampler.start_branch(native_b, native_b_rng),
    )
    assert (
        native_action_a.metadata["policy_sampling_seed"]
        != native_action_b.metadata["policy_sampling_seed"]
    )
    assert native_action_a.metadata["action_noise_std"] == 0.0
    assert native_action_b.metadata["action_noise_std"] == 0.0
    return {
        "shared_base_seed_without_native_sampling": True,
        "motion_only_structured_noise": True,
        "gripper_unchanged": True,
        "native_branch_seeds_differ": True,
        "policy_call_count": len(policy_calls),
    }


def main() -> None:
    detector_snapshot = _validate_detector_snapshot()

    coordinator, result, tool = _run()
    summary = result.to_summary()
    json.dumps(summary, sort_keys=True, allow_nan=False)
    branches = [result.branches[item] for item in result.execution_order]
    root = branches[0]

    assert root.status == BranchStatus.RECOVERABLE_FAILURE
    assert root.transitions
    root_transition_count = len(root.transitions)
    assert len(root.children) == 3
    assert root.rollback_selection is not None
    assert root.rollback_selection.strategy == "latest_safe"
    assert root.rollback_selection.selected_step == 0
    assert all(
        not candidate.eligible
        for candidate in root.rollback_selection.candidates
        if candidate.step >= root.rollback_selection.cutoff_step
    )

    # FIFO breadth-first execution: every depth-1 sibling runs before a child
    # spawned by a failed depth-1 branch.
    assert [branch.depth for branch in branches[:4]] == [0, 1, 1, 1]
    depth_one = branches[1:4]
    assert [branch.attempt_index for branch in depth_one] == [0, 1, 2]
    assert all(branch.parent_branch_id == root.branch_id for branch in depth_one)
    assert all(
        branch.fork_checkpoint_id == root.rollback_selection.selected_checkpoint_id
        for branch in depth_one
    )
    action_sampling = _validate_structured_action_sampling(
        depth_one, _FakePickPlaceEnv()._get_observations()
    )

    # Curation continues after the first successful sibling and retains the
    # failed sibling's transitions unchanged.
    assert depth_one[0].status == BranchStatus.SUCCESS
    assert depth_one[1].status == BranchStatus.RECOVERABLE_FAILURE
    assert depth_one[2].status == BranchStatus.SUCCESS
    assert depth_one[0].transitions[0].observation["robot0_eef_pos"][0] == 0.0
    assert depth_one[0].transitions[0].next_observation["robot0_eef_pos"][0] > 0.0
    assert len(root.transitions) == root_transition_count
    assert any(
        branch.execution_index > depth_one[0].execution_index for branch in branches
    )
    assert all(
        transition.action_metadata["environment_perturbed"] is False
        for branch in branches
        for transition in branch.transitions
    )

    # Every successful restore starts from the identical checkpoint state.
    assert tool.restored_states
    assert all(state["eef_x"] == 0.0 for state in tool.restored_states)
    assert all(
        branch.restore_report is None
        or branch.restore_report.get("restore_rng") is False
        for branch in branches
    )
    assert any(
        branch.expansion_outcome == "stopped_duplicate_failure" for branch in branches
    )

    # Warning-less fallback is explicitly temporal and makes no safe claim.
    temporal_decision = FailureDecision(
        step=root.failure_decision.step,
        status=FailureStatus.RECOVERABLE_FAILURE,
        risk_score=1.0,
        reason="fixture_temporal_failure",
        current_stage=root.failure_decision.current_stage,
        consecutive_steps=1,
        hard_failure=True,
        warning_onset_step=None,
        evidence={},
    )
    temporal = coordinator.select_rollback(root, temporal_decision)
    assert temporal.strategy == "latest_before_failure"
    assert temporal.selection_basis == "temporal_only"

    # The same seed produces the same IDs, actions, BFS order, and summaries.
    _, repeated, _ = _run()
    assert repeated.to_summary() == summary

    # A failed restore is quarantined as restore failure and does not mutate
    # the parent's candidate view or prevent remaining siblings from running.
    _, restore_failure_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=4,
            max_children_per_fork=3,
            max_attempts_per_checkpoint=3,
            max_depth=5,
            safety_margin_steps=1,
            duplicate_failure_limit=2,
        ),
        fail_restore_calls=(1,),
    )
    restore_branches = [
        restore_failure_result.branches[item]
        for item in restore_failure_result.execution_order
    ]
    assert restore_branches[1].status == BranchStatus.RESTORE_FAILED
    assert any(branch.status == BranchStatus.SUCCESS for branch in restore_branches[2:])
    assert restore_branches[0].active_checkpoint_ids == root.active_checkpoint_ids

    # No eligible checkpoint and exhausted global budget terminate cleanly.
    _, no_checkpoint_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=4,
            max_children_per_fork=3,
            max_attempts_per_checkpoint=3,
            max_depth=5,
            safety_margin_steps=3,
            duplicate_failure_limit=2,
        )
    )
    no_checkpoint_root = no_checkpoint_result.branches[
        no_checkpoint_result.execution_order[0]
    ]
    assert no_checkpoint_root.expansion_outcome == "no_eligible_checkpoint"
    assert len(no_checkpoint_result.branches) == 1

    _, budget_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=1,
            max_children_per_fork=3,
            max_attempts_per_checkpoint=3,
            max_depth=5,
            safety_margin_steps=1,
            duplicate_failure_limit=2,
        )
    )
    budget_root = budget_result.branches[budget_result.execution_order[0]]
    assert budget_root.expansion_outcome == "stopped_branch_budget"

    # Terminal done-without-success may branch from an earlier checkpoint when
    # configured, while simulation/observation corruption would quarantine.
    _, terminal_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=2,
            max_children_per_fork=1,
            max_attempts_per_checkpoint=1,
            max_depth=5,
            safety_margin_steps=0,
            duplicate_failure_limit=2,
        ),
        terminal_step=1,
    )
    terminal_root = terminal_result.branches[terminal_result.execution_order[0]]
    assert terminal_root.status == BranchStatus.TERMINAL_FAILURE
    assert terminal_root.children

    # Search depth counts fork edges, not trajectory steps: an unfailed root
    # branch runs every step through the configured full episode horizon.
    _, horizon_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=1,
            max_children_per_fork=1,
            max_attempts_per_checkpoint=1,
            max_depth=1,
            safety_margin_steps=1,
            duplicate_failure_limit=2,
        ),
        action_sampler=_HorizonSampler(),
    )
    horizon_branch = horizon_result.branches[horizon_result.execution_order[0]]
    assert horizon_branch.status == BranchStatus.HORIZON_EXHAUSTED
    assert len(horizon_branch.transitions) == 8

    # An info-only success signal is retained for audit but cannot create a
    # positive label under the production truth policy.
    _, untrusted_result, _ = _run(
        config=MonteCarloConfig(
            episode_horizon=8,
            max_total_branches=4,
            max_children_per_fork=3,
            max_attempts_per_checkpoint=3,
            max_depth=5,
            safety_margin_steps=1,
            duplicate_failure_limit=2,
        ),
        trusted_truth=False,
    )
    assert any(
        branch.outcome == "untrusted_success_signal"
        and branch.status == BranchStatus.QUARANTINED
        for branch in untrusted_result.branches.values()
    )

    report = {
        "validation_passed": True,
        "mode": "curation",
        "scheduler": "bounded_fifo_bfs",
        "branch_count": len(result.branches),
        "status_counts": summary["status_counts"],
        "detector_snapshot": detector_snapshot,
        "action_sampling": action_sampling,
        "checks": {
            "full_horizon_branching": True,
            "curation_continues_after_success": True,
            "failure_terminates_current_branch": True,
            "failure_transitions_preserved": True,
            "transition_state_action_alignment": True,
            "latest_safe_strict_cutoff": True,
            "temporal_fallback_not_called_safe": True,
            "bfs_sibling_order": True,
            "branch_rng_repeatable": True,
            "restore_rng_disabled": True,
            "sibling_checkpoint_views_isolated": True,
            "duplicate_failure_stopped": True,
            "restore_failure_isolated": True,
            "environment_not_perturbed": True,
            "summary_json_serializable": True,
            "untrusted_success_quarantined": True,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
