"""Validate rule-based pick-and-place failure detection only.

No checkpoint capture, selection, restore, or action replay is used here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from libero_rollback_tool import make_libero_env, step_env
from pick_place_checkpoint import load_pick_place_config
from pick_place_failure import (
    FailureStatus,
    PickPlaceFailureConfig,
    PickPlaceRuleFailureDetector,
)


def make_obs(
    *,
    target=(0.4, 0.0, 1.0),
    goal=(0.8, 0.0, 1.0),
    eef=(0.38, 0.0, 1.0),
    gripper=(0.02, -0.02),
):
    return {
        "akita_black_bowl_1_pos": np.asarray(target, dtype=np.float64),
        "plate_1_pos": np.asarray(goal, dtype=np.float64),
        "robot0_eef_pos": np.asarray(eef, dtype=np.float64),
        "robot0_gripper_qpos": np.asarray(gripper, dtype=np.float64),
    }


def compact_config(**changes):
    base = PickPlaceFailureConfig(
        history_window=4,
        confirmation_steps=2,
        approach_timeout=30,
        grasp_timeout=20,
        lift_timeout=20,
        place_timeout=20,
        release_timeout=20,
        below_initial_height_margin=0.50,
    )
    return dataclasses.replace(base, **changes)


def run_case(detector, observations, completed_events, *, done=False, info=None):
    decisions = []
    for step, obs in enumerate(observations):
        decisions.append(
            detector.update(
                obs,
                step=step,
                completed_events=completed_events,
                action=np.zeros(7, dtype=np.float32),
                done=done and step == len(observations) - 1,
                info=info if step == len(observations) - 1 else None,
            )
        )
    return decisions


def main() -> None:
    task_spec, _ = load_pick_place_config(
        Path("configs/pick_place/libero_spatial_task0.json")
    )
    results = {}

    normal_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    normal_observations = [
        make_obs(eef=(x, 0.0, 1.0))
        for x in (0.0, 0.08, 0.16, 0.24, 0.30, 0.34)
    ]
    normal_decisions = run_case(normal_detector, normal_observations, ())
    assert all(decision.status == FailureStatus.NORMAL for decision in normal_decisions)
    results["normal_approach"] = normal_decisions[-1].to_summary()

    stall_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    stall_decisions = run_case(
        stall_detector,
        [make_obs(eef=(0.0, 0.0, 1.0)) for _ in range(7)],
        (),
    )
    assert any(decision.status == FailureStatus.WARNING for decision in stall_decisions)
    assert stall_decisions[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert stall_decisions[-1].reason == "approach_stalled"
    results["approach_stalled"] = stall_decisions[-1].to_summary()

    grasp_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    missed_grasp = run_case(
        grasp_detector,
        [
            make_obs(eef=(0.0, 0.0, 1.0), gripper=(0.004, -0.004))
            for _ in range(3)
        ],
        ("approach_object",),
    )
    assert missed_grasp[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert missed_grasp[-1].reason == "missed_grasp"
    results["missed_grasp"] = missed_grasp[-1].to_summary()

    lift_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    no_lift = run_case(
        lift_detector,
        [make_obs(eef=(0.39, 0.0, 1.0)) for _ in range(7)],
        ("approach_object", "grasp_completed"),
    )
    assert no_lift[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert no_lift[-1].reason == "lift_no_progress"
    results["lift_no_progress"] = no_lift[-1].to_summary()

    drop_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    dropped = run_case(
        drop_detector,
        [
            make_obs(target=(0.4, 0.0, 1.10), eef=(0.4, 0.0, 1.12)),
            make_obs(target=(0.4, 0.0, 1.04), eef=(0.4, 0.0, 1.35)),
        ],
        ("approach_object", "grasp_completed", "object_lifted"),
    )
    assert dropped[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert dropped[-1].reason == "object_dropped"
    assert dropped[-1].hard_failure
    results["object_dropped"] = dropped[-1].to_summary()

    lowering_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    controlled_lowering = run_case(
        lowering_detector,
        [
            make_obs(target=(0.4, 0.0, 1.10), eef=(0.4, 0.0, 1.12)),
            make_obs(target=(0.4, 0.0, 1.04), eef=(0.4, 0.0, 1.06)),
        ],
        ("approach_object", "grasp_completed", "object_lifted"),
    )
    assert all(decision.status == FailureStatus.NORMAL for decision in controlled_lowering)
    results["controlled_lowering_not_drop"] = controlled_lowering[-1].to_summary()

    place_detector = PickPlaceRuleFailureDetector(
        task_spec,
        compact_config(place_timeout=5, min_goal_progress=0.0),
    )
    placement_timeout = run_case(
        place_detector,
        [make_obs(target=(0.2, 0.0, 1.1)) for _ in range(6)],
        ("approach_object", "grasp_completed", "object_lifted"),
    )
    assert placement_timeout[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert placement_timeout[-1].reason == "placement_timeout"
    results["placement_timeout"] = placement_timeout[-1].to_summary()

    release_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    bad_release = run_case(
        release_detector,
        [
            make_obs(target=(0.2, 0.0, 1.0), gripper=(0.03, -0.03))
            for _ in range(3)
        ],
        (
            "approach_object",
            "grasp_completed",
            "object_lifted",
            "place_completed",
        ),
    )
    assert bad_release[-1].status == FailureStatus.RECOVERABLE_FAILURE
    assert bad_release[-1].reason == "released_outside_goal"
    results["released_outside_goal"] = bad_release[-1].to_summary()

    invalid_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    invalid = invalid_detector.update(
        make_obs(target=(np.nan, 0.0, 1.0)),
        step=0,
    )
    assert invalid.status == FailureStatus.TERMINAL_FAILURE
    assert invalid.reason == "invalid_observation"
    results["invalid_observation"] = invalid.to_summary()

    done_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    terminal = done_detector.update(make_obs(), step=0, done=True, info={})
    assert terminal.status == FailureStatus.TERMINAL_FAILURE
    assert terminal.reason == "episode_terminated_without_success"
    results["terminal_without_success"] = terminal.to_summary()

    sim_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    simulation_error = sim_detector.update(
        {}, step=0, info={"sim_error": True}
    )
    assert simulation_error.status == FailureStatus.TERMINAL_FAILURE
    assert simulation_error.reason == "simulation_error"
    results["simulation_error"] = simulation_error.to_summary()

    collision_detector = PickPlaceRuleFailureDetector(task_spec, compact_config())
    collision = collision_detector.update(
        make_obs(),
        step=0,
        info={"collision": True, "failure_reason": "collision"},
    )
    assert collision.status == FailureStatus.RECOVERABLE_FAILURE
    assert collision.reason == "collision"
    results["explicit_collision"] = collision.to_summary()

    # Real LIBERO smoke check: valid observations and a short free-space prefix
    # must not produce a terminal or recoverable failure.  Long-window progress
    # rules are disabled here because zero/random actions are not a real policy.
    real_config = dataclasses.replace(
        compact_config(),
        history_window=50,
        approach_timeout=500,
    )
    real_detector = PickPlaceRuleFailureDetector(task_spec, real_config)
    env, obs, env_info = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=False,
    )
    real_decisions = []
    try:
        real_decisions.append(real_detector.update(obs, step=0))
        for step in range(1, 21):
            obs, reward, done, info = step_env(
                env, np.zeros(7, dtype=np.float32)
            )
            real_decisions.append(
                real_detector.update(
                    obs,
                    step=step,
                    reward=reward,
                    done=done,
                    info=info,
                )
            )
    finally:
        env.close()
    assert all(not decision.failed for decision in real_decisions)
    results["real_libero_smoke"] = {
        "environment": env_info,
        "steps": len(real_decisions),
        "decision_counts": real_detector.summary()["decision_counts"],
    }

    report = {
        "success": True,
        "scope": "failure_detection_only_no_checkpoint_restore_or_replay",
        "cases": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
