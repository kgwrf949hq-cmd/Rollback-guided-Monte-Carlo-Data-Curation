"""Validate pick-and-place event checkpoints and risk-aware replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from libero_rollback_tool import (
    capture_trace_point,
    compare_checkpoint_to_env,
    compare_replay_traces,
    get_observation,
    make_action,
    make_libero_env,
    step_env,
)
from pick_place_checkpoint import (
    EventEdgePolicy,
    PICK_PLACE_EVENTS,
    PickPlaceRiskAwareCheckpointManager,
    PickPlaceSubEventDetector,
    PickPlaceTaskSpec,
    load_pick_place_config,
)


def synthetic_event_detector_check() -> None:
    spec = PickPlaceTaskSpec(
        target_object="cube",
        goal_object="plate",
        approach_distance=0.20,
        grasp_distance=0.08,
        gripper_closed_aperture=0.02,
        gripper_open_aperture=0.06,
        lift_height=0.05,
        place_distance=0.08,
        place_height_tolerance=0.08,
        stable_object_motion=0.10,
        confirmation_steps=2,
    )
    detector = PickPlaceSubEventDetector(spec)

    def obs(eef, target, goal, gripper):
        return {
            "robot0_eef_pos": np.asarray(eef, dtype=np.float64),
            "robot0_gripper_qpos": np.asarray(gripper, dtype=np.float64),
            "cube_pos": np.asarray(target, dtype=np.float64),
            "plate_pos": np.asarray(goal, dtype=np.float64),
        }

    states = [
        obs([0.0, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
        obs([0.25, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
        obs([0.25, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
        obs([0.38, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.38, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.38, 0.0, 1.0], [0.4, 0.0, 1.08], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.38, 0.0, 1.0], [0.4, 0.0, 1.08], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.78, 0.0, 1.0], [0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.78, 0.0, 1.0], [0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.005, -0.005]),
        obs([0.78, 0.0, 1.0], [0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
        obs([0.78, 0.0, 1.0], [0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
        obs([0.78, 0.0, 1.0], [0.8, 0.0, 1.0], [0.8, 0.0, 1.0], [0.04, -0.04]),
    ]
    transitions = []
    for step, state in enumerate(states):
        transition = detector.update(state, step=step)
        if transition is not None:
            transitions.append(transition.event_name)
    assert transitions == list(PICK_PLACE_EVENTS), transitions


def main() -> None:
    synthetic_event_detector_check()

    task_spec, loaded_policies = load_pick_place_config(
        Path("configs/pick_place/libero_spatial_task0.json")
    )
    assert tuple(policy.to_event for policy in loaded_policies) == PICK_PLACE_EVENTS
    edge_policies = (
        EventEdgePolicy(None, "approach_object", 0.10, False, 2, 30),
        EventEdgePolicy("approach_object", "grasp_completed", 1.00, True, 1, 2),
        EventEdgePolicy("grasp_completed", "object_lifted", 1.00, True, 1, 2),
        EventEdgePolicy("object_lifted", "place_completed", 0.40, True, 3, 15),
        EventEdgePolicy("place_completed", "release_completed", 1.00, True, 1, 2),
    )
    manager = PickPlaceRiskAwareCheckpointManager(
        task_spec,
        selector_config=None,
        edge_policies=edge_policies,
    )
    env, obs, env_info = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=False,
    )

    action_rng = np.random.default_rng(321)
    actions_by_step = {}
    reference_by_step = {}
    failure_step = 15
    replay_end = 30
    failure_fixture = None
    try:
        manager.observe(env, 0, obs, info={}, metadata=env_info)
        scheduled_events = {
            8: "approach_object",
            12: "grasp_completed",
            16: "object_lifted",
            26: "place_completed",
            28: "release_completed",
        }
        expected_transition_steps = {
            event_name: step for step, event_name in scheduled_events.items()
        }
        for step in range(1, replay_end + 1):
            action = make_action(env, "random", action_rng, 0.012, step - 1)
            actions_by_step[step] = np.asarray(action, dtype=np.float32).copy()
            obs, reward, done, _ = step_env(env, action)
            info = {}
            if step in scheduled_events:
                info[f"{scheduled_events[step]}_completed"] = True
            reference_by_step[step] = capture_trace_point(env, obs, reward, done)
            manager.observe(
                env,
                step,
                obs,
                reward,
                done,
                info,
                action=action,
                rng=action_rng,
                metadata=env_info,
            )
            if step == failure_step:
                failure_fixture = manager.selector.tool.capture(
                    env,
                    step=step,
                    obs=obs,
                    rng=action_rng,
                    metadata={
                        "checkpoint_kind": "validation_failure_fixture",
                        "completed_event_count": manager.detector.current_event_index,
                    },
                )
            if done:
                raise RuntimeError(f"LIBERO terminated early at step {step}")

        transitions = manager.detector.transitions
        transition_steps = {transition.event_name: transition.step for transition in transitions}
        assert transition_steps == expected_transition_steps, transition_steps
        assert failure_fixture is not None

        # Put the simulator at the true failure boundary, inject a bad branch,
        # then recover using only manager-selected checkpoints.
        manager.selector.tool.restore(env, failure_fixture, rng=action_rng)
        bad_branch = []
        bad_rng = np.random.default_rng(999)
        for branch_step in range(4):
            bad_action = make_action(env, "sweep", bad_rng, 0.28, failure_step + branch_step)
            bad_obs, bad_reward, bad_done, _ = step_env(env, bad_action)
            bad_branch.append(capture_trace_point(env, bad_obs, bad_reward, bad_done))

        selected, restore_report = manager.restore_for_failure(
            env,
            failure_step,
            rng=action_rng,
        )
        restored_obs = get_observation(env, force_update=False)
        initial_report = compare_checkpoint_to_env(
            selected.checkpoint, env, restored_obs
        )
        reference_trace = [
            reference_by_step[step]
            for step in range(selected.step + 1, replay_end + 1)
        ]
        replay_trace = []
        for step in range(selected.step + 1, replay_end + 1):
            replay_obs, replay_reward, replay_done, _ = step_env(
                env, actions_by_step[step]
            )
            replay_trace.append(
                capture_trace_point(env, replay_obs, replay_reward, replay_done)
            )
            if replay_done:
                raise RuntimeError(f"Replay terminated at step {step}")
        replay_report = compare_replay_traces(
            reference_trace,
            replay_trace,
            tolerance=1e-10,
            observation_tolerance=1e-10,
        )

        keyframes = manager.selector.keyframes
        low_phase_steps = [keyframe.step for keyframe in keyframes if 0 < keyframe.step < 8]
        high_phase_steps = [keyframe.step for keyframe in keyframes if 8 < keyframe.step < 12]
        report = {
            "success": bool(
                transition_steps == expected_transition_steps
                and not low_phase_steps
                and high_phase_steps
                and selected.step < failure_step
                and initial_report["integration_state_max_abs_diff"] <= 1e-10
                and replay_report["success"]
            ),
            "environment": env_info,
            "scheduled_event_steps": expected_transition_steps,
            "detected_event_steps": transition_steps,
            "keyframe_steps": [keyframe.step for keyframe in keyframes],
            "low_risk_approach_checkpoints": low_phase_steps,
            "high_risk_grasp_checkpoints": high_phase_steps,
            "selected_keyframe": selected.to_summary(),
            "bad_branch_steps": len(bad_branch),
            "restore_report": restore_report,
            "initial_restore_report": initial_report,
            "replay_report": replay_report,
            "manager_summary": manager.summary(),
        }
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if not report["success"]:
            raise AssertionError("pick-place checkpoint validation failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
