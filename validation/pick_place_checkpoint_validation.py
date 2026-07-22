"""Validate production checkpoint config, task truth and deterministic replay.

Synthetic observation predicates are reported separately. No production
result is obtained by changing configured checkpoint intervals or injecting
sub-event flags through ``info``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from libero_rollback_tool import (
    EnvRollbackTool,
    capture_trace_point,
    compare_checkpoint_to_env,
    compare_replay_traces,
    get_observation,
    make_action,
    make_libero_env,
    step_env,
)
from pick_place_checkpoint import (
    PICK_PLACE_EVENTS,
    PickPlaceRiskAwareCheckpointManager,
    PickPlaceSubEventDetector,
    PickPlaceTaskSpec,
    extract_pick_place_task_truth,
    load_pick_place_config,
)


def synthetic_event_detector_check() -> dict:
    """Exercise geometry fallback only; this is not a production label test."""

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
        require_grasp_truth=False,
        require_place_truth=False,
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
    return {
        "scope": "synthetic_geometry_fallback_only",
        "production_evidence": False,
        "transitions": transitions,
    }


def real_bddl_contact_restore_check(task_spec: PickPlaceTaskSpec) -> dict:
    """Build a real MuJoCo contact state and restore it after contact removal."""

    env, obs, _ = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=False,
    )
    tool = EnvRollbackTool()
    try:
        initial_truth = extract_pick_place_task_truth(env, task_spec)
        initial_center_distance = float(
            np.linalg.norm(
                obs[f"{task_spec.target_object}_pos"]
                - obs[f"{task_spec.goal_object}_pos"]
            )
        )
        assert initial_center_distance < 0.15
        assert initial_truth.target_goal_contact is False
        assert initial_truth.target_on_goal is False
        assert initial_truth.task_success is False

        # Even a geometrically tempting observation must not produce a place
        # label when LIBERO's contact/On truth is false.
        false_positive_obs = {
            key: np.array(value, copy=True)
            if isinstance(value, np.ndarray)
            else value
            for key, value in obs.items()
        }
        false_positive_obs[f"{task_spec.target_object}_pos"] = np.array(
            false_positive_obs[f"{task_spec.goal_object}_pos"], copy=True
        )
        truth_gate_detector = PickPlaceSubEventDetector(task_spec)
        truth_gate_detector.restore_state(
            {
                "completed_events": list(PICK_PLACE_EVENTS[:3]),
                "confirmation_count": 0,
                "initial_target_height": float(
                    false_positive_obs[f"{task_spec.target_object}_pos"][2]
                ),
            },
            false_positive_obs,
        )
        assert all(
            truth_gate_detector.update(
                false_positive_obs, step=gate_step, task_truth=initial_truth
            )
            is None
            for gate_step in (1, 2)
        )

        core = env.env
        goal_pos = np.asarray(obs[f"{task_spec.goal_object}_pos"], dtype=np.float64)
        target_joint = core.get_object(task_spec.target_object).joints[-1]
        on_goal_qpos = np.asarray(
            [goal_pos[0], goal_pos[1], goal_pos[2] + 0.05, 1.0, 0.0, 0.0, 0.0]
        )
        core.sim.data.set_joint_qpos(target_joint, on_goal_qpos)
        core.sim.forward()
        done = False
        for step in range(1, 41):
            obs, reward, done, info = step_env(
                env, np.zeros(7, dtype=np.float32)
            )
            truth = extract_pick_place_task_truth(env, task_spec, info=info)
            if truth.target_on_goal and truth.task_success:
                break
        else:
            raise AssertionError("could not create a stable BDDL On contact state")

        checkpoint = tool.capture(env, step=step, obs=obs)
        contact_truth = extract_pick_place_task_truth(env, task_spec)
        assert contact_truth.target_goal_contact is True
        assert contact_truth.target_on_goal is True
        assert contact_truth.task_success is True
        assert done

        true_event_detector = PickPlaceSubEventDetector(task_spec)
        true_event_detector.restore_state(
            {
                "completed_events": list(PICK_PLACE_EVENTS[:3]),
                "confirmation_count": 0,
                "initial_target_height": float(
                    obs[f"{task_spec.target_object}_pos"][2]
                ),
            },
            obs,
        )
        truth_transitions = []
        for truth_step in range(4):
            transition = true_event_detector.update(
                obs, step=truth_step, task_truth=contact_truth
            )
            if transition is not None:
                truth_transitions.append(transition.event_name)
        assert truth_transitions == ["place_completed", "release_completed"]

        target_pos = np.asarray(obs[f"{task_spec.target_object}_pos"], dtype=np.float64)
        off_goal_qpos = np.asarray(
            [
                target_pos[0] + 0.30,
                target_pos[1],
                target_pos[2] + 0.15,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        core.sim.data.set_joint_qpos(target_joint, off_goal_qpos)
        core.sim.forward()
        removed_truth = extract_pick_place_task_truth(env, task_spec)
        assert removed_truth.target_on_goal is False

        restore_report = tool.restore(env, checkpoint)
        restored_truth = extract_pick_place_task_truth(env, task_spec)
        assert restore_report["success"], restore_report
        assert restored_truth.target_goal_contact is True
        assert restored_truth.target_on_goal is True
        assert restored_truth.task_success is True
        return {
            "scope": "real_mujoco_contact_and_libero_bddl_truth",
            "initial_center_distance": initial_center_distance,
            "initial_truth": initial_truth.as_dict(),
            "contact_truth": contact_truth.as_dict(),
            "truth_gated_transitions": truth_transitions,
            "removed_truth": removed_truth.as_dict(),
            "restored_truth": restored_truth.as_dict(),
            "restore_report": restore_report,
        }
    finally:
        env.close()


def main() -> None:
    synthetic_result = synthetic_event_detector_check()
    task_spec, production_policies = load_pick_place_config(
        Path("configs/pick_place/libero_spatial_task0.json")
    )
    assert tuple(policy.to_event for policy in production_policies) == PICK_PLACE_EVENTS
    assert not task_spec.allow_info_event_overrides
    assert task_spec.require_grasp_truth and task_spec.require_place_truth

    manager = PickPlaceRiskAwareCheckpointManager(
        task_spec,
        edge_policies=production_policies,
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
    failure_step = 25
    warning_onset_step = 23
    replay_end = 35
    failure_fixture = None
    try:
        manager.observe(
            env, 0, obs, info={}, metadata=env_info, rng=action_rng
        )
        for step in range(1, replay_end + 1):
            action = make_action(env, "random", action_rng, 0.012, step - 1)
            actions_by_step[step] = np.asarray(action, dtype=np.float32).copy()
            obs, reward, done, info = step_env(env, action)
            reference_by_step[step] = capture_trace_point(
                env, obs, reward, done
            )
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
                    metadata={"checkpoint_kind": "validation_branch_boundary"},
                )
            if done:
                raise RuntimeError(f"LIBERO terminated early at step {step}")

        assert not manager.detector.transitions, manager.detector.transitions
        assert failure_fixture is not None
        keyframes_before_rollback = [
            keyframe.step for keyframe in manager.selector.keyframes
        ]

        manager.selector.tool.restore(env, failure_fixture, rng=action_rng)
        bad_branch = []
        bad_rng = np.random.default_rng(999)
        for branch_step in range(4):
            bad_action = make_action(
                env, "sweep", bad_rng, 0.28, failure_step + branch_step
            )
            bad_obs, bad_reward, bad_done, _ = step_env(env, bad_action)
            bad_branch.append(
                capture_trace_point(env, bad_obs, bad_reward, bad_done)
            )

        selected, restore_report = manager.restore_for_failure(
            env,
            failure_step,
            strategy="latest_safe",
            warning_onset_step=warning_onset_step,
            safety_margin_steps=1,
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

        assert selected.step < warning_onset_step - 1
        assert restore_report["success"], restore_report
        assert initial_report["comparison_valid"]
        assert initial_report["integration_state_max_abs_diff"] <= 1e-10
        assert replay_report["success"], replay_report
        assert all(
            keyframe.step <= selected.step
            for keyframe in manager.selector.keyframes
        )
        production_result = {
            "scope": "production_config_real_observations_no_event_injection",
            "production_config_unchanged": True,
            "event_overrides_enabled": task_spec.allow_info_event_overrides,
            "detected_event_steps": {},
            "keyframe_steps_before_rollback": keyframes_before_rollback,
            "selected_keyframe": selected.to_summary(),
            "bad_branch_steps_preserved": len(bad_branch),
            "restore_report": restore_report,
            "initial_restore_report": initial_report,
            "replay_report": replay_report,
        }
    finally:
        env.close()

    contact_result = real_bddl_contact_restore_check(task_spec)
    report = {
        "validation_passed": True,
        "synthetic_event_predicates": synthetic_result,
        "production_checkpoint_replay": production_result,
        "real_task_truth": contact_result,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
