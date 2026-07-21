"""End-to-end validation of semantic keyframe selection and rollback.

The test deliberately corrupts a LIBERO rollout after a selected keyframe,
restores that keyframe, and replays the original actions.  A successful run
proves both policy-level selection (the selected step precedes failure) and
the low-level deterministic restore guarantee.
"""

from __future__ import annotations

import json

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
from semantic_checkpoint import SemanticKeyframeConfig, SemanticKeyframeSelector


def main() -> None:
    env, obs, env_info = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=False,
    )
    selector = SemanticKeyframeSelector(
        config=SemanticKeyframeConfig(
            min_interval=2,
            max_interval=8,
            score_threshold=0.75,
            force_score=3.0,
            motion_threshold=0.005,
            object_motion_threshold=0.005,
        )
    )
    action_rng = np.random.default_rng(123)
    reference_by_step = {}
    actions_by_step = {}
    failure_step = 18
    replay_end = 23
    failure_state_checkpoint = None

    try:
        selector.maybe_capture(env, 0, obs, info={"phase": "initial"})
        for step in range(1, replay_end + 1):
            action = make_action(
                env,
                "random",
                action_rng,
                scale=0.015,
                step_index=step - 1,
            )
            actions_by_step[step] = np.asarray(action, dtype=np.float32).copy()
            obs, reward, done, info = step_env(env, action)
            reference_by_step[step] = capture_trace_point(env, obs, reward, done)
            selector.maybe_capture(
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
                # Test fixture only: this state is deliberately not added to
                # selector.keyframes.  It lets the test produce the reference
                # continuation first and then place the simulator back at the
                # exact failure boundary before injecting bad actions.
                failure_state_checkpoint = selector.tool.capture(
                    env,
                    step=step,
                    obs=obs,
                    rng=action_rng,
                    metadata={"checkpoint_kind": "validation_fixture"},
                )
            if done:
                raise RuntimeError(
                    "LIBERO terminated before the validation horizon; "
                    f"got done at step {step}"
                )

        selected = selector.select_for_rollback(failure_step)
        if selected.step >= failure_step:
            raise AssertionError("rollback selector returned a checkpoint at/after failure")
        if failure_state_checkpoint is None:
            raise AssertionError("failure-state validation fixture was not captured")

        fixture_restore_report = selector.tool.restore(
            env,
            failure_state_checkpoint,
            rng=action_rng,
        )

        # Move the simulator into a deliberately bad branch.  The branch is
        # not evaluated for task quality; it only proves that restore removes
        # its state before replay starts.
        bad_rng = np.random.default_rng(999)
        bad_branch_steps = 4
        bad_branch_trace = []
        for branch_step in range(bad_branch_steps):
            bad_action = make_action(
                env,
                "sweep",
                bad_rng,
                scale=0.30,
                step_index=failure_step + branch_step,
            )
            bad_obs, bad_reward, bad_done, _ = step_env(env, bad_action)
            bad_branch_trace.append(capture_trace_point(env, bad_obs, bad_reward, bad_done))

        restored_keyframe, restore_report = selector.restore_for_failure(
            env,
            failure_step,
            rng=action_rng,
        )
        restored_obs = get_observation(env, force_update=False)
        initial_report = compare_checkpoint_to_env(
            restored_keyframe.checkpoint, env, restored_obs
        )

        reference_trace = [
            reference_by_step[step]
            for step in range(restored_keyframe.step + 1, replay_end + 1)
        ]
        replay_trace = []
        for step in range(restored_keyframe.step + 1, replay_end + 1):
            replay_obs, replay_reward, replay_done, _ = step_env(
                env, actions_by_step[step]
            )
            replay_trace.append(
                capture_trace_point(env, replay_obs, replay_reward, replay_done)
            )
            if replay_done:
                raise RuntimeError(f"replay terminated early at step {step}")

        replay_report = compare_replay_traces(
            reference_trace,
            replay_trace,
            tolerance=1e-10,
            observation_tolerance=1e-10,
        )
        report = {
            "success": bool(
                len(selector.keyframes) >= 2
                and selected.step < failure_step
                and initial_report.get("integration_state_max_abs_diff") <= 1e-10
                and replay_report["success"]
            ),
            "environment": env_info,
            "keyframe_summary": selector.summary(),
            "failure_step": failure_step,
            "selected_keyframe": selected.to_summary(),
            "failure_fixture_restore_report": fixture_restore_report,
            "restore_report": restore_report,
            "initial_restore_report": initial_report,
            "bad_branch_steps": len(bad_branch_trace),
            "replay_report": replay_report,
        }
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        if not report["success"]:
            raise AssertionError("semantic keyframe rollback validation failed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
