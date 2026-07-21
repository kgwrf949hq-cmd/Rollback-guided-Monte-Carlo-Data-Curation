import os

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from libero_rollback_tool import (
    EnvRollbackTool,
    capture_trace_point,
    compare_checkpoint_to_env,
    compare_replay_traces,
    get_observation,
    step_env,
)


def main() -> None:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    try:
        obs = env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(5):
            obs, _, _, _ = step_env(env, np.zeros(7, dtype=np.float32))

        tool = EnvRollbackTool()
        checkpoint = tool.capture(env, step=env.env.timestep, obs=obs)
        actions = [
            np.array([0.02, -0.01, 0.01, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            np.array([-0.01, 0.01, -0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        ] * 4

        reference = []
        for action in actions:
            obs, reward, done, _ = step_env(env, action)
            reference.append(capture_trace_point(env, obs, reward, done))

        restore_report = tool.restore(env, checkpoint, reload_xml=False)
        restored_obs = get_observation(env, force_update=False)
        initial_report = compare_checkpoint_to_env(checkpoint, env, restored_obs)

        replay = []
        for action in actions:
            obs, reward, done, _ = step_env(env, action)
            replay.append(capture_trace_point(env, obs, reward, done))

        replay_report = compare_replay_traces(
            reference,
            replay,
            tolerance=1e-10,
            observation_tolerance=1e-10,
        )
        print("restore_report", restore_report)
        print("initial_report", initial_report)
        print("replay_report", replay_report)
        assert restore_report["state_restore_method"].startswith("mujoco.mj_setState")
        assert restore_report["integration_state_restored"]
        assert initial_report["integration_state_max_abs_diff"] <= 1e-10
        assert replay_report["success"]
    finally:
        env.close()


if __name__ == "__main__":
    main()
