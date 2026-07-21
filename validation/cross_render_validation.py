import numpy as np

from libero_rollback_tool import (
    EnvRollbackTool,
    capture_trace_point,
    compare_checkpoint_to_env,
    compare_replay_traces,
    get_observation,
    make_libero_env,
    step_env,
)


def main():
    env_a, obs_a, _ = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=True,
    )
    env_b, _, _ = make_libero_env(
        benchmark_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        camera_width=64,
        camera_height=64,
        render=True,
    )
    try:
        for _ in range(2):
            obs_a, _, _, _ = step_env(env_a, np.zeros(7, dtype=np.float32))
        rng_a = np.random.default_rng(42)
        tool = EnvRollbackTool()
        checkpoint = tool.capture(
            env_a,
            step=env_a.env.timestep,
            obs=obs_a,
            rng=rng_a,
        )
        actions = [
            rng_a.uniform(-0.01, 0.01, size=7).astype(np.float32) for _ in range(3)
        ]
        reference = []
        for action in actions:
            obs_a, reward, done, _ = step_env(env_a, action)
            reference.append(capture_trace_point(env_a, obs_a, reward, done))

        rng_b = np.random.default_rng(999)
        restore_report = tool.restore(
            env_b,
            checkpoint,
            reload_xml=True,
            rng=rng_b,
        )
        restored_obs = get_observation(env_b, force_update=False)
        initial_report = compare_checkpoint_to_env(checkpoint, env_b, restored_obs)
        replay = []
        for action in actions:
            obs_b, reward, done, _ = step_env(env_b, action)
            replay.append(capture_trace_point(env_b, obs_b, reward, done))
        replay_report = compare_replay_traces(
            reference,
            replay,
            tolerance=1e-10,
            observation_tolerance=0.0,
        )
        print("restore_report", restore_report)
        print("initial_report", initial_report)
        print("replay_report", replay_report)
        assert restore_report["xml_restored"]
        assert initial_report["image_max_abs_diff"] == 0.0
        assert replay_report["success"]
        assert replay_report["image_max_abs_diff"] == 0.0
    finally:
        env_b.close()
        env_a.close()


if __name__ == "__main__":
    main()
