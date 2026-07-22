import os

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from libero_rollback_tool import (
    EnvRollbackTool,
    capture_runtime_state,
    capture_trace_point,
    compare_checkpoint_to_env,
    compare_replay_traces,
    get_observation,
    step_env,
)


def compare_nested(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                diffs.append((child, "missing"))
            else:
                diffs.extend(compare_nested(a[key], b[key], child))
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append((path, f"length {len(a)} != {len(b)}"))
        for idx, (x, y) in enumerate(zip(a, b)):
            diffs.extend(compare_nested(x, y, f"{path}[{idx}]"))
        return diffs
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            diffs.append((path, f"shape {a.shape} != {b.shape}"))
        elif a.size and np.max(np.abs(a.astype(float) - b.astype(float))) > 1e-12:
            diffs.append((path, float(np.max(np.abs(a.astype(float) - b.astype(float))))))
        return diffs
    try:
        if a != b:
            diffs.append((path, (repr(a)[:80], repr(b)[:80])))
    except Exception:
        pass
    return diffs


def make_env(bddl):
    return ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )


def model_snapshot(env):
    model = env.sim.model
    names = [
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "geom_friction",
        "geom_size",
        "geom_solref",
        "geom_solimp",
        "dof_armature",
        "dof_damping",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "qpos0",
        "qvel0",
        "site_pos",
        "site_quat",
        "site_rgba",
        "geom_rgba",
    ]
    return {name: np.asarray(getattr(model, name)).copy() for name in names if hasattr(model, name)}


def model_all_snapshot(env):
    model = env.sim.model
    result = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                result[name] = value.copy()
        except Exception:
            continue
    return result


def model_diffs(a, b):
    result = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or a[name].shape != b[name].shape:
            result[name] = "missing_or_shape"
        else:
            diff = float(np.max(np.abs(a[name].astype(float) - b[name].astype(float)))) if a[name].size else 0.0
            if diff > 1e-12:
                result[name] = diff
    return result


def main():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_a = make_env(bddl)
    env_b = make_env(bddl)
    try:
        init_state = suite.get_task_init_states(0)[0]
        env_a.reset()
        obs_a = env_a.set_init_state(init_state)
        for _ in range(5):
            obs_a, _, _, _ = step_env(env_a, np.zeros(7, dtype=np.float32))

        rng_a = np.random.default_rng(123)
        tool = EnvRollbackTool()
        checkpoint = tool.capture(
            env_a,
            step=env_a.env.timestep,
            obs=obs_a,
            rng=rng_a,
            rollout_state={"action_plan": [np.zeros(7, dtype=np.float32)]},
        )
        model_a_at_checkpoint = model_snapshot(env_a)
        model_a_all_at_checkpoint = model_all_snapshot(env_a)
        actions = [
            rng_a.uniform(-0.02, 0.02, size=7).astype(np.float32) for _ in range(6)
        ]
        reference = []
        for action in actions:
            obs_a, reward, done, _ = step_env(env_a, action)
            reference.append(capture_trace_point(env_a, obs_a, reward, done))

        env_b.reset()
        env_b.set_init_state(init_state)
        model_b_before_reload = model_snapshot(env_b)
        rng_b = np.random.default_rng(999)
        restore_report = tool.restore(
            env_b,
            checkpoint,
            reload_xml=True,
            rng=rng_b,
        )
        restored_obs = get_observation(env_b, force_update=False)
        initial_report = compare_checkpoint_to_env(checkpoint, env_b, restored_obs)
        model_b_after_reload = model_snapshot(env_b)
        model_b_all = model_all_snapshot(env_b)
        runtime_diffs = compare_nested(
            checkpoint.runtime_state, capture_runtime_state(env_b)
        )
        replay_actions = [
            rng_b.uniform(-0.02, 0.02, size=7).astype(np.float32) for _ in range(6)
        ]
        action_diff = max(
            float(np.max(np.abs(a - b))) for a, b in zip(actions, replay_actions)
        )
        replay = []
        for action in replay_actions:
            obs_b, reward, done, _ = step_env(env_b, action)
            replay.append(capture_trace_point(env_b, obs_b, reward, done))
        replay_report = compare_replay_traces(
            reference,
            replay,
            tolerance=1e-10,
            observation_tolerance=1e-10,
        )
        rng_probe_a = rng_a.uniform(-1, 1, size=8)
        rng_probe_b = rng_b.uniform(-1, 1, size=8)
        rng_diff = float(np.max(np.abs(rng_probe_a - rng_probe_b)))
        print("restore_report", restore_report)
        print("initial_report", initial_report)
        print("runtime_diffs", runtime_diffs[:80])
        print("model_diffs_before_reload", model_diffs(model_a_at_checkpoint, model_b_before_reload))
        print("model_diffs_after_reload", model_diffs(model_a_at_checkpoint, model_b_after_reload))
        print("model_all_diffs_after_reload", model_diffs(model_a_all_at_checkpoint, model_b_all))
        print("replay_report", replay_report)
        print("action_sequence_max_abs_diff", action_diff)
        print("rng_probe_max_abs_diff", rng_diff)
        assert restore_report["xml_restored"]
        assert restore_report["success"]
        assert restore_report["rng_state_restored"]
        assert initial_report["comparison_valid"]
        assert initial_report["integration_state_max_abs_diff"] <= 1e-10
        assert replay_report["success"]
        assert action_diff <= 1e-10
        assert rng_diff <= 1e-10
    finally:
        env_b.close()
        env_a.close()


if __name__ == "__main__":
    main()
