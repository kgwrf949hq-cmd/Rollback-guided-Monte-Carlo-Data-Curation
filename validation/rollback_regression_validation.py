"""Fast regressions for previously observed rollback false positives."""

from __future__ import annotations

import random

import numpy as np

from libero_rollback_tool import (
    EnvRollbackTool,
    compare_observations,
    compare_replay_traces,
    make_branch_rng,
    model_xml_sha256,
)


class _FakeState:
    def __init__(self, data):
        self.qpos = data.qpos.copy()
        self.qvel = data.qvel.copy()
        self.time = float(data.time)

    def flatten(self):
        return np.concatenate(
            [np.asarray([self.time]), self.qpos, self.qvel]
        )


class _FakeModel:
    def __init__(self):
        self.xml = "<mujoco model='legacy-fallback-test'/>"

    def get_xml(self):
        return self.xml


class _FakeData:
    def __init__(self):
        self.qpos = np.asarray([1.0, 2.0])
        self.qvel = np.asarray([0.1, 0.2])
        self.act = np.asarray([0.3, 0.4])
        self.ctrl = np.asarray([0.5, 0.6])
        self.mocap_pos = np.asarray([[0.7, 0.8, 0.9]])
        self.mocap_quat = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        self.time = 1.25


class _FakeLegacySim:
    """Legacy state API intentionally omits act/ctrl/mocap."""

    def __init__(self):
        self.data = _FakeData()
        self.model = _FakeModel()

    def get_state(self):
        return _FakeState(self.data)

    def set_state(self, state):
        self.data.qpos[...] = state.qpos
        self.data.qvel[...] = state.qvel
        self.data.time = state.time

    def forward(self):
        return None


class _FakeEnv:
    def __init__(self):
        self.sim = _FakeLegacySim()
        self.timestep = 7
        self.cur_time = 0.35
        self.done = False
        self._obs_cache = {}
        self._observables = {}

    def _get_observations(self):
        return {
            "state": np.concatenate(
                [self.sim.data.qpos, self.sim.data.qvel, self.sim.data.act]
            )
        }

    def check_success(self):
        return False


def _trace(obs):
    return {
        "integration_state": np.asarray([1.0]),
        "obs": obs,
        "reward": 0.0,
        "done": False,
        "success": False,
    }


def main() -> None:
    different_keys = compare_observations(
        {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)},
        {"other_image": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    assert not different_keys["comparison_valid"]
    assert not different_keys["observation_keys_match"]

    different_shapes = compare_observations(
        {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)},
        {"agentview_image": np.zeros((3, 2, 3), dtype=np.uint8)},
    )
    assert not different_shapes["comparison_valid"]
    assert different_shapes["observation_shape_mismatches"]

    replay_key_mismatch = compare_replay_traces(
        [_trace({"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)})],
        [_trace({"other_image": np.zeros((2, 2, 3), dtype=np.uint8)})],
        tolerance=0.0,
        observation_tolerance=0.0,
    )
    assert not replay_key_mismatch["success"]
    assert replay_key_mismatch["observation_key_mismatch_steps"] == [0]
    empty_replay = compare_replay_traces(
        [], [], tolerance=0.0, observation_tolerance=0.0
    )
    assert not empty_replay["success"]

    xml_a = '<mujoco><statistic meanmass="1"/></mujoco>'
    xml_b = '<mujoco><statistic meanmass="2"/></mujoco>'
    assert model_xml_sha256(xml_a) != model_xml_sha256(xml_b)
    statistic_env = _FakeEnv()
    statistic_env.sim.model.xml = xml_a
    statistic_checkpoint = EnvRollbackTool().capture(
        statistic_env,
        step=statistic_env.timestep,
        obs=statistic_env._get_observations(),
    )
    statistic_env.sim.model.xml = xml_b
    try:
        EnvRollbackTool().restore(statistic_env, statistic_checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("different explicit statistic XML was accepted")

    branch_a = make_branch_rng(17, "episode-0/branch-a").normal(size=8)
    branch_a_repeat = make_branch_rng(17, "episode-0/branch-a").normal(size=8)
    branch_b = make_branch_rng(17, "episode-0/branch-b").normal(size=8)
    assert np.array_equal(branch_a, branch_a_repeat)
    assert not np.array_equal(branch_a, branch_b)

    random.seed(11)
    np.random.seed(11)
    env = _FakeEnv()
    tool = EnvRollbackTool()
    obs = env._get_observations()
    checkpoint = tool.capture(env, step=env.timestep, obs=obs)
    expected_act = checkpoint.act.copy()
    # Simulate a checkpoint pickled before strict and compatibility XML
    # identities were split.
    del checkpoint.model_xml_compatibility_sha256
    env.sim.data.qpos[...] = -10.0
    env.sim.data.qvel[...] = -20.0
    env.sim.data.act[...] = -30.0
    env.sim.data.ctrl[...] = -40.0
    env.sim.data.mocap_pos[...] = -50.0
    env.sim.data.mocap_quat[...] = -60.0
    env.sim.data.time = 99.0
    env.timestep = 99
    env.cur_time = 9.9
    env.done = True
    report = tool.restore(env, checkpoint)
    assert report["state_restore_method"] == "sim.set_state(sim_state)"
    assert report["legacy_xml_signature"]
    assert np.array_equal(env.sim.data.act, expected_act)
    assert report["data_fields_restore"]["verified"]
    assert report["model_state_restore"]["success"]
    assert report["runtime_state_restore"]["verified"]
    assert report["observation_restore"]["comparison_valid"]
    assert report["success"], report

    print(
        {
            "success": True,
            "strict_observation_comparison": True,
            "strict_statistic_xml_hash": True,
            "legacy_act_restore_verified": True,
            "independent_branch_rng": True,
        }
    )


if __name__ == "__main__":
    main()
