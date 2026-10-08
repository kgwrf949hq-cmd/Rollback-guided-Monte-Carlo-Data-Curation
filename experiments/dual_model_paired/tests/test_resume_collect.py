"""Targeted resume collector checks; inherited synthetic harness, real HDF/readback.

These tests are explicitly synthetic and are not evidence of real-model control.
"""
import copy
import json
import hashlib
import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import dual_collect
import dual_runtime
from curation_store import save_payload
from dual_core import ActionContract, Proposal, canonicalize_actions, fingerprint
from trajectory_store import digest_obs


class Harness:
    def __init__(self, tmp):
        self.tmp = tmp; self.calls = []; self.clients = []; self.environments = []
        self.captures = []; self.restores = []; self.snapshots = {}
        self.end_after_aux = None; self.success_at = None; self.rule_after_aux = None
        self.mutate_comparison = False; self.model_error = None; self.aux_value = .8
        self.run = tmp / 'run'; self.run.mkdir()
        contract = ActionContract(verified=True, evidence=('MOCK integration OSC contract',)).as_dict()
        self.task = dict(suite='mock_suite', task_id=0, init_state_id=0, name='mock_task',
            bddl_sha256='mock_bddl', init_file_sha256='mock_init')
        self.cfg = dict(task_directory=str(tmp), warmup_steps=2, horizon=30, task=self.task,
            root_seed=123, camera_size=3, task_id='mock-task-owned', max_depth=3,
            fork_strategy='dual_disagreement', branch_action='aux_short_then_main', mode='active',
            action_contracts={'main': contract, 'aux': contract}, scales=dict(translation=1.,
                rotation=1., gripper=1., source='MOCK normalized command amplitudes'),
            d_min=.1, numeric_floor=1e-6, candidate_spacing=10, suffix_cooldown=10,
            models={r: {'identity': {'name': 'mock-' + r, 'weights_fingerprint': 'weight-' + r,
                'adapter_version': 'mock-adapter'}} for r in ('main', 'aux')}, interface_steps=10)

    def observation(self, env):
        value = (int(round(env.position * 10)) + env.t) % 255
        return dict(robot_state=np.array([env.t, env.position], np.float64),
            agentview_image=np.full((3, 3, 3), value, np.uint8),
            robot0_eye_in_hand_image=np.full((3, 3, 3), value + 1, np.uint8))

    def make_env(self, **kw):
        env = types.SimpleNamespace(t=0, position=0., aux_steps=0, restored=False, closed=False)
        env.data = types.SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))
        self.environments.append(env)
        return env, self.observation(env), {'task_language': 'synthetic placement task'}

    def step(self, env, action):
        env.t += 1; env.position += float(action[0]); env.data.qpos[:] = [env.t, env.position]
        env.data.qvel[:] = [1, float(action[0])]
        if env.restored and action[0] > .5:
            env.aux_steps += 1
        done = self.end_after_aux is not None and env.aux_steps >= self.end_after_aux
        return self.observation(env), 0., done, {'mock': True}

    def capture(self, env, obs, detector, path, branch, depth, step, task_id, contexts):
        assert step == env.t
        assert digest_obs(obs) == digest_obs(self.observation(env))
        self.captures.append((branch, step))
        self.snapshots[str(path)] = dict(t=env.t, position=env.position, detector=detector.snapshot_state())
        save_payload(path, self.snapshots[str(path)])
        return dict(path=str(path), fork_step=step, owner_branch_id=branch, owner_depth=depth,
            observation_sha256=digest_obs(obs), json_sha256='mock', npz_sha256='mock')

    def restore(self, env, detector, checkpoint, task_id):
        self.restores.append(checkpoint['path'])
        snapshot = self.snapshots[checkpoint['path']]
        env.t = snapshot['t']; env.position = snapshot['position']; env.restored = True
        env.data.qpos[:] = [env.t, env.position]
        detector.restore_state(snapshot['detector'])
        return self.observation(env), dict(success=True, cross_process=True, fresh_state_equal=True,
            fresh_render={'agentview_image': {'equal': True}, 'robot0_eye_in_hand_image': {'equal': True}})

    def collect(self, branch='root', selected=None, **kwargs):
        return dual_collect.collect(self.cfg, self.run, branch, selected, **kwargs)

    def trajectory(self, branch):
        with h5py.File(self.run / 'branches' / branch / 'trajectory.h5') as h:
            return dict(actions=h['executed_actions'][:], steps=h['env_steps'][:],
                observations=h['observation_sha256'][:],
                decisions=[json.loads(x) for x in h['decisions'][:]])


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path)
    class Truth:
        def __init__(self, env):
            self.task_success = h.success_at is not None and env.t >= h.success_at
        def as_dict(self):
            return {'task_success': self.task_success, 'mock_truth': True}
    class Detector:
        def __init__(self): self.steps = []
        def update(self, obs, **kw):
            self.steps.append(kw['step'])
            env = h.environments[-1]
            failed = h.rule_after_aux is not None and env.aux_steps >= h.rule_after_aux
            return types.SimpleNamespace(failed=failed, reason='mock_rule',
                to_summary=lambda: {'failed': failed, 'reason': 'mock_rule'})
        def snapshot_state(self): return {'steps': list(self.steps)}
        def restore_state(self, state): self.steps = list(state['steps'])
    class Adapter:
        def __init__(self, task, env): self.env = env
        def detector(self): return Detector()
        def truth(self, env): return Truth(env)
    class Client:
        def __init__(self, settings, task_id, contract, logdir, role):
            self.settings = settings; self.role = role; self.contract = contract; self.closed = False
            h.clients.append(self)
        def proposal(self, obs, instruction, request_id, seed):
            env = h.environments[-1]
            h.calls.append(dict(role=self.role, t=env.t, request=request_id, obs=digest_obs(obs), seed=seed))
            if h.model_error == self.role:
                raise TimeoutError('synthetic transport timeout')
            if h.mutate_comparison and self.role == 'aux':
                env.t += 1
            raw = np.full((7, 7), .1 if self.role == 'main' else h.aux_value, np.float32)
            return Proposal.create(raw, canonicalize_actions(raw, self.contract), self.settings['identity'],
                'mock-adapter', {'noise_seed': seed}, digest_obs(obs), 'input-' + digest_obs(obs),
                'mock-stateless-' + self.role, request_id, fingerprint(self.contract.as_dict()))
        def close(self): self.closed = True
    rollback = types.ModuleType('libero_rollback_tool')
    rollback.make_libero_env = h.make_env; rollback.step_env = h.step; rollback.find_sim = lambda env: env
    rollback.close_env = lambda env: setattr(env, 'closed', True)
    rollback.capture_integration_state = lambda sim: ('mock', np.array([sim.t, sim.position]))
    adapter_module = types.ModuleType('libero_task_adapter')
    adapter_module.describe_task = lambda *args: h.task
    adapter_module.TaskAdapter = Adapter
    monkeypatch.setitem(sys.modules, 'libero_rollback_tool', rollback)
    monkeypatch.setitem(sys.modules, 'libero_task_adapter', adapter_module)
    robosuite = types.ModuleType('robosuite'); robosuite.macros = types.SimpleNamespace(IMAGE_CONVENTION='opengl')
    monkeypatch.setitem(sys.modules, 'robosuite', robosuite)
    monkeypatch.setattr(dual_collect, 'VerifiedClient', Client)
    monkeypatch.setattr(dual_collect, 'capture_candidate', h.capture)
    monkeypatch.setattr(dual_runtime, 'capture_candidate', h.capture)
    monkeypatch.setattr(dual_collect, 'restore_candidate', h.restore)
    monkeypatch.setattr(dual_collect, 'probe_gripper_direction', lambda env: {'status': 'PASS', 'positive_command': 'close', 'mock': True})
    monkeypatch.setattr(dual_collect, 'guard', lambda base: None)
    monkeypatch.setattr(dual_collect.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(dual_collect.signal, 'alarm', lambda *args: None)
    return h



def extension(h):
    h.cfg.update(mode='shadow', root_seed=2026091901,
        calibration_source_id='calibration_shadow_extension_v1',
        calibration_shadow_seed=dual_runtime.stable_seed(2026091901, 'calibration_shadow_extension_v1'))
    return h


def test_extension_main_only_and_observations_stop_at_real_success(harness, monkeypatch):
    h = extension(harness); h.success_at = 8; env_parameters = []
    def tracked_env(**kw):
        env_parameters.append(kw); return h.make_env(**kw)
    monkeypatch.setattr(sys.modules['libero_rollback_tool'], 'make_libero_env', tracked_env)
    root = h.collect()
    assert root['outcome'] == 'success' and root['final_env_step'] == 8
    assert env_parameters[0]['seed'] == 2026091901
    assert all(c.role == 'main' for c in h.clients)
    assert all(c['role'] == 'main' for c in h.calls)
    assert root['candidates'] == [] and h.captures == [('root',2),('root',7)]
    from curation_store import load_payload
    states = [load_payload(p) for p in root['calibration_samples']]
    assert [s['state_step'] for s in states] == [2, 7]
    assert all(not s['terminal'] and not s['confirmed_failed'] and s['replan_boundary'] for s in states)
    assert all(s['source_id'] == 'calibration_shadow_extension_v1' for s in states)
    assert all(s['checkpoint']['fork_step']==s['state_step'] for s in states)
    assert [c['seed'] for c in h.calls] == [2080998210979712087, 16395419391564189420]
    assert all(d['action_source'] == 'main' for d in h.trajectory('root')['decisions'])
    assert all(e.closed for e in h.environments) and all(c.closed for c in h.clients)


def test_extension_directory_names_do_not_change_actual_control_seeds(harness):
    h = extension(harness); h.success_at = 8
    h.collect(); first = [c['seed'] for c in h.calls]; first_requests = [c['request'] for c in h.calls]
    h.run = h.tmp / 'different_directory_name'; h.run.mkdir(); h.calls.clear()
    h.collect()
    assert [c['seed'] for c in h.calls] == first
    assert [c['request'] for c in h.calls] != first_requests


def test_formal_main_and_aux_keep_original_seed_formula(harness):
    h = harness; h.cfg['root_seed'] = 2026091901; h.success_at = 8
    # Merely present extension seed metadata must not change a formal run stream.
    h.cfg['calibration_shadow_seed'] = 9623145367892944805
    h.collect()
    for c in h.calls:
        call_index = (c['t'] - h.cfg['warmup_steps']) // 5
        assert c['seed'] == dual_runtime.stable_seed(2026091901, h.run.name, 'root', c['role'], call_index)
    assert {c['role'] for c in h.calls} == {'main', 'aux'}


@pytest.mark.parametrize('change', [{'mode': 'active'}, {'calibration_shadow_seed': 1}])
def test_extension_rejects_wrong_mode_or_seed_before_environment(harness, change):
    h = extension(harness); h.cfg.update(change)
    with pytest.raises(RuntimeError): h.collect()
    assert h.environments == [] and h.clients == []


@pytest.mark.parametrize('kwargs', [{'branch': 'child'}, {'interface': True}])
def test_extension_rejects_nonroot_or_interface_before_environment(harness, kwargs):
    h = extension(harness)
    with pytest.raises(RuntimeError): h.collect(**kwargs)
    assert h.environments == [] and h.clients == []
