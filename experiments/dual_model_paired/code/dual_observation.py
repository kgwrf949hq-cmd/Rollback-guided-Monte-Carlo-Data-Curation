"""Separate policy history from a fresh image of the current integration state.

This adapter is deliberately restricted to the inspected robosuite backend.
MjData is copied, forwarded and rendered without forwarding live simulation
data or resampling live observables. No environment transition is performed.
"""
import copy
import hashlib
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dual_core import fingerprint
from trajectory_store import digest_obs

PROTOCOL = 'mujoco_forward_cloned_data_opengl_rgb_v1'
CAMERAS = (('agentview', 'agentview_image'), ('robot0_eye_in_hand', 'robot0_eye_in_hand_image'))
BINDING_SHA256 = 'f5c8aada3bc85d3e66b9c4719c62892f0a32e4ea3489f66cc9abcaa170282c04'
_RENDER_LOCK = threading.Lock()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _backend(sim):
    import mujoco
    from robosuite import macros
    from libero_rollback_tool import _native_mujoco_handles
    context = getattr(sim, '_render_context_offscreen', None)
    if context is None or macros.IMAGE_CONVENTION != 'opengl':
        raise RuntimeError('unverified fresh render backend or image convention')
    source = inspect.getsourcefile(type(context))
    if (mujoco.__version__ != '3.2.3' or not source or
            hashlib.sha256(Path(source).read_bytes()).hexdigest() != BINDING_SHA256):
        raise RuntimeError('unverified fresh render backend version/source')
    model, data = _native_mujoco_handles(sim)
    if model is None or data is None or getattr(context.data, '_data', None) is not data:
        raise RuntimeError('render context is not bound to live native data')
    return model, data, context, mujoco


def _unallocated_fields(data):
    """Do not read NULL arena pointers exposed as arrays by MuJoCo 3.2.3.

    The exact lists are MJDATA_ARENA_POINTERS_DUAL / _ISLAND in the installed
    mjxmacro.h. These are not state when their allocator is disabled. A plain
    repeated read already returns unrelated uninitialized arrays in that case.
    Active solver/island arrays remain subject to the same exact hash check.
    """
    model = getattr(data, 'model', None)
    if model is None:
        return set()
    import mujoco
    fields = set()
    if not mujoco.mj_isDual(model):
        fields.update(('efc_AR_rownnz', 'efc_AR_rowadr', 'efc_AR_colind', 'efc_AR'))
    if not int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ISLAND):
        fields.update(('dof_island', 'island_dofnum', 'island_dofadr', 'island_dofind',
            'dof_islandind', 'efc_island', 'island_efcnum', 'island_efcadr', 'island_efcind'))
    return fields


def _data_signature(data):
    """Hash allocated numeric native-data fields, including derived caches."""
    h = hashlib.sha256()
    unallocated = _unallocated_fields(data)
    h.update(repr(sorted(unallocated)).encode())
    for name in sorted(dir(data)):
        if name.startswith('_') or name in unallocated:
            continue
        value = getattr(data, name)
        if isinstance(value, np.ndarray) and value.dtype.kind in 'biufc':
            value = np.ascontiguousarray(value)
            h.update(name.encode()); h.update(str((value.dtype, value.shape)).encode()); h.update(value.tobytes())
        elif isinstance(value, (bool, int, float, np.number)):
            h.update(name.encode()); h.update(repr(value).encode())
    return h.hexdigest()


def _signature(env, sim, data):
    from libero_rollback_tool import (capture_integration_state, capture_model_state,
        capture_runtime_state, capture_rng_state, get_env_timestep, get_observation)
    spec, integration = capture_integration_state(sim)
    return dict(integration_spec=spec, integration_sha256=array_sha256(integration),
        native_data_sha256=_data_signature(data), model_sha256=fingerprint(capture_model_state(sim)),
        unallocated_native_fields=sorted(_unallocated_fields(data)),
        runtime_sha256=fingerprint(capture_runtime_state(env)), rng_sha256=fingerprint(capture_rng_state()),
        state_step=get_env_timestep(env), sim_time=float(data.time),
        policy_observation_sha256=digest_obs(get_observation(env, force_update=False)))


def fresh_reference(env, obs, state_step):
    """Read current-state cameras on cloned MjData; leave live history intact.

    The caller must serialize access to its environment, as the collector does.
    A nonblocking adapter lock rejects nested/concurrent use of this adapter.
    The inspected renderer reads only context.data._data, restored in finally.
    """
    from libero_rollback_tool import find_sim, capture_integration_state
    if not _RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent or nested fresh render adapter use')
    try:
        sim = find_sim(env)
        model, native, context, mujoco = _backend(sim)
        before = _signature(env, sim, native)
        if before['state_step'] != state_step or before['policy_observation_sha256'] != digest_obs(obs):
            raise RuntimeError('observation/state time alignment mismatch')
        spec, integration = capture_integration_state(sim)
        if spec is None or integration is None:
            raise RuntimeError('fresh render requires integration state')
        cloned = copy.copy(native)
        if cloned is native or np.shares_memory(cloned.qpos, native.qpos) or np.shares_memory(cloned.qvel, native.qvel):
            raise RuntimeError('native data clone aliases live state')
        mujoco.mj_forward(model, cloned)
        original_binding = context.data
        images = {}
        try:
            context.data = SimpleNamespace(_data=cloned)
            for camera, key in CAMERAS:
                expected = np.asarray(obs[key])
                if expected.ndim != 3 or expected.shape[2] != 3 or expected.dtype != np.uint8:
                    raise RuntimeError('invalid camera RGB shape/dtype')
                height, width = expected.shape[:2]
                image = np.array(sim.render(camera_name=camera, width=width, height=height), copy=True)
                if image.shape != expected.shape or image.dtype != expected.dtype:
                    raise RuntimeError('fresh camera shape/dtype mismatch')
                images[key] = image
        finally:
            context.data = original_binding
        after = _signature(env, sim, native)
        if before != after:
            raise RuntimeError('fresh render mutated live state: ' + ','.join(k for k in before if before[k] != after[k]))
        return dict(protocol=PROTOCOL, state_step=state_step, sim_time=before['sim_time'],
            integration_spec=spec, integration_sha256=array_sha256(integration),
            policy_observation_sha256=digest_obs(obs), image_convention='opengl',
            images=images, read_state_unchanged=True, live_before=before, live_after=after)
    finally:
        _RENDER_LOCK.release()


def validate_reference(cp):
    required = ('integration_state', 'integration_state_spec', 'runtime_state', 'rollout_state',
        'model_state', 'rng_state', 'env_timestep', 'env_cur_time', 'env_done', 'obs')
    if any(getattr(cp, name, None) is None for name in required):
        raise RuntimeError('checkpoint missing necessary restoration field')
    if 'failure_detector_state' not in cp.rollout_state:
        raise RuntimeError('checkpoint missing necessary failure history')
    ref = cp.metadata.get('fresh_render_reference')
    if not isinstance(ref, dict) or ref.get('protocol') != PROTOCOL:
        raise RuntimeError('checkpoint missing independent fresh render reference; original files must remain unchanged')
    binding = dict(state_step=cp.step, sim_time=cp.time_value, integration_spec=cp.integration_state_spec,
        integration_sha256=array_sha256(cp.integration_state), policy_observation_sha256=digest_obs(cp.obs),
        image_convention='opengl', read_state_unchanged=True)
    if cp.env_timestep != cp.step or any(ref.get(k) != v for k, v in binding.items()):
        raise RuntimeError('fresh reference state/time/observation binding mismatch')
    images = ref.get('images', {})
    if set(images) != {key for _, key in CAMERAS}:
        raise RuntimeError('fresh reference missing necessary camera')
    for _, key in CAMERAS:
        image = np.asarray(images[key]); policy = np.asarray(cp.obs[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or image.shape != policy.shape:
            raise RuntimeError('fresh reference camera shape/dtype mismatch')
    return ref


def compare_reference(saved, actual):
    for key in ('protocol', 'state_step', 'sim_time', 'integration_spec', 'integration_sha256',
                'policy_observation_sha256', 'image_convention', 'read_state_unchanged'):
        if saved.get(key) != actual.get(key):
            raise RuntimeError('fresh reference state/time/observation binding mismatch: ' + key)
    result = {}
    for _, key in CAMERAS:
        expected = np.asarray(saved['images'][key]); fresh = np.asarray(actual['images'][key])
        if expected.shape != fresh.shape or expected.dtype != fresh.dtype:
            raise RuntimeError('fresh simulator/camera mismatch: shape/dtype')
        delta = np.abs(fresh.astype(np.int16) - expected.astype(np.int16))
        result[key] = dict(equal=bool(np.array_equal(fresh, expected)), max_abs=int(delta.max()),
            mean_abs=float(delta.mean()), unequal_pixel_fraction=float(np.any(fresh != expected, axis=-1).mean()),
            shape=list(fresh.shape), dtype=str(fresh.dtype), fresh_sha256=array_sha256(fresh), saved_sha256=array_sha256(expected))
    if not all(value['equal'] for value in result.values()):
        raise RuntimeError('fresh simulator/camera mismatch: ' + str(result))
    return result
