#!/usr/bin/env python3
"""Prototype checkpoint/restore tool for LIBERO environments.

This script intentionally tests only the low-level rollback primitive:

1. Create an environment.
2. Step it for a while.
3. Capture a checkpoint.
4. Continue with a fixed action sequence.
5. Restore the checkpoint in place.
6. Replay the same actions and compare the resulting traces.

It does not decide when to rollback. That policy layer should be built after
this tool proves that environment copy/restore is reliable.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass
class Checkpoint:
    """Serializable-ish snapshot for one environment state.

    `integration_state` uses MuJoCo's native ``mjSTATE_INTEGRATION`` state
    vector when available. The older ``sim_state`` and manual fields remain
    as compatibility fallbacks for bindings that do not expose that API.
    """

    step: int
    created_at: float
    sim_state: Optional[Any]
    sim_state_flat: Optional[np.ndarray]
    integration_state: Optional[np.ndarray]
    integration_state_spec: Optional[int]
    model_state: Optional[Dict[str, np.ndarray]]
    qpos: Optional[np.ndarray]
    qvel: Optional[np.ndarray]
    act: Optional[np.ndarray]
    ctrl: Optional[np.ndarray]
    mocap_pos: Optional[np.ndarray]
    mocap_quat: Optional[np.ndarray]
    time_value: Optional[float]
    model_xml: Optional[str]
    model_xml_sha256: Optional[str]
    model_xml_compatibility_sha256: Optional[str]
    env_timestep: Optional[int]
    env_cur_time: Optional[float]
    env_done: Optional[bool]
    runtime_state: Optional[Dict[str, Any]]
    rng_state: Optional[Dict[str, Any]]
    rollout_state: Optional[Dict[str, Any]]
    obs: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return np.array(arr, copy=True)
    arr = arr.astype(np.float32)
    if arr.max(initial=0.0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def extract_frame(obs: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    key = find_image_key(obs)
    if key is None or obs is None:
        return None
    try:
        return ensure_uint8_image(obs[key])
    except Exception:
        return None


def write_ppm_frames(frames: Iterable[np.ndarray], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, frame in enumerate(frames):
        image = ensure_uint8_image(frame)
        path = output_dir / f"frame_{idx:06d}.ppm"
        with path.open("wb") as f:
            f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
            f.write(image.tobytes())
        count += 1
    return count


def write_video_or_frames(frames: Iterable[np.ndarray], output_path: Path, fps: int) -> Dict[str, Any]:
    frame_list = [ensure_uint8_image(frame) for frame in frames]
    if not frame_list:
        return {"ok": False, "error": "no frames captured"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(output_path), frame_list, fps=fps)
        return {
            "ok": True,
            "method": "imageio",
            "path": str(output_path),
            "frame_count": len(frame_list),
        }
    except Exception as exc:
        imageio_error = repr(exc)

    try:
        import cv2

        if suffix != ".mp4":
            cv2_path = output_path.with_suffix(".mp4")
        else:
            cv2_path = output_path
        height, width = frame_list[0].shape[:2]
        writer = cv2.VideoWriter(
            str(cv2_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open output file")
        for frame in frame_list:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return {
            "ok": True,
            "method": "cv2",
            "path": str(cv2_path),
            "frame_count": len(frame_list),
            "imageio_error": imageio_error,
        }
    except Exception as exc:
        cv2_error = repr(exc)

    frames_dir = output_path.with_suffix("")
    frame_count = write_ppm_frames(frame_list, frames_dir)
    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(npz_path, frames=np.stack(frame_list, axis=0), fps=np.array(fps))
    return {
        "ok": False,
        "method": "ppm_frames_and_npz_fallback",
        "frames_dir": str(frames_dir),
        "npz_path": str(npz_path),
        "frame_count": frame_count,
        "imageio_error": imageio_error,
        "cv2_error": cv2_error,
    }


def get_nested_attr(obj: Any, names: Iterable[str]) -> Optional[Any]:
    cur = obj
    for name in names:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def find_sim(env: Any) -> Any:
    """Find the MuJoCo sim object through common LIBERO/robosuite wrappers."""

    candidates = [
        ("sim",),
        ("env", "sim"),
        ("_env", "sim"),
        ("env", "env", "sim"),
        ("wrapped_env", "sim"),
        ("unwrapped", "sim"),
    ]
    for chain in candidates:
        sim = get_nested_attr(env, chain)
        if sim is not None:
            return sim
    raise RuntimeError("Could not find env.sim through known wrapper paths.")


def find_owner_with_method(env: Any, method_name: str) -> Optional[Any]:
    candidates = [
        (),
        ("env",),
        ("_env",),
        ("env", "env"),
        ("wrapped_env",),
        ("unwrapped",),
    ]
    for chain in candidates:
        owner = get_nested_attr(env, chain) if chain else env
        if owner is not None and hasattr(owner, method_name):
            return owner
    return None


def task_success_signal(
    env: Any, info: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[bool], str]:
    """Read task success from the environment truth API when available."""

    owner = find_owner_with_method(env, "check_success")
    if owner is not None:
        try:
            return bool(owner.check_success()), "env.check_success"
        except Exception:
            pass
    if isinstance(info, dict):
        for key in ("success", "is_success", "task_success"):
            if key in info:
                try:
                    return bool(info[key]), f"info.{key}"
                except Exception:
                    continue
    return None, "unavailable"


def maybe_copy_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        # ``MjData`` fields are numpy-compatible views. ``asarray(...).copy``
        # works across both the legacy and native robosuite bindings.
        return np.asarray(value).copy()
    except Exception:
        return None


def _native_mujoco_handles(sim: Any) -> Tuple[Any, Any]:
    """Return native ``mjModel`` / ``mjData`` handles for common bindings."""

    model = getattr(getattr(sim, "model", None), "_model", None)
    data = getattr(getattr(sim, "data", None), "_data", None)
    if model is None or data is None:
        raise RuntimeError("Simulator does not expose native MuJoCo handles")
    return model, data


def capture_integration_state(sim: Any) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Capture MuJoCo's full integration state when the native API is present.

    The compatibility ``MjSim.get_state`` exposed by this robosuite version
    only contains time/qpos/qvel. ``mjSTATE_INTEGRATION`` additionally covers
    activation, solver warm-start, controls, applied forces, mocap, equality
    state, userdata and plugin state.
    """

    try:
        import mujoco

        model, data = _native_mujoco_handles(sim)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(model, spec)
        state = np.empty(int(size), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        return int(spec), state
    except Exception:
        return None, None


def restore_integration_state(
    sim: Any, state_spec: Optional[int], state: Optional[np.ndarray]
) -> bool:
    """Restore a state captured by :func:`capture_integration_state`."""

    if state is None:
        return False
    try:
        import mujoco

        model, data = _native_mujoco_handles(sim)
        expected = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if state_spec is not None and int(state_spec) != expected:
            raise ValueError(
                "Unsupported MuJoCo state spec %s (expected %s)"
                % (state_spec, expected)
            )
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        expected_size = int(mujoco.mj_stateSize(model, spec))
        if int(np.asarray(state).size) != expected_size:
            raise ValueError(
                "State vector has size %s; current model expects %s"
                % (np.asarray(state).size, expected_size)
            )
        mujoco.mj_setState(model, data, np.asarray(state), spec)
        return True
    except Exception:
        return False


def model_xml_sha256(model_xml: Optional[str]) -> Optional[str]:
    """Return a strict digest of the exact XML text.

    In particular, ``<statistic>`` is part of this identity.  Older versions
    removed that element and could therefore report two genuinely different
    XML documents as identical.
    """

    if not model_xml:
        return None
    return hashlib.sha256(model_xml.encode("utf-8")).hexdigest()


def model_xml_compatibility_sha256(model_xml: Optional[str]) -> Optional[str]:
    """Return a restore-compatibility digest.

    MuJoCo may add a derived ``<statistic>`` element to XML serialized after
    ``mj_setConst``.  That serialization change must not prevent a second
    in-place restore, but it must also never be presented as an exact XML
    match. Restore uses this separate digest only for that narrow case.
    """

    if not model_xml:
        return None
    # ``mj_setConst`` may inject a derived ``<statistic>`` element into the
    # XML returned by robosuite.  It describes the compiled model, not its
    # identity, and therefore must not make a second restore look like a
    # cross-model restore.  Keep the hash strict for all other XML content.
    stable_xml = model_xml
    if re.search(r"<statistic\b", stable_xml, flags=re.IGNORECASE):
        # MuJoCo emits this as a standalone element surrounded by blank lines.
        # Remove the whole injected block while retaining the original XML
        # layout, so checkpoints made before this normalization remain valid.
        stable_xml = re.sub(
            r"\n[ \t]*\n[ \t]*<statistic\b[^>]*/>[ \t]*\n[ \t]*\n",
            "\n\n",
            stable_xml,
            flags=re.IGNORECASE,
        )
        stable_xml = re.sub(
            r"(?m)^[ \t]*<statistic\b[^>]*/>[ \t]*\r?\n?",
            "",
            stable_xml,
            flags=re.IGNORECASE,
        )
    return hashlib.sha256(stable_xml.encode("utf-8")).hexdigest()


def capture_rng_state(
    generator: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Capture process RNGs plus an optional rollout-local Generator."""

    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": copy.deepcopy(np.random.get_state()),
    }
    if generator is not None:
        state["generator"] = copy.deepcopy(generator.bit_generator.state)
    return state


def restore_rng_state(
    state: Optional[Dict[str, Any]],
    generator: Optional[np.random.Generator] = None,
) -> bool:
    if not state:
        return False
    try:
        if state.get("python_random") is not None:
            random.setstate(state["python_random"])
        if state.get("numpy_random") is not None:
            np.random.set_state(state["numpy_random"])
        if generator is not None and state.get("generator") is not None:
            generator.bit_generator.state = copy.deepcopy(state["generator"])
        return True
    except Exception:
        return False


def make_branch_rng(root_seed: int, branch_id: Any) -> np.random.Generator:
    """Create a deterministic RNG stream independent of checkpoint replay RNG.

    Monte Carlo callers should derive one generator per branch and restore the
    simulator with ``restore_rng=False``. Reusing the checkpoint's generator
    state for every child would intentionally reproduce the same samples.
    """

    identity = f"{int(root_seed)}:{branch_id}".encode("utf-8")
    digest = hashlib.sha256(identity).digest()
    entropy = [int(root_seed)] + [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, 16, 4)
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


MODEL_STATE_FIELDS = (
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
    # Compiled model constants used by forward dynamics and collision.
    "body_ipos",
    "body_iquat",
    "body_subtreemass",
    "body_invweight0",
    "body_sameframe",
    "body_simple",
    "dof_M0",
    "dof_invweight0",
    "geom_quat",
    "geom_sameframe",
    "bvh_aabb",
    "bvh_nodeid",
    "cam_mat0",
    "cam_pos",
    "cam_pos0",
    "cam_poscom0",
    "cam_quat",
    "light_dir",
    "light_dir0",
    "light_poscom0",
    "site_size",
)

MODEL_DERIVED_FIELDS = {
    "body_ipos",
    "body_iquat",
    "body_subtreemass",
    "body_invweight0",
    "body_sameframe",
    "body_simple",
    "dof_M0",
    "dof_invweight0",
    "geom_quat",
    "geom_sameframe",
    "bvh_aabb",
    "bvh_nodeid",
    "cam_mat0",
    "cam_pos",
    "cam_pos0",
    "cam_poscom0",
    "cam_quat",
    "light_dir",
    "light_dir0",
    "light_poscom0",
    "site_size",
}


def capture_model_state(sim: Any) -> Dict[str, np.ndarray]:
    model = getattr(getattr(sim, "model", None), "_model", None)
    if model is None:
        model = getattr(sim, "model", None)
    result: Dict[str, np.ndarray] = {}
    for name in MODEL_STATE_FIELDS:
        try:
            value = getattr(model, name)
            result[name] = np.asarray(value).copy()
        except Exception:
            continue
    return result


def restore_model_state(sim: Any, state: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    model = getattr(getattr(sim, "model", None), "_model", None)
    if model is None:
        model = getattr(sim, "model", None)
    report: Dict[str, Any] = {
        "attempted": sorted(state or {}),
        "written": [],
        "verified": [],
        "failed": [],
        "field_max_abs_diff": {},
        "success": False,
    }
    for name, value in (state or {}).items():
        try:
            target = getattr(model, name)
            if np.asarray(target).shape != np.asarray(value).shape:
                raise ValueError("shape mismatch")
            target[...] = value
            report["written"].append(name)
        except Exception as exc:
            report["failed"].append({"field": name, "error": repr(exc)})
    if report["written"]:
        try:
            import mujoco

            native_model, native_data = _native_mujoco_handles(sim)
            mujoco.mj_setConst(native_model, native_data)
            report["constants_recomputed"] = True
            # mj_setConst rebuilds these arrays from the model base fields. If
            # the source episode modified compiled constants, restore the
            # captured values after recomputation as well.
            for name in MODEL_DERIVED_FIELDS & set(state or {}):
                try:
                    target = getattr(model, name)
                    target[...] = state[name]
                    if name not in report["written"]:
                        report["written"].append(name)
                except Exception as exc:
                    report["failed"].append({"field": name, "error": repr(exc)})
        except Exception as exc:
            report["constants_recomputed"] = False
            report["constants_error"] = repr(exc)
    for name, value in (state or {}).items():
        try:
            target = np.asarray(getattr(model, name))
            reference = np.asarray(value)
            if target.shape != reference.shape:
                raise ValueError(
                    f"shape mismatch after restore: {target.shape} != {reference.shape}"
                )
            diff = max_abs_diff(reference, target)
            report["field_max_abs_diff"][name] = diff
            if diff != 0.0:
                raise ValueError(f"verification max_abs_diff={diff}")
            report["verified"].append(name)
        except Exception as exc:
            if not any(item.get("field") == name for item in report["failed"]):
                report["failed"].append({"field": name, "error": repr(exc)})
    report["success"] = bool(state is not None) and (
        len(report["verified"]) == len(state or {}) and not report["failed"]
    )
    # Compatibility for report consumers; this now means verified, not merely
    # that an assignment was attempted.
    report["restored"] = list(report["verified"])
    return report


def get_model_xml(env: Any, sim: Any) -> Optional[str]:
    model = getattr(sim, "model", None)
    if model is not None and hasattr(model, "get_xml"):
        try:
            return model.get_xml()
        except Exception:
            pass

    owner = find_owner_with_method(env, "model_xml")
    if owner is not None:
        try:
            xml = owner.model_xml
            if isinstance(xml, str):
                return xml
        except Exception:
            pass
    return None


def get_env_timestep(env: Any) -> Optional[int]:
    for chain in [("timestep",), ("env", "timestep"), ("_env", "timestep")]:
        value = get_nested_attr(env, chain)
        if value is not None:
            try:
                return int(value)
            except Exception:
                return None
    return None


def get_env_runtime_field(env: Any, name: str) -> Any:
    """Read a mutable robosuite environment field through common wrappers."""

    for chain in [(), ("env",), ("_env",), ("env", "env")]:
        owner = get_nested_attr(env, chain) if chain else env
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    return None


def set_env_runtime_field(env: Any, name: str, value: Any) -> bool:
    for chain in [(), ("env",), ("_env",), ("env", "env")]:
        owner = get_nested_attr(env, chain) if chain else env
        if owner is not None and hasattr(owner, name):
            try:
                setattr(owner, name, value)
                return True
            except Exception:
                pass
    return False


def set_env_timestep(env: Any, timestep: Optional[int]) -> None:
    if timestep is None:
        return
    for chain in [(), ("env",), ("_env",)]:
        owner = get_nested_attr(env, chain) if chain else env
        if owner is not None and hasattr(owner, "timestep"):
            try:
                setattr(owner, "timestep", int(timestep))
                return
            except Exception:
                pass


def refresh_controller_state(env: Any) -> bool:
    """Recompute controller-derived quantities without advancing physics."""

    owner = get_nested_attr(env, ("env",)) or env
    robots = getattr(owner, "robots", None)
    if robots is None:
        return False
    refreshed = False
    for robot in robots:
        controller = getattr(robot, "controller", None)
        if controller is None:
            continue
        try:
            controller.new_update = True
            controller.update(force=True)
            refreshed = True
        except Exception:
            # A controller without the robosuite update API can still be used
            # after the simulator state has been restored.
            continue
    return refreshed


def get_data_field(sim: Any, name: str) -> Optional[np.ndarray]:
    data = getattr(sim, "data", None)
    if data is None or not hasattr(data, name):
        return None
    return maybe_copy_array(getattr(data, name))


def set_data_field(sim: Any, name: str, value: Optional[np.ndarray]) -> bool:
    if value is None:
        return False
    data = getattr(sim, "data", None)
    if data is None or not hasattr(data, name):
        return False
    try:
        target = getattr(data, name)
        target[...] = value
        return True
    except Exception:
        return False


def get_observation(env: Any, *, force_update: bool = False) -> Optional[Dict[str, Any]]:
    """Read observations without advancing physics, when the env supports it."""

    for method in ["_get_observations", "_get_obs", "get_observation"]:
        owner = find_owner_with_method(env, method)
        if owner is None:
            continue
        try:
            if force_update and hasattr(owner, "_obs_cache"):
                owner._obs_cache = {}
            if force_update and method == "_get_observations":
                obs = getattr(owner, method)(force_update=True)
            else:
                obs = getattr(owner, method)()
            if isinstance(obs, dict):
                return obs
        except TypeError:
            try:
                obs = getattr(owner, method)()
                if isinstance(obs, dict):
                    return obs
            except Exception:
                pass
        except Exception:
            pass
    return None


def copy_obs(obs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obs, dict):
        return None
    copied: Dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        else:
            try:
                copied[key] = copy.deepcopy(value)
            except Exception:
                copied[key] = value
    return copied


_SKIP = object()


def _snapshot_value(value: Any, *, depth: int = 0) -> Any:
    """Copy small mutable runtime values without traversing simulator objects."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return copy.deepcopy(value)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            copied = _snapshot_value(item, depth=depth + 1)
            if copied is not _SKIP:
                result[copy.deepcopy(key)] = copied
        return result
    if isinstance(value, (list, tuple)):
        result = [_snapshot_value(item, depth=depth + 1) for item in value]
        return type(value)(item for item in result if item is not _SKIP)
    if depth < 2 and hasattr(value, "__dict__"):
        return {
            "__object_state__": _snapshot_fields(value, depth=depth + 1)
        }
    return _SKIP


def _snapshot_fields(obj: Any, *, depth: int = 0) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, value in vars(obj).items():
        if name in {"sim", "model", "robot_model", "viewer", "_render_context_offscreen"}:
            continue
        copied = _snapshot_value(value, depth=depth)
        if copied is not _SKIP:
            result[name] = copied
    return result


def _restore_value(target: Any, value: Any) -> Any:
    if isinstance(value, dict) and "__object_state__" in value:
        if target is not None and hasattr(target, "__dict__"):
            _restore_fields(target, value["__object_state__"])
            return target
        return target
    if isinstance(target, np.ndarray) and isinstance(value, np.ndarray):
        if target.shape == value.shape:
            target[...] = value
            return target
    return copy.deepcopy(value)


def _restore_fields(obj: Any, state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    for name, value in state.items():
        if not hasattr(obj, name):
            continue
        try:
            current = getattr(obj, name)
            restored = _restore_value(current, value)
            if restored is not current:
                setattr(obj, name, restored)
        except Exception:
            continue


def _snapshot_mismatches(reference: Any, current: Any, path: str = "") -> List[str]:
    """Return exact verification failures for two captured runtime snapshots."""

    location = path or "<root>"
    if isinstance(reference, np.ndarray):
        if not isinstance(current, np.ndarray):
            return [f"{location}:type"]
        if reference.shape != current.shape:
            return [f"{location}:shape:{reference.shape}!={current.shape}"]
        try:
            equal = np.array_equal(reference, current, equal_nan=True)
        except (TypeError, ValueError):
            equal = np.array_equal(reference, current)
        return [] if equal else [f"{location}:value"]
    if isinstance(reference, dict):
        if not isinstance(current, dict):
            return [f"{location}:type"]
        mismatches: List[str] = []
        for key in sorted(set(reference) | set(current), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in reference:
                mismatches.append(f"{child_path}:unexpected")
            elif key not in current:
                mismatches.append(f"{child_path}:missing")
            else:
                mismatches.extend(
                    _snapshot_mismatches(reference[key], current[key], child_path)
                )
        return mismatches
    if isinstance(reference, (list, tuple)):
        if not isinstance(current, type(reference)):
            return [f"{location}:type"]
        mismatches = []
        if len(reference) != len(current):
            mismatches.append(f"{location}:length:{len(reference)}!={len(current)}")
        for index, (expected, actual) in enumerate(zip(reference, current)):
            mismatches.extend(
                _snapshot_mismatches(expected, actual, f"{path}[{index}]")
            )
        return mismatches
    try:
        equal = bool(reference == current)
    except Exception:
        equal = False
    return [] if equal else [f"{location}:value"]


def capture_runtime_state(env: Any) -> Dict[str, Any]:
    """Capture controller, gripper and history-buffer state for replay."""

    owner = get_nested_attr(env, ("env",)) or env
    robots_state: List[Dict[str, Any]] = []
    for robot in getattr(owner, "robots", []) or []:
        buffer_names = (
            "recent_qpos",
            "recent_actions",
            "recent_torques",
            "recent_ee_forcetorques",
            "recent_ee_pose",
            "recent_ee_vel",
            "recent_ee_vel_buffer",
            "recent_ee_acc",
        )
        robots_state.append(
            {
                "torques": _snapshot_value(getattr(robot, "torques", None)),
                "buffers": {
                    name: _snapshot_fields(getattr(robot, name))
                    for name in buffer_names
                    if getattr(robot, name, None) is not None
                },
                "controller": (
                    _snapshot_fields(robot.controller)
                    if getattr(robot, "controller", None) is not None
                    else None
                ),
                "gripper_current_action": _snapshot_value(
                    getattr(getattr(robot, "gripper", None), "current_action", None)
                ),
            }
        )
    observables_state = {}
    for name, observable in getattr(owner, "_observables", {}).items():
        observables_state[name] = _snapshot_fields(observable)
    return {
        "robots": robots_state,
        "obs_cache": _snapshot_value(getattr(owner, "_obs_cache", {})),
        "observables": observables_state,
    }


def restore_runtime_state(
    env: Any, state: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not state:
        return {
            "attempted": False,
            "verified": False,
            "mismatches": ["runtime_state:not_captured"],
        }
    owner = get_nested_attr(env, ("env",)) or env
    if "obs_cache" in state:
        restored_cache = _restore_value(
            getattr(owner, "_obs_cache", None), state.get("obs_cache")
        )
        if restored_cache is not None:
            owner._obs_cache = restored_cache
    for name, observable_state in state.get("observables", {}).items():
        observable = getattr(owner, "_observables", {}).get(name)
        if observable is not None:
            _restore_fields(observable, observable_state)
    for robot, saved in zip(getattr(owner, "robots", []) or [], state.get("robots", [])):
        if saved.get("controller") and getattr(robot, "controller", None) is not None:
            _restore_fields(robot.controller, saved["controller"])
        for name, buffer_state in saved.get("buffers", {}).items():
            buffer = getattr(robot, name, None)
            if buffer is not None:
                _restore_fields(buffer, buffer_state)
        if "torques" in saved:
            robot.torques = _restore_value(getattr(robot, "torques", None), saved["torques"])
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and "gripper_current_action" in saved:
            gripper.current_action = _restore_value(
                getattr(gripper, "current_action", None), saved["gripper_current_action"]
            )
    current = capture_runtime_state(env)
    mismatches = _snapshot_mismatches(state, current)
    return {
        "attempted": True,
        "verified": not mismatches,
        "mismatches": mismatches,
    }


class EnvRollbackTool:
    """Capture and restore LIBERO/robosuite simulator states."""

    def capture(
        self,
        env: Any,
        *,
        step: int,
        obs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        rollout_state: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        sim = find_sim(env)

        sim_state = None
        sim_state_flat = None
        if hasattr(sim, "get_state"):
            try:
                sim_state = sim.get_state()
            except Exception:
                sim_state = None
            if sim_state is not None and hasattr(sim_state, "flatten"):
                try:
                    sim_state_flat = np.array(sim_state.flatten(), copy=True)
                except Exception:
                    sim_state_flat = None

        state_spec, integration_state = capture_integration_state(sim)
        xml = get_model_xml(env, sim)
        env_core = get_nested_attr(env, ("env",)) or env

        return Checkpoint(
            step=int(step),
            created_at=time.time(),
            sim_state=sim_state,
            sim_state_flat=sim_state_flat,
            integration_state=integration_state,
            integration_state_spec=state_spec,
            model_state=capture_model_state(sim),
            qpos=get_data_field(sim, "qpos"),
            qvel=get_data_field(sim, "qvel"),
            act=get_data_field(sim, "act"),
            ctrl=get_data_field(sim, "ctrl"),
            mocap_pos=get_data_field(sim, "mocap_pos"),
            mocap_quat=get_data_field(sim, "mocap_quat"),
            time_value=float(getattr(getattr(sim, "data", None), "time", 0.0))
            if getattr(sim, "data", None) is not None
            else None,
            model_xml=xml,
            model_xml_sha256=model_xml_sha256(xml),
            model_xml_compatibility_sha256=model_xml_compatibility_sha256(xml),
            env_timestep=get_env_timestep(env),
            env_cur_time=(
                float(getattr(env_core, "cur_time"))
                if hasattr(env_core, "cur_time")
                else None
            ),
            env_done=(bool(getattr(env_core, "done")) if hasattr(env_core, "done") else None),
            runtime_state=capture_runtime_state(env),
            rng_state=capture_rng_state(rng),
            rollout_state=copy.deepcopy(rollout_state) if rollout_state is not None else None,
            obs=copy_obs(obs),
            metadata=metadata or {},
        )

    def restore(
        self,
        env: Any,
        checkpoint: Checkpoint,
        *,
        reload_xml: bool = False,
        rng: Optional[np.random.Generator] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Restore a checkpoint, in place by default.

        Reloading XML reconstructs robosuite's controllers, buffers and
        observable cache. It is therefore reserved for cross-environment
        restores; normal rollback should preserve the existing environment and
        restore its state in place.
        """

        sim = find_sim(env)
        report: Dict[str, Any] = {
            "success": False,
            "xml_restored": False,
            "xml_exact_match": False,
            "xml_compatibility_match": False,
            "state_restore_method": None,
            "integration_state_restored": False,
            "model_state_restored": False,
            "timestep_restored": False,
            "cur_time_restored": False,
            "done_restored": False,
            "runtime_state_restored": False,
            "rng_state_restored": False,
            "rollout_state_available": getattr(checkpoint, "rollout_state", None) is not None,
            "observation_refreshed": False,
        }

        if reload_xml and checkpoint.model_xml:
            owner = find_owner_with_method(env, "reset_from_xml_string")
            if owner is None:
                raise RuntimeError("Environment cannot reload model XML")
            try:
                owner.reset_from_xml_string(checkpoint.model_xml)
                report["xml_restored"] = True
                sim = find_sim(env)
            except Exception as exc:
                report["xml_restore_error"] = repr(exc)
                raise

        current_xml = get_model_xml(env, sim)
        current_signature = model_xml_sha256(current_xml)
        current_compatibility_signature = model_xml_compatibility_sha256(current_xml)
        checkpoint_signature = getattr(checkpoint, "model_xml_sha256", None)
        checkpoint_compatibility_signature = getattr(
            checkpoint, "model_xml_compatibility_sha256", None
        )
        legacy_xml_signature = bool(
            checkpoint_signature and checkpoint_compatibility_signature is None
        )
        if legacy_xml_signature:
            # Before strict/compatibility identities were separated, this
            # field stored the statistic-stripped compatibility digest.
            checkpoint_compatibility_signature = checkpoint_signature
        report["legacy_xml_signature"] = legacy_xml_signature
        report["checkpoint_xml_sha256"] = checkpoint_signature
        report["current_xml_sha256"] = current_signature
        report["checkpoint_xml_compatibility_sha256"] = (
            checkpoint_compatibility_signature
        )
        report["current_xml_compatibility_sha256"] = current_compatibility_signature
        report["xml_exact_match"] = bool(
            not legacy_xml_signature
            and checkpoint_signature
            and current_signature
            and checkpoint_signature == current_signature
        )
        report["xml_compatibility_match"] = bool(
            checkpoint_compatibility_signature
            and current_compatibility_signature
            and checkpoint_compatibility_signature == current_compatibility_signature
        )
        checkpoint_statistic_count = len(
            re.findall(
                r"<statistic\b", checkpoint.model_xml or "", flags=re.IGNORECASE
            )
        )
        current_statistic_count = len(
            re.findall(r"<statistic\b", current_xml or "", flags=re.IGNORECASE)
        )
        statistic_serialization_only = bool(
            report["xml_compatibility_match"]
            and checkpoint_statistic_count != current_statistic_count
            and min(checkpoint_statistic_count, current_statistic_count) == 0
        )
        report["checkpoint_statistic_count"] = checkpoint_statistic_count
        report["current_statistic_count"] = current_statistic_count
        report["xml_compatibility_accepted"] = statistic_serialization_only
        if (
            checkpoint_signature
            and current_signature
            and checkpoint_signature != current_signature
            and not statistic_serialization_only
        ):
            raise ValueError(
                "Checkpoint model does not match the target environment; "
                "use reload_xml=True for a cross-environment restore"
            )
        if not report["xml_exact_match"] and statistic_serialization_only:
            report["xml_difference"] = "derived_statistic_serialization_only"

        model_restore_report = restore_model_state(
            sim, getattr(checkpoint, "model_state", None)
        )
        report["model_state_restore"] = model_restore_report
        report["model_state_restored"] = bool(model_restore_report["success"])

        if restore_integration_state(
            sim, checkpoint.integration_state_spec, checkpoint.integration_state
        ):
            report["state_restore_method"] = "mujoco.mj_setState(mjSTATE_INTEGRATION)"
            report["integration_state_restored"] = True
        elif checkpoint.sim_state is not None and hasattr(sim, "set_state"):
            try:
                sim.set_state(checkpoint.sim_state)
                report["state_restore_method"] = "sim.set_state(sim_state)"
            except Exception as exc:
                report["set_state_error"] = repr(exc)

        if report["state_restore_method"] is None:
            if checkpoint.sim_state_flat is not None and hasattr(
                sim, "set_state_from_flattened"
            ):
                try:
                    sim.set_state_from_flattened(checkpoint.sim_state_flat)
                    report["state_restore_method"] = "sim.set_state_from_flattened"
                except Exception as exc:
                    report["set_state_flat_error"] = repr(exc)

        # Supplemental fields are restored even when the compatibility
        # sim.set_state path succeeds. The old implementation only did this
        # in its final fallback branch, leaving ctrl/mocap stale.
        if report["state_restore_method"] is None:
            ok_qpos = set_data_field(sim, "qpos", checkpoint.qpos)
            ok_qvel = set_data_field(sim, "qvel", checkpoint.qvel)
            if ok_qpos and ok_qvel:
                report["state_restore_method"] = "manual qpos/qvel assignment"

        if report["state_restore_method"] is None:
            raise RuntimeError(f"Failed to restore MuJoCo state: {report}")

        # Restore every compatibility field regardless of which state API was
        # used. In particular, legacy sim.set_state objects omit ``act``.
        data_field_values = {
            "act": checkpoint.act,
            "ctrl": checkpoint.ctrl,
            "mocap_pos": checkpoint.mocap_pos,
            "mocap_quat": checkpoint.mocap_quat,
        }
        data_field_writes = {
            name: set_data_field(sim, name, value)
            for name, value in data_field_values.items()
            if value is not None
        }
        time_written = False
        if checkpoint.time_value is not None and hasattr(sim, "data"):
            try:
                sim.data.time = checkpoint.time_value
                time_written = True
            except Exception:
                time_written = False

        if hasattr(sim, "forward"):
            sim.forward()
        refresh_controller_state(env)
        runtime_restore_report = restore_runtime_state(env, checkpoint.runtime_state)
        report["runtime_state_restore"] = runtime_restore_report
        report["runtime_state_restored"] = bool(
            runtime_restore_report.get("verified")
        )
        # ``mj_forward`` can update warm-start acceleration. Restore the
        # integration vector once more after all forward calls so the saved
        # state remains exact while derived quantities stay synchronized with
        # the same qpos/qvel.
        if report["integration_state_restored"]:
            second_restore_ok = restore_integration_state(
                sim, checkpoint.integration_state_spec, checkpoint.integration_state
            )
            report["integration_state_restored"] = bool(second_restore_ok)

        data_field_diffs = {
            name: max_abs_diff(value, get_data_field(sim, name))
            for name, value in data_field_values.items()
            if value is not None
        }
        time_diff = (
            abs(float(checkpoint.time_value) - float(sim.data.time))
            if checkpoint.time_value is not None and hasattr(sim, "data")
            else None
        )
        report["data_fields_restore"] = {
            "writes": data_field_writes,
            "max_abs_diff": data_field_diffs,
            "time_written": time_written,
            "time_abs_diff": time_diff,
            "verified": all(diff == 0.0 for diff in data_field_diffs.values())
            and (time_diff is None or time_diff == 0.0),
        }
        current_spec, current_integration = capture_integration_state(sim)
        integration_diff = max_abs_diff(
            checkpoint.integration_state, current_integration
        )
        report["integration_state_verification"] = {
            "checkpoint_spec": checkpoint.integration_state_spec,
            "current_spec": current_spec,
            "max_abs_diff": integration_diff,
            "verified": bool(
                checkpoint.integration_state is not None
                and current_integration is not None
                and checkpoint.integration_state_spec == current_spec
                and integration_diff == 0.0
            ),
        }

        before = get_env_timestep(env)
        set_env_timestep(env, checkpoint.env_timestep)
        after = get_env_timestep(env)
        report["timestep_restored"] = (
            checkpoint.env_timestep is not None and after == checkpoint.env_timestep
        )
        report["timestep_before_restore"] = before
        report["timestep_after_restore"] = after

        before_cur_time = get_env_runtime_field(env, "cur_time")
        if checkpoint.env_cur_time is not None:
            cur_time_written = set_env_runtime_field(
                env, "cur_time", float(checkpoint.env_cur_time)
            )
        report["cur_time_before_restore"] = before_cur_time
        report["cur_time_after_restore"] = get_env_runtime_field(env, "cur_time")
        report["cur_time_restored"] = bool(
            checkpoint.env_cur_time is not None
            and cur_time_written
            and report["cur_time_after_restore"] == checkpoint.env_cur_time
        )

        if checkpoint.env_done is not None:
            done_written = set_env_runtime_field(
                env, "done", bool(checkpoint.env_done)
            )
        report["done_after_restore"] = get_env_runtime_field(env, "done")
        report["done_restored"] = bool(
            checkpoint.env_done is not None
            and done_written
            and report["done_after_restore"] == checkpoint.env_done
        )

        if restore_rng:
            rng_write_ok = restore_rng_state(
                getattr(checkpoint, "rng_state", None), rng
            )
            current_rng_state = capture_rng_state(rng)
            checkpoint_rng_state = getattr(checkpoint, "rng_state", None)
            if isinstance(checkpoint_rng_state, dict):
                # A caller may supply a Generator at restore time even when
                # the checkpoint captured only process RNGs. Extra current
                # sources do not invalidate verification of captured sources.
                current_rng_state = {
                    key: current_rng_state.get(key) for key in checkpoint_rng_state
                }
            rng_mismatches = _snapshot_mismatches(
                checkpoint_rng_state, current_rng_state
            )
            report["rng_state_restore"] = {
                "write_ok": rng_write_ok,
                "verified": bool(rng_write_ok and not rng_mismatches),
                "mismatches": rng_mismatches,
            }
            report["rng_state_restored"] = bool(
                report["rng_state_restore"]["verified"]
            )

        # The observable values, sampling clocks and cache were restored above,
        # so reading without a forced sample returns the checkpoint observation
        # and preserves the future sampling schedule. A forced sample here
        # would advance observable clocks and break deterministic replay.
        restored_observation = get_observation(env, force_update=False)
        report["observation_refreshed"] = restored_observation is not None
        observation_restore_report = compare_observations(
            checkpoint.obs, restored_observation
        )
        report["observation_restore"] = observation_restore_report
        report["observation_restored"] = bool(
            observation_restore_report.get("comparison_valid")
            and observation_restore_report.get("observation_max_abs_diff") == 0.0
        )
        state_verified = (
            report["integration_state_verification"]["verified"]
            if checkpoint.integration_state is not None
            else max_abs_diff(checkpoint.qpos, get_data_field(sim, "qpos")) == 0.0
            and max_abs_diff(checkpoint.qvel, get_data_field(sim, "qvel")) == 0.0
        )
        required_checks = [
            bool(state_verified),
            bool(report["data_fields_restore"]["verified"]),
        ]
        if checkpoint.model_state is not None:
            required_checks.append(bool(report["model_state_restored"]))
        if checkpoint.runtime_state is not None:
            required_checks.append(bool(report["runtime_state_restored"]))
        if checkpoint.env_timestep is not None:
            required_checks.append(bool(report["timestep_restored"]))
        if checkpoint.env_cur_time is not None:
            required_checks.append(bool(report["cur_time_restored"]))
        if checkpoint.env_done is not None:
            required_checks.append(bool(report["done_restored"]))
        if restore_rng:
            required_checks.append(bool(report["rng_state_restored"]))
        if checkpoint.obs is not None:
            required_checks.append(bool(report["observation_restored"]))
        report["success"] = all(required_checks)
        return report


def make_libero_suite(benchmark_name: str) -> Any:
    from libero.libero import benchmark

    if hasattr(benchmark, "get_benchmark_dict"):
        benchmark_dict = benchmark.get_benchmark_dict()
        if benchmark_name not in benchmark_dict:
            raise KeyError(
                f"Unknown benchmark {benchmark_name!r}. "
                f"Available: {sorted(benchmark_dict.keys())}"
            )
        return benchmark_dict[benchmark_name]()

    if hasattr(benchmark, "get_benchmark"):
        return benchmark.get_benchmark(benchmark_name)()

    raise RuntimeError("Unsupported LIBERO benchmark API.")


def make_libero_env(
    *,
    benchmark_name: str,
    task_id: int,
    init_state_id: int,
    seed: int,
    camera_width: int,
    camera_height: int,
    render: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs import OffScreenRenderEnv

    suite = make_libero_suite(benchmark_name)
    task = suite.get_task(task_id)
    bddl_file_name = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    if render:
        env_args = {
            "bddl_file_name": bddl_file_name,
            "camera_heights": camera_height,
            "camera_widths": camera_width,
            "camera_depths": False,
        }
        env = OffScreenRenderEnv(**env_args)
    else:
        env = ControlEnv(
            bddl_file_name=bddl_file_name,
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            hard_reset=False,
        )
    if hasattr(env, "seed"):
        env.seed(seed)

    obs = env.reset()
    init_states = suite.get_task_init_states(task_id)
    init_state = init_states[init_state_id]
    if hasattr(env, "set_init_state"):
        obs = env.set_init_state(init_state)

    info = {
        "benchmark_name": benchmark_name,
        "task_id": task_id,
        "init_state_id": init_state_id,
        "seed": seed,
        "bddl_file_name": bddl_file_name,
        "task_language": getattr(task, "language", None),
        "task_problem_folder": getattr(task, "problem_folder", None),
        "task_bddl_file": getattr(task, "bddl_file", None),
    }
    return env, obs, info


def action_dim(env: Any) -> int:
    for chain in [("action_dim",), ("env", "action_dim"), ("_env", "action_dim")]:
        value = get_nested_attr(env, chain)
        if value is not None:
            return int(value)

    owner = find_owner_with_method(env, "action_spec")
    if owner is not None:
        spec = owner.action_spec
        try:
            low, _high = spec() if callable(spec) else spec
            return int(np.asarray(low).shape[0])
        except Exception:
            pass

    return 7


def make_action(
    env: Any,
    mode: str,
    rng: np.random.Generator,
    scale: float,
    step_index: int = 0,
) -> np.ndarray:
    dim = action_dim(env)
    if mode == "zero":
        return np.zeros(dim, dtype=np.float32)
    if mode == "random":
        return rng.uniform(low=-scale, high=scale, size=dim).astype(np.float32)
    if mode == "sweep":
        action = np.zeros(dim, dtype=np.float32)
        if dim > 0:
            # Cycle through xyz-like control axes so motion is visible in video.
            axis_count = min(3, dim)
            segment = max(1, int(20))
            axis = (step_index // segment) % axis_count
            sign = 1.0 if ((step_index // (segment * axis_count)) % 2 == 0) else -1.0
            action[axis] = sign * scale
        if dim > 0:
            action[-1] = -1.0 if (step_index // 30) % 2 == 0 else 1.0
        return action
    raise ValueError(f"Unknown action mode: {mode}")


def step_env(
    env: Any, action: np.ndarray
) -> Tuple[Optional[Dict[str, Any]], float, bool, Dict[str, Any]]:
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), dict(info)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info)
    raise RuntimeError(f"Unsupported env.step return value: {type(result)}")


def close_env(env: Any) -> None:
    if env is None:
        return
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def max_abs_diff(
    a: Optional[np.ndarray], b: Optional[np.ndarray]
) -> Optional[float]:
    if a is None or b is None:
        return None
    if np.asarray(a).shape != np.asarray(b).shape:
        return None
    if np.asarray(a).size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def find_image_key(obs: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(obs, dict):
        return None
    preferred = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "image",
    ]
    for key in preferred:
        if key in obs and isinstance(obs[key], np.ndarray) and obs[key].ndim >= 3:
            return key
    for key, value in obs.items():
        if key.endswith("_image") and isinstance(value, np.ndarray) and value.ndim >= 3:
            return key
    return None


def compare_observations(
    reference: Optional[Dict[str, Any]], current: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Strictly compare every observation field without advancing the env.

    Missing keys, extra keys, type changes and shape changes are comparison
    failures. A report with no comparable fields is invalid rather than a
    vacuous match.
    """

    result: Dict[str, Any] = {
        "comparison_valid": False,
        "observation_keys_match": False,
        "observation_max_abs_diff": None,
        "image_max_abs_diff": None,
        "image_mean_abs_diff": None,
        "observation_missing_keys": [],
        "observation_extra_keys": [],
        "observation_shape_mismatches": {},
        "observation_type_mismatches": {},
        "observation_value_mismatches": [],
        "compared_field_count": 0,
        "compared_array_field_count": 0,
        "compared_image_field_count": 0,
    }
    if not isinstance(reference, dict) or not isinstance(current, dict):
        result["comparison_error"] = "reference/current must both be dictionaries"
        return result

    ref_keys = set(reference)
    cur_keys = set(current)
    result["observation_keys_match"] = ref_keys == cur_keys
    result["observation_missing_keys"] = sorted(ref_keys - cur_keys)
    result["observation_extra_keys"] = sorted(cur_keys - ref_keys)
    diffs: List[float] = []
    image_diffs: List[np.ndarray] = []
    for key in sorted(ref_keys & cur_keys):
        ref_value, cur_value = reference[key], current[key]
        ref_is_array = isinstance(ref_value, np.ndarray)
        cur_is_array = isinstance(cur_value, np.ndarray)
        if ref_is_array != cur_is_array:
            result["observation_type_mismatches"][key] = {
                "reference": type(ref_value).__name__,
                "current": type(cur_value).__name__,
            }
            continue
        if ref_is_array and cur_is_array:
            if ref_value.shape != cur_value.shape:
                result["observation_shape_mismatches"][key] = {
                    "reference": list(ref_value.shape),
                    "current": list(cur_value.shape),
                }
                continue
            result["compared_field_count"] += 1
            result["compared_array_field_count"] += 1
            is_image = key.endswith("_image") or "image" in key
            if is_image:
                result["compared_image_field_count"] += 1
            if np.issubdtype(ref_value.dtype, np.number) and np.issubdtype(
                cur_value.dtype, np.number
            ):
                diff = np.abs(
                    ref_value.astype(np.float64) - cur_value.astype(np.float64)
                )
                field_max = float(np.max(diff)) if diff.size else 0.0
                diffs.append(field_max)
                if is_image:
                    image_diffs.append(diff)
            else:
                equal = np.array_equal(ref_value, cur_value)
                diffs.append(0.0 if equal else float("inf"))
                if not equal:
                    result["observation_value_mismatches"].append(key)
            continue
        if isinstance(ref_value, (bool, int, float, np.number)) and isinstance(
            cur_value, (bool, int, float, np.number)
        ):
            result["compared_field_count"] += 1
            diffs.append(abs(float(ref_value) - float(cur_value)))
            continue
        try:
            equal = bool(ref_value == cur_value)
        except Exception:
            equal = False
        result["compared_field_count"] += 1
        if not equal:
            result["observation_value_mismatches"].append(key)
            diffs.append(float("inf"))
    if diffs:
        result["observation_max_abs_diff"] = max(diffs)
    if image_diffs:
        result["image_max_abs_diff"] = max(float(np.max(x)) for x in image_diffs)
        result["image_mean_abs_diff"] = float(
            np.mean(np.concatenate([x.reshape(-1) for x in image_diffs]))
        )
    result["comparison_valid"] = bool(
        ref_keys
        and result["observation_keys_match"]
        and not result["observation_shape_mismatches"]
        and not result["observation_type_mismatches"]
        and not result["observation_value_mismatches"]
        and result["compared_field_count"] == len(ref_keys)
    )
    return result


def compare_model_states(
    reference: Optional[Dict[str, np.ndarray]],
    current: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    diffs: Dict[str, float] = {}
    for name in set(reference or {}) & set(current):
        a, b = reference[name], current[name]
        if a.shape != b.shape:
            diffs[name] = float("inf")
        elif a.size:
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if diff > 0.0:
                diffs[name] = diff
    return {
        "model_state_max_abs_diff": max(diffs.values()) if diffs else 0.0,
        "model_state_field_diffs": diffs,
    }


def compare_checkpoint_to_env(
    checkpoint: Checkpoint, env: Any, restored_obs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    sim = find_sim(env)
    current_spec, current_integration = capture_integration_state(sim)
    integration_diff = max_abs_diff(checkpoint.integration_state, current_integration)
    report = {
        "integration_state_spec": checkpoint.integration_state_spec,
        "integration_state_size": (
            int(np.asarray(checkpoint.integration_state).size)
            if checkpoint.integration_state is not None
            else None
        ),
        "integration_state_max_abs_diff": integration_diff,
        "qpos_max_abs_diff": max_abs_diff(checkpoint.qpos, get_data_field(sim, "qpos")),
        "qvel_max_abs_diff": max_abs_diff(checkpoint.qvel, get_data_field(sim, "qvel")),
        "act_max_abs_diff": max_abs_diff(checkpoint.act, get_data_field(sim, "act")),
        "ctrl_max_abs_diff": max_abs_diff(checkpoint.ctrl, get_data_field(sim, "ctrl")),
        "mocap_pos_max_abs_diff": max_abs_diff(
            checkpoint.mocap_pos, get_data_field(sim, "mocap_pos")
        ),
        "mocap_quat_max_abs_diff": max_abs_diff(
            checkpoint.mocap_quat, get_data_field(sim, "mocap_quat")
        ),
        "env_timestep_checkpoint": checkpoint.env_timestep,
        "env_timestep_restored": get_env_timestep(env),
        "env_cur_time_checkpoint": checkpoint.env_cur_time,
        "env_cur_time_restored": get_env_runtime_field(env, "cur_time"),
        "env_done_checkpoint": checkpoint.env_done,
        "env_done_restored": get_env_runtime_field(env, "done"),
    }
    report.update(compare_observations(checkpoint.obs, restored_obs))
    report.update(compare_model_states(checkpoint.model_state, capture_model_state(sim)))
    report["current_integration_state_spec"] = current_spec
    return report


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def maybe_save_checkpoint(path: Optional[Path], checkpoint: Checkpoint) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(checkpoint, f)


def capture_trace_point(
    env: Any,
    obs: Optional[Dict[str, Any]],
    reward: float,
    done: bool,
) -> Dict[str, Any]:
    """Capture the state and outputs at one policy step for replay checks."""

    sim = find_sim(env)
    state_spec, integration_state = capture_integration_state(sim)
    task_success, success_source = task_success_signal(env)
    return {
        "state_spec": state_spec,
        "integration_state": integration_state,
        "obs": copy_obs(obs),
        "reward": float(reward),
        "done": bool(done),
        "success": task_success,
        "success_source": success_source,
    }


def compare_replay_traces(
    reference: Sequence[Dict[str, Any]],
    replay: Sequence[Dict[str, Any]],
    *,
    tolerance: float,
    observation_tolerance: float,
) -> Dict[str, Any]:
    """Compare two fixed-action rollouts after an in-place restore."""

    report: Dict[str, Any] = {
        "success": False,
        "reference_steps": len(reference),
        "replay_steps": len(replay),
        "steps_compared": min(len(reference), len(replay)),
        "integration_state_max_abs_diff": None,
        "observation_max_abs_diff": None,
        "image_max_abs_diff": None,
        "image_mean_abs_diff": None,
        "reward_max_abs_diff": None,
        "done_mismatches": None,
        "success_mismatches": None,
        "success_signal_unavailable_steps": [],
        "observation_comparison_valid_steps": 0,
        "observation_mismatch_steps": [],
        "observation_shape_mismatch_steps": [],
        "observation_key_mismatch_steps": [],
        "image_fields_compared": 0,
    }
    if len(reference) != len(replay):
        report["length_mismatch"] = True
    if not reference:
        report["comparison_error"] = "empty reference trace is not validation evidence"
        return report

    state_diffs: List[float] = []
    obs_diffs: List[float] = []
    image_diffs: List[np.ndarray] = []
    reward_diffs: List[float] = []
    done_mismatches = 0
    success_mismatches = 0
    success_signal_unavailable_steps: List[int] = []
    observation_valid_steps = 0
    observation_mismatch_steps: List[int] = []
    observation_shape_mismatch_steps: List[int] = []
    observation_key_mismatch_steps: List[int] = []
    image_fields_compared = 0
    for step_index, (ref, got) in enumerate(zip(reference, replay)):
        diff = max_abs_diff(ref["integration_state"], got["integration_state"])
        if diff is not None:
            state_diffs.append(diff)
        obs_report = compare_observations(ref["obs"], got["obs"])
        if obs_report["comparison_valid"]:
            observation_valid_steps += 1
        else:
            observation_mismatch_steps.append(step_index)
        if not obs_report["observation_keys_match"]:
            observation_key_mismatch_steps.append(step_index)
        if obs_report["observation_shape_mismatches"]:
            observation_shape_mismatch_steps.append(step_index)
        image_fields_compared += int(obs_report["compared_image_field_count"])
        if obs_report["observation_max_abs_diff"] is not None:
            obs_diffs.append(obs_report["observation_max_abs_diff"])
        for key in set(ref.get("obs") or {}) & set(got.get("obs") or {}):
            a, b = ref["obs"].get(key), got["obs"].get(key)
            if (
                isinstance(a, np.ndarray)
                and isinstance(b, np.ndarray)
                and a.shape == b.shape
                and (key.endswith("_image") or "image" in key)
            ):
                image_diffs.append(np.abs(a.astype(np.float64) - b.astype(np.float64)))
        reward_diffs.append(abs(float(ref["reward"]) - float(got["reward"])))
        done_mismatches += int(bool(ref["done"]) != bool(got["done"]))
        if ref.get("success") is None or got.get("success") is None:
            success_signal_unavailable_steps.append(step_index)
        success_mismatches += int(ref.get("success") != got.get("success"))

    report["integration_state_max_abs_diff"] = max(state_diffs) if state_diffs else None
    report["observation_max_abs_diff"] = max(obs_diffs) if obs_diffs else None
    if image_diffs:
        report["image_max_abs_diff"] = max(float(np.max(x)) for x in image_diffs)
        report["image_mean_abs_diff"] = float(
            np.mean(np.concatenate([x.reshape(-1) for x in image_diffs]))
        )
    report["reward_max_abs_diff"] = max(reward_diffs) if reward_diffs else None
    report["done_mismatches"] = done_mismatches
    report["success_mismatches"] = success_mismatches
    report["success_signal_unavailable_steps"] = success_signal_unavailable_steps
    report["observation_comparison_valid_steps"] = observation_valid_steps
    report["observation_mismatch_steps"] = observation_mismatch_steps
    report["observation_shape_mismatch_steps"] = observation_shape_mismatch_steps
    report["observation_key_mismatch_steps"] = observation_key_mismatch_steps
    report["image_fields_compared"] = image_fields_compared

    report["success"] = (
        len(reference) == len(replay)
        and len(state_diffs) == len(reference)
        and max(state_diffs) <= tolerance
        and observation_valid_steps == len(reference)
        and len(obs_diffs) == len(reference)
        and max(obs_diffs) <= observation_tolerance
        and (not image_diffs or report["image_max_abs_diff"] <= observation_tolerance)
        and (not reward_diffs or max(reward_diffs) <= tolerance)
        and done_mismatches == 0
        and success_mismatches == 0
        and not success_signal_unavailable_steps
    )
    return report


def run_test(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tool = EnvRollbackTool()
    env = None
    video_frames = []
    try:
        log("[1/5] Creating environment...")
        env, obs, env_info = make_libero_env(
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            seed=args.seed,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            render=not args.no_render,
        )
        log(f"      task language: {env_info.get('task_language')}")
        frame = extract_frame(obs)
        if frame is not None:
            video_frames.append(frame)

        log(f"[2/5] Stepping to checkpoint ({args.pre_steps} steps)...")
        done = False
        global_step = 0
        for _ in range(args.pre_steps):
            action = make_action(
                env, args.action_mode, rng, args.random_action_scale, global_step
            )
            obs, _reward, done, _info = step_env(env, action)
            frame = extract_frame(obs)
            if frame is not None:
                video_frames.append(frame)
            global_step += 1
            if done:
                break

        checkpoint_step = get_env_timestep(env) or args.pre_steps
        log("[3/5] Capturing checkpoint and recording a fixed-action trace...")
        checkpoint = tool.capture(
            env,
            step=checkpoint_step,
            obs=obs,
            rng=rng,
            metadata={
                **env_info,
                "pre_steps_requested": args.pre_steps,
                "action_mode": args.action_mode,
                "restore_mode": "in_place",
            },
        )
        maybe_save_checkpoint(
            Path(args.save_checkpoint) if args.save_checkpoint else None, checkpoint
        )

        actions: List[np.ndarray] = []
        reference_trace: List[Dict[str, Any]] = []
        for _ in range(args.post_steps):
            action = make_action(
                env, args.action_mode, rng, args.random_action_scale, global_step
            )
            actions.append(np.array(action, copy=True))
            obs, reward, done, _info = step_env(env, action)
            reference_trace.append(capture_trace_point(env, obs, reward, done))
            frame = extract_frame(obs)
            if frame is not None:
                video_frames.append(frame)
            global_step += 1
            if done:
                break

        log("[4/5] Restoring in place and replaying RNG-generated actions...")
        restore_report = tool.restore(env, checkpoint, reload_xml=False, rng=rng)
        restored_obs = get_observation(env, force_update=False)
        if restored_obs is None:
            raise RuntimeError("Could not regenerate observation after restore")
        if args.record_restored_frames:
            frame = extract_frame(restored_obs)
            if frame is not None:
                video_frames.append(frame)
        compare_report = compare_checkpoint_to_env(checkpoint, env, restored_obs)

        replay_trace: List[Dict[str, Any]] = []
        replay_action_diffs: List[float] = []
        step_ok = True
        step_error = None
        replay_global_step = checkpoint_step
        for reference_action in actions:
            try:
                action = make_action(
                    env,
                    args.action_mode,
                    rng,
                    args.random_action_scale,
                    replay_global_step,
                )
                replay_action_diffs.append(
                    float(np.max(np.abs(np.asarray(reference_action) - np.asarray(action))))
                )
                replay_global_step += 1
                obs, reward, done, _info = step_env(env, action)
                replay_trace.append(capture_trace_point(env, obs, reward, done))
                if args.record_restored_frames:
                    frame = extract_frame(obs)
                    if frame is not None:
                        video_frames.append(frame)
            except Exception as exc:
                step_ok = False
                step_error = repr(exc)
                break
            if done:
                break
        replay_report = compare_replay_traces(
            reference_trace,
            replay_trace,
            tolerance=args.tolerance,
            observation_tolerance=args.observation_tolerance,
        )

        log("[5/5] Writing validation report...")
        initial_state_ok = (
            bool(restore_report.get("success"))
            and bool(compare_report.get("comparison_valid"))
            and compare_report.get("observation_max_abs_diff") is not None
            and compare_report["observation_max_abs_diff"]
            <= args.observation_tolerance
            and compare_report.get("integration_state_max_abs_diff") is not None
            and compare_report["integration_state_max_abs_diff"] <= args.tolerance
            and compare_report.get("env_timestep_checkpoint")
            == compare_report.get("env_timestep_restored")
            and compare_report.get("env_cur_time_checkpoint")
            == compare_report.get("env_cur_time_restored")
            and compare_report.get("env_done_checkpoint")
            == compare_report.get("env_done_restored")
            and bool(restore_report.get("rng_state_restored"))
        )
        action_sequence_max_abs_diff = max(replay_action_diffs) if replay_action_diffs else 0.0
        success = bool(
            step_ok
            and initial_state_ok
            and action_sequence_max_abs_diff <= args.tolerance
            and replay_report["success"]
        )

        video_report = None
        if args.video_output:
            log(f"      Writing video/frames to {args.video_output} ...")
            video_report = write_video_or_frames(
                video_frames, Path(args.video_output), args.video_fps
            )

        report = {
            "success": success,
            "env_info": env_info,
            "checkpoint": {
                "step": checkpoint.step,
                "env_timestep": checkpoint.env_timestep,
                "env_cur_time": checkpoint.env_cur_time,
                "env_done": checkpoint.env_done,
                "has_sim_state": checkpoint.sim_state is not None,
                "has_sim_state_flat": checkpoint.sim_state_flat is not None,
                "has_integration_state": checkpoint.integration_state is not None,
                "integration_state_spec": checkpoint.integration_state_spec,
                "model_state_fields": sorted(checkpoint.model_state or {}),
                "integration_state_size": (
                    int(np.asarray(checkpoint.integration_state).size)
                    if checkpoint.integration_state is not None
                    else None
                ),
                "has_runtime_state": checkpoint.runtime_state is not None,
                "has_rng_state": checkpoint.rng_state is not None,
                "has_local_generator_state": bool(
                    checkpoint.rng_state and checkpoint.rng_state.get("generator") is not None
                ),
                "has_rollout_state": checkpoint.rollout_state is not None,
                "model_xml_sha256": checkpoint.model_xml_sha256,
                "model_xml_compatibility_sha256": getattr(
                    checkpoint, "model_xml_compatibility_sha256", None
                ),
                "has_model_xml": checkpoint.model_xml is not None,
                "obs_keys": sorted(checkpoint.obs.keys()) if checkpoint.obs else None,
            },
            "restore_report": restore_report,
            "compare_report": compare_report,
            "replay_report": replay_report,
            "action_sequence_max_abs_diff": action_sequence_max_abs_diff,
            "step_after_restore": {
                "ok": step_ok,
                "error": step_error,
            },
            "video": video_report,
            "tolerance": args.tolerance,
            "observation_tolerance": args.observation_tolerance,
        }

        if args.output:
            save_json(Path(args.output), report)
        return report
    finally:
        close_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LIBERO environment checkpoint/restore rollback primitive."
    )
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pre-steps", type=int, default=20)
    parser.add_argument("--post-steps", type=int, default=10)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Use a CPU-only environment without camera observations.",
    )
    parser.add_argument("--action-mode", choices=["zero", "random", "sweep"], default="zero")
    parser.add_argument("--random-action-scale", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--observation-tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed absolute difference for replay observations/images.",
    )
    parser.add_argument("--output", default="rollback_restore_report.json")
    parser.add_argument("--save-checkpoint", default="")
    parser.add_argument("--video-output", default="")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--record-restored-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_test(args)
    except Exception as exc:
        log(f"ERROR: {exc!r}")
        return 2

    log(json.dumps(report, indent=2, sort_keys=True))
    if report["success"]:
        log("RESULT: PASS - checkpoint restored and fixed-action replay matched.")
        return 0

    log("RESULT: FAIL - restore ran, but state comparison exceeded tolerance.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
