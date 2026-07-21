#!/usr/bin/env python3
"""pi0.5 LIBERO rollout demo with checkpoint capture and rollback video.

This script keeps the official LIBERO evaluation path intact and adds a small
demonstration layer:

* query pi0.5 through the OpenPI websocket policy server,
* save a simulator checkpoint every N environment steps,
* intentionally enter a short "failure branch" for the video demo,
* restore the latest checkpoint,
* continue rollout from the restored state with pi0.5 actions.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from libero_rollback_tool import (
    EnvRollbackTool,
    close_env,
    compare_checkpoint_to_env,
    ensure_uint8_image,
    get_observation,
    make_action,
    make_libero_env,
    step_env,
    write_video_or_frames,
)
from semantic_checkpoint import SemanticKeyframeConfig, SemanticKeyframeSelector
from openpi_client import image_tools
from openpi_client import websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--benchmark", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--camera-resolution", type=int, default=256)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument(
        "--checkpoint-policy",
        choices=["semantic", "periodic"],
        default="semantic",
        help="Select semantic keyframes or preserve the fixed-interval policy.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--checkpoint-min-interval", type=int, default=3)
    parser.add_argument("--checkpoint-max-interval", type=int, default=20)
    parser.add_argument("--checkpoint-max-count", type=int, default=64)
    parser.add_argument("--checkpoint-score-threshold", type=float, default=1.0)
    parser.add_argument("--demo-failure-step", type=int, default=70)
    parser.add_argument("--failure-branch-steps", type=int, default=18)
    parser.add_argument("--failure-action-mode", default="sweep", choices=["zero", "random", "sweep"])
    parser.add_argument("--failure-action-scale", type=float, default=0.28)
    parser.add_argument("--post-restore-steps", type=int, default=55)
    parser.add_argument("--video-output", default="artifacts/pi05_libero_rollback_demo.mp4")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output", default="artifacts/pi05_libero_rollback_demo_report.json")
    parser.add_argument("--hold-frames", type=int, default=8)
    parser.add_argument(
        "--restore-action-plan",
        action="store_true",
        help="Restore the pending policy action chunk at rollback (default: replan).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def policy_element(obs: Dict[str, Any], task_description: str, resize_size: int) -> Dict[str, Any]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }


def frame_from_obs(obs: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(obs, dict) or "agentview_image" not in obs:
        return None
    return ensure_uint8_image(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def annotate_frame(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    color: Tuple[int, int, int],
    *,
    width: int = 512,
) -> np.ndarray:
    image = Image.fromarray(ensure_uint8_image(frame)).convert("RGB")
    ratio = float(width) / float(image.width)
    height = max(1, int(round(image.height * ratio)))
    image = image.resize((width, height), Image.BILINEAR)

    banner_h = 72
    border = 6
    canvas = Image.new("RGB", (width, height + banner_h), (16, 18, 24))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, banner_h], fill=color)
    canvas.paste(image, (0, banner_h))
    draw.rectangle([0, banner_h, width - 1, height + banner_h - 1], outline=color, width=border)

    font = ImageFont.load_default()
    title_w, title_h = text_size(draw, title, font)
    subtitle_w, subtitle_h = text_size(draw, subtitle, font)
    draw.text(((width - title_w) // 2, 16), title, fill=(255, 255, 255), font=font)
    draw.text(((width - subtitle_w) // 2, 42), subtitle, fill=(255, 255, 255), font=font)
    return np.asarray(canvas)


def append_frame(
    frames: List[np.ndarray],
    obs: Optional[Dict[str, Any]],
    title: str,
    subtitle: str,
    color: Tuple[int, int, int],
    repeats: int = 1,
) -> None:
    frame = frame_from_obs(obs)
    if frame is None:
        return
    annotated = annotate_frame(frame, title, subtitle, color)
    for _ in range(max(1, int(repeats))):
        frames.append(np.array(annotated, copy=True))


def capture_checkpoint(
    rollback: EnvRollbackTool,
    env: Any,
    step: int,
    obs: Dict[str, Any],
    checkpoint_steps: List[int],
    metadata: Dict[str, Any],
    *,
    rng: Optional[np.random.Generator] = None,
    action_plan: Optional[Deque[np.ndarray]] = None,
    policy_queries: int = 0,
    restored_policy_steps: int = 0,
) -> Any:
    rollout_state = {
        "action_plan": [
            np.asarray(action, dtype=np.float32).copy()
            for action in (action_plan or ())
        ],
        "policy_queries": int(policy_queries),
        "restored_policy_steps": int(restored_policy_steps),
        # The current websocket policy is feed-forward. This explicit field
        # makes the schema ready for recurrent/cache state later.
        "policy_state": None,
    }
    checkpoint = rollback.capture(
        env,
        step=step,
        obs=obs,
        metadata=metadata,
        rng=rng,
        rollout_state=rollout_state,
    )
    checkpoint_steps.append(int(step))
    logging.info("Captured checkpoint at env step %s", step)
    return checkpoint


def capture_semantic_checkpoint(
    selector: SemanticKeyframeSelector,
    env: Any,
    step: int,
    obs: Dict[str, Any],
    reward: float,
    done: bool,
    info: Dict[str, Any],
    checkpoint_steps: List[int],
    metadata: Dict[str, Any],
    *,
    action: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    action_plan: Optional[Deque[np.ndarray]] = None,
    policy_queries: int = 0,
    restored_policy_steps: int = 0,
) -> Optional[Any]:
    """Evaluate one state and return its checkpoint only when selected."""

    rollout_state = {
        "action_plan": [
            np.asarray(pending_action, dtype=np.float32).copy()
            for pending_action in (action_plan or ())
        ],
        "policy_queries": int(policy_queries),
        "restored_policy_steps": int(restored_policy_steps),
        "policy_state": None,
    }
    keyframe = selector.maybe_capture(
        env,
        step,
        obs,
        reward,
        done,
        info,
        action=action,
        rng=rng,
        rollout_state=rollout_state,
        metadata=metadata,
    )
    if keyframe is None:
        return None
    checkpoint_steps.append(int(step))
    logging.info(
        "Captured semantic checkpoint at env step %s (%s, score=%.3f)",
        step,
        ",".join(keyframe.labels),
        keyframe.score,
    )
    return keyframe.checkpoint


def restore_action_plan(action_plan: Deque[np.ndarray], checkpoint: Any) -> bool:
    state = getattr(checkpoint, "rollout_state", None) or {}
    saved_actions = state.get("action_plan")
    if saved_actions is None:
        return False
    action_plan.clear()
    for action in saved_actions:
        action_plan.append(np.asarray(action, dtype=np.float32).copy())
    return True


def infer_next_action(
    client: websocket_client_policy.WebsocketClientPolicy,
    action_plan: Deque[np.ndarray],
    obs: Dict[str, Any],
    task_description: str,
    resize_size: int,
    replan_steps: int,
) -> Tuple[np.ndarray, bool]:
    queried_policy = False
    if not action_plan:
        element = policy_element(obs, task_description, resize_size)
        action_chunk = client.infer(element)["actions"]
        if len(action_chunk) < replan_steps:
            raise RuntimeError(
                "Policy returned %d actions, fewer than replan_steps=%d"
                % (len(action_chunk), replan_steps)
            )
        for action in action_chunk[:replan_steps]:
            action_plan.append(np.asarray(action, dtype=np.float32))
        queried_policy = True
    return action_plan.popleft(), queried_policy


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    env = None
    frames: List[np.ndarray] = []
    checkpoint_steps: List[int] = []
    restore_report: Optional[Dict[str, Any]] = None
    restore_validation: Optional[Dict[str, Any]] = None
    rollback_done = False
    done = False
    policy_queries = 0
    restored_policy_steps = 0
    action_plan_restored = False
    selected_keyframe_summary: Optional[Dict[str, Any]] = None
    latest_checkpoint = None
    action_plan: Deque[np.ndarray] = collections.deque()

    try:
        env, obs, info = make_libero_env(
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            seed=args.seed,
            camera_width=args.camera_resolution,
            camera_height=args.camera_resolution,
        )
        task_description = str(info.get("task_language") or "")
        client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        rollback = EnvRollbackTool()
        semantic_selector = SemanticKeyframeSelector(
            tool=rollback,
            config=SemanticKeyframeConfig(
                min_interval=args.checkpoint_min_interval,
                max_interval=args.checkpoint_max_interval,
                max_keyframes=args.checkpoint_max_count,
                score_threshold=args.checkpoint_score_threshold,
            ),
        )
        if args.checkpoint_policy == "semantic":
            latest_checkpoint = capture_semantic_checkpoint(
                semantic_selector,
                env,
                0,
                obs,
                0.0,
                False,
                {},
                checkpoint_steps,
                {"phase": "initial", "task_description": task_description},
                rng=rng,
                action_plan=action_plan,
                policy_queries=policy_queries,
                restored_policy_steps=restored_policy_steps,
            )

        append_frame(
            frames,
            obs,
            "pi0.5 LIBERO rollout",
            "initial state",
            (44, 92, 160),
            repeats=args.hold_frames,
        )

        t = 0
        while t < args.max_steps:
            if t < args.num_steps_wait:
                obs, _reward, done, _info = step_env(env, LIBERO_DUMMY_ACTION)
                t += 1
                if args.checkpoint_policy == "semantic":
                    semantic_checkpoint = capture_semantic_checkpoint(
                        semantic_selector,
                        env,
                        t,
                        obs,
                        _reward,
                        done,
                        _info,
                        checkpoint_steps,
                        {"phase": "settling", "task_description": task_description},
                        action=np.asarray(LIBERO_DUMMY_ACTION, dtype=np.float32),
                        rng=rng,
                        action_plan=action_plan,
                        policy_queries=policy_queries,
                        restored_policy_steps=restored_policy_steps,
                    )
                    if semantic_checkpoint is not None:
                        latest_checkpoint = semantic_checkpoint
                append_frame(frames, obs, "settling objects", "env step %03d" % t, (88, 96, 112))
                continue

            should_checkpoint = args.checkpoint_policy == "periodic" and (
                latest_checkpoint is None
                or (t - args.num_steps_wait) % args.checkpoint_interval == 0
            )
            if should_checkpoint:
                latest_checkpoint = capture_checkpoint(
                    rollback,
                    env,
                    t,
                    obs,
                    checkpoint_steps,
                    {"phase": "policy_rollout", "task_description": task_description},
                    rng=rng,
                    action_plan=action_plan,
                    policy_queries=policy_queries,
                    restored_policy_steps=restored_policy_steps,
                )
                append_frame(
                    frames,
                    obs,
                    "CHECKPOINT CAPTURED",
                    "saved simulator state at step %03d" % t,
                    (34, 120, 184),
                    repeats=args.hold_frames,
                )

            if (
                args.demo_failure_step >= 0
                and not rollback_done
                and latest_checkpoint is not None
                and t >= args.demo_failure_step
            ):
                if args.checkpoint_policy == "semantic":
                    selected_keyframe = semantic_selector.select_for_rollback(t + 1)
                    latest_checkpoint = selected_keyframe.checkpoint
                    selected_keyframe_summary = selected_keyframe.to_summary()
                append_frame(
                    frames,
                    obs,
                    "FAILURE DETECTED",
                    "branching from checkpoint step %03d" % latest_checkpoint.step,
                    (190, 58, 52),
                    repeats=args.hold_frames,
                )
                failed_obs = obs
                failure_error = None
                for branch_step in range(args.failure_branch_steps):
                    try:
                        bad_action = make_action(
                            env,
                            args.failure_action_mode,
                            rng,
                            args.failure_action_scale,
                            step_index=branch_step,
                        )
                        failed_obs, _reward, _done, _info = step_env(env, bad_action)
                        append_frame(
                            frames,
                            failed_obs,
                            "FAILURE BRANCH",
                            "bad rollout frame %02d" % (branch_step + 1),
                            (190, 58, 52),
                        )
                    except Exception as exc:  # noqa: BLE001 - recorded in report.
                        failure_error = repr(exc)
                        logging.exception("Failure branch raised an exception")
                        break

                append_frame(
                    frames,
                    failed_obs,
                    "ROLLBACK REQUESTED",
                    "restore latest checkpoint before continuing",
                    (218, 131, 44),
                    repeats=args.hold_frames,
                )
                restore_report = rollback.restore(env, latest_checkpoint, rng=rng)
                restored_obs = get_observation(env, force_update=False)
                if restored_obs is None:
                    raise RuntimeError("Could not read observation after restore")
                if args.checkpoint_policy == "semantic":
                    semantic_selector.reset_history(restored_obs)
                restore_validation = compare_checkpoint_to_env(
                    latest_checkpoint, env, restored_obs
                )
                obs = restored_obs
                t = int(latest_checkpoint.step)
                if args.restore_action_plan:
                    action_plan_restored = restore_action_plan(
                        action_plan, latest_checkpoint
                    )
                    if not action_plan_restored:
                        raise RuntimeError(
                            "Checkpoint does not contain a saved policy action plan"
                        )
                else:
                    action_plan.clear()
                rollback_done = True
                append_frame(
                    frames,
                    obs,
                    "ROLLBACK RESTORED",
                    "returned to checkpoint step %03d" % t,
                    (42, 146, 96),
                    repeats=args.hold_frames * 2,
                )
                if failure_error:
                    logging.info("Failure branch error was recorded: %s", failure_error)
                continue

            action, queried = infer_next_action(
                client,
                action_plan,
                obs,
                task_description,
                args.resize_size,
                args.replan_steps,
            )
            policy_queries += int(queried)
            obs, _reward, done, _info = step_env(env, action.tolist())
            t += 1
            if args.checkpoint_policy == "semantic":
                semantic_checkpoint = capture_semantic_checkpoint(
                    semantic_selector,
                    env,
                    t,
                    obs,
                    _reward,
                    done,
                    _info,
                    checkpoint_steps,
                    {"phase": "policy_rollout", "task_description": task_description},
                    action=action,
                    rng=rng,
                    action_plan=action_plan,
                    policy_queries=policy_queries,
                    restored_policy_steps=restored_policy_steps,
                )
                if semantic_checkpoint is not None:
                    latest_checkpoint = semantic_checkpoint
            if rollback_done:
                restored_policy_steps += 1
                phase = "RESTORED PI0.5 ROLLOUT"
                color = (42, 146, 96)
            else:
                phase = "PI0.5 POLICY ROLLOUT"
                color = (68, 98, 172)
            append_frame(frames, obs, phase, "env step %03d" % t, color)

            if done:
                append_frame(
                    frames,
                    obs,
                    "TASK SUCCESS",
                    "env reported done at step %03d" % t,
                    (42, 146, 96),
                    repeats=args.hold_frames,
                )
                break
            if rollback_done and restored_policy_steps >= args.post_restore_steps:
                append_frame(
                    frames,
                    obs,
                    "DEMO COMPLETE",
                    "continued %d policy steps after restore" % restored_policy_steps,
                    (42, 146, 96),
                    repeats=args.hold_frames,
                )
                break

        video_result = write_video_or_frames(frames, pathlib.Path(args.video_output), args.video_fps)
        report = {
            "success": True,
            "task_done": bool(done),
            "rollback_performed": bool(rollback_done),
            "rollback_trigger": "demo_failure_step",
            "checkpoint_policy": args.checkpoint_policy,
            "checkpoint_interval": args.checkpoint_interval,
            "checkpoint_steps": checkpoint_steps,
            "semantic_keyframes": (
                semantic_selector.summary()
                if args.checkpoint_policy == "semantic"
                else None
            ),
            "selected_keyframe": selected_keyframe_summary,
            "restore_report": restore_report,
            "restore_validation": restore_validation,
            "final_env_step": int(t),
            "post_restore_policy_steps": int(restored_policy_steps),
            "policy_queries": int(policy_queries),
            "action_plan_restore_requested": bool(args.restore_action_plan),
            "action_plan_restored": bool(action_plan_restored),
            "frame_count": len(frames),
            "video": video_result,
            "task": info,
        }
    except Exception as exc:  # noqa: BLE001 - report should survive demo failures.
        logging.exception("Rollback rollout demo failed")
        report = {
            "success": False,
            "error": repr(exc),
            "rollback_performed": bool(rollback_done),
            "checkpoint_steps": checkpoint_steps,
            "restore_report": restore_report,
            "restore_validation": restore_validation,
            "frame_count": len(frames),
        }
        if frames:
            report["video"] = write_video_or_frames(
                frames, pathlib.Path(args.video_output), args.video_fps
            )
    finally:
        close_env(env)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    started = time.time()
    report = run_demo(args)
    elapsed = time.time() - started
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Elapsed seconds: %.2f" % elapsed)
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
