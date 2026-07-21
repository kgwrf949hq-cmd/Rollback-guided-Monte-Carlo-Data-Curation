# 远程服务器验证步骤

本说明只验证 checkpoint/restore 基础能力，不依赖网络，也不决定什么时候回退或回退到哪个 checkpoint。

## 1. 上传仓库

在本地把仓库复制到远程服务器（服务器地址和用户名按实际配置替换）：

```bash
scp -r Rollback-guided-Monte-Carlo-Data-Curation user@server:/path/to/
ssh user@server
cd /path/to/Rollback-guided-Monte-Carlo-Data-Curation
```

确认当前 shell 使用已经安装 LIBERO、robosuite 和 MuJoCo 的 Python 环境。服务器无法联网时，不要在运行阶段安装依赖。

## 2. 无渲染最小验证

```bash
PYTHONPATH=. python libero_rollback_tool.py \
  --no-render \
  --pre-steps 5 \
  --post-steps 8 \
  --action-mode random \
  --random-action-scale 0.02 \
  --tolerance 1e-10 \
  --observation-tolerance 1e-10 \
  --output rollback_restore_report.json
```

成功时报告中的 `success` 为 `true`，并且以下量应接近 0：

- `integration_state_max_abs_diff`
- `qpos/qvel/ctrl/mocap` 差异
- `observation_max_abs_diff`
- `action_sequence_max_abs_diff`
- `reward_max_abs_diff`

## 3. 三类回归验证

```bash
PYTHONPATH=. python validation/round1_norender_validation.py
PYTHONPATH=. python validation/cross_env_validation.py
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH=. python validation/cross_render_validation.py
PYTHONPATH=. python validation/semantic_keyframe_rollback_validation.py
PYTHONPATH=. python validation/pick_place_checkpoint_validation.py
```

第三个测试使用 OSMesa 软件渲染，不占用 CUDA；若服务器已正确配置 EGL，也可以改用 `MUJOCO_GL=egl` 进行短测试。

语义关键帧测试会运行一段确定性 rollout，在每一步评估语义关键帧，故意执行 4 步大幅扰动，再回退到严格早于失败步的最近关键帧，并 replay 原始动作。报告中的 `selected_keyframe.step` 必须小于 `failure_step`，`replay_report.success` 必须为 `true`。

Pick-and-place 测试还会验证五个子事件的顺序、低风险接近阶段的稀疏 checkpoint、高风险抓取阶段的加密 checkpoint，以及风险系统选点后的确定性回退。

## 4. pi0.5 演示

先启动 OpenPI policy server，再执行：

```bash
PYTHONPATH=. python pi05_libero_rollback_rollout.py \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint-policy semantic \
  --video-output ./artifacts/pi05_rollback.mp4 \
  --output ./artifacts/pi05_rollback_report.json
```

默认回退后重新规划动作。追加 `--restore-action-plan` 可测试恢复 checkpoint 中保存的 pending action chunk。

## 5. 常见问题

- `ModuleNotFoundError: libero`：切换到已配置 LIBERO 的 Python 环境，并确认 `PYTHONPATH=.`。
- OpenGL/EGL 错误：先使用 `MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=""`，确认状态恢复逻辑后再排查 EGL。
- 模型不匹配：跨环境恢复时使用 `reload_xml=True`；工具会用 XML SHA-256 拒绝不兼容的目标环境。
- 状态差异不为 0：检查 task、init state、seed、MuJoCo 版本，以及是否在恢复后强制刷新 observable（这会改变采样时钟）。
