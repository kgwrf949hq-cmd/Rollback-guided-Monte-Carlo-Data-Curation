# Rollback-guided Monte Carlo Data Curation

用于 LIBERO/robosuite/MuJoCo 机器人轨迹筛选的 checkpoint/rollback 基础实现。它把一次 rollout 的仿真状态完整保存下来，在检测到失败分支后恢复到最近 checkpoint，并继续执行或重新规划动作，为后续的蒙特卡洛采样、启发式回退和数据集构造提供可靠的状态恢复层。

## 架构概览

```text
LIBERO environment
        │
        ▼
EnvRollbackTool.capture()
        │  Checkpoint
        ├─ MuJoCo integration state (mjSTATE_INTEGRATION)
        ├─ model arrays / XML signature
        ├─ environment runtime state
        ├─ RNG states
        ├─ observation cache and metadata
        └─ pending policy action chunk
        │
        ▼
failure detector / sampling policy
        │
        ▼
EnvRollbackTool.restore()
        │
        ├─ in-place restore (normal rollback)
        └─ XML reload + restore (cross-environment replay)
        │
        ▼
replay actions or query pi0.5 again
```

实现分为两层：

1. `libero_rollback_tool.py` 是与策略无关的状态快照和恢复层。
2. `semantic_checkpoint.py` 是 checkpoint 选择层：根据运动、夹爪、物体、奖励、任务进度和终止信号保存语义关键帧，并保证最大 checkpoint 间隔。
3. `pi05_libero_rollback_rollout.py` 是示例策略层：通过 OpenPI websocket 调用 pi0.5，定期保存 checkpoint，模拟一段失败分支，回退后继续 rollout，并可生成带阶段标注的视频或帧序列。

## 核心功能

- 使用 MuJoCo 原生 `mjSTATE_INTEGRATION` 保存和恢复积分状态；在旧绑定中自动回退到 `sim_state` 或手动字段恢复。
- 同步 `ctrl`、mocap、activation、solver warm-start、applied forces 等容易被普通 `sim.set_state()` 遗漏的字段。
- 保存并恢复 model 基础数组及编译派生常量，调用 `mj_setConst` 重建 MuJoCo model 常量。
- 默认原地恢复，避免重新加载 XML 导致 controller、gripper、history buffer 和 observable cache 被重置；需要跨环境恢复时可显式使用 `reload_xml=True`。
- 校验模型 XML SHA-256，防止把 checkpoint 恢复到不兼容的模型。
- 对 MuJoCo 恢复时自动注入的派生 `<statistic>` XML 节点做稳定化处理，支持同一环境连续多次回退。
- 保存环境 `timestep`、`cur_time`、`done`、控制器/夹爪运行状态、observable cache 和采样时钟。
- 保存 Python `random`、NumPy 全局 RNG、局部 `np.random.Generator`，保证随机动作 replay 可重复。
- 可选保存 pi0.5 的 pending action chunk；rollout 中用 `--restore-action-plan` 选择恢复动作块，否则回退后重新请求策略。
- 提供无渲染、跨环境和双相机渲染验证脚本。

## 语义关键帧选择

`SemanticKeyframeSelector` 在每个环境状态上计算一个轻量语义分数：

```text
semantic score = motion + gripper transition + object motion
               + reward/progress change + direction turn
               + terminal/success signal
```

满足语义阈值时保存 checkpoint；高于 `force_score` 的强语义事件或距离上一个 checkpoint 达到 `max_interval` 时强制保存。`min_interval` 用于抑制连续重复状态，episode start、success、terminal、强语义事件和 max-interval 事件可以突破该限制。失败发生在 `failure_step` 后时，`select_for_rollback()` 只从严格早于失败的 checkpoint 中选择，默认使用最近的安全关键帧，最大化已保留任务进展。

基础用法：

```python
from semantic_checkpoint import SemanticKeyframeSelector

selector = SemanticKeyframeSelector()
selector.maybe_capture(env, 0, obs)
for step in range(1, horizon + 1):
    obs, reward, done, info = env.step(action)
    selector.maybe_capture(env, step, obs, reward, done, info, action=action)

keyframe, report = selector.restore_for_failure(env, failure_step=step)
```

该选择器只负责“保存哪些状态”和“回退到哪个状态”，不绑定具体失败检测器。后续可直接把碰撞检测、任务进度模型或 learned failure score 接入 `info`，再调整 `SemanticKeyframeConfig` 的阈值。

## 环境要求

代码针对以下环境验证：

- Python 3.8/3.11
- LIBERO
- robosuite 1.4
- MuJoCo 3.2.3
- NumPy
- （pi0.5 示例）`openpi_client`、Pillow

依赖应从已有的 LIBERO/OpenPI 环境安装；仿真服务器可以完全离线运行，不需要在运行时访问网络。

## 快速验证

在已配置 LIBERO 的环境中，从仓库根目录执行：

```bash
PYTHONPATH=. python libero_rollback_tool.py \
  --no-render \
  --pre-steps 5 \
  --post-steps 8 \
  --action-mode random \
  --random-action-scale 0.02 \
  --tolerance 1e-10 \
  --observation-tolerance 1e-10
```

验证脚本：

```bash
PYTHONPATH=. python validation/round1_norender_validation.py

# 需要 MuJoCo 的 OSMesa 软件渲染支持
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH=. python validation/cross_render_validation.py

# 跨两个独立 LIBERO 环境恢复（无相机）
PYTHONPATH=. python validation/cross_env_validation.py

# 语义关键帧保存、故意失败分支、回退和确定性 replay
PYTHONPATH=. python validation/semantic_keyframe_rollback_validation.py
```

验证报告可通过 `--output report.json` 保存；视频和渲染不可用时，工具会自动回退为 PPM 帧和 NPZ 文件。

## pi0.5 rollout 示例

先启动 OpenPI policy server，再在同一环境中执行：

```bash
PYTHONPATH=. python pi05_libero_rollback_rollout.py \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint-policy semantic \
  --checkpoint-max-interval 20 \
  --demo-failure-step 70 \
  --video-output ./artifacts/pi05_rollback.mp4 \
  --output ./artifacts/pi05_rollback_report.json
```

默认使用语义关键帧策略；`--checkpoint-policy periodic` 可切换回固定间隔基线。默认流程是“回退后清空尚未执行的动作块并重新规划”。若要验证 checkpoint 中保存的动作块恢复，可追加：

```bash
--restore-action-plan
```

该脚本中的失败分支是用于演示 rollback 的可控扰动，不代表最终的失败检测器。实际研究流程应将 `demo_failure_step` 替换为任务成功判定、碰撞/越界检测或 learned failure detector，并在其上实现 checkpoint 选择策略。

## 设计边界

本仓库解决的是“恢复到同一仿真状态并可继续 replay”的基础问题，不预设 checkpoint 间隔、失败评分或蒙特卡洛搜索策略。后续可在 `EnvRollbackTool` 之上增加：

- 根据任务进展和失败概率自适应设置不均匀 checkpoint 间隔；
- 失败检测与候选 checkpoint 选择；
- 多分支 rollout、轨迹评分和去重；
- 将成功分支及其恢复元数据导出为训练数据集。

## 已验证结果

在远程 LIBERO/MuJoCo 环境中已完成：

- 原地无渲染恢复：integration state、观测、动作序列和 reward 差异均为 `0.0`；
- 跨环境无渲染恢复：model state、integration state、runtime state、RNG 和 6 步 replay 通过；
- OSMesa 双相机原地及跨环境恢复：初始图像和 replay 图像最大绝对差均为 `0.0`；
- 语义关键帧端到端测试：生成 5 个关键帧（0、2、4、12、20），在失败步 18 选择第 12 步关键帧，执行 4 步扰动后恢复，11 步 replay 全部一致；
- `py_compile`、pi0.5 CLI help 及 action-plan 恢复单元检查通过。

CUDA/EGL 渲染不是上述验证的必要条件；共享 GPU 忙碌时建议使用 OSMesa，避免影响其他任务。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `libero_rollback_tool.py` | checkpoint 数据结构、捕获/恢复、比较和验证 CLI |
| `semantic_checkpoint.py` | 语义关键帧检测、checkpoint 密度控制和回退点选择 |
| `pi05_libero_rollback_rollout.py` | pi0.5 policy rollout 与 rollback 演示 |
| `validation/round1_norender_validation.py` | 原地无渲染确定性 replay |
| `validation/cross_env_validation.py` | 新建环境后的 XML/model/runtime/RNG 恢复 |
| `validation/cross_render_validation.py` | 双相机渲染一致性验证 |
| `validation/semantic_keyframe_rollback_validation.py` | 关键帧选择到回退 replay 的端到端验证 |
| `docs/REMOTE_TEST_STEPS.md` | 远程服务器运行说明与故障排查 |
| `docs/SEMANTIC_KEYFRAME_DESIGN.md` | 需求、评分、选择策略和参数调节说明 |

## License

MIT License，详见 [LICENSE](LICENSE)。
