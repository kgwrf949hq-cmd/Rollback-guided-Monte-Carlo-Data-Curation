# Rollback-guided Monte Carlo Data Curation

用于 LIBERO/robosuite/MuJoCo 机器人轨迹筛选的 checkpoint/rollback 原型。当前证据支持的准确定位是：在当前 LIBERO/MuJoCo 配置上完成了单任务、短轨迹、原生 `mjSTATE_INTEGRATION` 路径的确定性恢复，并实现了单进程、内存态、预算受限 BFS 的第一版蒙特卡洛 curation coordinator；尚未接入真实 pi0.5 自动分支闭环或数据集 writer，不能称为完整状态系统或优质数据集。

## 架构概览

```text
LIBERO environment
        │
        ▼
EnvRollbackTool.capture()
        │  Checkpoint
        ├─ MuJoCo integration state (mjSTATE_INTEGRATION)
        ├─ selected model arrays / strict + compatibility XML signatures
        ├─ selected environment/controller runtime state
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

实现分层如下：

1. `libero_rollback_tool.py` 是与策略无关的状态快照和恢复层。
2. `semantic_checkpoint.py` 是 checkpoint 选择层：根据运动、夹爪、物体、奖励、任务进度和终止信号保存语义关键帧，并保证最大 checkpoint 间隔。
3. `pick_place_checkpoint.py` 和 `pick_place_failure.py` 分别提供 pick-and-place 事件锚点与无副作用失败判定。
4. `rollback_policy.py` 是独立回退点策略层：默认按失败原因选择阶段锚点，锚点不可用时回退到严格时间策略，本身不 restore 环境。
5. `monte_carlo_coordinator.py` 管理 curation-only FIFO/BFS、lineage、预算和所有 rollout 副作用；`pi05_libero_rollback_rollout.py` 仍是尚未接入 coordinator 的单次演示脚本。

## 核心功能

- 使用 MuJoCo 原生 `mjSTATE_INTEGRATION` 保存和恢复积分状态；旧绑定回退路径会另行恢复并读回核验 `act/ctrl/mocap/time`。
- 同步 `ctrl`、mocap、activation、solver warm-start、applied forces 等容易被普通 `sim.set_state()` 遗漏的字段。
- 保存并恢复 model 基础数组及编译派生常量，调用 `mj_setConst` 重建 MuJoCo model 常量。
- 默认原地恢复，避免重新加载 XML 导致 controller、gripper、history buffer 和 observable cache 被重置；需要跨环境恢复时可显式使用 `reload_xml=True`。
- 使用严格 XML SHA-256 标识实际 XML；另设只忽略 MuJoCo 派生 `<statistic>` 的兼容签名。兼容恢复不会再被报告成严格哈希相同。
- replay 对 observation key、类型和 shape 做严格比较；没有可比较字段、缺 key 或 shape 不同都不能通过。
- 保存并逐字段读回核验白名单中的环境 `timestep/cur_time/done`、控制器/夹爪运行状态、observable cache 和采样时钟。该白名单不代表任意 wrapper 的完整 Python 状态。
- 保存 Python `random`、NumPy 全局 RNG、局部 `np.random.Generator`，保证随机动作 replay 可重复。
- 提供 `make_branch_rng(root_seed, branch_id)` 派生独立分支 RNG；蒙特卡洛分支应使用 `restore_rng=False`，避免每个子分支重放同一随机序列。
- 可选保存 pi0.5 的 pending action chunk；rollout 中用 `--restore-action-plan` 选择恢复动作块，否则回退后重新请求策略。
- 提供无渲染、跨环境和双相机渲染验证脚本。
- 提供 pick-and-place 子事件状态机；生产配置中的 grasp 使用 robosuite `_check_grasp`，place/release 使用 LIBERO BDDL `On`、contact 和完整 `check_success` 真值。
- 子事件完成点强制保存；每条子事件边独立配置风险系数、启发式开关和 checkpoint 间隔。

## 语义关键帧选择

`SemanticKeyframeSelector` 在每个环境状态上计算一个轻量语义分数：

```text
semantic score = motion + gripper transition + object motion
               + reward/progress change + direction turn
               + terminal/success signal
```

满足语义阈值时保存 checkpoint；高于 `force_score` 的强语义事件或距离上一个 checkpoint 达到 `max_interval` 时强制保存。默认 `latest_before_failure` 只表示“时间上早于报警”，不声称安全。`latest_safe` 必须由调用方提供 `warning_onset_step` 和安全余量；恢复成功后会剪掉选中点之后、属于废弃未来分支的 checkpoint。

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

## Pick-and-place 子事件与风险策略

`pick_place_checkpoint.py` 在现有 selector 上增加任务语义层。接近与运动特征来自 observation；生产 grasp/place/release 标签还必须通过 robosuite/LIBERO 的接触与 BDDL 真值：

```python
from pick_place_checkpoint import (
    PickPlaceRiskAwareCheckpointManager,
    PickPlaceTaskSpec,
)

manager = PickPlaceRiskAwareCheckpointManager(
    PickPlaceTaskSpec(
        target_object="akita_black_bowl_1",
        goal_object="plate_1",
    )
)
manager.observe(env, step, obs, reward, done, info, action=action)
```

默认风险策略会让接近物体阶段保持稀疏，仅保留起点、事件完成点和最大间隔点；抓取、抬升、放置和释放边启用更密集的启发式 checkpoint。策略不是按动作位移大小直接保存，而是由当前子事件边的 `risk_coefficient` 调整分数阈值和最大间隔。

## 第一版失败检测

`pick_place_failure.py` 是独立的阶段感知规则检测器，不保存/选择 checkpoint，也不调用 restore 或 replay：

```python
from pick_place_failure import PickPlaceRuleFailureDetector

detector = PickPlaceRuleFailureDetector(task_spec)
decision = detector.update(
    obs,
    step=step,
    completed_events=event_detector.completed_events,
    action=action,
    reward=reward,
    done=done,
    info=info,
    env=env,
)
```

输出包含 `normal / warning / recoverable_failure / terminal_failure`、风险分数、失败原因、当前阶段、连续确认步数和证据。第一版覆盖接近无进展、抓取失败、抬升无进展、物体掉落、运输/放置停滞、放置超时、错误释放、工作空间越界、无效 observation、仿真异常及未成功终止。

失败检测器提供可序列化的 `snapshot_state()/restore_state()`。回退到 checkpoint 时会恢复 history window、stage 起点、warning onset 和连续确认状态；审计用 decisions 由 branch record 单独保存，不会从失败 sibling 污染新分支。

## 第一版蒙特卡洛 curation coordinator

`monte_carlo_coordinator.py` 在上述组件外管理所有副作用。第一版固定使用 curation 语义：每条 branch 运行到 verified success、确认失败、环境终止或完整 episode horizon；成功只结束当前 branch，不取消 FIFO frontier 中的其他 sibling。

主要约束：

- 单进程顺序复用一个环境，使用预算受限 FIFO/BFS，先完成同层 sibling 再扩展下一层；
- stable branch/checkpoint ID、完整 transitions、lineage、rollback selection 和 restore report 全部保存在内存中；
- 每个 child 只继承 fork checkpoint 之前的活动 checkpoint view，全局 registry 和失败轨迹不删除；
- 默认使用阶段感知回退策略：例如 missed grasp 回到 `approach_object`，drop 优先回到 `object_lifted`，release failure 回到 `place_completed`；
- 阶段锚点必须是当前 lineage 中真实的 subevent-completion checkpoint，且仍必须严格早于 warning cutoff；锚点缺失时自动回退到 `latest_safe`/`latest_before_failure`；
- child 使用 `make_branch_rng()` 和 `restore_rng=False`，并恢复 subevent、selector 与 failure detector 的 branch-local 状态；
- sibling 的环境状态保持完全相同，只允许模型原生采样或结构化动作扰动；代码不提供环境扰动入口；
- 默认只有 `env.check_success`（LIBERO 任务真值）能产生正标签，info-only success 会进入 quarantine；
- runner/observation/simulation corruption 进入 quarantine，不伪装成任务失败。

动作生成通过 `BranchActionSampler` 解耦。`StructuredActionChunkSampler` 接收一个 `(observation, sampling_seed) -> action_chunk` 回调：服务端支持可控原生采样时，每个 branch 使用独立 seed 且不加人工噪声；否则同一 fork 共享基础 policy seed，attempt 0 只重新规划，attempt 1/2 分别在前六个运动维度加入低/中强度时间相关噪声，gripper 维度保持策略原值。

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

# pick-and-place 子事件、风险边策略和回退 replay
PYTHONPATH=. python validation/pick_place_checkpoint_validation.py

# 仅验证失败检测规则，不运行 checkpoint/restore/replay
PYTHONPATH=. python validation/pick_place_failure_validation.py

# 历史 observation/XML/旧 fallback 假阳性回归
PYTHONPATH=. python validation/rollback_regression_validation.py

# 无需 LIBERO 或 policy server 的蒙特卡洛 BFS/lineage/终止验证
PYTHONPATH=. python validation/monte_carlo_coordinator_validation.py

# 独立回退策略的阶段锚点、严格 cutoff 和 temporal fallback 验证
PYTHONPATH=. python validation/rollback_policy_validation.py
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

Pick-and-place 任务可启用事件风险策略：

```bash
PYTHONPATH=. python pi05_libero_rollback_rollout.py \
  --checkpoint-policy pick_place \
  --pick-place-config configs/pick_place/libero_spatial_task0.json
```

也可以不使用配置文件，直接传入 `--target-object akita_black_bowl_1 --goal-object plate_1`。JSON 配置把人工定义的物体绑定、事件阈值和每条事件边的风险系数集中保存，新增同类任务时不需要修改 Python 代码。

该脚本中的失败分支是用于演示 rollback 的可控扰动，不代表最终的失败检测器。报告将运行是否异常记为 `run_ok`，将任务真值单列为 `task_success` 与 `outcome`，不再把脚本未抛异常写成任务成功。

## 设计边界

本仓库已经在 `EnvRollbackTool` 之上提供可解释的 checkpoint、失败检测和第一版 BFS curation coordinator；当前仍不提供学习式 checkpoint/branch 评分或 durable writer。后续可以增加：

checkpoint 中的 Python/NumPy 进程级 RNG 只用于单进程确定性 replay，不应作为并行环境的共享随机状态。

- 根据任务进展和失败概率自适应设置不均匀 checkpoint 间隔；
- learned failure risk 与候选 checkpoint/branch 质量评分；
- action proposal 去重、winner 选择和真实 pi0.5 shadow/active rollout；
- 保留成功、可恢复失败、终止失败和不确定分支及其 lineage；不确定或验证器分歧样本应隔离，而不是冒充成功/失败标签；
- 导出带 provenance、版本、校验和及 train/validation 划分的数据集。

## 已验证结果

在远程 LIBERO/MuJoCo 环境中已完成：

- 原地无渲染恢复：integration state、观测、动作序列和 reward 差异均为 `0.0`；
- 跨环境无渲染恢复：model state、integration state、runtime state、RNG 和 6 步 replay 通过；
- OSMesa 双相机原地及跨环境恢复：初始图像和 replay 图像最大绝对差均为 `0.0`；
- 语义关键帧端到端测试：生成 5 个关键帧（0、2、4、12、20），在失败步 18 选择第 12 步关键帧，执行 4 步扰动后恢复，11 步 replay 全部一致；
- 生产 pick-and-place 配置在无事件注入的短轨迹中生成 checkpoint `[0, 30]`；这是恢复测试，不是五个真实事件有效性测试；
- task0 初态中心距约 `0.11693m` 时真实 `contact=False/On=False/success=False`；构造 MuJoCo 接触态后 `contact=True/On=True/success=True`，移除接触并恢复后真值重新一致；
- 失败规则的合成行为与真实零动作 liveness 通过；零动作会触发 stall，明确不作为 precision、recall 或低误报证据；
- observation key/shape 假阳性、`<statistic>` 严格哈希、旧 fallback 的 `act` 恢复均有独立回归；
- `py_compile` 与当前 CLI 检查通过；真实 pi0.5 policy server 闭环仍未验证。

CUDA/EGL 渲染不是上述验证的必要条件；共享 GPU 忙碌时建议使用 OSMesa，避免影响其他任务。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `libero_rollback_tool.py` | checkpoint 数据结构、捕获/恢复、比较和验证 CLI |
| `semantic_checkpoint.py` | 语义关键帧检测、checkpoint 密度控制和回退点选择 |
| `pick_place_checkpoint.py` | pick-and-place 子事件状态机和风险边 checkpoint 管理器 |
| `pick_place_failure.py` | 独立的阶段感知规则失败检测器 |
| `rollback_policy.py` | 无环境副作用的阶段感知/时间回退点策略与审计结果 |
| `monte_carlo_coordinator.py` | curation-only BFS、branch records、稳定 ID、策略调度、预算/重复失败保护及动作采样接口 |
| `configs/pick_place/*.json` | 人工定义的目标物体、目标区域、事件阈值和风险边配置 |
| `pi05_libero_rollback_rollout.py` | pi0.5 policy rollout 与 rollback 演示 |
| `validation/round1_norender_validation.py` | 原地无渲染确定性 replay |
| `validation/cross_env_validation.py` | 新建环境后的 XML/model/runtime/RNG 恢复 |
| `validation/cross_render_validation.py` | 双相机渲染一致性验证 |
| `validation/semantic_keyframe_rollback_validation.py` | 关键帧选择到回退 replay 的端到端验证 |
| `validation/pick_place_checkpoint_validation.py` | pick-and-place 子事件、风险密度和回退 replay 验证 |
| `validation/pick_place_failure_validation.py` | 失败规则、时序确认和真实零动作 liveness 验证 |
| `validation/rollback_regression_validation.py` | observation/XML/fallback 历史假阳性回归 |
| `validation/monte_carlo_coordinator_validation.py` | 无 policy server 的 BFS、lineage、RNG、终止、隔离和可序列化验证 |
| `validation/rollback_policy_validation.py` | 失败原因到阶段锚点、strict cutoff 与 temporal fallback 验证 |
| `docs/REMOTE_TEST_STEPS.md` | 远程服务器运行说明与故障排查 |
| `docs/SEMANTIC_KEYFRAME_DESIGN.md` | 需求、评分、选择策略和参数调节说明 |
| `docs/FAILURE_DETECTION.md` | 失败规则、输出语义、验证范围和未来风险融合公式 |
| `docs/HANDOFF.md` | 修复版事实状态、失败分支数据语义及蒙特卡洛 coordinator 实施约束 |

## License

MIT License，详见 [LICENSE](LICENSE)。
