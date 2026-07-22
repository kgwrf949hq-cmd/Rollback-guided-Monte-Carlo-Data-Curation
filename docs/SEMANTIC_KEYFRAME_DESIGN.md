# 语义关键帧 checkpoint 设计

## 1. 整体需求

当前系统已在指定 LIBERO/MuJoCo 配置的短轨迹上验证原生积分状态与白名单运行态恢复；这不等于任意 wrapper 的完整状态。checkpoint 选择层需要解决的是“什么时候保存”和“失败后回到哪里”。

### 功能需求

1. 在 rollout 状态流中识别有任务意义的状态变化，并保存语义关键帧。
2. 没有明显事件时仍然以最大步数间隔保存恢复点，避免长区间无 checkpoint。
3. 抑制连续重复保存，控制存储和捕获开销。
4. 每个 checkpoint 记录触发原因、分数、特征和动作元数据，支持数据集筛选和论文分析。
5. 纯时间策略只声称 checkpoint 严格早于失败步；声称“安全”时必须提供 warning onset 与安全余量。
6. 选择层不改变现有 `Checkpoint` 格式，实际状态恢复继续由 `EnvRollbackTool` 完成。

### 非功能需求

- 不依赖相机渲染，优先使用低维 observation 和环境返回的 `info`。
- 不绑定某一个 LIBERO task；有标准字段时使用标准字段，没有时使用数值 observation 兜底。
- 每步只做小规模向量差分，避免把 checkpoint 选择变成 rollout 瓶颈。
- 有限内存预算；超过 `max_keyframes` 时优先删除最旧的普通关键帧，保留 episode 边界和最新关键帧。
- 同样的输入状态流和阈值应产生同样的关键帧序列。
- 允许后续接入碰撞检测、任务进度模型或 learned failure detector，而不修改底层恢复逻辑。

底层同时保存严格 XML SHA-256 和恢复兼容签名。只有兼容签名忽略 MuJoCo 可能注入的派生 `<statistic>`；严格哈希始终区分不同 statistic XML。

## 2. 关键帧语义

`SemanticKeyframeSelector` 将当前状态与上一个状态比较，计算：

```text
S_t = w_m · motion_t
    + w_g · gripper_transition_t
    + w_o · object_motion_t
    + w_r · reward_change_t
    + w_p · progress_change_t
    + w_turn · direction_turn_t
    + terminal/success bonuses
```

当前实现的语义事件包括：

- `episode_start`：第一个状态，保证每个 rollout 至少有一个恢复点；
- `eef_motion`：末端执行器位移超过阈值；
- `motion_turn`：连续运动方向发生明显改变，常对应接近、绕行或放置动作阶段切换；
- `gripper_transition`：夹爪状态发生变化；
- `object_motion`：物体低维状态发生变化；
- `reward_change` / `progress_change`：任务反馈或显式进度信号变化；
- `success` / `terminal`：成功或终止状态；
- `max_interval`：没有语义事件时的周期性兜底。

其中 `score_threshold` 决定普通语义事件是否保存，`force_score` 决定强语义事件是否可以突破 `min_interval`，`max_interval` 是恢复覆盖的硬上限。

## 3. 选择和回退算法

每个环境状态执行以下步骤：

```text
observe state_t
    │
    ├─ extract eef / gripper / object / reward / progress features
    ├─ compare with state_(t-1)
    ├─ calculate semantic score and labels
    ├─ if start/success/terminal/high-score/max-interval: capture
    ├─ else if score >= score_threshold and interval >= min_interval: capture
    └─ otherwise skip
```

检测到失败步 `f` 后：

```python
eligible = [k for k in keyframes if k.step < f and not k.checkpoint.env_done]
rollback_keyframe = max(eligible, key=lambda k: k.step)  # latest_before_failure
EnvRollbackTool.restore(env, rollback_keyframe.checkpoint)
```

`best_score` 用于比较“最近状态”和“语义置信度最高状态”。`latest_safe` 需要 `warning_onset_step`，并在选点后剪除废弃分支上更晚的 checkpoint。

## 4. 调用接口

```python
selector = SemanticKeyframeSelector(
    config=SemanticKeyframeConfig(
        min_interval=3,
        max_interval=15,
        score_threshold=1.0,
    )
)

selector.maybe_capture(env, 0, obs)
for step in range(1, horizon + 1):
    obs, reward, done, info = step_env(env, action)
    selector.maybe_capture(
        env,
        step,
        obs,
        reward,
        done,
        info,
        action=action,
        rng=rng,
    )

keyframe, restore_report = selector.restore_for_failure(
    env,
    failure_step=failure_step,
    strategy="latest_safe",
    warning_onset_step=warning_onset_step,
    safety_margin_steps=2,
    rng=rng,
)
```

每个关键帧的 `Checkpoint.metadata` 中会保存：

- `semantic_score`；
- `semantic_labels`；
- `semantic_features`；
- 捕获时 reward、done 和 action；
- 外部 task、seed、rollout phase 等元数据。

## 5. 参数调节建议

- 机器人动作频率较高、状态变化平滑时，提高 `min_interval`；
- 任务有短暂接触或夹爪事件时，降低 `gripper_threshold` 和 `object_motion_threshold`；
- 任务长、失败代价高时，降低 `max_interval`；
- 长 rollout 或多分支采样时，设置合适的 `max_keyframes`，避免 checkpoint 内存无限增长；
- 关键动作必须保留时，降低 `score_threshold` 或提高对应事件权重；
- 先用验证脚本观察 `keyframe_summary` 的 label 分布，再在多任务数据上统计 checkpoint 数量、恢复成功率和平均回退距离。

当前算法是可解释的启发式基线，不替代最终的 learned failure detector。后续蒙特卡洛数据构造应保留成功、可恢复失败、终止失败与不确定分支，并记录分支 lineage、warning onset、回退点和原始真值；失败分支是训练与偏好数据，不应被默认丢弃。

## 6. Pick-and-place 第一版

`pick_place_checkpoint.py` 将上述策略具体化为五个有序子事件：

```text
approach_object
      ↓
grasp_completed
      ↓
object_lifted
      ↓
place_completed
      ↓
release_completed
```

人工为任务实例绑定 `target_object` 和 `goal_object`，并调整距离、高度、夹爪开合和稳定步数阈值。生产配置默认 `require_grasp_truth=true`、`require_place_truth=true`、`allow_info_event_overrides=false`：grasp 使用 robosuite `_check_grasp`，place/release 使用 LIBERO BDDL `On` 和 contact。`info` 事件覆写只能在显式开启的合成 fixture 中使用。

人工配置放在 `configs/pick_place/*.json`，可以同时设置任务阈值和下列事件边策略；`load_pick_place_config()` 负责校验顺序并加载，不需要为每个 LIBERO task 改 Python 源码。

每条事件边都有独立 `EventEdgePolicy`：

```text
start → approach       risk=0.15, heuristic=off, max_interval=30
approach → grasp       risk=1.00, heuristic=on,  max_interval=6
grasp → lifted         risk=0.90, heuristic=on,  max_interval=7
lifted → placed        risk=0.45, heuristic=on,  max_interval=15
placed → released      risk=1.00, heuristic=on,  max_interval=5
```

事件完成点总是强制保存；低风险接近边不使用运动分数，只按最大间隔兜底；高风险边通过风险系数降低分数阈值并缩短最大间隔。这样可以把 checkpoint 预算集中到抓取、抬升、放置和释放等真正容易失败的阶段。
