# Rollback-guided Monte Carlo Data Curation 交接文档

最后核对：2026-07-22（第一版 coordinator 本地验证后）

## 0. 新对话先读结论

当前项目的准确定位是：

> 已完成“单任务、短轨迹、当前 LIBERO/MuJoCo 配置、原生 `mjSTATE_INTEGRATION` 路径”的确定性恢复原型，并实现单进程、内存态、预算受限 FIFO/BFS 的第一版蒙特卡洛 curation coordinator；尚未接入真实 pi0.5 自动分支闭环、真实策略有效性评估或数据集 writer。

第一版 coordinator 已通过无需 LIBERO/policy server 的确定性 fixture；下一阶段应接入真实 pi0.5 shadow mode 和小规模远端验证，但不应开始批量采集，也不应把当前内存产物称为优质数据集。

最重要的数据约定：

> 失败轨迹是有价值的数据，必须保留。回退后的“分支剪枝”只从当前 checkpoint 候选集合中移除废弃未来分支的 checkpoint，绝不删除已经发生的失败轨迹、transition、动作、真值、失败原因或 branch lineage。

只有以下样本不应直接进入带确定标签的训练集：状态/观测损坏、runner 或基础设施错误、真值未知、规则与任务真值冲突、验证器分歧。它们应进入 quarantine，而不是伪装成成功或失败样本。

---

## 1. 版本与 source of truth

本地仓库：

```text
/Users/chenxiaofu/Desktop/程设/Rollback-guided-Monte-Carlo-Data-Curation
```

远端运行目录：

```text
openpi-server:/home/chenxiaofu/openpi
```

版本关系：

- 本轮修复基于 `2df5f11` 开发；同步前 GitHub `main=a31e0ea`，该远端提交只删除了旧的 `docs/HANDOFF.md`，功能代码没有变化。
- 本文档所在的后续 `main` revision 包含严格 replay/XML/restore、任务真值、回退语义、分支 RNG 和验证修复；实际 revision 以 `git rev-parse HEAD` 为准。
- `/home/chenxiaofu/openpi` 不是 Git 仓库；修复后的功能代码和验证脚本已通过 `scp` 同步并在远端验证。
- GitHub `main` 是完成同步后的代码 source of truth；不要用远端运行目录中的旧文件反向覆盖 Git 仓库。

开始新功能前先执行：

```bash
cd /Users/chenxiaofu/Desktop/程设/Rollback-guided-Monte-Carlo-Data-Curation
git status --short --branch
git diff --check
git diff
```

同步完成后这里应是 clean worktree。若不 clean，先分辨已有修改的来源，不要 reset 或覆盖；蒙特卡洛功能应使用独立、可审查的后续提交。

当前核心文件 SHA-256（用于确认远端是否仍与本地修复版一致）：

| 文件 | SHA-256 |
| --- | --- |
| `libero_rollback_tool.py` | `03d9687a0ddb3f0af2de3e3a6a294a6b38d0877cd8049cacbc0dbf668f5fdfef` |
| `semantic_checkpoint.py` | `e406ef1f35475a06edc9fbf305adc02251de13eb7170299fc9173c0f508adc44` |
| `pick_place_checkpoint.py` | `4d4019944363aea367c6a53cd9d715cca5b0e5cd105641d712c1c5cd98f7bde9` |
| `pick_place_failure.py` | `97d7bc87d9a833f4fc32f6d43fc263ae22c7aa265ca9874a52241e410c52f311` |
| `rollback_policy.py` | `83a1c3923c07cba9da94b557dfdceeea994c882c88112b3cc73edfbc1d7e3683` |
| `monte_carlo_coordinator.py` | `df908135808553ff4bc40519f80641373c645784d22daf56c5405bd8065aa9bf` |
| `pi05_libero_rollback_rollout.py` | `ed6ad6fcebf64b15e9a26d253d4c6e24496fff738f77572587cab30a245af31a` |
| `configs/pick_place/libero_spatial_task0.json` | `e270d6664497a7f363eeb4530207d782e1afdd25d897b4fc0d35e8f346b1db4e` |
| `validation/rollback_regression_validation.py` | `ef86d188ff2d8a31d911f3aa23c7ee602672cb6102f382a520186a08b3e42527` |
| `validation/monte_carlo_coordinator_validation.py` | `914ba252854c0bbc174ade3d6ba0fbfd8f49a59611dcca860aca1cbd6ca205d8` |
| `validation/rollback_policy_validation.py` | `7e31a4c6a02de05bcd9c15a9ac3fb1e547971b42918b3973e77bfdabc8b7466b` |

---

## 2. 当前代码分层

```text
EnvRollbackTool                 状态 capture / restore / 严格 replay 比较
        ↑
SemanticKeyframeSelector        通用 checkpoint 捕获与时间/告警选点
        ↑
PickPlaceRiskAwareCheckpointManager
                                pick-place 事件、任务真值与边风险

PickPlaceRuleFailureDetector    无副作用失败检测，只返回 FailureDecision

RollbackPolicy                  无副作用回退点选择，返回可审计 RollbackSelection

MonteCarloCoordinator           curation-only FIFO/BFS、lineage、预算与副作用

pi05_libero_rollback_rollout.py 单次演示脚本；仍不是蒙特卡洛 coordinator
```

coordinator 已将上述层次组成可测试的内存闭环。失败检测器和回退策略都不直接 restore、采样或写数据；所有副作用由 coordinator 管理。真实 pi0.5 demo 尚未切换到该 coordinator。

---

## 3. 已修复的问题

### 3.1 replay 假阳性

`libero_rollback_tool.py` 现在严格检查 observation：

- key 集合不同：失败；
- 同名字段类型或 shape 不同：失败；
- 没有任何可比较 observation：失败；
- 空 replay trace：失败；
- 只有确实比较过的数值或图像字段满足容差时才能 PASS。

因此旧报告中的 `success=True` 不能自动继承可信度，必须用修复后的比较器重跑。

### 3.2 XML 身份误判

模型 XML 现在保留两个概念：

- 严格 SHA-256：任何显式 `<statistic>` 差异都会改变哈希；
- 窄范围 compatibility signature：只兼容“原 XML 无 statistic，MuJoCo 恢复后自动派生 statistic”的重复恢复情况。

两个显式但不同的 `<statistic>` XML 会被拒绝，不再报告成相同模型。

### 3.3 restore report 过度乐观

restore report 改为逐字段读回核验，不再把“字段存在”或“尝试写入”当成恢复成功。核验覆盖当前白名单中的：

- integration state 与 `qpos/qvel/act/ctrl/mocap/time`；
- 选定 model state；
- environment/controller/gripper/observable runtime state；
- Python、NumPy 与可选局部 Generator RNG；
- 恢复后的 observation。

旧 MuJoCo `sim.set_state` fallback 会额外补回 `act` 等遗漏字段并读回验证。

边界仍然存在：model/runtime 是白名单子集，不是任意 Python wrapper 的“完整环境状态”。准确表述始终是“当前配置的短轨迹恢复原型”。

### 3.4 pick-place 标签真值

生产配置 `configs/pick_place/libero_spatial_task0.json` 现在默认：

```json
{
  "require_grasp_truth": true,
  "require_place_truth": true,
  "allow_info_event_overrides": false
}
```

- grasp 使用 robosuite `_check_grasp`；
- place/release 使用 LIBERO BDDL `On`、真实 contact、目标绑定及 `env.check_success()`；
- 生产路径不接受人工 `info["<event>_completed"]` 冒充事件；
- 合成 predicate fixture 可以显式关闭真值要求，但结果与生产配置分开报告。

task0 初态的已核验真值：

```text
碗与盘中心距离 = 0.1169336095 m
contact          = False
On               = False
task success     = False
```

因此中心距离小于旧规则的 `0.15m` 不再足以产生“安全释放”标签。构造真实 MuJoCo 接触态后，`On=True`、`success=True`；移除接触并恢复该 checkpoint 后，真值也能恢复为真。

### 3.5 checkpoint 选点的安全语义

`semantic_checkpoint.py` 的默认策略已改为：

```python
strategy="latest_before_failure"
```

它只声称 checkpoint 在失败报警之前，不声称安全。

只有调用方提供 `warning_onset_step` 时才能使用：

```python
strategy="latest_safe"
cutoff = warning_onset_step - safety_margin_steps
selected.step < cutoff
```

失败检测的 `FailureDecision` 已新增 `warning_onset_step`。restore 成功后，selector 会从当前活动候选集合中剪掉 selected step 之后的 checkpoint，避免第二次回退选到废弃未来分支；restore 失败时不剪。

这里的 `latest_safe` 只是“位于已知 warning onset 之前并留有离散步数余量”的操作性约定，不是碰撞、接触或可达性层面的物理安全证明。

再次强调：`prune_after()` 只处理 checkpoint 候选，不是轨迹数据删除 API。

### 3.6 失败检测和成功标签

- 失败检测器可读取 `env` / `PickPlaceTaskTruth`，会调用真实 `check_success()`；
- `done=True` 且没有已验证成功会标为 terminal failure；
- 软失败会记录最初 warning step，连续确认后再升级；
- 21 步零动作真实 smoke 不再被包装成“零误报”，而是主动触发 stall 的 liveness 验证；
- pi0.5 demo 报告已拆为 `run_ok`、`task_success`、`task_success_source` 和 `outcome`。`run_ok=True` 只代表脚本正常结束，不是任务成功。

### 3.7 分支随机数

`libero_rollback_tool.py` 新增：

```python
make_branch_rng(root_seed, branch_id)
```

相同 `(root_seed, branch_id)` 可重复，不同 `branch_id` 产生不同随机序列。未来从同一 checkpoint 派生子分支时应：

```python
branch_rng = make_branch_rng(root_seed, branch_id)
tool.restore(env, checkpoint, rng=branch_rng, restore_rng=False)
```

`restore_rng=False` 很重要：仿真状态仍恢复到同一点，但不会用 checkpoint 中的旧 Generator 状态覆盖独立分支 RNG。进程级 Python/NumPy RNG 恢复只适合单进程确定性 replay，不应作为并行环境的共享随机状态。

---

## 4. 当前验证证据

修复版已在 `openpi-server` 上通过以下验证：

- 历史假阳性回归：不同图像 key/shape、空 trace 均不再 PASS；不同显式 statistic XML 被拒绝；旧 fallback 的 `act` 恢复通过；分支 RNG 可重复且分支间不同。
- 原地无渲染恢复：严格 restore/replay PASS，已比较字段差异为 `0.0`。
- 跨环境恢复：model/runtime/RNG/replay PASS。
- OSMesa 双相机：确实比较图像字段，像素最大差异为 `0.0`。
- 通用语义回退：失败步 18 选择 step 12，restore 后剪除 step 20，replay PASS。
- 无事件注入、未改生产间隔的 task0 短轨迹：checkpoint 为 `[0, 30]`；warning onset 为 23、安全余量为 1 时选中 step 0，并剪除 step 30。
- task0 真实 BDDL/contact 真值与接触态 restore 验证通过。
- place/release 真值门控能产生 `place_completed/release_completed`；中心距 `0.11693m` 但 `On=False/contact=False` 的开放夹爪状态会正确判为错误释放。
- 21 步零动作 liveness：`normal=7, warning=1, recoverable_failure=13`。这不是 precision、recall 或低误报证据。
- 系统 Python 3.8 已能在同一进程导入 LIBERO 与仓库内的 OpenPI client。

必须正确解释这些结果：

- 合成五事件 fixture 只证明状态机/门控逻辑，不证明真实 pi0.5 能完成事件；
- 构造接触态只证明真值读取与恢复，不等于真实 grasp/transport/drop rollout 覆盖；
- 零差异只覆盖已比较字段和当前短轨迹，不等于所有 wrapper 状态完整；
- 当前没有真实策略轨迹上的 precision、recall、warning lead time、误回退率或恢复后成功率。

---

## 5. 尚未实现

以下内容仍然不存在：

- pi0.5 policy server 闭环；
- 真实 pi0.5 sampling seed / 多 action candidate 协议；
- 候选分支质量评分、winner 选择与 action proposal 去重；
- durable episode/transition writer、原子落盘和断点恢复；
- provenance、代码/配置版本、校验和；
- train/validation 按 root episode 或 init state 隔离划分；
- 真实成功、missed grasp、drop、collision、wrong placement 的人工真值与指标。

`pi05_libero_rollback_rollout.py` 仍然只是一次人工坏动作分支的演示，不能扩展几行循环后就称为蒙特卡洛系统。

当前已经实现但只在内存/fake environment 中验收的内容包括：稳定 branch/checkpoint ID、完整 transition/lineage、branch-local checkpoint view、failure/selector/subevent 状态恢复、full-horizon FIFO/BFS、最大分支/深度/重试预算、重复失败保护、quarantine、可重复 branch RNG 和 JSON 可序列化摘要。

---

## 6. 失败分支的数据语义

“是否继续扩展某条分支”和“是否保留该分支为数据”是两个独立决定：

| 分支结果 | 是否保留原始轨迹 | 是否作为确定标签 | 是否继续扩展 |
| --- | --- | --- | --- |
| verified success | 保留 | 正样本 | 可停止或作为优质候选 |
| recoverable failure | 保留 | 负样本/恢复偏好样本 | 可从更早 checkpoint 新建 sibling |
| terminal failure | 保留 | 负样本 | 当前分支停止，可按预算从祖先另开分支 |
| uncertain / validator disagreement | 保留到 quarantine | 否 | 由审计策略决定 |
| runner/simulation corruption | 保留诊断元数据 | 否 | 停止并隔离，不当作任务失败 |
| duplicate failure | 保留最小 lineage/签名，轨迹按去重策略存一次 | 失败标签不变 | 停止重复扩展 |

失败轨迹至少可用于：失败检测训练、偏好对、credit assignment、恢复点学习、hard-negative mining、行为克隆中的反例过滤和失败模式覆盖分析。

不得只保存回退后的成功后缀。应保存：

```text
共同前缀
├── 失败 child：完整动作/状态/真值/失败原因
└── 恢复 child：从同一或更早 checkpoint 开始的新尝试
```

两条 child 通过 `parent_branch_id`、`fork_checkpoint_id` 和 `attempt_index` 关联，才能形成有价值的 counterfactual pair。

---

## 7. 已实现：第一版蒙特卡洛分支 coordinator

已新增：

```text
monte_carlo_coordinator.py
rollback_policy.py
validation/monte_carlo_coordinator_validation.py
validation/rollback_policy_validation.py
```

当前第一版只做单进程、小规模、内存中的 branch records；不包含大型 writer，也不依赖 policy server 才能测试。运行语义已经确定为 curation-only：成功只结束当前 branch，不取消 frontier 中的 sibling；每条 branch 都运行到成功、确认失败、done 或完整 episode horizon。BFS depth 是 fork/rollback 层数，不是轨迹步数。

### 7.1 必需的数据结构

至少定义：

- `BranchRecord`
  - `episode_id`
  - `branch_id`
  - `parent_branch_id`
  - `fork_checkpoint_id`
  - `fork_step`
  - `depth`
  - `attempt_index`
  - `root_seed` 与可重建的 branch seed/key
  - `status/outcome`
  - ordered transitions
  - `failure_decision`
  - `rollback_selection`
  - `restore_report`
- `TransitionRecord`
  - branch-local step 与 environment step
  - observation/action/reward/done
  - task truth 和 truth source
  - event stage
  - failure status/reason/risk/warning onset
  - checkpoint ID（若该步捕获）
- `RollbackSelection`
  - failure step、warning onset、安全余量
  - strategy 与 selection basis
  - 所有候选 checkpoint ID/step 及拒绝原因
  - 最终 checkpoint ID/step
  - 是否只有时间保证，禁止模糊写成 safe

ID 必须稳定、可序列化，不能使用 Python 对象地址。推荐由 `episode_id + parent_branch_id + fork checkpoint + attempt_index` 派生。

现有 `SemanticKeyframe.step` 在分支树中不唯一。coordinator 必须给 checkpoint 包一层稳定的 `checkpoint_id=(branch_id, step, ordinal)`，并且只能从当前 branch 及其祖先链选择候选点；不得把 sibling branch 上相同步数的 checkpoint 混入候选集合。

### 7.2 coordinator 责任

coordinator 才能执行副作用：

1. 记录当前 branch 的每个 transition；
2. 调用 checkpoint manager 观察并捕获候选点；
3. 调用失败检测器取得无副作用 `FailureDecision`；
4. 将原失败 branch 标记结束但完整保留；
5. 向独立 rollback policy 提供 branch-local checkpoint 候选并执行其选择结果；
6. 创建新的 child/sibling branch record；
7. 用 `restore_rng=False` 恢复仿真，用 `make_branch_rng()` 初始化该 child；
8. 恢复或重建 checkpoint manager、subevent detector 和 failure detector 的 branch-local 历史，避免 sibling 污染；
9. 默认清空旧 policy action chunk 并重新规划；
10. restore 成功后让 child 只继承 fork 之前的活动 checkpoint view，不删除全局 checkpoint 或父分支数据；
11. 根据最大深度、分支数、重试数和重复失败签名停止扩展。

subevent、semantic selector 和 failure detector 现在都具备 coordinator 所需的 snapshot/restore。failure detector 会恢复 history、初始/最大物体高度、stage 起点和 warning 连续计数；审计 decisions 留在原 branch，不复制到 sibling。

不要让 `PickPlaceRuleFailureDetector`、`SemanticKeyframeSelector`、`RollbackPolicy` 或数据 writer 反向控制环境。

### 7.3 已实现的回退选择规则

`rollback_policy.py` 已将“回退到哪一步”从 coordinator 抽成可注入的纯策略。默认使用 `StageAwareRollbackPolicy`，不需要人工为每次失败填 step：

```text
missed grasp / grasp timeout
    → approach_object 事件完成点

lift failure
    → grasp_completed，缺失时再尝试 approach_object

object dropped
    → object_lifted，缺失时再尝试 grasp_completed

placement failure
    → object_lifted，缺失时再尝试 grasp_completed

release failure
    → place_completed，缺失时再尝试 object_lifted

未配置的 failure reason
    → 根据 current_stage 自动推导 predecessor anchor
```

事件锚点只认真实 `subevent_completion`、对应的 `subevent_transition.event_name` 或事件 label；不会把“历史上完成过该事件”的更晚普通 checkpoint 误当成锚点。所有语义锚点还必须满足时间硬约束：

```text
warning_onset_step 已知
    → checkpoint.step < warning_onset_step - safety_margin_steps

warning_onset_step 未知
    → checkpoint.step < failure_step
```

请求的语义锚点缺失或不在 cutoff 前时，自动委托 `TemporalRollbackPolicy`：前一种情况使用严格 `latest_safe`，warning onset 缺失时使用 `latest_before_failure` 且 `selection_basis="temporal_only"`。没有 eligible checkpoint 时不回退也不删除失败轨迹。

`RollbackSelection` 保留 requested/selected anchor、fallback 原因、cutoff、每个候选的拒绝原因、评分分解与排名。当前每次失败仍只选一个 checkpoint，尚未同时比较多个回退点或实现 learned scoring。后续可增加：

```text
progress retained
+ semantic confidence
+ distance from warning onset
- predicted failure risk
- repeated failure penalty
- rollback distance/cost
```

学习式评分没有真实标签前不要冒充已实现。

### 7.4 重复失败保护

至少用以下字段构造稳定 failure signature：

```text
(fork_checkpoint_id, event_stage, failure_reason, quantized object/EEF state)
```

同一祖先 checkpoint 出现相同签名超过上限时停止扩展，记录 `stopped_duplicate_failure`。停止扩展不等于删除失败数据。

### 7.5 第一版验收测试

无需 policy server 的确定性测试必须覆盖：

1. 同一 root seed、同一 branch ID 可重复；不同 branch ID 动作不同。
2. 从同一 checkpoint 恢复两个 child，初始仿真状态相同，独立动作后轨迹分叉。
3. `latest_safe` 绝不选择 `warning_onset_step - margin` 及之后的点。
4. warning onset 缺失时不能偷偷声称 safe。
5. restore 失败时不剪 checkpoint；成功时只剪活动候选集合。
6. 失败 branch 的 transitions 在新 child 创建后仍完整存在。
7. lineage 的 parent/fork/attempt/depth 正确。
8. terminal failure、无 eligible checkpoint、重复失败、超预算都能干净终止。
9. `run_ok=True` 不能生成 success 标签；只能使用 `check_success()`/BDDL 真值。
10. coordinator 输出可序列化摘要，且两次同 seed 运行结构一致。
11. 从 sibling A 回到祖先再创建 sibling B 时，checkpoint 候选、subevent 状态和 failure history 不受 sibling A 污染。

上述 coordinator 测试由 `validation/monte_carlo_coordinator_validation.py` 的确定性 fake environment 覆盖；回退规则由 `validation/rollback_policy_validation.py` 单独覆盖。它们不是 LIBERO/pi0.5 有效性证据；下一步才是接入真实 pi0.5 shadow mode。

---

## 8. 远端运行状态快照

以下是 2026-07-22 最后只读核验的动态快照，重新实验前必须复查：

```text
policy server  127.0.0.1:8000 未监听
GPU 0/1        利用率 100%，各占用约 20 GB
GPU 2/3        利用率 100%，各占用约 38 GB
/home          余量约 38 GB
/data          余量约 34 GB
```

- GPU 上有其他用户进程，不要终止或干扰。
- PyPI 当时可访问；“服务器完全无法联网”已过时，但网络可用性不是稳定保证。
- 系统 Python 3.8 可导入 LIBERO；rollout 脚本在 `openpi_client` 未安装时会自动发现仓库中的 `packages/openpi-client/src`，同一进程导入已通过。
- 没有运行中的 pi0.5/rollback 采集任务。
- 没有新增可视为数据集的 episode/transition 文件。
- 当前磁盘和 GPU 状态均不适合批量采集。

OSMesa 短验证可避免占用 CUDA：

```bash
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python3 <validation_script.py>
```

不要把 policy server 未启动、GPU 忙、磁盘不足误写成“部署小问题”；虽然内存态 coordinator 已实现，真实策略闭环和 writer 仍不存在。

---

## 9. 修复版验证入口

本地仓库中的入口：

```bash
PYTHONPATH=. python validation/rollback_regression_validation.py
PYTHONPATH=. python validation/round1_norender_validation.py
PYTHONPATH=. python validation/cross_env_validation.py
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  PYTHONPATH=. python validation/cross_render_validation.py
PYTHONPATH=. python validation/semantic_keyframe_rollback_validation.py
PYTHONPATH=. python validation/pick_place_checkpoint_validation.py
PYTHONPATH=. python validation/pick_place_failure_validation.py
PYTHONPATH=. python validation/rollback_policy_validation.py
PYTHONPATH=. python validation/monte_carlo_coordinator_validation.py
python -m py_compile \
  libero_rollback_tool.py semantic_checkpoint.py \
  pick_place_checkpoint.py pick_place_failure.py rollback_policy.py \
  monte_carlo_coordinator.py \
  pi05_libero_rollback_rollout.py
```

远端验证脚本过去有被平铺到 `/home/chenxiaofu/openpi` 根目录的情况；运行前先确认实际路径，不要因路径差异误判为代码缺失。

---

## 10. 开始真实采集前的门槛

完成 coordinator 代码并不等于可以批量采集。至少还需要：

1. 启动并验证 policy server；
2. shadow mode 收集真实成功、可恢复失败、终止失败和不确定轨迹，不自动改写环境；
3. 人工或任务真值标注 grasp、drop、collision、wrong placement；
4. 统计 precision、recall、warning lead time、误回退率和恢复后成功率；
5. 小规模 writer 支持 provenance、版本、校验和、去重和原子写入；
6. train/validation 必须按 root episode、init state 或 scene seed 分组，禁止 sibling branch 泄漏到不同划分；
7. 确认存储配额和共享 GPU 使用窗口。

达到这些门槛后，才讨论“优质数据集”和批量采集。

---

## 11. 给下一次新对话的建议任务描述

可直接把下面内容交给新对话：

> 请先完整阅读 `docs/HANDOFF.md`，确认第一版 curation coordinator 的本地验证通过且不要覆盖现有修复。下一步把 `StructuredActionChunkSampler` 通过 callback 接到真实 pi0.5 shadow mode，先验证相同 checkpoint/seed 的动作可重复性、失败检测指标和 rollback proposal，不启动批量采集、不实现大型 writer。若 policy server 暴露原生 sampling seed，则使用原生采样且不叠加人工动作噪声；否则同一 fork 共享基础 policy seed，只改变结构化动作扰动。环境多样性放在不同 root episode/init state，不在同一 sibling 中同时改变环境与动作。

## 12. 禁止继续使用的旧结论

以下说法均不准确，不应再出现在报告、README、论文或新代码注释中：

- “已经恢复完整环境状态”；
- “latest checkpoint 就是安全回退点”；
- “生产配置验证了五个真实子事件”；
- “21 步真实 smoke 证明零误报”；
- “图像差异缺失也可视为 replay PASS”；
- “run_ok/success=True 表示 LIBERO 任务成功”；
- “失败分支应剪掉或不进入数据集”；
- “当前已经完成真实 pi0.5 自动蒙特卡洛采集或数据集生成系统”。
