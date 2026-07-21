# Rollback-guided Monte Carlo Data Curation 交接文档

最后核对时间：2026-07-21 23:33 CST

## 1. 项目与代码状态

- GitHub：<https://github.com/kgwrf949hq-cmd/Rollback-guided-Monte-Carlo-Data-Curation>
- 分支：`main`
- 功能基线提交：`a8c9f83 Add stage-aware pick-place failure detection`（交接文档为后续文档提交）。
- 本地工作树：交接文档编写前为 clean，功能基线已与 GitHub 同步。
- 远程运行目录：`/home/chenxiaofu/openpi`
- 远程运行目录不是 Git 仓库；远程文件由 `scp` 同步，GitHub 仓库是代码的 source of truth。

当前工作分为三个独立流程：

```text
流程一：完整状态捕获/恢复
流程二：子事件与风险感知 checkpoint 选取
流程三：阶段感知失败检测
```

三者尚未组成最终自动闭环。特别是流程三目前只输出 `FailureDecision`，不会调用流程二，也不会触发流程一。

---

## 2. 流程一：完整状态捕获与恢复

### 2.1 入口

文件：`libero_rollback_tool.py`

核心接口：

```python
tool = EnvRollbackTool()
checkpoint = tool.capture(env, step=step, obs=obs, rng=rng)
restore_report = tool.restore(env, checkpoint, rng=rng)
```

### 2.2 捕获内容

- MuJoCo `mjSTATE_INTEGRATION`；
- `qpos/qvel/act/ctrl/mocap/time`；
- model 基础数组与编译派生常量；
- XML 稳定签名；
- 环境 `timestep/cur_time/done`；
- controller、gripper、robot history buffers；
- observable cache 与采样时钟；
- Python random、NumPy global RNG、局部 `Generator`；
- rollout action chunk 与外部 metadata。

默认在同一环境原地恢复；跨环境时显式使用 `reload_xml=True`。MuJoCo 首次 restore 可能注入派生 `<statistic>` XML 节点，哈希逻辑已忽略这一派生节点，因此同一环境支持连续多次 rollback，同时仍拒绝真正不同的模型。

### 2.3 已验证

- 原地无渲染：integration state、observation、action、reward 差异均为 `0.0`；
- 跨环境：model/runtime/RNG 恢复一致，6 步 replay 通过；
- OSMesa 双相机原地与跨环境：图像最大差异 `0.0`；
- 连续两次 restore：通过；
- Python 编译：通过。

### 2.4 边界

该流程只回答“如何恢复”，不决定何时保存、选择哪个 checkpoint，也不判断失败。

---

## 3. 流程二：checkpoint 选取

### 3.1 通用语义基线

文件：`semantic_checkpoint.py`

`SemanticKeyframeSelector` 根据末端运动、方向变化、夹爪变化、物体变化、reward/progress、success/terminal 和最大间隔选择 checkpoint。它是通用启发式基线，不等同于显式子事件完成。

### 3.2 Pick-and-place 子事件状态机

文件：`pick_place_checkpoint.py`

人工定义五个有序事件：

```text
approach_object
→ grasp_completed
→ object_lifted
→ place_completed
→ release_completed
```

人工只配置任务语义、观测字段绑定和阈值；程序自动读取目标物体位置、目标区域位置、末端位置、夹爪 aperture、物体高度和稳定性。环境若已有更可靠事件信号，也可通过 `info["<event>_completed"]` 输入。

任务配置：`configs/pick_place/libero_spatial_task0.json`

当前绑定：

```text
target_object = akita_black_bowl_1
goal_object   = plate_1
```

### 3.3 子事件边风险策略

`PickPlaceRiskAwareCheckpointManager` 让事件完成点强制保存，并按当前事件边调整启发式阈值和最大间隔：

```text
start → approach       risk=0.15  heuristic=off  max_interval=30
approach → grasp       risk=1.00  heuristic=on   max_interval=6
grasp → lifted         risk=0.90  heuristic=on   max_interval=7
lifted → placed        risk=0.45  heuristic=on   max_interval=15
placed → released      risk=1.00  heuristic=on   max_interval=5
```

高风险边通过较高风险系数降低启发式分数门槛，并缩短最大间隔；低风险 approach 边关闭运动启发式，只保留起点、事件完成点和最大间隔兜底。

### 3.4 已验证

真实 LIBERO 状态上的验证结果：

```text
event steps      = 8, 12, 16, 26, 28
checkpoint steps = 0, 8, 10, 12, 14, 16, 26, 28, 30
approach 1–7     = 无中间 checkpoint
grasp edge       = 第 10 步增加风险点
failure boundary = 15
selected point   = 14
restore diff     = 0.0
replay steps     = 16，全部一致
```

验证由两部分组成：

1. 合成 observation 序列验证五个 observation predicate；
2. 真实 LIBERO 环境通过明确的 `info` 事件标记隔离验证风险密度、checkpoint 保存和恢复。

尚未用真实成功 pi0.5 rollout 验证五个事件能否全部仅靠 observation 自动触发。

### 3.5 pi0.5 接口

`pi05_libero_rollback_rollout.py` 已支持：

```bash
--checkpoint-policy semantic
--checkpoint-policy pick_place
--checkpoint-policy periodic
```

Pick-and-place 示例：

```bash
PYTHONPATH=. python pi05_libero_rollback_rollout.py \
  --checkpoint-policy pick_place \
  --pick-place-config configs/pick_place/libero_spatial_task0.json
```

注意：该 CLI 已完成编译和 `--help` 检查，但完整 policy inference 目前受服务器环境和 GPU 状态阻塞，见第 6 节。

---

## 4. 流程三：失败检测

### 4.1 入口与边界

文件：`pick_place_failure.py`

```python
detector = PickPlaceRuleFailureDetector(task_spec)
decision = detector.update(
    obs,
    step=step,
    completed_events=event_detector.completed_events,
    action=action,
    reward=reward,
    done=done,
    info=info,
)
```

输出：

```text
normal
warning
recoverable_failure
terminal_failure
```

同时返回 `risk_score/reason/current_stage/consecutive_steps/hard_failure/evidence`。

该模块是可复用的 pick-and-place 检测器，不是验证脚本；但它只检测，不捕获/选择 checkpoint、不 restore、不 replay、不调用策略。

### 4.2 第一版规则

- approach 无进展与超时；
- missed grasp 与 grasp timeout；
- lift 无进展、抓取丢失与 lift timeout；
- transport/place 无进展与 placement timeout；
- object dropped；
- released outside goal 与 release timeout；
- observation/action 非有限值；
- 工作空间越界；
- 显式碰撞和仿真异常；
- done 但未成功。

软规则先进入 `warning`，连续确认后升级；硬规则立即触发。`object_dropped` 要求高度显著下降、物体已与末端分离且不在目标区域，以避免把正常受控下降误判为掉落。

### 4.3 已验证

验证脚本：`validation/pick_place_failure_validation.py`

覆盖：

- 正常接近；
- approach stalled；
- missed grasp；
- lift no progress；
- object dropped；
- 正常下降不误判；
- placement timeout；
- released outside goal；
- 显式 collision；
- invalid observation；
- simulation error；
- done without success。

真实 LIBERO smoke：

```text
steps                 = 21
normal                = 21
warning               = 0
recoverable_failure   = 0
terminal_failure      = 0
```

验证报告范围明确为：

```text
failure_detection_only_no_checkpoint_restore_or_replay
```

### 4.4 Learned risk 尚未实现

当前只有 `rule_risk`。如果未来 `learned_risk` 也表示失败风险，不能相减。推荐先使用归一化加权和：

```python
final_risk = (
    rule_weight * rule_risk + model_weight * learned_risk
) / (rule_weight + model_weight)
```

也可研究 `max(rule_risk, learned_risk)` 或概率并集。确定性硬规则应独立拥有最高优先级。

---

## 5. 三个流程的当前连接关系

目前实际状态：

```text
流程一 EnvRollbackTool
    ↑ 被流程二的验证/演示调用

流程二 PickPlaceRiskAwareCheckpointManager
    ↑ 已接入 pi0.5 demo 的 checkpoint policy

流程三 PickPlaceRuleFailureDetector
    独立运行，尚未接入流程二或流程一
```

建议后续新增单独 coordinator，不把副作用写进失败检测器：

```text
obs/action
   ├─→ SubEventDetector ─→ CheckpointManager
   └─→ FailureDetector  ─→ FailureDecision

FailureDecision.warning
   → 可请求临时提高 checkpoint 密度

FailureDecision.recoverable_failure
   → coordinator 请求选点
   → EnvRollbackTool.restore
   → 清空 action chunk 并重新调用策略

FailureDecision.terminal_failure
   → coordinator 终止当前 rollout
```

该闭环尚未实现，不应在交接时描述为已完成。

---

## 6. openpi-server 实时状态

### 6.1 连接与目录

```text
SSH alias : openpi-server
Host      : 10.77.110.127
User      : chenxiaofu
Hostname  : ubuntu
Work dir  : /home/chenxiaofu/openpi
Network   : 服务器无法联网
```

远程目录不是 Git 仓库。不要在远程执行 `git pull`；应从 GitHub/local clone 选择明确文件后 `scp` 同步。

### 6.2 Python 环境分裂（重要）

系统 Python：

```text
Python       3.8.10
robosuite    1.4.0
MuJoCo       3.2.3
NumPy        1.24.4
LIBERO       可导入
openpi_client 不可导入
```

OpenPI venv：

```text
/home/chenxiaofu/openpi/.venv/bin/python
Python       3.11.15
openpi_client 可导入
LIBERO       不可导入
```

因此：

- 系统 Python 可以运行 LIBERO/rollback/checkpoint/failure validation；
- `.venv` 可以检查 pi0.5 CLI 和使用 `openpi_client`；
- 完整 `pi05_libero_rollback_rollout.py` 同时需要两套依赖，目前没有单一解释器能直接满足。

下一位接手者应优先解决离线环境合并，例如把已有 LIBERO 安装以兼容方式加入 OpenPI venv，或建立包含 OpenPI client 的 Python 3.8 环境。不要在无网服务器直接尝试在线安装。

### 6.3 GPU 状态（动态快照）

核对时间：2026-07-21 23:33 CST

| GPU | 型号 | 显存占用 | 利用率 | 温度 |
| --- | --- | ---: | ---: | ---: |
| 0 | RTX A6000 | 22305 / 49140 MiB | 100% | 82°C |
| 1 | RTX A6000 | 22304 / 49140 MiB | 100% | 85°C |
| 2 | RTX A6000 | 41194 / 49140 MiB | 100% | 85°C |
| 3 | RTX A6000 | 41197 / 49140 MiB | 100% | 79°C |

占用来自其他用户 `fengyou+` 的 Ray/FSDP/VLLM/RoboTwin 训练，主要进程已持续约 1.5 小时。不要终止这些进程，也不要运行 EGL/CUDA 验证干扰共享任务。相机验证可使用：

```bash
MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES=""
```

### 6.4 端口状态

```text
127.0.0.1:8000  未监听（没有 OpenPI policy server）
127.0.0.1:8265  Ray dashboard，属于其他用户作业
```

### 6.5 磁盘状态（重要）

```text
/home  5.6T，剩余约 40G，Use%=100%
/data  1.7T，剩余约 34G，Use%=98%
/      191G，剩余约 53G，Use%=72%
```

磁盘余量很低。不要在 `/home` 或 `/data` 生成大规模 rollout 视频、checkpoint dump 或数据集；短期验证使用 `/tmp`，大规模实验前先与服务器维护者确认存储目录和清理方案。

### 6.6 远程文件与 source-of-truth

远程关键文件位于 `/home/chenxiaofu/openpi/`，验证脚本在远程被平铺到根目录，配置仍在 `configs/pick_place/`：

```text
libero_rollback_tool.py
semantic_checkpoint.py
pick_place_checkpoint.py
pick_place_failure.py
pi05_libero_rollback_rollout.py
semantic_keyframe_rollback_validation.py
pick_place_checkpoint_validation.py
pick_place_failure_validation.py
cross_env_validation.py
configs/pick_place/libero_spatial_task0.json
```

本地与远程关键文件 SHA-256 已核对一致：

| 文件 | SHA-256 |
| --- | --- |
| `libero_rollback_tool.py` | `68da9dd5a613848e7f4ecebfca72cb3c52c4d9fc0421d135c7fc93731a4347e7` |
| `semantic_checkpoint.py` | `71c8f8b73601760234b0436b764f0e87899a28e61d9d0f93f28fd87aaf309224` |
| `pick_place_checkpoint.py` | `86f7edd9b31fe73d962ba37d5ef10fa857565d5263f41138794cade77928b6ad` |
| `pick_place_failure.py` | `b2134eec86a52576b8d5cf22ee1f78a4274ddc31157f141c9a02749247a44f4d` |
| `pi05_libero_rollback_rollout.py` | `4721a65d1e92c1cdac49e2541d0a22f7d414adadc2de845dd6e2cc6e9deb5b68` |

远程存在 7 月 13–14 日的旧报告和约 200 KB 演示视频：

```text
rollback_restore_report*.json
pi05_libero_rollback_demo_report.json
pi05_libero_rollback_demo.mp4
```

它们不是最新完整闭环实验结果，不应与本轮验证混淆。

robosuite 每次启动会警告缺少 private macro file；当前验证中该警告不影响结果。

---

## 7. 远程验证命令

进入服务器：

```bash
ssh openpi-server
cd /home/chenxiaofu/openpi
```

无 GPU 验证：

```bash
PYTHONPATH=. MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  python3 semantic_keyframe_rollback_validation.py

PYTHONPATH=. MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  python3 pick_place_checkpoint_validation.py

PYTHONPATH=. MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  python3 pick_place_failure_validation.py

PYTHONPATH=. MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES="" \
  python3 cross_env_validation.py
```

检查 pi0.5 CLI（只检查参数，不代表完整运行环境已打通）：

```bash
.venv/bin/python pi05_libero_rollback_rollout.py --help
```

---

## 8. 已知限制

1. 当前子事件和失败规则只覆盖 pick-and-place 任务族。
2. 目标区域用目标物体中心距离近似，尚未利用 BDDL 区域几何和接触关系。
3. missed grasp/drop 等规则主要使用位置、高度和夹爪 aperture；接触传感器尚未接入。
4. 失败检测尚未在真实 pi0.5 失败轨迹上统计 precision、recall、检测延迟和误报率。
5. Learned failure model 尚未实现。
6. 失败检测尚未接入 checkpoint manager 和 restore coordinator。
7. 完整 pi0.5 inference + checkpoint + failure + rollback + replan 尚未验证。
8. 当前服务器 GPU 全忙、8000 无 policy server、Python 环境分裂、磁盘接近满载。

---

## 9. 推荐下一步

按优先级：

1. 解决 OpenPI client 与 LIBERO 的单一 Python 运行环境；
2. 等待/协调空闲 GPU，启动 policy server；
3. 在成功和人为失败的真实 pi0.5 rollout 中只记录 `FailureDecision`，暂不自动回退；
4. 标注 missed grasp、drop、stall、wrong placement，校准规则阈值；
5. 统计 precision、recall、warning lead time、false-positive rate；
6. 新增无副作用 coordinator，将 warning 接到 checkpoint 密度，将 recoverable failure 接到选点和 restore；
7. 回退后默认清空 action chunk 并重新规划，记录重复失败签名和最大重试次数；
8. 最后再实现 learned risk，并与 rule risk 使用加权和/最大值/概率融合比较。

## 10. 关键提交

```text
a8c9f83  Add stage-aware pick-place failure detection
0bb2fcb  Add risk-aware pick-place event checkpoints
51f1b53  Make repeated rollback model validation stable
de94171  Add semantic keyframe checkpoint selection
60b6426  Add MuJoCo rollback checkpoint and replay tools
```
