# 已验收的双模型配对采集原型

本目录冻结 `paired_fresh_20261007_r2` 的已验收实现与小规模结果。Main 为 `pi05_libero`，Aux 为 `pi0_libero`。它用于从稳定动作分歧处采集可比较的分叉轨迹；目前已经证明流程可用，尚未证明采集效率、成功率或下游训练收益。

## 确定的策略

1. 用当前修复后的代码重新执行 Main，按每 10 个控制步比较一次两模型；不复用旧 A 中缺失完整 snapshot 的候选。
2. 保存完整 checkpoint，并验证可恢复。候选须通过原有动作契约、非终止等检查，以及稳定分歧筛选：同一 observation 下，每模型使用 5 个推理种子采样；跨模型动作距离 `D` 的中位数必须大于 `max(Main 内部 P90, Aux 内部 P90)`，且至少 80% 的跨模型距离达到 `D_min=0.07465353239106164`。阈值保持不变。
3. 每个合格候选从同一个 checkpoint 独立恢复两次，双方都执行 `sim.forward()`、controller refresh、清空 observation cache、observation refresh。起始 observation、qpos/qvel、xpos/site_xpos 等精确一致。
4. Control 全程执行 Main；Dual 由 Aux 执行前 `K=5` 步，再交还 Main。两条轨迹均运行至成功、失败或 episode horizon；5 步仅指介入窗口。
5. 交还后按绝对重规划步匹配 Main 推理种子；不同重复使用不同种子。同种子并不保证不同 observation 下产生相同动作。
6. 保存 checkpoint ID、候选步、分歧指标、完整轨迹、视频、reward/task progress、success/failure 和 terminal reason。这里只运行有预算的小规模配对实验。

## 已有证据

- 状态恢复：统一 refresh 后，同进程与跨进程共 4 个验收用例均 PASS，初始状态和相同动作 20 步重放精确一致。终止标志验收覆盖这 20 步；完整 rollout 由下面的配对实验另行验证。
- 小规模配对：同一 LIBERO 任务、2 个初始化；19 个检查点得到 2 个合格候选，2 个推理种子/候选，共 4 对、8 条完整轨迹。每个初始化在找到首个合格候选后停止，因此 2/19 不是总体候选产出率估计。
- 两候选的跨模型动作分歧中位数分别为模型内部较大 P90 的 **4.59 倍、3.30 倍**；Control 与 Dual 均 **4/4 成功**，观察到的成功率差为 **0 个百分点**。
- Aux 交还后仍有轨迹差异，部分目标完成时序改变；尚未证明比 Main 自身的随机分叉更有采集价值。

完整数值与结论边界见 [PAIRED_BRANCH_REPORT.md](PAIRED_BRANCH_REPORT.md)。下一步优先增加“Main 仅前 5 步换种子，随后回到配对 Main 种子”的同窗口对照，再扩大任务、候选和种子覆盖。

## 文件入口

| 路径 | 用途 |
|---|---|
| `scripts/paired_worker.py` | 新候选采集、稳定筛选、同起点 Control/Dual 完整 rollout |
| `scripts/run_paired.py` | 有限预算、模型进程启动与清理、配对调度 |
| `scripts/verify_common_refresh.py` | 统一计算边界的恢复验收 |
| `scripts/verify_state_replay.py` | 相同动作重放及共享状态比较函数 |
| `scripts/analyze_pairs.py`、`verify_artifacts.py` | 读取原始轨迹分析、检查起点/种子/动作来源/哈希/视频 |
| `code/` | 与已验收版本逐字节一致的运行依赖；含既有 saver |
| `tests/` | 既有 checkpoint 路由、采集语义、稳定分歧与 Aux 交还回归测试 |
| `experiment_manifest.json` | 原试验冻结配置，包含原机器路径、截止时间和模型指纹 |
| `results/` | 原始 JSON 汇总、恢复证据、配对核验及小型 CSV 数据表 |
| `SOURCE_MANIFEST.json` | 上述冻结源文件及证据文件的 SHA-256 清单 |

`code/probe_runner.py`、`steps123_run.py` 等保留的是既有回归测试所需依赖；本次配对实验入口是 `scripts/paired_worker.py`。该目录独立于仓库根目录的早期核心基线，便于保留已验收代码的准确版本。

## 验证与复用

无需 GPU 的既有回归测试，从本目录运行：

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

打包验证记录见 [VALIDATION.md](VALIDATION.md)。这些合成回归测试不代替真实仿真验收，真实验收证据在 `results/restoration/`，真实配对结果在 `results/paired/`。

真实运行需要原先的 LIBERO/robosuite/MuJoCo、OpenPI 模型环境及权重。此版本仍保留原实验的机器路径绑定，**不是下载后即可直接运行的通用安装包**。准备下一次小规模实验时：

1. 将本目录复制到新的实验目录；保留本归档与 `SOURCE_MANIFEST.json` 作为基准。
2. 在新副本中配置 `experiment_manifest.json` 的新 run ID、有限截止时间、模型权重位置、端口和已有恢复验收路径；核对原模型/任务指纹。策略参数、筛选阈值与 saver 保持已验收版本。
3. 对照 `scripts/run_paired.py` 中的模型 Python、仿真 Python、OpenPI client 路径、GPU 编号，以及媒体脚本中的 ffmpeg 路径，绑定实际服务器环境。恢复验收脚本和原始轨迹分析脚本还引用了历史证据目录；跨机器复用时也须迁移这些证据或明确重映射路径。
4. 准备完成后，在新副本执行 `python scripts/run_paired.py`。该命令会启动真实模型并采集数据；本次 Git 归档未执行它。

完整 checkpoint、HDF5 轨迹、视频和模型权重不纳入 Git。它们仍在原实验目录与本地交付包内；JSON 中的原始路径和哈希用于定位与核验。仅下载此 Git 分支可以查看实现、运行合成回归测试和阅读结果；重新计算轨迹指标还需要原始 HDF5 等附件。
