# Git 归档验证记录

归档日期：2026-10-08。此次只整理已验收实现与证据，没有启动模型、仿真采集或大规模树实验。

## 本次实际执行

- 复制 19 个已验收代码依赖、7 个配对/验收/分析脚本、3 个既有测试文件；29 个 Python 文件语法解析通过。
- `SOURCE_MANIFEST.json` 中 42 个冻结源文件/证据文件全部 SHA-256 匹配本地原交付。
- 恢复验收记录中的 `libero_rollback_tool.py`、`dual_runtime.py`、`dual_observation.py` 三个哈希与归档源码完全一致。没有修改 saver、动作距离或筛选阈值。
- 在归档目录执行 `python -m pytest -q tests`：**19 passed in 5.60s**。
- 测试环境：Python 3.12.14、NumPy 2.5.3、h5py 3.16.0、pytest 9.1.1。依赖安装在本任务临时目录，未修改服务器模型环境。

这 19 项为选入本目录的既有合成回归测试，验证 checkpoint 路由、采集约束、稳定分歧和短程 Aux 交还等语义。它们不是新增的真实模型 rollout，也不代表本次重新执行了历史完整测试集。

## 保留的既有真实验收证据

- `results/restoration/summary.json`：两个来源用例 × 同进程/跨进程，共 4 个 PASS；统一 refresh 后初始状态与 20 步 same-actions replay 精确一致，共 80 次 replay transition 比较。终止标志结论限于这 20 步。
- `results/paired/paired_results.json`：2 个新候选、4 组配对、8 条完整轨迹；Control 与 Dual 均 4/4 成功。
- `results/paired/artifact_verification.json`：同 checkpoint、起始状态精确一致、Main 种子匹配、动作来源、轨迹哈希和视频检查 PASS。

本次未重新运行这些真实仿真/模型实验。完整 HDF5、checkpoint 二进制、视频和模型权重保留在原交付中，Git 保存实现、配置、汇总与验证记录。
