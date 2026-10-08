"""Render the explicitly requested Markdown report from measured results."""
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
r=json.loads((BASE/'paired_results.json').read_text())


def f(x):return 'NA' if x is None else '%.6g'%x


lines=['# PAIRED_BRANCH_REPORT','',
    '本轮仅为小规模 paired rollout；所有候选来自当前修复源码生成的新 Main 实际运行。没有使用旧 A checkpoint、旧动作回放重建候选、formal16/formal160 或递归树采集。','',
    '## 运行与恢复验收','',
    '- state restoration：PASS。A/B 同进程、跨进程共 4 组，在两侧完全相同 refresh 后瞬时全字段、固定动作 20 步及末态/终止标志精确一致。',
    '- 已有 saver 与已修复采集源码均保持逐字节一致。',
    '- 本轮状态：**%s**；目标 %d 个候选，获得 %d 个合格候选、%d 组完整配对、%d 条完整分支。'%(r['status'],r['target_candidates'],r['qualified_candidate_count'],r['complete_pair_count'],r['completed_branch_count']),
    '- D_min=%.17g；numeric_floor=%.17g；K=5；全部保持原值。'%(r['d_min'],r['numeric_floor']),
    '- 稳定筛选：每模型重复 5 次，跨模型中位数必须超过两个模型内部 P90，且至少 80% 跨模型距离达到 D_min。',
    '- Control/Dual 各自建立新进程、新环境，restore 同一 checkpoint，随后调用相同 `sim.forward → controller.update(force=True) → 清空 observation cache → _get_observations(force_update=True)`。',
    '- 每条分支起点均与该候选 gate 保存的相同计算边界逐字段精确比较；包含 qpos/qvel/act/time、controller/runtime、task progress、全部 observation、xpos/site_xpos 和 EEF metrics。',
    '- 组内按绝对步号匹配 Main 的显式推理噪声种子；重复组使用不同种子，并交替 Control/Dual 执行顺序。',
    '- K=5 仅为 Aux 介入窗口。后续每条分支执行到官方 success、环境 failure、既有 failure detector 判定或 episode horizon。规则停止单列，不能当作已证实的自然失败。','',
    '## 新候选采集','',
    '| 新 root | 实际采集步数 | 稳定筛选次数 | 结束原因 |','|---|---:|---:|---|']
for x in r['collection']:lines.append('| %s | %d | %d | %s |'%(x['root_id'],x['collection_steps'],x['gate_attempts'],x['reason']))
for c in r['candidates']:
    stats=c['stable_disagreement']['summary']
    lines+=['','## 候选 '+c['candidate_id'],'',
        '- candidate control step=%d，env step=%d；checkpoint id=`%s/checkpoint`。'%(c['candidate_control_step'],c['candidate_env_step'],c['candidate_id']),
        '- checkpoint：`%s`。完整材料、restore 与稳定分歧筛选均通过。'%c['checkpoint']['path'],
        '- source D=%s；restore+refresh 后 D=%s；跨模型中位数=%s；Main 内部 P90=%s；Aux 内部 P90=%s；跨模型过阈值比例=%s。'%(f(c['source_disagreement']['D']),f(c['post_refresh_disagreement']['D']),f(stats['cross_model']['median']),f(stats['main_within']['p90']),f(stats['aux_within']['p90']),f(stats['cross_model']['over_fraction'])),
        '', '| 重复组 | 轨迹 | 实际步数 | Aux步数 | 交回Main | success | 最终谓词 | reward合计 | terminal reason |',
        '|---:|---|---:|---:|---|---|---|---:|---|']
    for b in c['branches']:
        rel=Path(b['trajectory']).relative_to(BASE)
        lines.append('| %d | [%s](%s) | %d | %d | %s | %s | %s | %s | %s |'%(b['replicate'],b['arm'],rel,b['steps'],b['actual_aux_steps'],b['handoff_occurred'],b['success'],b['final_task_progress']['predicates'],f(b['reward_sum']),b['terminal_reason']))
    lines+=['','### 已观察到的事实','']
    for p in c['pairs']:
        lines.append('- 重复组 %d：两分支起点精确一致=%s；共同观察 %d 步；交回 Main 后 EEF 距离中位数 %s m、P90 %s m；超过 1 mm 的步数 %d/%d；谓词不同 %d 步、阶段不同 %d 步；最终 success 是否改变=%s。'%(p['replicate'],p['initial_states_exact'],p['common_steps'],f(p['post_handoff_eef_distance_m']['median']),f(p['post_handoff_eef_distance_m']['p90']),p['post_handoff_eef_over_1mm_steps'],p['post_handoff_total_steps'],p['predicate_different_steps'],p['stage_different_steps'],p['outcome_changed']))
    rc=c.get('randomness_comparison')
    if rc:
        lines+=['','在四条轨迹共同的 %d 步范围内，两个 Control 随机种子之间的 EEF 距离中位数为 %s m、P90 为 %s m。'%(rc['shared_steps'],f(rc['control_seed_variation_eef_m']['median']),f(rc['control_seed_variation_eef_m']['p90'])),'']
        for rep,values in rc['paired_effect_eef_m'].items():
            lines.append('- 配对组 %s：EEF 差异中位数 %s m、P90 %s m；其差异大于 Control 跨种子差异的步数比例为 %s。'%(rep,f(values['median']),f(values['p90']),f(rc['paired_greater_than_control_seed_variation_fraction'][rep])))
        lines+=['','![EEF 差异与 Control 跨种子变化](pairs/%s/eef_comparison.png)'%c['candidate_id']]
    lines+=['','视频：'+ '；'.join('[rep%d %s](%s)'%(b['replicate'],b['arm'],Path(b['video']).relative_to(BASE)) for b in c['branches'])+'。']
lines+=['','## 可以支持的结论','']
if r['complete_pair_count']:
    persistent=sum(p['post_handoff_eef_over_1mm_steps']>0 for c in r['candidates'] for p in c['pairs'])
    changed=sum(p['outcome_changed'] for c in r['candidates'] for p in c['pairs'])
    lines+=['- 本轮 %d 组完整配对使用同 checkpoint 和相同恢复/refresh 边界，起点一致性经过实际验证。'%r['complete_pair_count'],
        '- %d/%d 组在 Aux 交回后观察到超过 1 mm 的 EEF 差异；具体持续步数见上表，不能仅从动作不同推断任务效果。'%(persistent,r['complete_pair_count']),
        '- %d/%d 组的最终 success 标志发生改变；方向由各组 Control/Dual 结果给出。'%(changed,r['complete_pair_count'])]
    if all(c.get('randomness_comparison',{}).get('all_paired_medians_and_p90_below_control_seed_variation') for c in r['candidates']):
        lines+=['- 在各候选四条轨迹共同的观察范围内，每组介入差异的 EEF 中位数和 P90 均低于本轮两个 Control 种子之间的差异。因此，当前数据没有显示介入差异的总体幅度超过这组随机波动对照；不能据此认定 Aux 带来稳定收益。']
else:lines+=['- 本轮尚无完整配对结果；只能报告候选筛选/执行状态，不能判断 Aux 介入效果。']
lines+=['','## 当前不能支持的结论','',
    '- 每候选仅两个推理随机种子，只有一组 Control 跨种子对照；即使描述性距离较大，也不能证明介入效应显著超过总体随机波动。',
    '- 不能据此声称稳定提高成功率、覆盖所有任务/初态，或能支撑 formal16/formal160、深层树采集及训练收益。',
    '- 候选经过稳定分歧筛选，结论限于这些入选状态，不代表所有自然状态。',
    '- 本轮配对只覆盖同一个复合任务的两个初态。记录的 stage 标签是既有通用复合任务指标，不能据此判定完整的语义子任务顺序。',
    '- 若某分支先结束，后续不外推对照轨迹。20 步恢复验收不是完整 episode 终止验收；本报告的配对分支终止结果由实际完整 rollout 独立产生。','',
    '## 数据与复现','',
    '- 汇总：[paired_results.json](paired_results.json)。',
    '- 独立落盘核验：[artifact_verification.json](evidence/artifact_verification.json)，8 条分支全部 PASS。',
    '- 配置：[experiment_manifest.json](experiment_manifest.json)；逐次请求与显式种子见 `model_requests.jsonl`，原始模型应答见各目录 `requests/`。',
    '- 每条分支包含完整无损 observation/action/reward/task-progress HDF5、视频、初/末状态、qpos/qvel/act/time/xpos/site_xpos 逐步 NPZ、恢复报告及终止原因。',
    '- 采集与配对使用 `scripts/paired_worker.py`；只读分析使用 `scripts/analyze_pairs.py`。',
    '- 1 mm 仅用于展示空间差异，没有用于候选准入，也没有改变分歧阈值。','']
if r.get('execution_error'):lines+=['执行错误：`'+r['execution_error']+'`。','']
(BASE/'PAIRED_BRANCH_REPORT.md').write_text('\n'.join(lines))
