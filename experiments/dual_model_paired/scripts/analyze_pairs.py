"""Analyze saved complete trajectories without any model or simulator calls."""
import json
from pathlib import Path
import sys
import hashlib
import collections

import h5py
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from curation_store import atomic_json
from steps123_logic import distance
M=json.loads((BASE/'experiment_manifest.json').read_text())


def load_trace(directory):
    with h5py.File(directory/'trajectory.h5') as h:
        decisions=[json.loads(x) for x in h['decisions'].asstr()[:]]
        return dict(eef=np.array([x['task_truth']['metrics']['eef'] for x in decisions]),
            stages=[x['task_truth']['stage'] for x in decisions],
            predicates=[x['task_truth']['predicates'] for x in decisions],
            qpos=h['qpos'][:],actions=h['executed_actions'][:],rewards=h['rewards'][:],
            action_sources=[x['action_source'] for x in decisions],seeds=[x['seed'] for x in decisions])


def stats(values):
    a=np.asarray(values)
    if not len(a):return dict(n=0,median=None,p90=None,maximum=None)
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.percentile(a,90)),maximum=float(np.max(a)))


def analyze():
    state=json.loads((BASE/'run_status.json').read_text())
    source=Path('/data/chenxiaofu/dual_tasks/state_saver_audit_20261007/code')
    changed=[p.name for p in (BASE/'code').glob('*.py') if p.read_bytes()!=(source/p.name).read_bytes()]
    result=dict(status=state['status'],restoration_status='PASS',saver_modified=False,
        original_source_files_changed=changed,run_id=M['run_id'],remote_root=str(BASE),d_min=M['d_min'],numeric_floor=M['numeric_floor'],K=M['K'],
        stable_filter='cross-model median > max(main within P90, aux within P90), and cross-model fraction >= D_min at least 0.8; five independent proposals per model',
        target_candidates=M['target_candidates'],replicates=M['replicates'],fresh_collection=True,
        old_A_checkpoints_used=False,old_executed_actions_replayed_for_candidate_collection=False,
        branch_start_protocol=['restore same complete checkpoint into a fresh process/environment','sim.forward()','controller.update(force=True)','clear observation cache','_get_observations(force_update=True)'],
        candidates=[],collection=[],limitations=['Only two paired seeds per candidate: no significance or stable success-rate estimate.',
            'Inference noise is matched within pairs at absolute Main replan steps, varied across replicate seeds.',
            'Failure-detector rule stops are reported separately from official simulator failure.',
            'Distances use only observed common prefixes; no extrapolation after either trajectory ends.',
            'Post-handoff EEF distance > 1 mm is descriptive, never a candidate selection threshold.'])
    for p in (BASE/'collection').glob('*/result.json'):result['collection'].append(json.loads(p.read_text()))
    for path in state['candidates']:
        candidate_path=Path(path);item=json.loads(candidate_path.read_text());cid=item['candidate_id']
        gate=json.loads((candidate_path.parent/'gate/result.json').read_text())
        entry=dict(candidate_id=cid,checkpoint=item['checkpoint'],candidate_control_step=item['candidate_control_step'],
            candidate_env_step=item['candidate_env_step'],source_disagreement=item['source_disagreement'],
            post_refresh_disagreement=gate['post_refresh_D'],stable_disagreement=gate['stats'],
            complete_checkpoint=gate['complete_checkpoint'],restore_pass=gate['restore_pass'],pairs=[],branches=[])
        traces={};directories={}
        for rep in range(M['replicates']):
            for arm in ['control','dual']:
                directory=BASE/'pairs'/cid/('rep%d_%s'%(rep,arm));p=directory/'result.json'
                if not p.exists():continue
                branch=json.loads(p.read_text())
                branch['checkpoint_id']=cid+'/checkpoint'
                branch['trajectory_relative']=str(Path(branch['trajectory']).relative_to(BASE))
                branch['video_relative']=str(Path(branch['video']).relative_to(BASE))
                entry['branches'].append(branch)
                traces[rep,arm]=load_trace(directory);directories[rep,arm]=directory
            if (rep,'control') not in traces or (rep,'dual') not in traces:continue
            c=traces[rep,'control'];d=traces[rep,'dual'];n=min(len(c['eef']),len(d['eef']))
            distances=np.linalg.norm(c['eef'][:n]-d['eef'][:n],axis=1)
            pair=dict(replicate=rep,common_steps=n,post_handoff_eef_distance_m=stats(distances[5:]),
                post_handoff_eef_over_1mm_steps=int(np.count_nonzero(distances[5:]>0.001)),
                post_handoff_total_steps=max(0,n-5),first_qpos_difference_step=next((i+1 for i in range(n) if not np.array_equal(c['qpos'][i],d['qpos'][i])),None),
                predicate_different_steps=sum(a!=b for a,b in zip(c['predicates'][:n],d['predicates'][:n])),
                stage_different_steps=sum(a!=b for a,b in zip(c['stages'][:n],d['stages'][:n])))
            m=min(5,n)
            pair['D_executed']=distance(c['actions'][:m],d['actions'][:m],m) if m else None
            pair['executed_intervention_comparison_steps']=m
            pair['control_all_main']=all(role=='main' for role in c['action_sources'])
            pair['dual_aux5_then_main']=d['action_sources']==['aux']*min(5,len(d['action_sources']))+['main']*max(0,len(d['action_sources'])-5)
            pair['main_seeds_paired_after_handoff']=c['seeds'][5:n]==d['seeds'][5:n]
            cb=next(b for b in entry['branches'] if b['replicate']==rep and b['arm']=='control')
            db=next(b for b in entry['branches'] if b['replicate']==rep and b['arm']=='dual')
            pair.update(outcome_changed=cb['success']!=db['success'],terminal_reason_changed=cb['terminal_reason']!=db['terminal_reason'],
                control_success=cb['success'],dual_success=db['success'],initial_states_exact=cb['initial_exact'] and db['initial_exact'])
            entry['pairs'].append(pair)
        if len(traces)==2*M['replicates'] and M['replicates']==2:
            n=min(len(x['eef']) for x in traces.values())
            baseline=np.linalg.norm(traces[0,'control']['eef'][:n]-traces[1,'control']['eef'][:n],axis=1)
            curves={str(rep):np.linalg.norm(traces[rep,'control']['eef'][:n]-traces[rep,'dual']['eef'][:n],axis=1) for rep in range(2)}
            entry['randomness_comparison']=dict(shared_steps=n,control_seed_variation_eef_m=stats(baseline[5:]),
                paired_effect_eef_m={r:stats(v[5:]) for r,v in curves.items()},
                paired_greater_than_control_seed_variation_fraction={r:float(np.mean(v[5:]>baseline[5:])) if n>5 else None for r,v in curves.items()},
                statistical_significance='NOT_ESTABLISHED_TWO_SEEDS_ONLY')
            entry['randomness_comparison']['all_paired_medians_and_p90_below_control_seed_variation']=all(
                stats(v[5:])['median']<stats(baseline[5:])['median'] and stats(v[5:])['p90']<stats(baseline[5:])['p90'] for v in curves.values()) if n>5 else None
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(8,4))
            ax.plot(np.arange(1,n+1),baseline,label='Control seed 0 vs seed 1',color='gray',ls='--')
            for rep,values in curves.items():ax.plot(np.arange(1,n+1),values,label='Control vs Dual, seed '+rep)
            ax.axvline(5,color='black',lw=1,label='Aux handoff (K=5)')
            ax.set(xlabel='Steps after restored checkpoint',ylabel='End-effector distance (m)',title=cid)
            ax.legend(fontsize=8);fig.tight_layout();fig.savefig(BASE/'pairs'/cid/'eef_comparison.png',dpi=160);plt.close(fig)
        result['candidates'].append(entry)
    result['qualified_candidate_count']=len(result['candidates'])
    result['completed_branch_count']=sum(len(x['branches']) for x in result['candidates'])
    result['complete_pair_count']=sum(len(x['pairs']) for x in result['candidates'])
    result['all_branch_initial_states_exact']=all(b['initial_exact'] for c in result['candidates'] for b in c['branches']) if result['completed_branch_count'] else None
    result['model_requests']=sum(1 for _ in (BASE/'model_requests.jsonl').open()) if (BASE/'model_requests.jsonl').exists() else 0
    if 'error' in state:result['execution_error']=state['error']
    atomic_json(BASE/'paired_results.json',result)
    return result


if __name__=='__main__':
    r=analyze();print(json.dumps({k:r[k] for k in ['status','qualified_candidate_count','complete_pair_count','completed_branch_count','all_branch_initial_states_exact','model_requests']}))
