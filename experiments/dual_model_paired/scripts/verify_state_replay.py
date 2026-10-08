"""Bounded real-LIBERO replay; no policy service, sampling or tree collection.

Uses saved executed commands to reach S, then compares a fixed 20-command
suffix directly, after serialized same-process restore, and in a fresh worker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

import h5py
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from curation_store import atomic_json,save_payload,load_payload
from dual_core import fingerprint
from dual_runtime import save_diagnostic_state,restore_candidate
from libero_rollback_tool import (make_libero_env,step_env,find_sim,close_env,
    capture_integration_state,capture_runtime_state,capture_rng_state,
    capture_model_state,get_env_timestep,get_env_runtime_field)
from libero_task_adapter import TaskAdapter,describe_task
from trajectory_store import digest_obs

HIST=Path('/data/chenxiaofu/dual_tasks/calibration_continue_20260920T020734Z/sample_review')
LATEST=HIST/'task_complexity_probe_20260924T031445Z_steps123'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def leaf_hashes(value,path=''):
    if isinstance(value,dict):
        result={}
        for k,v in value.items():result.update(leaf_hashes(v,path+'.'+str(k)))
        return result
    if isinstance(value,(list,tuple)):
        result={}
        for i,v in enumerate(value):result.update(leaf_hashes(v,path+'[%d]'%i))
        return result
    return {path:fingerprint(value)}


def snapshot(env,obs,adapter,detector):
    sim=find_sim(env);spec,integration=capture_integration_state(sim)
    return dict(integration_spec=int(spec),integration=integration.copy(),
        simulator={k:np.asarray(getattr(sim.data,k)).copy() for k in
                   ['qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','xpos','site_xpos']},
        sim_time=float(sim.data.time),env_timestep=get_env_timestep(env),
        env_cur_time=get_env_runtime_field(env,'cur_time'),env_done=get_env_runtime_field(env,'done'),
        observation={k:np.asarray(v).copy() for k,v in obs.items()},
        observation_sha256=digest_obs(obs),task_progress=adapter.truth(env).as_dict(),
        runtime_hashes=leaf_hashes(capture_runtime_state(env)),
        model_state_hash=fingerprint(capture_model_state(sim)),
        rng_hash=fingerprint(capture_rng_state()),detector=detector.snapshot_state())


def compare(a,b,path=''):
    differences=[]
    if isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype:
            return [dict(path=path,reason='array shape/dtype/type')]
        if not np.array_equal(a,b):
            delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
            differences.append(dict(path=path,reason='array value',max_abs=float(delta.max()) if delta.size else 0.,different_elements=int(np.count_nonzero(a!=b))))
    elif isinstance(a,dict):
        if not isinstance(b,dict):return [dict(path=path,reason='type')]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:differences.append(dict(path=path+'.'+k,reason='missing key'))
            else:differences.extend(compare(a[k],b[k],path+'.'+k))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):return [dict(path=path,reason='sequence type/length')]
        for i,(x,y) in enumerate(zip(a,b)):differences.extend(compare(x,y,path+'[%d]'%i))
    elif a!=b:
        d=dict(path=path,reason='scalar value')
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):d['max_abs']=abs(a-b)
        else:d.update(expected=str(a)[:180],actual=str(b)[:180])
        differences.append(d)
    return differences


def advance(env,obs,adapter,detector,actions,start_step):
    trace=[snapshot(env,obs,adapter,detector)]
    for i,action in enumerate(actions):
        obs,reward,done,extra=step_env(env,np.asarray(action,np.float32))
        truth=adapter.truth(env)
        detector.update(obs,step=start_step+i+1,task_truth=truth,done=done,action=action,reward=reward,info=extra)
        row=snapshot(env,obs,adapter,detector)
        row.update(reward=float(reward),done=bool(done),action=np.asarray(action,np.float32))
        trace.append(row)
        if done or truth.task_success:
            raise RuntimeError('fixed test suffix unexpectedly terminal; preserve evidence and stop')
    return obs,trace


def make_case_env(case):
    root=case['root'];task=case['task']
    random.seed(root['env_seed']);np.random.seed(root['env_seed'])
    current=describe_task('libero_10',task['task_id'],root['init_index'])
    for key in ['name','bddl_sha256','init_file_sha256']:
        if current[key]!=task[key]:raise RuntimeError('task identity changed: '+key)
    env,obs,_=make_libero_env(benchmark_name='libero_10',task_id=task['task_id'],
        init_state_id=root['init_index'],seed=root['env_seed'],camera_width=256,camera_height=256)
    adapter=TaskAdapter(current,env)
    return env,obs,adapter,adapter.detector()


def comparison_report(reference,actual):
    rows=[]
    if len(reference)!=len(actual):raise RuntimeError('trace length mismatch')
    for i,(a,b) in enumerate(zip(reference,actual)):
        diffs=compare(a,b)
        rows.append(dict(offset=i,equal=not diffs,differences=diffs))
    return dict(exact_equal=all(r['equal'] for r in rows),
        initial_equal=rows[0]['equal'],final_equal=rows[-1]['equal'],
        steps=len(rows)-1,first_mismatch_offset=next((r['offset'] for r in rows if not r['equal']),None),rows=rows)


def child(directory):
    case=json.loads((directory/'case.json').read_text())
    env=None
    try:
        env,obs,adapter,detector=make_case_env(case)
        sample=load_payload(directory/'diagnostic_state')
        obs,restore=restore_candidate(env,detector,sample['checkpoint'],case['run_id'])
        atomic_json(directory/'cross_process_restore.json',restore)
        actions=load_payload(directory/'fixed_actions')['actions']
        _,trace=advance(env,obs,adapter,detector,actions,case['save_env_step'])
        reference=load_payload(directory/'reference_trace')['trace']
        result=comparison_report(reference,trace)
        result.update(pid=os.getpid(),parent_pid=case['parent_pid'],cross_process=restore['cross_process'])
        atomic_json(directory/'cross_process_result.json',result)
        return 0 if result['exact_equal'] and restore['cross_process'] else 2
    except BaseException as e:
        atomic_json(directory/'cross_process_result.json',dict(exact_equal=False,error=repr(e),traceback=traceback.format_exc()))
        return 2
    finally:
        if env is not None:close_env(env)


def parent():
    out=BASE/'evidence/real_replay';out.mkdir(parents=True,exist_ok=False)
    m=json.loads((LATEST/'experiment_manifest.json').read_text())
    summary=dict(status='RUNNING',model_calls=0,new_collected_branches=0,
        d_min=m['d_min'],K=m['K'],camera_size=256,fixed_suffix_steps=20,cases=[])
    for rid,save_step in [('A_init1',140),('B_init0',85)]:
        directory=out/rid;directory.mkdir()
        root=next(r for r in m['source_roots']+m['roots'] if r['root_id']==rid)
        source=Path(root.get('source_directory',str(LATEST/'roots'/rid)))
        case=dict(root=root,task=m['tasks'][root['task_label']],save_control_step=save_step,
            save_env_step=save_step+10,run_id='state_saver_audit_20261007',parent_pid=os.getpid(),
            source_directory=str(source),trajectory_sha256=sha(source/'trajectory.h5'),
            warmup_sha256=sha(source/'warmup.h5'))
        atomic_json(directory/'case.json',case)
        with h5py.File(source/'warmup.h5','r') as h:warmup=h['executed_actions'][:]
        with h5py.File(source/'trajectory.h5','r') as h:actions=h['executed_actions'][:save_step+20]
        fixed=np.asarray(actions[save_step:save_step+20],np.float32)
        if fixed.shape!=(20,7):raise RuntimeError('incomplete fixed suffix')
        save_payload(directory/'fixed_actions',dict(actions=fixed,sha256=fingerprint(fixed)))
        env=None
        try:
            env,obs,adapter,detector=make_case_env(case)
            for a in warmup:obs,_,done,_=step_env(env,a)
            for i,a in enumerate(actions[:save_step]):
                obs,reward,done,extra=step_env(env,a)
                detector.update(obs,step=11+i,task_truth=adapter.truth(env),done=done,action=a,reward=reward,info=extra)
                if done or adapter.truth(env).task_success:raise RuntimeError('terminal while reaching S')
            before=snapshot(env,obs,adapter,detector)
            cp=save_diagnostic_state(env,obs,detector,directory/'diagnostic_state',
                directory/'checkpoint','root',0,save_step+10,case['run_id'],
                {'main':'fixed_saved_commands_no_inference','aux':'not_called'},
                dict(state_step=save_step+10,control_step=save_step,truth=adapter.truth(env).as_dict()))
            after=snapshot(env,obs,adapter,detector)
            capture_differences=compare(before,after)
            payload=load_payload(directory/'checkpoint')
            atomic_json(directory/'saved_schema.json',dict(fields=sorted(payload),
                runtime_fields=sorted(payload['runtime_state']),
                controller_fields=sorted(payload['runtime_state']['robots'][0]['controller']),
                history_buffers=sorted(payload['runtime_state']['robots'][0]['buffers']),
                observable_count=len(payload['runtime_state']['observables']),
                rollout_fields=sorted(payload['rollout_state']),
                rng_fields=sorted(payload['rng_state']),
                detector_history_length=len(payload['rollout_state']['failure_detector_state']['history']),
                capture_differences=capture_differences,
                checkpoint_bytes=sum((directory/('checkpoint'+s)).stat().st_size for s in ['.json','.npz'])))
            if capture_differences:raise RuntimeError('capture mutated live state')
            obs,reference=advance(env,obs,adapter,detector,fixed,save_step+10)
            save_payload(directory/'reference_trace',dict(trace=reference))
            obs,restore=restore_candidate(env,detector,cp,case['run_id'])
            atomic_json(directory/'same_process_restore.json',restore)
            _,replayed=advance(env,obs,adapter,detector,fixed,save_step+10)
            result=comparison_report(reference,replayed)
            atomic_json(directory/'same_process_result.json',result)
        except BaseException as e:
            result=dict(exact_equal=False,error=repr(e),traceback=traceback.format_exc())
            atomic_json(directory/'same_process_result.json',result)
        finally:
            if env is not None:close_env(env)
        if (directory/'reference_trace.json').exists():
            with (directory/'cross_process.log').open('w') as f:
                proc=subprocess.run([sys.executable,__file__,'--child',str(directory)],stdout=f,stderr=subprocess.STDOUT,timeout=180)
            cross=json.loads((directory/'cross_process_result.json').read_text())
        else:cross=dict(exact_equal=False,error='reference trace unavailable')
        summary['cases'].append(dict(root_id=rid,save_control_step=save_step,
            same_process_exact=result['exact_equal'],cross_process_exact=cross['exact_equal']))
        atomic_json(out/'summary.json',summary)
        print(json.dumps(summary['cases'][-1]),flush=True)
    summary['status']='PASS' if all(c['same_process_exact'] and c['cross_process_exact'] for c in summary['cases']) else 'FAIL'
    summary['planned_environment_steps']=sum([150+60,95+60])
    atomic_json(out/'summary.json',summary)
    return 0 if summary['status']=='PASS' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path);args=parser.parse_args()
    sys.exit(child(args.child) if args.child else parent())
