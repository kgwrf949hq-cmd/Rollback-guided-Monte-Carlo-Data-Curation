"""Final acceptance with identical refresh boundaries; never modifies the saver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import h5py
import numpy as np

from verify_state_replay import (BASE, make_case_env, snapshot, compare, advance,
    comparison_report, sha)
from curation_store import atomic_json, load_payload, save_payload
from dual_runtime import restore_candidate
from libero_rollback_tool import (find_sim, find_owner_with_method, step_env,
    close_env, capture_runtime_state)

SOURCE = Path('/data/chenxiaofu/dual_tasks/state_saver_audit_20261007')
OUT = BASE/'evidence/common_refresh'
PROTOCOL = ['sim.forward()', 'controller.new_update=True; controller.update(force=True)',
            'observation_owner._obs_cache={}',
            'observation_owner._get_observations(force_update=True)']


def refresh(env):
    """Exactly the same operations and order on original and restored states."""
    sim = find_sim(env)
    before = {k: np.asarray(getattr(sim.data, k)).copy() for k in ['qpos','qvel','act']}
    time_before = float(sim.data.time)
    sim.forward()
    owner = find_owner_with_method(env, '_get_observations')
    if owner is None or not hasattr(owner, '_obs_cache'):
        raise RuntimeError('required observation refresh API unavailable')
    robots = getattr(owner, 'robots', None)
    if not robots:
        raise RuntimeError('controller refresh owner unavailable')
    for robot in robots:
        robot.controller.new_update = True
        robot.controller.update(force=True)
    owner._obs_cache = {}
    obs = owner._get_observations(force_update=True)
    if not isinstance(obs, dict):
        raise RuntimeError('refresh did not return observations')
    for k, value in before.items():
        if not np.array_equal(value, getattr(sim.data, k)):
            raise RuntimeError('refresh changed physical coordinate '+k)
    if float(sim.data.time) != time_before:
        raise RuntimeError('refresh advanced simulator time')
    return obs


def normalized_snapshot(env, obs, adapter, detector):
    state = snapshot(env, obs, adapter, detector)
    state['runtime_state'] = capture_runtime_state(env)
    return state


def restore_and_run(directory, cross_process):
    case = json.loads((directory/'case.json').read_text())
    old = SOURCE/'evidence/real_replay'/case['root']['root_id']
    env = None
    mode = 'cross_process' if cross_process else 'same_process'
    try:
        env, obs, adapter, detector = make_case_env(case)
        cp = load_payload(old/'diagnostic_state')['checkpoint']
        obs, report = restore_candidate(env, detector, cp, case['run_id'])
        atomic_json(directory/(mode+'_restore.json'), report)
        obs = refresh(env)
        state = normalized_snapshot(env, obs, adapter, detector)
        initial = compare(load_payload(directory/'normalized_original')['state'], state)
        save_payload(directory/(mode+'_normalized'), {'state':state})
        actions = load_payload(old/'fixed_actions')['actions']
        _, trace = advance(env, obs, adapter, detector, actions, case['save_env_step'])
        reference = load_payload(directory/'reference_trace')['trace']
        result = comparison_report(reference, trace)
        result['normalized_initial_differences'] = initial
        result['normalized_initial_exact'] = not initial
        result['replay_20_steps_exact'] = all(r['equal'] for r in result['rows'][1:])
        result['terminal_flags_exact'] = all(
            a['env_done'] == b['env_done'] and a.get('done') == b.get('done') and
            a['task_progress'] == b['task_progress'] for a,b in zip(reference,trace))
        result['pid'] = os.getpid()
        result['reference_pid'] = case['reference_pid']
        result['fresh_process_vs_reference'] = os.getpid() != case['reference_pid']
        result['status'] = 'PASS' if (result['exact_equal'] and not initial and
            result['terminal_flags_exact'] and result['fresh_process_vs_reference'] == cross_process) else 'FAIL'
        save_payload(directory/(mode+'_trace'), {'trace':trace})
        atomic_json(directory/(mode+'_result.json'), result)
        return result
    finally:
        if env is not None: close_env(env)


def same_environment_run(directory, env, adapter, detector, case):
    old = SOURCE/'evidence/real_replay'/case['root']['root_id']
    cp = load_payload(old/'diagnostic_state')['checkpoint']
    obs, report = restore_candidate(env, detector, cp, case['run_id'])
    atomic_json(directory/'same_process_restore.json', report)
    obs = refresh(env)
    state = normalized_snapshot(env, obs, adapter, detector)
    initial = compare(load_payload(directory/'normalized_original')['state'], state)
    save_payload(directory/'same_process_normalized', {'state':state})
    _, trace = advance(env, obs, adapter, detector,
        load_payload(old/'fixed_actions')['actions'], case['save_env_step'])
    reference = load_payload(directory/'reference_trace')['trace']
    result = comparison_report(reference, trace)
    result.update(normalized_initial_differences=initial, normalized_initial_exact=not initial,
        replay_20_steps_exact=all(r['equal'] for r in result['rows'][1:]),
        terminal_flags_exact=all(a['env_done']==b['env_done'] and a.get('done')==b.get('done')
            and a['task_progress']==b['task_progress'] for a,b in zip(reference,trace)),
        pid=os.getpid(), reference_pid=case['reference_pid'], fresh_process_vs_reference=False)
    result['status'] = 'PASS' if result['exact_equal'] and not initial and result['terminal_flags_exact'] else 'FAIL'
    save_payload(directory/'same_process_trace', {'trace':trace})
    atomic_json(directory/'same_process_result.json', result)
    return result


def parent():
    OUT.mkdir(parents=True, exist_ok=False)
    files = ['code/libero_rollback_tool.py','code/dual_runtime.py','code/dual_observation.py']
    before = {name:sha(BASE/name) for name in files}
    summary = dict(status='RUNNING', protocol=PROTOCOL, exact_comparison=True,
        tolerance=0, saver_modified=False, model_calls=0, cases=[],
        previous_results_sha256=sha(SOURCE/'RESTORE_TEST_RESULTS.json'),
        source=str(SOURCE), code_sha256=before)
    atomic_json(OUT/'summary.json', summary)
    for rid in ['A_init1','B_init0']:
        directory=OUT/rid; directory.mkdir()
        old=SOURCE/'evidence/real_replay'/rid
        case=json.loads((old/'case.json').read_text())
        case['reference_pid']=os.getpid()
        atomic_json(directory/'case.json', case)
        env=None
        try:
            env,obs,adapter,detector=make_case_env(case)
            source=Path(case['source_directory'])
            for name in ['trajectory','warmup']:
                if sha(source/(name+'.h5')) != case[name+'_sha256']:
                    raise RuntimeError('original action source changed')
            with h5py.File(source/'warmup.h5') as h: warmup=h['executed_actions'][:]
            with h5py.File(source/'trajectory.h5') as h: prefix=h['executed_actions'][:case['save_control_step']]
            for a in warmup: obs,_,_,_=step_env(env,a)
            for i,a in enumerate(prefix):
                obs,reward,done,extra=step_env(env,a)
                detector.update(obs,step=11+i,task_truth=adapter.truth(env),done=done,action=a,reward=reward,info=extra)
            raw=snapshot(env,obs,adapter,detector)
            previous=load_payload(old/'reference_trace')['trace']
            raw_differences=compare(previous[0],raw)
            atomic_json(directory/'original_reconstruction.json',dict(exact=not raw_differences,differences=raw_differences))
            if raw_differences: raise RuntimeError('original checkpoint state reconstruction not exact')
            obs=refresh(env)
            save_payload(directory/'normalized_original',dict(state=normalized_snapshot(env,obs,adapter,detector)))
            fixed=load_payload(old/'fixed_actions')['actions']
            _,reference=advance(env,obs,adapter,detector,fixed,case['save_env_step'])
            save_payload(directory/'reference_trace',dict(trace=reference))
            atomic_json(directory/'refresh_vs_previous_suffix.json',comparison_report(previous[1:],reference[1:]))
            same=same_environment_run(directory,env,adapter,detector,case)
        finally:
            if env is not None: close_env(env)
        with (directory/'cross_process.log').open('w') as log:
            process=subprocess.run([sys.executable,__file__,'--child',str(directory)],
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        cross=json.loads((directory/'cross_process_result.json').read_text())
        row=dict(root_id=rid,same_process_status=same['status'],cross_process_status=cross['status'],
            initial_exact=same['normalized_initial_exact'] and cross['normalized_initial_exact'],
            replay_20_steps_exact=same['replay_20_steps_exact'] and cross['replay_20_steps_exact'],
            terminal_flags_exact=same['terminal_flags_exact'] and cross['terminal_flags_exact'],
            final_exact=same['final_equal'] and cross['final_equal'],cross_process_returncode=process.returncode)
        summary['cases'].append(row)
        atomic_json(OUT/'summary.json',summary)
        print(json.dumps(row),flush=True)
    after={name:sha(BASE/name) for name in files}
    summary['saver_unchanged']=before==after
    summary['status']='PASS' if before==after and all(r['same_process_status']=='PASS' and
        r['cross_process_status']=='PASS' for r in summary['cases']) else 'FAIL'
    summary['environment_steps']=365
    summary['compared_replay_transitions']=80
    summary['terminal_scope']='done/success/task-progress flags at all 20 replay steps; no claim of full-episode terminal coverage'
    atomic_json(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='PASS' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    try:
        result=restore_and_run(args.child,True) if args.child else None
        sys.exit((0 if result['status']=='PASS' else 2) if args.child else parent())
    except Exception as error:
        out=args.child/'cross_process_result.json' if args.child else OUT/'execution_error.json'
        atomic_json(out,dict(status='ERROR',error=repr(error),traceback=traceback.format_exc()))
        raise
