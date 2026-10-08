"""Small fresh-candidate experiment; all branches restore the same checkpoint."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from curation_store import atomic_json,load_payload,save_payload
from dual_core import ActionContract,DisagreementScales,disagreement,fingerprint,require_compatible_contracts
from dual_runtime import VerifiedClient,stable_seed,save_diagnostic_state,restore_candidate,probe_gripper_direction
from trajectory_store import Trajectory,digest_obs,readback
from libero_rollback_tool import step_env,find_sim,close_env,capture_integration_state
from verify_state_replay import make_case_env,compare
from verify_common_refresh import refresh,normalized_snapshot
from steps123_logic import pair_stats

M=json.loads((BASE/'experiment_manifest.json').read_text())
CONTRACTS={r:ActionContract(**x) for r,x in M['action_contracts'].items()}
require_compatible_contracts(CONTRACTS['main'],CONTRACTS['aux'])


def ledger(event,**fields):
    with (BASE/'experiment_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=event,at=time.time(),pid=os.getpid(),**fields))+'\n')
        f.flush()


def guard():
    if time.time()>M['deadline_epoch']-180:raise RuntimeError('bounded experiment deadline')
    if shutil.disk_usage(BASE).free<20*1024**3:raise RuntimeError('disk reserve')
    if sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())>M['max_artifact_bytes']:
        raise RuntimeError('bounded artifact budget')


def clients(directory):
    return {r:VerifiedClient(M['models'][r],M['run_id'],CONTRACTS[r],directory/'requests',r)
            for r in ['main','aux']}


def ask(client,obs,case,request,seed):
    guard()
    record=BASE/'model_requests.jsonl'
    if record.exists() and sum(1 for _ in record.open())>=M['max_model_requests']:
        raise RuntimeError('bounded model request budget')
    with record.open('a') as f:f.write(json.dumps(dict(request_id=request,seed=seed,role=client.role))+'\n')
    return client.proposal(obs,case['task']['language'],request,seed)


def frame_video(directory):
    from PIL import Image,ImageDraw
    destination=directory/'trajectory.mp4'
    command=['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo',
        '-pix_fmt','rgb24','-s','640x360','-r','20','-i','pipe:0','-an','-c:v','libx264',
        '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    n=0
    try:
        with h5py.File(directory/'trajectory.h5') as h:
            prior={}
            for i in range(len(h['observation_sha256'])):
                canvas=Image.new('RGB',(640,360),(18,22,30));draw=ImageDraw.Draw(canvas)
                draw.text((8,8),directory.name+' | offset '+str(i)+' | recorded observations',fill='white')
                for j,key in enumerate(['agentview_image','robot0_eye_in_hand_image']):
                    ds=h['observations'][key]
                    raw=np.bitwise_xor(ds[i],prior[key]) if i and ds.attrs['xor_previous'] else ds[i]
                    prior[key]=raw
                    canvas.paste(Image.fromarray(raw[::-1]).resize((320,320)),(320*j,40))
                process.stdin.write(np.asarray(canvas).tobytes());n+=1
    finally:
        process.stdin.close()
        code=process.wait()
    if code:raise RuntimeError('video encoding failed')
    atomic_json(directory/'video.json',dict(frames=n,fps=20,source='trajectory.h5',display='vertical flip of raw OpenGL observations'))


def key_state(env):
    data=find_sim(env).data
    return {k:np.asarray(getattr(data,k)).copy() for k in ['qpos','qvel','act','xpos','site_xpos','time']}


def gate(candidate_path):
    candidate=json.loads(candidate_path.read_text());case=candidate['case']
    directory=candidate_path.parent/'gate';directory.mkdir()
    env=None;cs={}
    try:
        cs=clients(directory)
        env,obs,adapter,detector=make_case_env(case)
        obs,restore=restore_candidate(env,detector,candidate['checkpoint'],M['run_id'])
        atomic_json(directory/'restore.json',restore)
        obs=refresh(env)
        initial=normalized_snapshot(env,obs,adapter,detector)
        save_payload(directory/'normalized_initial',dict(state=initial))
        props={r:[] for r in cs}
        for rep in range(5):
            for role in ['main','aux']:
                request=candidate['candidate_id']+'_gate_'+role+'_'+str(rep)
                p=ask(cs[role],obs,case,request,stable_seed(M['master_seed'],candidate['candidate_id'],'gate',role,rep))
                props[role].append(p.as_dict())
        stats=pair_stats(props,M['d_min'])
        static=disagreement(props['main'][0]['canonical_actions'],props['aux'][0]['canonical_actions'],DisagreementScales(**M['scales']),5)
        stable=bool(stats['noise_separated'] and static['D']>=M['d_min'] and static['D']>M['numeric_floor'])
        unchanged=not compare(initial,normalized_snapshot(env,obs,adapter,detector))
        if not unchanged:raise RuntimeError('gate inference changed candidate state')
        result=dict(status='QUALIFIED' if stable else 'REJECTED_STABLE_DISAGREEMENT',
            complete_checkpoint=True,restore_pass=restore['success'],cross_process_restore=restore['cross_process'],
            observation_sha256=digest_obs(obs),stats=stats,post_refresh_D=static,
            post_refresh_static_threshold_pass=static['D']>=M['d_min'],state_unchanged=unchanged,
            pre_refresh_D=candidate['source_disagreement'],candidate_id=candidate['candidate_id'])
        atomic_json(directory/'result.json',result)
        return result
    finally:
        for client in cs.values():client.close()
        if env is not None:close_env(env)


def collect(root_id):
    root=next(r for r in M['roots'] if r['root_id']==root_id)
    case=dict(root=root,task=M['tasks'][root['task_label']],run_id=M['run_id'])
    directory=BASE/'collection'/root_id;directory.mkdir(parents=True)
    env=writer=None;cs={};selected=None;reason='horizon';n=0;admitted=[];gates=0
    try:
        cs=clients(directory)
        env,obs,adapter,detector=make_case_env(case)
        interface=probe_gripper_direction(env)
        if interface['status']!='PASS':raise RuntimeError('action interface verification failed')
        atomic_json(directory/'interface.json',interface)
        for _ in range(10):obs,_,done,_=step_env(env,np.array([0,0,0,0,0,0,-1],np.float32))
        writer=Trajectory(directory/'trajectory.h5',obs)
        failure=False;seen=set()
        while n<520:
            before=key_state(env);observation=digest_obs(obs)
            request=M['run_id']+'_'+root_id+'_main_'+str(n)
            p=ask(cs['main'],obs,case,request,stable_seed(M['master_seed'],root_id,'collection_main',n))
            if n%10==0 and 520-n>=10 and gates<M['max_gate_attempts_per_root']:
                q=ask(cs['aux'],obs,case,M['run_id']+'_'+root_id+'_aux_'+str(n),stable_seed(M['master_seed'],root_id,'collection_aux',n))
                score=disagreement(p.canonical_actions,q.canonical_actions,DisagreementScales(**M['scales']),5)
                state=fingerprint(dict(integration=capture_integration_state(find_sim(env))[1],obs=observation))
                cid=root_id+'_s%06d'%n
                target=directory/'candidates'/cid;target.mkdir(parents=True)
                cp=save_diagnostic_state(env,obs,detector,target/'diagnostic_state',target/'checkpoint',
                    root_id,0,n+10,M['run_id'],{'main':p.context_fingerprint,'aux':q.context_fingerprint},
                    dict(control_step=n,env_step=n+10,source='fresh_live_main_collection',state_fingerprint=state))
                static=dict(replan_boundary=n%5==0,nonterminal=not adapter.truth(env).task_success,
                    same_observation=p.raw_obs_fingerprint==q.raw_obs_fingerprint==observation,
                    finite_actions=bool(np.isfinite(p.canonical_actions).all() and np.isfinite(q.canonical_actions).all()),
                    remaining_horizon_at_least_2K=520-n>=10,threshold=score['D']>=M['d_min'],
                    numeric_floor=score['D']>M['numeric_floor'],unique_state=state not in seen,
                    candidate_spacing=all(abs(n-old)>=10 for old in admitted),failure_filter=not failure)
                item=dict(candidate_id=cid,checkpoint=cp,candidate_control_step=n,candidate_env_step=n+10,
                    source_disagreement=score,static_filters=static,case=case,
                    source='new_main_policy_trajectory_no_historical_snapshot_or_action_replay')
                atomic_json(target/'candidate.json',item)
                if all(static.values()):
                    admitted.append(n);seen.add(state);gates+=1
                    with (target/'gate.log').open('w') as f:
                        result=subprocess.run([sys.executable,__file__,'gate',str(target/'candidate.json')],stdout=f,stderr=subprocess.STDOUT,timeout=600)
                    if result.returncode:raise RuntimeError('candidate gate execution failed: '+cid)
                    result=json.loads((target/'gate/result.json').read_text())
                    ledger('candidate_screened',candidate_id=cid,status=result['status'],D=score['D'])
                    if result['status']=='QUALIFIED':
                        selected=str(target/'candidate.json');reason='qualified_candidate_found';break
            if compare(before,key_state(env)) or digest_obs(obs)!=observation:
                raise RuntimeError('collection inference or candidate gate mutated source environment')
            for k in range(min(5,520-n)):
                a=np.asarray(p.canonical_actions[k],np.float32)
                obs,r,done,extra=step_env(env,a);n+=1
                truth=adapter.truth(env);decision=detector.update(obs,step=n+10,task_truth=truth,done=done,action=a,reward=r,info=extra)
                writer.step(obs,a,np.asarray(p.raw_actions[k],np.float32),n+10,r,done,dict(source='main',task_truth=truth.as_dict(),failure=decision.to_summary()),find_sim(env))
                failure=decision.failed
                if truth.task_success:reason='success';break
                if done:reason='environment_failure';break
                if failure:reason='failure_detector:'+str(decision.reason);break
            atomic_json(directory/'progress.json',dict(control_steps=n,gate_attempts=gates,reason=reason))
            if reason!='horizon':break
        writer.close();writer=None
        result=dict(root_id=root_id,candidate=selected,collection_steps=n,reason=reason,gate_attempts=gates,
            terminal=bool(adapter.truth(env).task_success or failure),source='fresh_model_rollout')
        atomic_json(directory/'result.json',result);ledger('collection_finished',**result)
        return result
    finally:
        if writer is not None:writer.close()
        for client in cs.values():client.close()
        if env is not None:close_env(env)


def branch(candidate_path,rep,arm):
    item=json.loads(candidate_path.read_text());case=item['case'];cid=item['candidate_id']
    directory=BASE/'pairs'/cid/('rep%d_%s'%(rep,arm));directory.mkdir(parents=True)
    gate_result=json.loads((candidate_path.parent/'gate/result.json').read_text())
    if gate_result['status']!='QUALIFIED':raise RuntimeError('unqualified candidate')
    env=writer=None;cs={};keyframes=[];offset=0;actual_aux=0;handoff=False;outcome='failure';reason='episode_horizon'
    try:
        cs=clients(directory)
        env,obs,adapter,detector=make_case_env(case)
        obs,restore=restore_candidate(env,detector,item['checkpoint'],M['run_id'])
        atomic_json(directory/'restore.json',restore)
        if not restore['cross_process']:raise RuntimeError('branch did not independently restore')
        obs=refresh(env)
        initial=normalized_snapshot(env,obs,adapter,detector)
        baseline=load_payload(candidate_path.parent/'gate/normalized_initial')['state']
        diffs=compare(baseline,initial)
        atomic_json(directory/'initial_equality.json',dict(exact=not diffs,differences=diffs))
        save_payload(directory/'initial_state',dict(state=initial))
        if diffs:raise RuntimeError('branch initial state mismatch')
        start_sha=digest_obs(obs);writer=Trajectory(directory/'trajectory.h5',obs)
        keyframes.append(key_state(env));total_reward=0.;remaining=520-item['candidate_control_step']
        while offset<remaining:
            role='aux' if arm=='dual' and offset<5 else 'main'
            if arm=='dual' and role=='main' and not handoff:
                handoff=True;ledger('handoff',candidate_id=cid,replicate=rep,offset=offset)
            # Main seeds are paired by absolute step, never by arm or execution order.
            absolute=item['candidate_control_step']+offset
            seed=stable_seed(M['master_seed'],cid,'pair',rep,role,absolute)
            request=M['run_id']+'_'+cid+'_r%d_%s_%s_%d'%(rep,arm,role,absolute)
            p=ask(cs[role],obs,case,request,seed)
            for k in range(min(5,remaining-offset)):
                a=np.asarray(p.canonical_actions[k],np.float32)
                obs,r,done,extra=step_env(env,a);offset+=1;total_reward+=float(r)
                if role=='aux':actual_aux+=1
                truth=adapter.truth(env)
                decision=detector.update(obs,step=item['candidate_env_step']+offset,task_truth=truth,done=done,action=a,reward=r,info=extra)
                writer.step(obs,a,np.asarray(p.raw_actions[k],np.float32),item['candidate_env_step']+offset,r,done,
                    dict(action_source=role,request_id=request,seed=seed,task_truth=truth.as_dict(),failure=decision.to_summary()),find_sim(env))
                keyframes.append(key_state(env))
                if truth.task_success:outcome='success';reason='official_env.check_success';break
                if done:reason='environment_terminated_without_success';break
                if decision.failed:reason='failure_detector:'+str(decision.reason);break
            atomic_json(directory/'progress.json',dict(steps=offset,actual_aux_steps=actual_aux,task_progress=truth.as_dict()))
            if truth.task_success or done or decision.failed:break
        writer.close();writer=None
        validation=readback(directory/'trajectory.h5',item['candidate_env_step'],start_sha)
        np.savez_compressed(directory/'key_states.npz',**{k:np.stack([s[k] for s in keyframes]) for k in keyframes[0]})
        save_payload(directory/'final_state',dict(state=normalized_snapshot(env,obs,adapter,detector)))
        frame_video(directory)
        result=dict(status='COMPLETED',candidate_id=cid,checkpoint=item['checkpoint'],candidate_step=item['candidate_control_step'],
            source_disagreement=item['source_disagreement'],post_refresh_disagreement=gate_result['post_refresh_D'],
            stable_disagreement=gate_result['stats']['summary'],replicate=rep,arm=arm,steps=offset,
            actual_aux_steps=actual_aux,handoff_occurred=handoff,success=outcome=='success',failure=outcome!='success',
            outcome=outcome,terminal_reason=reason,official_env_success=truth.task_success,env_done=bool(done),
            rule_cutoff=reason.startswith('failure_detector:'),natural_failure_confirmed=bool(done and not truth.task_success),
            final_task_progress=truth.as_dict(),reward_sum=total_reward,initial_exact=True,readback=validation,
            trajectory=str(directory/'trajectory.h5'),video=str(directory/'trajectory.mp4'),pid=os.getpid())
        atomic_json(directory/'result.json',result);ledger('branch_finished',candidate_id=cid,replicate=rep,arm=arm,steps=offset,outcome=outcome,terminal_reason=reason)
        return result
    finally:
        if writer is not None:writer.close()
        for client in cs.values():client.close()
        if env is not None:close_env(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','gate','branch']);p.add_argument('target');p.add_argument('--rep',type=int);p.add_argument('--arm',choices=['control','dual']);a=p.parse_args()
    try:
        result=collect(a.target) if a.mode=='collect' else gate(Path(a.target)) if a.mode=='gate' else branch(Path(a.target),a.rep,a.arm)
        print(json.dumps(result),flush=True)
    except Exception as error:
        ledger('execution_error',mode=a.mode,target=a.target,error=repr(error),traceback=traceback.format_exc())
        raise
