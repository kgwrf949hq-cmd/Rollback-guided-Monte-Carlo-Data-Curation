"""One root or cached-aux continuation on the current r3 simulator/writer."""
from __future__ import annotations
import argparse,json,os,random,shutil,time,traceback,signal
from pathlib import Path
import numpy as np
from curation_store import atomic_json,save_payload,load_payload
from trajectory_store import Trajectory,digest_obs,readback
from dual_core import (ActionContract,DisagreementScales,DisagreementForkSelector,
    AuxThenMainExecutor,Proposal,executed_disagreement,fingerprint,require_compatible_contracts)
from dual_runtime import VerifiedClient,stable_seed,capture_candidate,restore_candidate,probe_gripper_direction,save_diagnostic_state


def guard(base):
    if shutil.disk_usage(base).free<20*1024**3:raise RuntimeError('disk reserve below 20 GiB')
    total=sum(p.stat().st_size for p in base.rglob('*') if p.is_file())
    if total>20*1024**3:raise RuntimeError('task output limit reached')


def continuation_guard(cfg):
    if not cfg.get('resume_authorization_path'):return
    from dual_resume_budget import read_authorization,read_resource_ledger,remaining_seconds
    a=read_authorization(cfg);ledger=read_resource_ledger(cfg) or {}
    total=sum(p.lstat().st_size for p in Path(cfg['task_directory']).rglob('*') if p.is_file())
    combined=a['cumulative_resource_ledger']['storage']['combined_conservative_upper_bytes']+total+ledger.get('new_local_high_water_bytes',0)
    if combined>20*1024**3:raise RuntimeError('cumulative artifact limit reached')
    if remaining_seconds(cfg)<=cfg.get('finalization_reserve_seconds',600):raise RuntimeError('safe finalization time reached')


def collect(cfg,run_dir,branch_id='root',selected=None,interface=False):
    if selected and cfg['mode']=='shadow':raise RuntimeError('shadow cannot restore or execute a child')
    dual=cfg['fork_strategy']=='dual_disagreement'
    calibration_extension=cfg.get('calibration_source_id')=='calibration_shadow_extension_v1'
    if calibration_extension and (not dual or selected or interface or cfg['mode']!='shadow' or branch_id!='root'):
        raise RuntimeError('calibration-only seed is restricted to main-only shadow root')
    if calibration_extension and cfg.get('calibration_shadow_seed')!=stable_seed(cfg['root_seed'],'calibration_shadow_extension_v1'):
        raise RuntimeError('calibration-only seed differs from predeclared derivation')
    if selected and selected.get('fork_strategy')!=cfg['fork_strategy']:raise RuntimeError('selected candidate strategy mismatch')
    from libero_rollback_tool import make_libero_env,step_env,find_sim,close_env,capture_integration_state
    from libero_task_adapter import describe_task,TaskAdapter
    run_dir=Path(run_dir);base=Path(cfg['task_directory']);out=run_dir/'branches'/branch_id
    out.mkdir(parents=True,exist_ok=False);guard(base);continuation_guard(cfg)
    started=time.time();env=writer=main=aux=None;n=0;phase='initialization';fork=cfg['warmup_steps'];depth=0;parent=None;candidates=[];samples=[];executor=None;restore_report=None;gripper_direction=None
    def deadline(*a):raise TimeoutError('branch wall budget')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(min(2400,cfg.get('max_branch_wall_seconds',1800)))
    try:
        task=cfg['task'];random.seed(cfg['root_seed']);np.random.seed(cfg['root_seed'])
        current=describe_task(task['suite'],task['task_id'],task['init_state_id'])
        for k in ['name','bddl_sha256','init_file_sha256']:
            if task[k]!=current[k]:raise RuntimeError('task identity mismatch '+k)
        if (cfg.get('task_outcome_policy')=='official_horizon_success_else_failure_v1' and
            task.get('default_horizon')!=current.get('default_horizon')):raise RuntimeError('official task horizon identity mismatch')
        env,obs,info=make_libero_env(benchmark_name=task['suite'],task_id=task['task_id'],init_state_id=task['init_state_id'],seed=cfg['root_seed'],camera_width=cfg['camera_size'],camera_height=cfg['camera_size'])
        adapter=TaskAdapter(task,env);detector=adapter.detector()
        if selected:
            phase='restore';fork=selected['step'];depth=selected['depth'];parent=selected['parent_branch_id']
            if (depth>cfg['max_depth'] or selected['checkpoint']['owner_branch_id']!=parent or
                selected['checkpoint']['fork_step']!=fork or selected['checkpoint']['owner_depth']+1!=depth):raise RuntimeError('invalid parent ownership or fork coordinates')
            obs,restore_report=restore_candidate(env,detector,selected['checkpoint'],cfg['task_id'])
            atomic_json(out/'restore.json',restore_report)
            if not restore_report['cross_process']:raise RuntimeError('child requires cross-process restored snapshot')
            if adapter.truth(env).task_success:raise RuntimeError('terminal fork state')
            if dual:
                main_proposal=Proposal.from_dict(selected['main_proposal']);aux_proposal=Proposal.from_dict(selected['aux_proposal'])
                if main_proposal.raw_obs_fingerprint!=digest_obs(obs) or aux_proposal.raw_obs_fingerprint!=digest_obs(obs):raise RuntimeError('cached proposals do not match the restored observation')
                executor=AuxThenMainExecutor(aux_proposal,fork_step=fork)
        else:
            warm=Trajectory(out/'stabilization.h5',obs)
            try:
                for t in range(1,fork+1):
                    a=np.array([0,0,0,0,0,0,-1],dtype=np.float32);obs,r,d,x=step_env(env,a);warm.step(obs,a,a,t,r,d,{'action_source':'stabilization'},find_sim(env))
                    if d or adapter.truth(env).task_success:raise RuntimeError('terminal state reached during stabilization')
            finally:warm.close()
        initial=digest_obs(obs);start_truth=adapter.truth(env).as_dict()
        if start_truth['task_success']:raise RuntimeError('terminal initial state cannot produce model proposals')
        if interface:
            gripper_direction=probe_gripper_direction(env);atomic_json(out/'gripper_direction.json',gripper_direction)
            if gripper_direction['status']!='PASS':raise RuntimeError('gripper direction mapping failed')
            if cfg['action_contracts']['aux']['gripper_positive']!=gripper_direction['positive_command']:raise RuntimeError('configured gripper direction differs from deployed mapping')
        writer=Trajectory(out/'trajectory.h5',obs)
        contracts={r:ActionContract(**c) for r,c in cfg['action_contracts'].items()}
        if dual:require_compatible_contracts(contracts['main'],contracts['aux'])
        if not interface:main=VerifiedClient(cfg['models']['main'],cfg['task_id'],contracts['main'],out/'requests','main')
        # No auxiliary client is constructed in legacy mode.
        if (dual and not calibration_extension) or interface:aux=VerifiedClient(cfg['models']['aux'],cfg['task_id'],contracts['aux'],out/'requests','aux')
        scales=DisagreementScales(**cfg['scales']) if dual else None
        selector=DisagreementForkSelector(d_min=cfg['d_min'],numeric_floor=cfg['numeric_floor'],scales=scales,max_depth=cfg['max_depth'],spacing=cfg['candidate_spacing'],cooldown=cfg['suffix_cooldown'],mode=cfg['mode']) if dual and not interface and not calibration_extension else None
        t=fork;calls=0;chunk=None;offset=0;outcome='horizon';reason='shared_path_horizon';end=cfg['warmup_steps']+cfg['horizon'];replan_steps=5 if dual or interface else cfg.get('replan_steps',5)
        if interface:end=min(end,fork+cfg['interface_steps'])
        while t<end:
            if n%5==0:guard(base);continuation_guard(cfg)
            action=executor.next_action() if executor and executor.actual_k<5 and not executor.terminal else None
            if action is not None:
                source='aux';raw=np.asarray(executor.proposal.raw_actions[executor.actual_k],dtype=np.float32)
            else:
                if chunk is None or offset>=replan_steps:
                    phase='model';role='aux' if interface else 'main';client=aux if interface else main
                    rid='%s_%s_%s_%05d'%(run_dir.name,branch_id,role,calls)
                    _,frozen=capture_integration_state(find_sim(env));before=digest_obs(obs)
                    control_seed=(stable_seed(cfg['calibration_shadow_seed'],'root','main',calls) if calibration_extension
                                  else stable_seed(cfg['root_seed'],run_dir.name,branch_id,role,calls))
                    prop=client.proposal(obs,info['task_language'],rid,control_seed)
                    if calibration_extension:
                        _,after=capture_integration_state(find_sim(env))
                        if not np.array_equal(frozen,after) or digest_obs(obs)!=before:raise RuntimeError('environment advanced during main inference')
                        sample_path=out/'calibration_observations'/('%06d'%t)
                        save_diagnostic_state(env,obs,detector,sample_path,
                            out/'checkpoints'/('cp_%06d'%t),branch_id,depth,t,cfg['task_id'],
                            {'main':prop.context_fingerprint},
                            {'state_step':t,'state_fingerprint':fingerprint({'obs':before,'integration':frozen}),
                            'source_id':'calibration_shadow_extension_v1','source_run':str(run_dir),
                            'terminal':False,'confirmed_failed':False,'replan_boundary':True})
                        samples.append(str(sample_path))
                    if len(prop.canonical_actions)<replan_steps:raise RuntimeError('policy chunk shorter than replan interval')
                    if executor and executor.needs_main_replan:executor.mark_handoff(t,digest_obs(obs))
                    if selector is not None and depth<cfg['max_depth']:
                        rid_aux='%s_%s_aux_%05d'%(run_dir.name,branch_id,calls)
                        aprop=aux.proposal(obs,info['task_language'],rid_aux,stable_seed(cfg['root_seed'],run_dir.name,branch_id,'aux',calls))
                        _,after=capture_integration_state(find_sim(env))
                        if not np.array_equal(frozen,after) or digest_obs(obs)!=before:raise RuntimeError('environment advanced during comparison')
                        cp_path=out/'checkpoints'/('cp_%06d'%t)
                        capture=lambda:capture_candidate(env,obs,detector,cp_path,branch_id,depth,t,cfg['task_id'],{'main':prop.context_fingerprint,'aux':aprop.context_fingerprint})
                        checkpoint=None
                        if cfg['mode']=='shadow':
                            sample_path=out/'calibration_observations'/('%06d'%t)
                            checkpoint=save_diagnostic_state(env,obs,detector,sample_path,cp_path,
                                branch_id,depth,t,cfg['task_id'],
                                {'main':prop.context_fingerprint,'aux':aprop.context_fingerprint},
                                {'state_step':t,'state_fingerprint':fingerprint({'obs':before,'integration':frozen})})
                            samples.append(str(sample_path))
                        choice=selector.consider(branch_id=branch_id,depth=depth,step=t,branch_fork_step=fork,horizon=end,state_fingerprint=fingerprint({'obs':before,'integration':frozen}),main=prop,aux=aprop,checkpoint=checkpoint if checkpoint is not None else capture,terminal=False,confirmed_failed=False,recoverable=True)
                        comparison=dict(state_step=t,main_proposal=prop.as_dict(),aux_proposal=aprop.as_dict(),admitted=choice['admitted'],reason=choice['reason'],checkpoint=checkpoint if checkpoint is not None else (dict(choice['candidate'].checkpoint) if choice['admitted'] else None))
                        atomic_json(out/'comparisons'/('%06d.json'%t),comparison)
                        if choice['admitted']:
                            c=choice['candidate'];cd=dict(fork_strategy='dual_disagreement',fork_key=c.candidate_id,parent_branch_id=branch_id,depth=depth+1,step=t,D=c.score['D'],D_components=c.score,main_proposal=prop.as_dict(),aux_proposal=aprop.as_dict(),checkpoint=dict(c.checkpoint),checkpoint_path=c.checkpoint['path'],state_fingerprint=c.state_fingerprint,branch_fork_step=fork,selection_reason='disagreement_and_all_hard_gates')
                            candidates.append(cd);atomic_json(out/'candidates.json',candidates)
                    chunk=prop;offset=0;calls+=1
                    save_payload(out/'policy_chunks'/('%05d'%calls),prop.as_dict())
                action=np.asarray(chunk.canonical_actions[offset],dtype=np.float32);raw=np.asarray(chunk.raw_actions[offset],dtype=np.float32);offset+=1;source='aux' if interface else 'main'
            phase='environment';obs,reward,done,extra=step_env(env,action);t+=1;n+=1
            if not all(np.isfinite(np.asarray(v)).all() for v in obs.values()):raise RuntimeError('nonfinite observation')
            truth=adapter.truth(env);decision=detector.update(obs,step=t,task_truth=truth,done=done,action=action,reward=reward,info=extra)
            phase='storage';writer.step(obs,action,raw,t,reward,done,dict(action_source=source,task_truth=truth.as_dict(),failure=decision.to_summary()),find_sim(env))
            if executor and source=='aux':executor.record_transition(t,terminal=bool(done),success=truth.task_success,evidence=dict(env_done=bool(done),task_truth=truth.as_dict()) if done or truth.task_success else None)
            if truth.task_success:outcome='success';reason='env.check_success';break
            if done:outcome='task_failure';reason='environment_terminated_without_success';break
            if decision.failed:
                if decision.reason in ['invalid_observation','simulation_error']:raise RuntimeError(decision.reason)
                outcome='rule_stop';reason=decision.reason;break
            if (not dual and not interface and depth<cfg['max_depth'] and t<end and
                len(candidates)<cfg.get('max_checkpoints_per_node',6) and
                (n==1 or n%cfg.get('checkpoint_interval',30)==0)):
                # Current r3 legacy periodic checkpoint policy remains separate.
                cp_path=out/'checkpoints'/('cp_%06d'%t)
                checkpoint=capture_candidate(env,obs,detector,cp_path,branch_id,depth,t,cfg['task_id'],{'main':'fresh_seeded_chunk_at_fork'})
                _,integration=capture_integration_state(find_sim(env))
                cd=dict(fork_strategy='legacy',fork_key=fingerprint({'checkpoint':checkpoint,'strategy':'legacy'}),
                    parent_branch_id=branch_id,depth=depth+1,step=t,D=0.,D_components=None,
                    checkpoint=checkpoint,checkpoint_path=checkpoint['path'],
                    state_fingerprint=fingerprint({'obs':digest_obs(obs),'integration':integration}),
                    branch_fork_step=fork,selection_reason='legacy_periodic_checkpoint')
                candidates.append(cd);atomic_json(out/'candidates.json',candidates)
        if (not interface and outcome=='horizon' and t==end and
            cfg.get('task_outcome_policy')=='official_horizon_success_else_failure_v1' and
            cfg['horizon']==task.get('default_horizon')):
            outcome='task_failure';reason='official_horizon_exhausted'
        if not n:raise RuntimeError('zero length branch')
        writer.close();writer=None;validation=readback(out/'trajectory.h5',fork,initial);atomic_json(out/'readback.json',validation)
        d_exec={'D':None,'m':0,'valid':False};exploration_fingerprint=None;duplicate_of=None
        if selected and dual:
            import h5py
            pp=run_dir/'branches'/parent
            pr=json.loads((pp/'branch.json').read_text());idx=fork-pr['fork_step']
            with h5py.File(pp/'trajectory.h5','r') as p,h5py.File(out/'trajectory.h5','r') as c:
                parent_actions=p['executed_actions'][idx:idx+5];child_actions=c['executed_actions'][:executor.actual_k]
                # Parent's actual suffix must match the very proposal that earned this D.
                expected=np.asarray(selected['main_proposal']['canonical_actions'],dtype=np.float32)[:len(parent_actions)]
                if not np.array_equal(parent_actions,expected):raise RuntimeError('parent executed different main proposal')
                d_exec=executed_disagreement(parent_actions,child_actions,scales,cfg['d_min'],cfg['numeric_floor'])
                if d_exec['m']:
                    exploration_fingerprint=fingerprint({'initial_observation_sha256':initial,
                        'executed_aux_prefix':child_actions[:d_exec['m']].astype(np.float32).tolist()})
        # Rule stops within K have no normal terminal proof and cannot pass intervention validation.
        inter=executor.report() if executor else dict(planned_K=0,actual_K=0,handoff_occurred=False,handoff_state_step=None)
        intervention_ok=(not dual or not selected or inter['actual_K']==5 or inter.get('terminal',False))
        valid=bool(selected and validation['readback_passed'] and intervention_ok and (d_exec['valid'] if dual else True))
        if exploration_fingerprint:
            for saved in sorted((run_dir/'branches').glob('*/branch.json'),key=lambda p:(p.stat().st_mtime_ns,str(p))):
                prior=json.loads(saved.read_text())
                if prior.get('execution_valid') and prior.get('exploration_fingerprint')==exploration_fingerprint:
                    duplicate_of=prior.get('duplicate_of') or prior['branch_id'];valid=False;break
        record=dict(schema='dual_branch_v1',branch_id=branch_id,parent_branch_id=parent,depth=depth,fork_step=fork,final_env_step=t,control_steps=n,outcome=outcome,termination_reason=reason,
            task_id=cfg['task_id'],run_id=run_dir.name,mode=cfg['mode'],fork_strategy=cfg['fork_strategy'],selected_candidate=selected,initial_observation_sha256=initial,
            task_default_horizon=task.get('default_horizon'),task_outcome_policy=cfg.get('task_outcome_policy'),
            execution_valid=bool(validation['readback_passed'] and intervention_ok),valid_nonroot=valid,D_executed=d_exec['D'],executed_components=d_exec,m=d_exec['m'],
            exploration_fingerprint=exploration_fingerprint,duplicate_of=duplicate_of,
            restore_report=restore_report,readback=validation,candidates=candidates,calibration_samples=samples,gripper_direction=gripper_direction,initial_task_truth=start_truth,final_task_truth=adapter.truth(env).as_dict(),wall_seconds=time.time()-started,**inter)
        atomic_json(out/'branch.json',record);print(json.dumps({'branch_id':branch_id,'outcome':outcome,'steps':n,'valid':valid,'candidates':len(candidates)}),flush=True);return record
    except BaseException as exc:
        atomic_json(out/'error.json',dict(status='error',phase=phase,error=repr(exc),traceback=traceback.format_exc(),counts_as_task_failure=False,executed_steps=n,at=time.time()));raise
    finally:
        signal.alarm(0)
        if writer is not None:writer.close()
        for client in [main,aux]:
            if client is not None:client.close()
        if env is not None:close_env(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('request');a=p.parse_args();r=json.loads(Path(a.request).read_text());collect(r['config'],r['run_dir'],r.get('branch_id','root'),r.get('selected'),r.get('interface',False))
