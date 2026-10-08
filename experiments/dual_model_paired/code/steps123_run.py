"""Bounded saved-state repeats and one fixed intervention per source root."""
import json,os,random,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from curation_store import atomic_json,load_payload,save_payload
from steps123_logic import candidate_subset,controls,pair_stats,select_fork,distance
from steps123_budget import reserve,step as budget_step

def sources(p):
 return [dict(r,directory=r['source_directory']) for r in p.M['source_roots']]+[dict(r,directory=str(p.B/'roots'/r['root_id'])) for r in p.M['roots']]

def rows_for(root):
 return [json.loads(f.read_text()) for f in sorted((Path(root['directory'])/'comparisons').glob('*.json'))]

def freeze(p,name,value):
 path=p.B/name
 if path.exists():raise RuntimeError('selection already frozen:'+name)
 atomic_json(path,value);p.append('run_ledger.jsonl',dict(event='selection_frozen',name=name,sha256=p.sha(path)))

def run_repeats(p):
 chosen=[];missing=[];a_count=0
 for root in sources(p):
  rows=rows_for(root);c=candidate_subset(rows,root['task_label'])
  if root['task_label']=='A':a_count+=len(c)
  seen=set()
  for kind,subset in [('candidate',c),('noncandidate_control',controls(rows))]:
   for row in subset:
    step=row['control_step'];key=(row['task_name'],row['init_index'],row['state_sha256'])
    if key in seen:continue
    seen.add(key)
    path=Path(root['directory'])/'states'/('%06d'%step)
    if not path.with_suffix('.json').exists() or not path.with_suffix('.npz').exists():
     missing.append(dict(root_id=root['root_id'],control_step=step,reason='saved_state_missing'));continue
    chosen.append(dict(root=root,kind=kind,control_step=step,comparison=str(Path(root['directory'])/'comparisons'/('%06d.json'%step)),comparison_sha256=p.sha(Path(root['directory'])/'comparisons'/('%06d.json'%step)),state=str(path),state_json_sha256=p.sha(path.with_suffix('.json')),state_npz_sha256=p.sha(path.with_suffix('.npz')),state_sha256=row['state_sha256'],observation_sha256=row['observation_sha256'],input_fingerprints={role:row[role]['input_fingerprint'] for role in ['main','aux']},original_static_candidate=row['static_candidate'],original_over_threshold=row['over_threshold'],control_anchor_step=row.get('control_anchor_step'),control_same_predicates=row.get('control_same_predicates')))
 if a_count!=15:raise RuntimeError('historical_A_candidate_count_changed:'+str(a_count))
 if len(chosen)>33:raise RuntimeError('repeat_state_budget')
 freeze(p,'repeat_selection.json',dict(rules=p.M['repeat_selection_rules'],states=chosen,missing=missing,expected_requests=10*len(chosen)))
 cs={r:p.client(r,p.B/'repeat_requests') for r in ['main','aux']};count=0
 try:
  for c in chosen:
   p.guard();state=load_payload(c['state']);row=json.loads(Path(c['comparison']).read_text());props={r:[] for r in cs}
   from trajectory_store import digest_obs
   assert digest_obs(state['obs'])==c['observation_sha256']==state['observation_sha256']
   for role in ['main','aux']:
    for rep in range(5):
     rid=p.M['run_id']+'_'+c['root']['root_id']+'_repeat_'+str(c['control_step'])+'_'+role+'_'+str(rep)
     prop=p.ask(cs[role],c['root'],state['obs'],rid,'diagnostic_repeat_'+role,rep)
     assert not prop.cache_hit and prop.input_fingerprint==c['input_fingerprints'][role],'repeat_input_or_cache_mismatch'
     assert prop.raw_obs_fingerprint==c['observation_sha256']
     props[role].append(prop.as_dict());count+=1
   result=dict(root_id=c['root']['root_id'],task_label=c['root']['task_label'],control_step=c['control_step'],selection_kind=c['kind'],state_sha256=c['state_sha256'],observation_sha256=c['observation_sha256'],proposals=props,**pair_stats(props,p.M['d_min']))
   atomic_json(p.B/'repeats'/(c['root']['root_id']+'_%06d.json'%c['control_step']),result)
   atomic_json(p.B/'repeat_progress.json',dict(completed_requests=count,completed_states=count//10,expected_requests=10*len(chosen)))
  return count
 finally:
  for c in cs.values():c.close()

def restorable(row):
 cp=row.get('checkpoint')
 if not cp:return False
 path=Path(cp['path'])
 if not path.with_suffix('.json').exists() or not path.with_suffix('.npz').exists():return False
 d=load_payload(path)
 return all(d.get(k) is not None for k in ['runtime_state','rollout_state','rng_state','model_state','integration_state','obs']) and 'fresh_render_reference' in d.get('metadata',{})

def run_forks(p,owned):
 src={r['root_id']:r for r in sources(p)};selection={}
 for rid in p.M['fork_order']:
  root=src[rid];rows=rows_for(root)
  repeats={x['control_step']:x for f in (p.B/'repeats').glob(rid+'_*.json') for x in [json.loads(f.read_text())]}
  row,excluded=select_fork(rows,repeats,restorable)
  if row:
   cp=row['checkpoint']
   for ext,key in [('.json','json_sha256'),('.npz','npz_sha256')]:assert p.sha(Path(cp['path']).with_suffix(ext))==cp[key]
   parent=Path(root['directory'])/'trajectory.h5'
   if not parent.exists():raise RuntimeError('parent_trajectory_missing')
   import h5py
   from trajectory_store import digest_obs
   with h5py.File(parent,'r') as h:
    n=row['control_step'];m=min(5,len(h['executed_actions'])-n)
    assert m>0 and h['observation_sha256'][n].decode()==row['observation_sha256']
    assert np.array_equal(h['executed_actions'][n:n+m],np.asarray(row['main']['canonical_actions'],np.float32)[:m])
   selection[rid]=dict(status='SELECTED',root=root,row=row,parent_trajectory_sha256=p.sha(parent),excluded=excluded)
  else:selection[rid]=dict(status='BLOCKED' if root['task_label']=='A' else 'NO_ELIGIBLE_POINT',reason='no candidate meets frozen entry conditions',root=root,excluded=excluded)
 freeze(p,'fork_selection.json',selection)
 results={};blocked_shared=False
 for rid in p.M['fork_order']:
  item=selection[rid]
  if item['status']!='SELECTED':results[rid]={'status':item['status'],'reason':item['reason']};continue
  if blocked_shared:results[rid]={'status':'BLOCKED_CORRECTNESS','reason':'prior shared recovery/execution error'};continue
  p.guard()
  cmd=['/usr/bin/python3',str(p.B/'code/steps123_run.py'),'fork',rid]
  with (p.B/'logs'/(rid+'_fork.log')).open('x') as log:
   worker=p.subprocess.Popen(cmd,cwd=p.B,env=os.environ.copy(),stdout=log,stderr=p.subprocess.STDOUT,start_new_session=True)
  identity=p.identity(worker,'fork_'+rid);owned.append((worker,identity));p.append('process_ledger.jsonl',dict(event='started',identity=identity,argv=cmd))
  while worker.poll() is None:
   if time.time()>=p.DEADLINE-180:raise RuntimeError('deadline')
   time.sleep(1)
  path=p.B/'forks'/rid/'result.json';result=json.loads(path.read_text()) if path.exists() else {'status':'ERROR','reason':'worker_no_result'}
  results[rid]=result
  if worker.returncode or result.get('status') in ['ERROR','BLOCKED_CORRECTNESS']:blocked_shared=True
  atomic_json(p.B/'fork_progress.json',results)
 return results

def decode_observations(h,end):
 previous={};result={}
 for i in range(end+1):
  obs={k:np.bitwise_xor(ds[i],previous[k]) if ds.attrs['xor_previous'] and i else ds[i] for k,ds in h['observations'].items()}
  result[i]=obs;previous=obs
 return result

def fork_worker(p,rid):
 import h5py
 from libero_rollback_tool import make_libero_env,find_sim,close_env,capture_integration_state
 from libero_task_adapter import TaskAdapter
 from dual_runtime import restore_candidate,probe_gripper_direction
 from trajectory_store import Trajectory,digest_obs,readback
 item=json.loads((p.B/'fork_selection.json').read_text())[rid];assert item['status']=='SELECTED'
 root=item['root'];row=item['row'];task=p.M['tasks'][root['task_label']];n0=row['control_step'];t0=row['env_step']
 out=p.B/'forks'/rid;out.mkdir(parents=True,exist_ok=False)
 env=writer=main=None;actual_k=0;main_steps=0;phase='restore';attempt=False;matched=0;newseeds=0
 signal.signal(signal.SIGTERM,p.stop_signal);signal.signal(signal.SIGALRM,p.stop_signal);signal.alarm(max(1,int(p.DEADLINE-time.time()-180)))
 try:
  p.guard();reserve(p.B,'recovery_points',rid,4)
  parent=Path(root['directory'])/'trajectory.h5';assert p.sha(parent)==item['parent_trajectory_sha256']
  with h5py.File(parent,'r') as h:
   m=min(5,len(h['executed_actions'])-n0);actions=h['executed_actions'][n0:n0+m];qpos=h['qpos'][n0:n0+m];qvel=h['qvel'][n0:n0+m];rewards=h['rewards'][n0:n0+m];dones=h['done'][n0:n0+m]
   refs=decode_observations(h,n0+m)
   decisions=[json.loads(x) for x in h['decisions'][n0:n0+m]]
  random.seed(root['env_seed']);np.random.seed(root['env_seed'])
  env,obs,info=make_libero_env(benchmark_name='libero_10',task_id=task['task_id'],init_state_id=root['init_index'],seed=root['env_seed'],camera_width=256,camera_height=256)
  assert info['task_language']==task['language']
  adapter=TaskAdapter(task,env);detector=adapter.detector()
  assert probe_gripper_direction(env)['status']=='PASS'
  obs,restore=restore_candidate(env,detector,row['checkpoint'],p.M['run_id'])
  assert restore['cross_process'] and digest_obs(obs)==row['observation_sha256']
  atomic_json(out/'restore_initial.json',restore)
  replay=[]
  for k,a in enumerate(actions):
   obs,reward,done,_=budget_step(p,env,a,rid+'_recovery_replay',str(k),5);truth=adapter.truth(env)
   detector.update(obs,step=t0+k+1,task_truth=truth,done=done)
   sim=find_sim(env);record=dict(offset=k,observation_sha256=digest_obs(obs),reference_observation_sha256=digest_obs(refs[n0+k+1]),qpos_max_abs=float(np.max(np.abs(sim.data.qpos-qpos[k]))),qvel_max_abs=float(np.max(np.abs(sim.data.qvel-qvel[k]))),predicates=truth.predicates,reference_predicates=decisions[k]['truth']['predicates'],reward_abs=abs(float(reward)-float(rewards[k])),done=bool(done),reference_done=bool(dones[k]))
   record['pass']=record['observation_sha256']==record['reference_observation_sha256'] and record['qpos_max_abs']<=1e-8 and record['qvel_max_abs']<=1e-8 and record['predicates']==record['reference_predicates'] and record['reward_abs']<=1e-8 and record['done']==record['reference_done']
   replay.append(record);atomic_json(out/'recovery_replay.json',dict(initialization_env_steps=0,steps=replay,tolerance=1e-8,observation_tolerance=0))
   if not record['pass']:raise RuntimeError('saved_parent_replay_mismatch')
   if done or truth.task_success:break
  obs,again=restore_candidate(env,detector,row['checkpoint'],p.M['run_id']);assert digest_obs(obs)==row['observation_sha256']
  atomic_json(out/'restore_before_aux.json',again)
  phase='aux_then_main';reserve(p.B,'child_attempts',rid,4);attempt=True
  p.append('attempt_ledger.jsonl',dict(event='child_attempt_reserved',root_id=rid,control_step=n0,source_checkpoint=row['checkpoint'],parent_trajectory_sha256=item['parent_trajectory_sha256']))
  writer=Trajectory(out/'trajectory.h5',obs);main=p.client('main',out/'requests');n=n0;t=t0;chunk=None;offset=0;outcome='horizon';reason='shared_root_horizon';terminal=False
  parent_comparisons={x['control_step']:x for x in rows_for(root)};executed_aux=[]
  while n<520:
   if actual_k<5:
    source='aux';a=np.asarray(row['aux']['canonical_actions'][actual_k],np.float32);raw=np.asarray(row['aux']['raw_actions'][actual_k],np.float32);request_id=row['aux']['request_id'];off=actual_k;seed_match=None
   else:
    source='main'
    if chunk is None or offset>=5:
     prefix=p.M['run_id']+'_'+rid+'_child_'+str(n)
     orig=parent_comparisons.get(n);override=orig['main']['inference_settings']['noise_seed'] if orig else None
     chunk=p.ask(main,root,obs,prefix,'child_main',seed_override=override);offset=0
     seed_match=override is not None
     if seed_match:matched+=1
     else:newseeds+=1
    a=np.asarray(chunk.canonical_actions[offset],np.float32);raw=np.asarray(chunk.raw_actions[offset],np.float32);request_id=chunk.request_id;off=offset;offset+=1
   obs,reward,done,extra=budget_step(p,env,a,rid+'_child',str(n),520-n0);n+=1;t+=1
   assert all(np.isfinite(np.asarray(v)).all() for v in obs.values()),'nonfinite_child_observation'
   truth=adapter.truth(env);failure=detector.update(obs,step=t,task_truth=truth,done=done).status.value
   if source=='aux':actual_k+=1;executed_aux.append(a.copy())
   else:main_steps+=1
   writer.step(obs,a,raw,t,reward,done,dict(source=source,request_id=request_id,chunk_offset=off,truth=truth.as_dict(),matched_parent_noise_seed=seed_match),find_sim(env))
   atomic_json(out/'progress.json',dict(actual_K=actual_k,main_steps=main_steps,control_step=n))
   if truth.task_success:outcome='success';reason='official_success';terminal=True;break
   if done:outcome='task_unsuccessful';reason='environment_terminated_without_success';terminal=True;break
   if failure in ['recoverable_failure','terminal_failure']:outcome='rule_stopped';reason='existing_failure_detector';break
  writer.close();writer=None
  common=min(5,len(actions),actual_k);d=distance(actions,executed_aux,common) if common else None
  execution_valid=actual_k==5 or terminal
  result=dict(root_id=rid,status='COMPLETED',restore_pass=True,recovery_replay_steps=len(replay),initialization_env_steps=0,child_attempt=True,control_step=n0,env_step=t0,actual_K=actual_k,handoff_occurred=main_steps>0,main_steps=main_steps,child_control_steps=n-n0,final_control_step=n,m=common,D_executed=d,execution_valid=execution_valid,effective_difference=bool(execution_valid and d is not None and d>=p.M['d_min'] and d>p.M['numeric_floor']),outcome=outcome,termination_reason=reason,matched_parent_seed_requests=matched,fallback_child_seed_requests=newseeds,readback=readback(out/'trajectory.h5',t0,row['observation_sha256']),parent_trajectory=str(parent),parent_trajectory_sha256=item['parent_trajectory_sha256'])
  atomic_json(out/'result.json',result);p.append('attempt_ledger.jsonl',dict(event='child_finished',**result))
 except BaseException as e:
  atomic_json(out/'result.json',dict(root_id=rid,status='BLOCKED_CORRECTNESS',phase=phase,error=repr(e),traceback=traceback.format_exc(),child_attempt=attempt,actual_K=actual_k,main_steps=main_steps,counts_as_task_failure=False))
  raise
 finally:
  if writer:writer.close()
  if main:main.close()
  if env:close_env(env)

if __name__=='__main__':
 import probe_runner as p
 if sys.argv[1]=='fork':fork_worker(p,sys.argv[2])
