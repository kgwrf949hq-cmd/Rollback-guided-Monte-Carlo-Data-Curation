"""Restricted four-root diagnostic. Never invokes production tree entry/gates."""
import os,sys,json,time,hashlib,signal,shutil,subprocess,fcntl,traceback,random,socket
from pathlib import Path
import numpy as np
from curation_store import atomic_json,save_payload,load_payload
from dual_core import ActionContract,DisagreementScales,disagreement,fingerprint,require_compatible_contracts
from dual_runtime import VerifiedClient,stable_seed,probe_gripper_direction
from probe_control import Control, process_identity, same_process
from steps123_budget import reserve, step as budget_step
from dual_runtime import save_diagnostic_state
B=Path(__file__).resolve().parent.parent
M=json.loads((B/'experiment_manifest.json').read_text())
import datetime
DEADLINE=datetime.datetime.fromisoformat(M['authorization']['deadline_utc'].replace('Z','+00:00')).timestamp()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def append(name,data):
 data=dict(data,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with (B/name).open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def guard():
 if time.time()>=DEADLINE-180:raise RuntimeError('deadline_finalization_reserve')
 size=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());free=shutil.disk_usage(B).free
 # 512 MiB is reserved for all local files, work and delivery copies, audited on transfer.
 if size+512*1024**2>3*1024**3:raise RuntimeError('new_artifact_limit')
 if size+512*1024**2+M['historical_cumulative_upper_bytes']>20*1024**3:raise RuntimeError('historical_cumulative_limit')
 if free<20*1024**3:raise RuntimeError('remote_disk_reserve')
 local=json.loads((B/'evidence/local_resource_snapshot.json').read_text())
 if local['free_bytes']-512*1024**2<20*1024**3:raise RuntimeError('local_disk_reserve')
 if time.time()-local['at']>1800:raise RuntimeError('local_resource_snapshot_stale')
 append('resource_ledger.jsonl',dict(remote_bytes=size,local_reserved_bytes=512*1024**2,remote_free_bytes=free,local_snapshot=local,deadline=DEADLINE))
def stop_signal(*args):raise TimeoutError('continuous_deadline_or_termination')
def seed(root,role,rid,rep=0):return stable_seed(M['master_seed'],root['task_name'],root['init_index'],role,rid,rep)
def client(role,out):return VerifiedClient(M['models'][role],M['run_id'],ActionContract(**M['action_contracts'][role]),out,role)
def ask(c,root,obs,rid,role,rep=0,seed_override=None):
 guard();s=seed(root,role,rid,rep) if seed_override is None else seed_override;
 kind='repeat_requests' if role.startswith('diagnostic_repeat') else ('child_requests' if role=='child_main' else 'shadow_requests')
 reserve(B,kind,rid,330 if kind=='repeat_requests' else 416,'total_logical_requests',1164)
 before_np=np.random.get_state();before_py=random.getstate()
 append('request_ledger.jsonl',dict(event='logical_started',root_id=root['root_id'],role=role,request_id=rid,replicate=rep,seed=s))
 try:
  p=c.proposal(obs,M['tasks'][root['task_label']]['language'],rid,s)
  after_np=np.random.get_state()
  assert before_np[0]==after_np[0] and np.array_equal(before_np[1],after_np[1]) and before_np[2:]==after_np[2:] and before_py==random.getstate(),'client RNG side effect'
  append('request_ledger.jsonl',dict(event='logical_completed',root_id=root['root_id'],request_id=rid,seed=s,proposal_sha=p.proposal_fingerprint,cache_hit=p.cache_hit))
  return p
 except BaseException as e:
  append('request_ledger.jsonl',dict(event='logical_error',root_id=root['root_id'],request_id=rid,error=repr(e),retries=0));raise

def orientation(obs,root,out):
 from pi05_libero_rollback_rollout import policy_element
 from openpi_client import image_tools
 from PIL import Image,ImageDraw
 from trajectory_store import digest_obs
 from robosuite import macros
 before=digest_obs(obs);element=policy_element(obs,M['tasks'][root['task_label']]['language'],224)
 assert before==digest_obs(obs)
 panels=[];records={}
 for rawkey,wirekey in [('agentview_image','observation/image'),('robot0_eye_in_hand_image','observation/wrist_image')]:
  raw=obs[rawkey];expected=image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw[::-1,::-1]),224,224))
  assert np.array_equal(expected,element[wirekey]),'policy_preprocessing_mismatch'
  records[rawkey]={'raw_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'input_sha256':hashlib.sha256(element[wirekey].tobytes()).hexdigest(),'video_sha256':hashlib.sha256(raw[::-1].tobytes()).hexdigest()}
  for title,a in [('raw '+rawkey,raw),('actual policy input (180 deg)',element[wirekey]),('video (vertical flip)',raw[::-1])]:
   panel=Image.new('RGB',(320,350),(22,25,30));panel.paste(Image.fromarray(a).resize((320,320)),(0,30));ImageDraw.Draw(panel).text((5,8),title,fill='white');panels.append(panel)
 canvas=Image.new('RGB',(960,700))
 for i,p in enumerate(panels):canvas.paste(p,((i%3)*320,(i//3)*350))
 canvas.save(out/'orientation.png');save_payload(out/'actual_input',element)
 report=dict(status='AUTOMATED_MAPPING_PASS_PENDING_VISUAL',convention=macros.IMAGE_CONVENTION,raw_observation_sha256=before,panels=records,policy_transform='raw[::-1,::-1] then resize_with_pad(224)',display_transform='raw[::-1]',same_saved_observation=True,physical_fresh_render_used=False)
 atomic_json(out/'orientation.json',report)
 control=Control(B)
 request=control.begin_review(root['root_id'],before,out/'orientation.png')
 report['review_request']=request
 atomic_json(out/'orientation.json',report)
 append('run_ledger.jsonl',dict(event='orientation_ready',root_id=root['root_id'],request_id=request['request_id'],path=str(out/'orientation.png')))
 control.wait(root['root_id'],guard)
 report['status']='PASS';report['visual_review']=str(control.directory(root['root_id'])/'review_response.json');atomic_json(out/'orientation.json',report)


def root_run(rid):
 from libero_rollback_tool import make_libero_env,step_env,find_sim,close_env,capture_integration_state
 from libero_task_adapter import TaskAdapter,describe_task
 from trajectory_store import Trajectory,digest_obs,readback
 from libero.libero.envs.predicates import eval_predicate_fn
 root=next(r for r in M['roots'] if r['root_id']==rid);task=M['tasks'][root['task_label']]
 out=B/'roots'/rid;out.mkdir(parents=True,exist_ok=False)
 env=writer=warm=main=aux=None;n=0;t=0;comparisons=0;failed=0;phase='initialization';start=time.time();admitted=[];seen=set();reason='horizon/task_unsuccessful';outcome='horizon'
 signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGALRM,stop_signal);signal.alarm(max(1,int(DEADLINE-time.time()-180)))
 try:
  guard();Control(B).authorize_worker(rid)
  current=describe_task('libero_10',task['task_id'],root['init_index'])
  for key in ['name','language','bddl_sha256','init_file_sha256']:assert current[key]==task[key],key
  random.seed(root['env_seed']);np.random.seed(root['env_seed'])
  env,obs,info=make_libero_env(benchmark_name='libero_10',task_id=task['task_id'],init_state_id=root['init_index'],seed=root['env_seed'],camera_width=256,camera_height=256)
  assert info['task_language']==task['language'];adapter=TaskAdapter(task,env);detector=adapter.detector();last_failure='normal'
  initial_truth=adapter.truth(env).as_dict()
  extra={}
  if root['task_label']=='B':
   extra['microwave_initially_open']=bool(eval_predicate_fn('open',adapter.core.object_states_dict['microwave_1']))
   assert extra['microwave_initially_open'],'BDDL_actual_initial_open_mismatch'
  interface=probe_gripper_direction(env);assert interface['status']=='PASS'
  require_compatible_contracts(ActionContract(**M['action_contracts']['main']),ActionContract(**M['action_contracts']['aux']))
  atomic_json(out/'initialization.json',dict(info=info,initial_truth=initial_truth,interface=interface,extra=extra))
  warm=Trajectory(out/'warmup.h5',obs)
  for t in range(1,11):
   a=np.array([0,0,0,0,0,0,-1],np.float32);obs,reward,done,x=budget_step(sys.modules[__name__],env,a,rid+'_warmup',str(t),10);warm.step(obs,a,a,t,reward,done,{'source':'warmup'},find_sim(env))
   if done or env.check_success():raise RuntimeError('terminal_during_warmup')
  warm.close();warm=None
  phase='orientation';orientation(obs,root,out)
  phase='control';initial_sha=digest_obs(obs);writer=Trajectory(out/'trajectory.h5',obs)
  main=client('main',out/'requests');aux=client('aux',out/'requests')
  calls=0
  while n<520:
   assert not env.check_success()
   _,physical=capture_integration_state(find_sim(env));physical=physical.copy();before=digest_obs(obs)
   Control(B).boundary(rid,n,guard)
   _,after_pause=capture_integration_state(find_sim(env))
   if not np.array_equal(physical,after_pause) or before!=digest_obs(obs):raise RuntimeError('state_changed_during_terminal_pause')
   state_sha=fingerprint({'observation':before,'integration':physical})
   binding=fingerprint({'name':task['name'],'language':task['language'],'bddl':task['bddl_sha256'],'initial':task['initial_states'][str(root['init_index'])]['sha256'],'obs':before,'state':state_sha})
   prefix=M['run_id']+'_'+rid+'_'+binding[:16]+'_'+str(n)
   try:
    p=ask(main,root,obs,prefix+'_main','main_control');q=ask(aux,root,obs,prefix+'_aux','aux_shadow')
   except BaseException:
    failed+=1;raise
   _,after=capture_integration_state(find_sim(env));assert np.array_equal(physical,after) and before==digest_obs(obs),'state_changed_during_comparison'
   score=disagreement(p.canonical_actions,q.canonical_actions,M['scales'],5)
   static={'replan_boundary':n%5==0,'nonterminal':not env.check_success(),'same_observation':p.raw_obs_fingerprint==q.raw_obs_fingerprint==before,'finite_actions':True,'remaining_horizon_at_least_2K':520-n>=10,'threshold':score['D']>=M['d_min'],'numeric_floor':score['D']>M['numeric_floor'],'unique_state':state_sha not in seen,'candidate_spacing':all(abs(n-old)>=10 for old in admitted)}
   eligible=all(static.values())
   # Save every completed shadow comparison with the shared full saver. A
   # later repeated diagnostic can promote a point rejected by this one draw.
   checkpoint=save_diagnostic_state(env,obs,detector,out/'states'/('%06d'%n),
    out/'full_states'/('%06d'%n),rid,0,t,M['run_id'],
    {'main':p.context_fingerprint,'aux':q.context_fingerprint},
    dict(integration=physical,state_sha256=state_sha,observation_sha256=before,
     control_step=n,env_step=t,prompt=task['language'],binding=binding,
     truth=adapter.truth(env).as_dict()))
   if eligible:
    admitted.append(n);seen.add(state_sha)
   record=dict(root_id=rid,task_label=root['task_label'],task_name=task['name'],init_index=root['init_index'],control_step=n,env_step=t,state_sha256=state_sha,observation_sha256=before,binding=binding,main=p.as_dict(),aux=q.as_dict(),D_proposed=score['D'],components=score,over_threshold=static['threshold'],static_filters=static,static_candidate=eligible,not_evaluated=M['not_evaluated'],D_executed='not_applicable',actual_K_aux='not_applicable',valid_nonroot='not_applicable',observed_child_depth='not_applicable',truth=adapter.truth(env).as_dict())
   record.update(checkpoint=checkpoint,failure_filter_pass=last_failure not in ['recoverable_failure','terminal_failure'])
   atomic_json(out/'comparisons'/('%06d.json'%n),record);comparisons+=1;calls+=1
   for k in range(min(5,520-n)):
    a=np.asarray(p.canonical_actions[k],np.float32);raw=np.asarray(p.raw_actions[k],np.float32)
    obs,reward,done,x=budget_step(sys.modules[__name__],env,a,rid+'_control',str(n),520);n+=1;t+=1
    assert all(np.isfinite(np.asarray(v)).all() for v in obs.values()),'nonfinite_observation'
    truth_obj=adapter.truth(env);last_failure=detector.update(obs,step=t,task_truth=truth_obj,done=done).status.value
    truth=truth_obj.as_dict();writer.step(obs,a,raw,t,reward,done,dict(source='main',request_id=p.request_id,chunk_offset=k,truth=truth),find_sim(env))
    if truth['task_success']:outcome='success';reason='official_env.check_success';break
    if done:outcome='task_unsuccessful';reason='environment_terminated_without_success';break
    if last_failure in ['recoverable_failure','terminal_failure']:outcome='rule_stopped';reason='existing_failure_detector';break
   atomic_json(out/'progress.json',dict(control_steps=n,comparisons=comparisons,static_candidates=len(admitted)))
   if outcome!='horizon':break
  writer.close();writer=None
  validation=readback(out/'trajectory.h5',10,initial_sha)
  atomic_json(out/'result.json',dict(root_id=rid,status='completed',outcome=outcome,termination_reason=reason,control_steps=n,env_steps=t,comparisons=comparisons,failed_comparisons=failed,static_candidates=len(admitted),final_truth=adapter.truth(env).as_dict(),readback=validation,wall_seconds=time.time()-start,aux_takeovers=0))
  Control(B).finish(rid,'COMPLETED',reason)
 except BaseException as e:
  Control(B).finish(rid,'EXPIRED' if 'deadline' in str(e) else 'FAILED',repr(e))
  atomic_json(out/'result.json',dict(root_id=rid,status='error',outcome='error',phase=phase,error=repr(e),traceback=traceback.format_exc(),control_steps=n,env_steps=t,comparisons=comparisons,failed_comparisons=failed,static_candidates=len(admitted),wall_seconds=time.time()-start,aux_takeovers=0))
  raise
 finally:
  for w in [writer,warm]:
   if w is not None:w.close()
  for c in [main,aux]:
   if c is not None:c.close()
  if env is not None:close_env(env)

def repeats():
 from steps123_run import run_repeats
 return run_repeats(sys.modules[__name__])

def identity(proc,kind):
 value=process_identity(proc.pid)
 if value is None:raise RuntimeError('child_identity_unavailable')
 return dict(value,kind=kind,started_at=time.time())
def alive(x):return same_process(x)

def schedule():
 (B/'root_slots').mkdir(exist_ok=True);(B/'logs').mkdir(exist_ok=True)
 lock=(B/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads((B/'run_status.json').read_text());owned=[]
 if (B/'evidence/scheduler_started.json').exists():raise RuntimeError('No automatic restart; audit consumed roots first. Exact in-root recovery not validated.')
 Control(B).check_deadline()
 atomic_json(B/'evidence/scheduler_started.json',dict(process_identity(),at=time.time(),manifest_sha256=sha(B/'experiment_manifest.json')))
 signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGALRM,stop_signal);signal.alarm(max(1,int(DEADLINE-time.time()-120)))
 try:
  guard()
  codefreeze={p.name:sha(p) for p in (B/'code').glob('*.py')};atomic_json(B/'evidence/execution_code_freeze.json',codefreeze)
  for role,gpu in [('main',0),('aux',1)]:
   guard();settings=M['models'][role]
   gpuinfo=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True);assert int(gpuinfo.strip())>=22000
   with socket.socket() as s:s.bind(('127.0.0.1',settings['port']))
   cmd=['/home/chenxiaofu/openpi/.venv/bin/python',str(B/'code/dual_model_server.py'),'--config',settings['identity']['name'],'--weights',settings['identity']['weights_path'],'--identity',str(B/'evidence'/(role+'_model_identity.json')),'--port',str(settings['port']),'--task-id',M['run_id']]
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.40',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
   log=(B/'logs'/(role+'_service.log')).open('x');p=subprocess.Popen(cmd,cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();x=identity(p,role);owned.append((p,x));append('process_ledger.jsonl',dict(event='started',identity=x,argv=cmd))
  state['phase']='LOADING_MODELS';atomic_json(B/'run_status.json',state)
  until=min(time.time()+900,DEADLINE-600)
  while True:
   for p,x in owned:assert p.poll() is None,'service_failed_'+x['kind']
   if all((B/'evidence'/(r+'_model_identity.json')).exists() for r in ['main','aux']):
    ready=True
    for r in ['main','aux']:
     try:
      c=client(r,B/'health_requests');c.close()
     except OSError:ready=False
    if ready:break
   if time.time()>until:raise RuntimeError('model_load_timeout')
   time.sleep(2)
  for root in M['roots']:
   guard();rid=root['root_id'];state['phase']='ROOT_'+rid
   reservation=dict(root_id=rid,at=time.time(),manifest_sha256=sha(B/'experiment_manifest.json'),consumed=True)
   with (B/'root_slots'/rid).open('x') as f:json.dump(reservation,f);f.flush();os.fsync(f.fileno())
   append('run_ledger.jsonl',dict(event='root_slot_consumed',**reservation));state['root_slots_consumed']+=1;state['roots'][rid]={'status':'started'};atomic_json(B/'run_status.json',state)
   cmd=['/usr/bin/python3',str(B/'code/probe_runner.py'),'root',rid]
   log=(B/'logs'/(rid+'.log')).open('x');p=subprocess.Popen(cmd,cwd=B,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();x=identity(p,rid);owned.append((p,x));append('process_ledger.jsonl',dict(event='started',identity=x,argv=cmd))
   while p.poll() is None:
    if time.time()>DEADLINE-180:raise RuntimeError('deadline')
    time.sleep(1)
   result_path=B/'roots'/rid/'result.json';result=json.loads(result_path.read_text()) if result_path.exists() else {'status':'error','error':'worker_no_result'}
   state['roots'][rid]=result;atomic_json(B/'run_status.json',state);append('run_ledger.jsonl',dict(event='root_ended',root_id=rid,result=result))
   if p.returncode or result.get('status')!='completed':
    raise RuntimeError('root_failed_scheduler_halted_without_consuming_next_slot:'+rid)
  state['phase']='REPEAT_DIAGNOSTICS';atomic_json(B/'run_status.json',state);state['repeat_requests']=repeats();state['phase']='FORK_VALIDATION';atomic_json(B/'run_status.json',state)
  from steps123_run import run_forks
  state['fork_results']=run_forks(sys.modules[__name__],owned);state['phase']='COMPLETED'
 except BaseException as e:
  state['phase']='STOPPED_PARTIAL';state['stop_reason']=repr(e);state['traceback']=traceback.format_exc();append('run_ledger.jsonl',dict(event='scheduler_stopped',error=repr(e)))
 finally:
  for p,x in reversed(owned):
   if alive(x):
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
     if alive(x):os.killpg(p.pid,signal.SIGKILL)
     p.wait(timeout=10)
   append('process_ledger.jsonl',dict(event='stopped',identity=x,alive=alive(x),returncode=p.poll()))
  state['owned_processes_stopped']=all(not alive(x) for p,x in owned);state['ended_at']=time.time();atomic_json(B/'run_status.json',state);lock.close()
if __name__=='__main__':
 if sys.argv[1]=='schedule':schedule()
 elif sys.argv[1]=='root':root_run(sys.argv[2])
