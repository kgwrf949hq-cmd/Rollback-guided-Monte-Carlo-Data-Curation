import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from steps123_logic import candidate_subset,controls,pair_stats,select_fork,distance
from steps123_budget import reserve

def row(n,candidate=True,d=1.,pred=False):
 return dict(control_step=n,static_candidate=candidate,over_threshold=candidate,D_proposed=d,static_filters={'threshold':candidate,'candidate_spacing':True,'nonterminal':True},truth={'predicates':[pred]})

def test_selection_time_sampling_and_controls():
 rows=[row(n,n%10==0,d=100-n,pred=n>=30) for n in range(0,61,5)]
 assert [x['control_step'] for x in candidate_subset(rows,'B')]==[0,10,30,40,60]
 assert [x['control_step'] for x in controls(rows)]==[5,55]
 assert [x['control_step'] for x in controls([row(n,False) for n in range(9)])]==[2,5]

def test_noise_complete_and_correlated_pair_counts():
 props={r:[{'canonical_actions':np.full((5,7),v).tolist()} for _ in range(5)] for r,v in [('main',0.),('aux',.5)]}
 stats=pair_stats(props,.1)
 assert stats['noise_separated'] is True
 assert [stats['summary'][x]['pairs'] for x in ['main_within','aux_within','cross_model']]==[10,10,25]
 props['aux'].pop();assert pair_stats(props,.1)['noise_separated']=='UNDETERMINED'

def test_selection_requires_recovery_and_fixed_original_rank():
 rows=[row(20,d=.9),row(10,d=.9),row(30,d=2)]
 chosen,excluded=select_fork(rows,{n:{'noise_separated':True} for n in [10,20,30]},lambda r:r['control_step']!=30)
 assert chosen['control_step']==10 and excluded[0]['control_step']==30
 assert select_fork(rows,{},lambda r:True)[0] is None

def test_atomic_budget_no_identity_or_slot_reuse(tmp_path):
 reserve(tmp_path,'child_attempts','B0',1)
 with pytest.raises(RuntimeError):reserve(tmp_path,'child_attempts','B0',1)
 with pytest.raises(RuntimeError):reserve(tmp_path,'child_attempts','B1',1)
 assert json.loads((tmp_path/'budget_counters.json').read_text())['counts']=={'child_attempts':1}

def test_comparable_prefix_zero_and_clipped_commands():
 with pytest.raises(ValueError):distance([],[],0)
 assert distance(np.ones((5,7)),np.ones((5,7)))==0

@pytest.mark.parametrize('terminal_at,expected_k,expected_main',[(10,5,5),(2,2,0)])
def test_actual_fixed_entry_replay_cached_aux_handoff(monkeypatch,tmp_path,terminal_at,expected_k,expected_main):
 import h5py
 import libero_rollback_tool as tool
 import libero_task_adapter as adapter_mod
 import dual_runtime as runtime
 from curation_store import atomic_json
 from trajectory_store import Trajectory,digest_obs
 from steps123_run import fork_worker
 class Env:
  def __init__(self):self.n=0;self.pos=0.;self.data=SimpleNamespace(qpos=np.array([0.]),qvel=np.array([0.]))
  def obs(self):return {'position':np.array([self.pos],np.float64)}
 env=Env()
 def step(e,a):
  e.n+=1;e.pos+=float(a[0]);e.data.qpos[:]=e.pos
  return e.obs(),0.,e.n>=terminal_at,{}
 class Truth:
  def __init__(self,e):self.predicates=[e.n>=terminal_at];self.task_success=e.n>=terminal_at
  def as_dict(self):return dict(predicates=self.predicates,task_success=self.task_success)
 class Detector:
  def update(self,*a,**k):return SimpleNamespace(status=SimpleNamespace(value='normal'))
 class Adapter:
  def __init__(self,t,e):self.e=e
  def detector(self):return Detector()
  def truth(self,e):return Truth(e)
 def restore(e,*args):
  e.n=0;e.pos=0.;e.data.qpos[:]=0
  return e.obs(),{'cross_process':True,'success':True}
 monkeypatch.setattr(tool,'make_libero_env',lambda **k:(env,env.obs(),{'task_language':'test'}))
 monkeypatch.setattr(tool,'step_env',step);monkeypatch.setattr(tool,'find_sim',lambda e:e);monkeypatch.setattr(tool,'close_env',lambda e:None)
 monkeypatch.setattr(adapter_mod,'TaskAdapter',Adapter);monkeypatch.setattr(runtime,'restore_candidate',restore);monkeypatch.setattr(runtime,'probe_gripper_direction',lambda e:{'status':'PASS'})
 monkeypatch.setattr('signal.signal',lambda *a:None);monkeypatch.setattr('signal.alarm',lambda *a:None)
 parent=tmp_path/'roots/B_init0';parent.mkdir(parents=True)
 writer=Trajectory(parent/'trajectory.h5',env.obs())
 for i in range(terminal_at):
  obs,r,d,_=step(env,np.zeros(7,np.float32));writer.step(obs,np.zeros(7),np.zeros(7),11+i,r,d,{'truth':Truth(env).as_dict()},env)
 writer.close();restore(env)
 prop=lambda v:{'canonical_actions':np.full((5,7),v).tolist(),'raw_actions':np.full((5,7),v).tolist(),'request_id':'cached_'+str(v),'inference_settings':{'noise_seed':99}}
 rr={'root_id':'B_init0','task_label':'B','directory':str(parent),'env_seed':1,'task_name':'test','init_index':0}
 r={'control_step':0,'env_step':10,'checkpoint':{},'observation_sha256':digest_obs(env.obs()),'main':prop(0.),'aux':prop(1.)}
 atomic_json(parent/'comparisons/000005.json',dict(control_step=5,main=prop(0.)))
 import hashlib,time
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 atomic_json(tmp_path/'fork_selection.json',{'B_init0':{'status':'SELECTED','root':rr,'row':r,'parent_trajectory_sha256':sha(parent/'trajectory.h5')}})
 calls=[]
 def ask(c,root,obs,rid,role,seed_override=None):
  calls.append((float(obs['position'][0]),seed_override,role));return SimpleNamespace(canonical_actions=np.zeros((5,7)),raw_actions=np.zeros((5,7)),request_id=rid)
 p=SimpleNamespace(B=tmp_path,M={'run_id':'fake','tasks':{'B':{'task_id':9,'language':'test'}},'d_min':.1,'numeric_floor':1e-6},DEADLINE=time.time()+600,guard=lambda:None,sha=sha,stop_signal=lambda *a:None,append=lambda *a:None,client=lambda *a:SimpleNamespace(close=lambda:None),ask=ask)
 fork_worker(p,'B_init0')
 result=json.loads((tmp_path/'forks/B_init0/result.json').read_text())
 assert result['actual_K']==expected_k and result['main_steps']==expected_main and result['effective_difference']
 with h5py.File(tmp_path/'forks/B_init0/trajectory.h5','r') as h:assert np.array_equal(h['executed_actions'][:expected_k],np.ones((expected_k,7)))
 if expected_main:assert calls==[(5.,99,'child_main')]
 else:assert calls==[]
