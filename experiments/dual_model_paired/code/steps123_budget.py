import fcntl,json,os,time
from pathlib import Path
from curation_store import atomic_json

def reserve(base,kind,identity,limit,group=None,group_limit=None):
 base=Path(base)
 with (base/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  path=base/'budget_counters.json';s=json.loads(path.read_text()) if path.exists() else {'counts':{},'identities':{}}
  key=kind+':'+identity
  if key in s['identities']:raise RuntimeError('budget identity already reserved:'+key)
  if s['counts'].get(kind,0)>=limit:raise RuntimeError('budget exceeded:'+kind)
  if group and s['counts'].get(group,0)>=group_limit:raise RuntimeError('budget exceeded:'+group)
  s['counts'][kind]=s['counts'].get(kind,0)+1
  if group:s['counts'][group]=s['counts'].get(group,0)+1
  s['identities'][key]=time.time();atomic_json(path,s)
  with (base/'budget_ledger.jsonl').open('a') as f:f.write(json.dumps(dict(event='reserved',kind=kind,identity=identity,group=group,at=time.time()))+'\n');f.flush();os.fsync(f.fileno())

def step(p,env,action,kind,identity,limit):
 from libero_rollback_tool import step_env
 p.guard();reserve(p.B,kind,identity,limit,'total_env_steps',3200)
 result=step_env(env,action)
 p.append('environment_ledger.jsonl',dict(event='completed',kind=kind,identity=identity))
 return result
