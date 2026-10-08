"""Task-owned, stateless explicit-noise OpenPI service. No dependency mutation."""
from __future__ import annotations
import argparse,hashlib,json,os,signal,time,threading
from pathlib import Path
import numpy as np


def digest(value):
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,dict):
            for k in sorted(v):h.update(str(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:add(x)
        elif isinstance(v,str):h.update(v.encode())
        else:
            a=np.ascontiguousarray(v);h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    add(value);return h.hexdigest()


class OwnedPolicyAdapter:
    """Task-isolated protocol around the actual, single input-transform call."""
    def __init__(self,policy,task_id,weights_fingerprint,noise_shape,inference_settings):
        self.policy=policy;self.task_id=task_id;self.weights_fingerprint=weights_fingerprint
        self.noise_shape=tuple(noise_shape);self.inference_settings=dict(inference_settings)
        self._lock=threading.Lock();self._seen=set();self._observed={};self._transform_calls=0
        original=policy._input_transform
        def audited_transform(data):
            self._transform_calls+=1
            if self._transform_calls!=1:raise RuntimeError('policy input transform called more than once')
            prepared=original(data);self._observed['sha256']=digest(prepared);return prepared
        policy._input_transform=audited_transform

    def infer(self,obs):
        with self._lock:
            obs=dict(obs);request_id=obs.pop('dual_request_id');operation=obs.pop('dual_operation','infer')
            if obs.pop('dual_task_id')!=self.task_id:raise ValueError('wrong task service')
            if not isinstance(request_id,str) or not request_id:raise ValueError('missing request identity')
            if operation=='fingerprint_input':
                if request_id in self._seen:raise ValueError('request identity already executed or attempted')
                if len(self._seen)>=10000:raise RuntimeError('task service request ledger limit')
                self._seen.add(request_id);self._observed.clear();self._transform_calls=0
                self.policy._input_transform(obs)
                return dict(dual_request_id=request_id,prepared_input_sha256=self._observed['sha256'],
                    model_fingerprint=self.weights_fingerprint,inference_settings=self.inference_settings,
                    context_protocol='stateless_explicit_noise_v1',preprocess_only=True,cache_hit=False)
            if operation!='infer':raise ValueError('unknown operation')
            noise_seed=obs.pop('dual_noise_seed')
            if type(noise_seed) is not int or not 0<=noise_seed<2**64:raise ValueError('invalid explicit noise seed')
            if request_id in self._seen:raise ValueError('request identity already executed or attempted')
            if len(self._seen)>=10000:raise RuntimeError('task service request ledger limit')
            self._seen.add(request_id);self._observed.clear();self._transform_calls=0
            noise=np.random.default_rng(noise_seed).standard_normal(self.noise_shape).astype(np.float32)
            has_rng=hasattr(self.policy,'_rng');prior_rng=getattr(self.policy,'_rng',None)
            ledger=Path(__file__).resolve().parent.parent/'service_requests.jsonl'
            def record(event):
                with ledger.open('a') as f:
                    f.write(json.dumps(dict(event=event,request_id=request_id,seed=noise_seed,weights=self.weights_fingerprint,pid=os.getpid(),at=time.time()))+'\n');f.flush();os.fsync(f.fileno())
            record('actual_execution_started')
            try:
                result=dict(self.policy.infer(obs,noise=noise));record('actual_execution_completed')
            finally:
                if has_rng:self.policy._rng=prior_rng
            if self._transform_calls!=1:raise RuntimeError('policy did not expose exactly one actual input transform')
            result.update(dual_request_id=request_id,dual_noise_seed=noise_seed,prepared_input_sha256=self._observed['sha256'],
                noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),model_fingerprint=self.weights_fingerprint,
                inference_settings=self.inference_settings,cache_hit=False)
            return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--weights',required=True);p.add_argument('--identity',required=True);p.add_argument('--port',required=True,type=int);p.add_argument('--task-id',required=True);a=p.parse_args()
    base=Path(__file__).resolve().parent.parent
    runtime_budget=json.loads((base/'evidence/runtime_budget.json').read_text())
    runtime_cfg=json.loads((base/'configs/shadow.json').read_text()) if (base/'configs/shadow.json').exists() else {}
    if runtime_cfg.get('resume_authorization_path'):
        from dual_resume_budget import remaining_seconds
        remaining=remaining_seconds(runtime_cfg)
    else:
        remaining=runtime_budget.get('max_real_wall_seconds',14400)-(time.time()-runtime_budget['real_epoch'])
    if remaining<=60:raise RuntimeError('task model wall budget exhausted')
    signal.alarm(max(1,int(remaining-60)))
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.models import pi0,pi0_config,model
    from openpi.policies import policy as policy_module,libero_policy
    from openpi import transforms
    cfg=config.get_config(a.config);weights=Path(a.weights).resolve()
    files=[]
    for f in sorted(weights.rglob('*')):
        if not f.is_file():continue
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
        files.append(dict(path=str(f.relative_to(weights)),bytes=f.stat().st_size,sha256=h.hexdigest()))
    identity=dict(name=a.config,weights_path=str(weights),weights_fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files=files,
        task_id=a.task_id,pid=os.getpid(),context_protocol='stateless_explicit_noise_v1',noise_shape=[cfg.model.action_horizon,cfg.model.action_dim],
        adapter_version=digest({str(f):f.read_bytes().hex() for f in [Path(m.__file__) for m in [policy_config,config,policy_module,libero_policy,transforms,pi0,pi0_config,model]]}),pi0_source_sha256=hashlib.sha256(Path(pi0.__file__).read_bytes()).hexdigest())
    # This operational identity is separate from frozen behavior identity, so
    # old real proposals remain reusable after lifecycle-only server repairs.
    identity['service_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    policy=policy_config.create_trained_policy(cfg,weights)
    identity['loaded']=True
    # Freeze the documented Pi0 sampling default explicitly instead of merely
    # claiming num_steps=10 in client audit metadata.
    policy._sample_kwargs=dict(policy._sample_kwargs,num_steps=10)
    identity['inference_settings']=dict(policy._sample_kwargs)
    owned=OwnedPolicyAdapter(policy,a.task_id,identity['weights_fingerprint'],identity['noise_shape'],identity['inference_settings'])
    Path(a.identity).write_text(json.dumps(identity,indent=2))
    WebsocketPolicyServer(owned,host='127.0.0.1',port=a.port,metadata=identity).serve_forever()
if __name__=='__main__':main()
