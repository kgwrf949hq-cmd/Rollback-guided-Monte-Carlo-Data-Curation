"""Verified model transport and exact snapshot hooks, shared by run modes."""
from __future__ import annotations
import copy,hashlib,inspect,json,os,re,time
from pathlib import Path
import numpy as np
from curation_store import atomic_json,save_payload,load_payload
from trajectory_store import digest_obs
from dual_core import Proposal,canonicalize_actions,fingerprint


def stable_seed(*values):return int.from_bytes(hashlib.sha256(':'.join(map(str,values)).encode()).digest()[:8],'little')


def probe_gripper_direction(env):
    """Verify deployed Panda mapping on deep copies without stepping any env.

    Panda XML has two slide-joint position actuators with ranges [0,.04] and
    [-.04,0]; opening is sum(abs(joint target)). This check is deliberately
    specific to that verified geometry and fails closed for other grippers.
    The live format_action is never called because it mutates current_action.
    """
    from libero_rollback_tool import find_sim,capture_integration_state
    owner=getattr(env,'env',env);robots=getattr(owner,'robots',[])
    if len(robots)!=1:raise RuntimeError('gripper verification requires exactly one robot')
    robot=robots[0];gripper=robot.gripper;sim=find_sim(env)
    if type(gripper).__name__!='PandaGripper':raise RuntimeError('unsupported gripper direction geometry')
    indexes=[sim.model.actuator_name2id(a) for a in gripper.actuators]
    ranges=np.asarray(sim.model.actuator_ctrlrange[indexes],dtype=np.float64)
    if ranges.shape!=(2,2) or not np.allclose(ranges,[[0.,.04],[-.04,0.]],rtol=0,atol=1e-12):
        raise RuntimeError('Panda actuator ranges differ from verified geometry')
    original=np.asarray(gripper.current_action).copy();ctrl=np.asarray(sim.data.ctrl).copy()
    _,before=capture_integration_state(sim);before=np.asarray(before).copy()
    bias=ranges.mean(axis=1);weight=(ranges[:,1]-ranges[:,0])/2
    targets={};formatted={}
    for name,sign in [('negative_open',-1),('positive_close',1)]:
        isolated=copy.deepcopy(gripper)
        isolated.current_action=np.zeros_like(np.asarray(isolated.current_action),dtype=np.float64)
        value=np.asarray(isolated.format_action(np.array([sign],dtype=np.float64)))
        if value.shape!=(2,) or not np.isfinite(value).all():raise RuntimeError('invalid gripper mapping')
        formatted[name]=value.tolist();targets[name]=(bias+weight*value).tolist()
    _,after=capture_integration_state(sim)
    unchanged=bool(np.array_equal(original,gripper.current_action) and np.array_equal(ctrl,sim.data.ctrl) and np.array_equal(before,after))
    widths={name:float(np.abs(value).sum()) for name,value in targets.items()}
    neutral=float(np.abs(bias).sum())
    valid=unchanged and widths['positive_close']<neutral<widths['negative_open']
    return dict(status='PASS' if valid else 'FAIL',method='deepcopy of actual PandaGripper; actual actuator ranges and robot normalized-control mapping; no simulator step',
        gripper_class=type(gripper).__module__+'.'+type(gripper).__name__,
        format_action_sha256=hashlib.sha256(inspect.getsource(type(gripper).format_action).encode()).hexdigest(),
        robot_grip_action_sha256=hashlib.sha256(inspect.getsource(type(robot).grip_action).encode()).hexdigest(),
        actuator_names=list(gripper.actuators),actuator_ctrlrange=ranges.tolist(),formatted=formatted,
        joint_position_targets=targets,aperture_targets_m=widths,neutral_aperture_m=neutral,
        aperture_definition='sum(abs(two Panda slide-joint targets)); exact deployed XML ranges checked',
        positive_command='close',negative_command='open',original_state_unchanged=unchanged,
        simulation_steps=0,scope='control interface mapping; not proof of object contact or successful grasp')


class VerifiedClient:
    def __init__(self,settings,task_id,contract,logdir,role):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        self.codec=msgpack_numpy;self.settings=settings;self.task_id=task_id;self.contract=contract;self.role=role;self.logdir=Path(logdir);self.logdir.mkdir(parents=True,exist_ok=True)
        contract.validate()
        self.ws=connect('ws://127.0.0.1:%d'%settings['port'],compression=None,max_size=None,open_timeout=15,close_timeout=5)
        try:
            self.identity=self.codec.unpackb(self.ws.recv(timeout=120))
            for k in ['name','weights_fingerprint','adapter_version']:
                if self.identity.get(k)!=settings['identity'][k]:raise RuntimeError('wrong model service: '+k)
            if (self.identity.get('task_id')!=task_id or self.identity.get('context_protocol')!='stateless_explicit_noise_v1'
                or self.identity.get('loaded') is not True):raise RuntimeError('unverified service isolation or model load')
            shape=self.identity.get('noise_shape')
            if not isinstance(shape,(list,tuple)) or len(shape)!=2 or any(type(x) is not int or x<=0 for x in shape) or shape[0]<5 or shape[1]<7:
                raise RuntimeError('unverified model noise shape')
            if self.identity.get('inference_settings')!={'num_steps':10}:raise RuntimeError('unverified inference settings')
        except BaseException:
            self.close();raise
    def fingerprint_input(self,obs,instruction,request_id):
        from pi05_libero_rollback_rollout import policy_element
        if not isinstance(request_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',request_id):raise ValueError('unsafe request identity')
        raw_sha=digest_obs(obs);element=policy_element(obs,instruction,224);wire=fingerprint(element)
        if digest_obs(obs)!=raw_sha:raise RuntimeError('preprocessing mutated raw observation')
        p=self.logdir/(request_id+'.json')
        if p.exists():raise RuntimeError('request identity reuse')
        audit=dict(request_id=request_id,role=self.role,status='started',raw_observation_sha256=raw_sha,
            wire_input_sha256=wire,context_fingerprint=fingerprint({'protocol':'stateless_explicit_noise_v1','task_id':self.task_id,'role':self.role}),
            history_semantics='stateless_no_history',instruction=instruction,preprocess_only=True,started_at=time.time())
        atomic_json(p,audit)
        try:
            self.ws.send(self.codec.Packer().pack(dict(element,dual_task_id=self.task_id,
                dual_request_id=request_id,dual_operation='fingerprint_input')))
            reply=self.ws.recv(timeout=120)
            if isinstance(reply,str):raise RuntimeError('model server preprocessing error')
            reply=self.codec.unpackb(reply)
            if (reply.get('dual_request_id')!=request_id or reply.get('preprocess_only') is not True
                or reply.get('cache_hit') is not False or reply.get('model_fingerprint')!=self.identity['weights_fingerprint']
                or reply.get('inference_settings')!=self.identity['inference_settings']
                or reply.get('context_protocol')!='stateless_explicit_noise_v1'
                or not re.fullmatch(r'[0-9a-f]{64}',reply.get('prepared_input_sha256',''))):raise RuntimeError('preprocessing identity mismatch')
            if digest_obs(obs)!=raw_sha or fingerprint(element)!=wire:raise RuntimeError('preprocessing mutated raw input')
            audit.update(reply,status='completed',ended_at=time.time());atomic_json(p,audit);return audit
        except BaseException as e:
            audit.update(status='error',error=repr(e),ended_at=time.time());atomic_json(p,audit);raise
    def proposal(self,obs,instruction,request_id,seed):
        from pi05_libero_rollback_rollout import policy_element
        if not isinstance(request_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}',request_id):raise ValueError('unsafe request identity')
        if type(seed) is not int or not 0<=seed<2**64:raise ValueError('invalid explicit noise seed')
        raw_sha=digest_obs(obs);element=policy_element(obs,instruction,224)
        if digest_obs(obs)!=raw_sha:raise RuntimeError('preprocessing mutated raw observation')
        before=fingerprint(element);start=time.time()
        audit=dict(request_id=request_id,role=self.role,seed=seed,started_at=start,status='started',raw_observation_sha256=raw_sha,wire_input_sha256=before,cache_hit=False)
        p=self.logdir/(request_id+'.json')
        if p.exists():raise RuntimeError('request identity reuse')
        atomic_json(p,audit)
        try:
            packet=dict(element,dual_task_id=self.task_id,dual_request_id=request_id,dual_noise_seed=seed)
            self.ws.send(self.codec.Packer().pack(packet));reply=self.ws.recv(timeout=120)
            if isinstance(reply,str):raise RuntimeError('model server error')
            reply=self.codec.unpackb(reply)
            if reply['dual_request_id']!=request_id or reply['dual_noise_seed']!=seed or reply['model_fingerprint']!=self.identity['weights_fingerprint'] or reply.get('cache_hit') is not False:raise RuntimeError('model response identity mismatch')
            expected_noise=np.random.default_rng(seed).standard_normal(tuple(self.identity['noise_shape'])).astype(np.float32)
            if reply.get('noise_sha256')!=hashlib.sha256(expected_noise.tobytes()).hexdigest():raise RuntimeError('model explicit noise hash mismatch')
            if reply.get('inference_settings')!=self.identity['inference_settings']:raise RuntimeError('model inference settings mismatch')
            if not isinstance(reply.get('prepared_input_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',reply['prepared_input_sha256']):raise RuntimeError('invalid server preprocessing fingerprint')
            raw=np.asarray(reply['actions'],dtype=np.float32);canonical=canonicalize_actions(raw,self.contract)
            settings=dict(reply['inference_settings'],noise_seed=seed,noise_sha256=reply['noise_sha256'],noise_shape=self.identity['noise_shape'])
            prop=Proposal.create(raw,canonical,self.settings['identity'],self.identity['adapter_version'],settings,raw_sha,reply['prepared_input_sha256'],fingerprint({'protocol':'stateless_explicit_noise_v1','task_id':self.task_id,'role':self.role}),request_id,fingerprint(self.contract.as_dict()))
            if digest_obs(obs)!=raw_sha or fingerprint(element)!=before:raise RuntimeError('model mutated input state')
            audit.update(status='completed',ended_at=time.time(),proposal=prop.as_dict());atomic_json(p,audit);return prop
        except BaseException as e:
            audit.update(status='error',ended_at=time.time(),error=repr(e),server_may_have_executed=True);atomic_json(p,audit);raise
    def close(self):self.ws.close()


def capture_candidate(env,obs,detector,path,branch_id,depth,step,task_id,model_context):
    from libero_rollback_tool import EnvRollbackTool
    from dual_observation import fresh_reference,validate_reference
    cp=EnvRollbackTool().capture(env,step=step,obs=obs,metadata=dict(task_id=task_id,owner_branch_id=branch_id,owner_depth=depth,checkpoint_id=path.name,capture_pid=os.getpid()),rollout_state=dict(failure_detector_state=detector.snapshot_state(),model_context=model_context,pending_actions=[]))
    if cp.integration_state is None or cp.runtime_state is None or cp.rollout_state is None:raise RuntimeError('incomplete snapshot')
    # Policy-visible observable history and a fresh current-state render are
    # distinct references. The cloned-data render changes neither live state.
    cp.metadata['fresh_render_reference']=fresh_reference(env,obs,step)
    validate_reference(cp)
    if path.with_suffix('.json').exists():raise RuntimeError('snapshot overwrite forbidden')
    save_payload(path,{k:v for k,v in cp.__dict__.items() if k!='sim_state'})
    return dict(path=str(path),fork_step=step,owner_branch_id=branch_id,owner_depth=depth,observation_sha256=digest_obs(obs),json_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),npz_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest())


def save_diagnostic_state(env,obs,detector,sample_path,checkpoint_path,
                          branch_id,depth,step,task_id,model_context,sample_fields):
    """Index a diagnostic observation with the existing full rollback saver.

    Diagnostics may later become fork points, including points rejected by a
    single noisy proposal. Capture before stepping, independently of admission.
    This is routing only: capture_candidate/EnvRollbackTool own the snapshot
    schema, serialization and restore checks. Existing observation readers keep
    their old fields, and candidates reuse the very same checkpoint reference.
    """
    if Path(sample_path).with_suffix('.json').exists():
        raise RuntimeError('diagnostic state overwrite forbidden')
    if {'obs','checkpoint','checkpoint_schema'} & set(sample_fields):
        raise ValueError('diagnostic metadata cannot replace checkpoint fields')
    checkpoint=capture_candidate(env,obs,detector,Path(checkpoint_path),
        branch_id,depth,step,task_id,model_context)
    save_payload(sample_path,dict(sample_fields,obs=obs,checkpoint=checkpoint,
        checkpoint_schema='EnvRollbackTool.Checkpoint'))
    return checkpoint


def restore_candidate(env,detector,checkpoint,task_id):
    from libero_rollback_tool import EnvRollbackTool,Checkpoint,get_observation,find_sim,capture_integration_state
    from robosuite import macros
    from dual_observation import fresh_reference,validate_reference,compare_reference
    if macros.IMAGE_CONVENTION!='opengl':raise RuntimeError('fresh render validation requires verified opengl observation convention')
    p=Path(checkpoint['path'])
    for ext,key in [('.json','json_sha256'),('.npz','npz_sha256')]:
        if hashlib.sha256(p.with_suffix(ext).read_bytes()).hexdigest()!=checkpoint[key]:raise RuntimeError('checkpoint checksum changed')
    d=load_payload(p);d['sim_state']=None;cp=Checkpoint(**d)
    if cp.metadata['task_id']!=task_id or cp.metadata['owner_branch_id']!=checkpoint['owner_branch_id'] or cp.step!=checkpoint['fork_step']:raise RuntimeError('checkpoint identity changed')
    if cp.env_done:raise RuntimeError('terminal checkpoint cannot be restored as a fork')
    reference=validate_reference(cp)
    report=EnvRollbackTool().restore(env,cp,reload_xml=False)
    if not report['success']:raise RuntimeError('snapshot restore failed')
    # Verify live integration and independently forward/render cloned live data.
    # Never compare an instantaneous render to historical sensor samples.
    sim=find_sim(env);_,integration=capture_integration_state(sim)
    report['fresh_state_equal']=bool(np.array_equal(integration,cp.integration_state))
    if not report['fresh_state_equal']:raise RuntimeError('fresh simulator/camera mismatch: integration state')
    obs=get_observation(env,force_update=False)
    current=fresh_reference(env,obs,cp.step)
    images=compare_reference(reference,current)
    report['fresh_render']=images;report['image_convention']=macros.IMAGE_CONVENTION
    report['fresh_reference_protocol']=current['protocol']
    report['fresh_read_state_unchanged']=current['read_state_unchanged']
    report['policy_observation_source']='restored_observable_history; not fresh render'
    report['state_step']=cp.step;report['sim_time']=current['sim_time']
    if not report['fresh_state_equal'] or not all(x['equal'] for x in images.values()):raise RuntimeError('fresh simulator/camera mismatch: '+json.dumps(report))
    # Renderer cannot advance physical transitions; preserve observable timing from snapshot.
    obs=get_observation(env,force_update=False)
    if digest_obs(obs)!=checkpoint['observation_sha256']:raise RuntimeError('restored observation differs')
    detector.restore_state(cp.rollout_state['failure_detector_state'])
    if detector.snapshot_state()!=cp.rollout_state['failure_detector_state']:raise RuntimeError('detector restore mismatch')
    report.update(capture_pid=cp.metadata['capture_pid'],restore_pid=os.getpid(),cross_process=cp.metadata['capture_pid']!=os.getpid(),success=True)
    return obs,report
