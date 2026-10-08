"""Routing regressions; real MuJoCo replay evidence is produced separately."""
import json
import sys
import ast
import signal
import time
import traceback
import random
from types import ModuleType,SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from curation_store import load_payload
from test_resume_collect import harness
from dual_runtime import save_diagnostic_state


@pytest.mark.parametrize('aux_value', [.1,.8])
def test_shadow_saves_rejected_and_admitted_states_once(harness,aux_value):
    h=harness
    h.cfg['mode']='shadow'
    h.aux_value=aux_value
    result=h.collect()
    samples=[load_payload(p) for p in result['calibration_samples']]
    assert len(samples)==6
    assert h.captures==[('root',t) for t in range(2,32,5)]
    for sample in samples:
        step=sample['state_step']
        row=json.loads((h.run/'branches/root/comparisons'/('%06d.json'%step)).read_text())
        assert sample['checkpoint']==row['checkpoint']
        assert sample['checkpoint']['fork_step']==step
        assert Path(sample['checkpoint']['path']).with_suffix('.npz').is_file()
    for candidate in result['candidates']:
        sample=next(s for s in samples if s['state_step']==candidate['step'])
        assert sample['checkpoint']==candidate['checkpoint']
    if aux_value==.1:
        assert result['candidates']==[]
    else:
        assert len(result['candidates'])==3
    assert len(h.environments)==1 and h.restores==[]
    assert all(d['action_source']=='main' for d in h.trajectory('root')['decisions'])


def test_diagnostic_refuses_overwrite_and_metadata_collision(harness):
    h=harness;h.cfg['mode']='shadow';h.aux_value=.1
    result=h.collect()
    sample_path=result['calibration_samples'][0]
    with pytest.raises(RuntimeError,match='overwrite'):
        save_diagnostic_state(None,None,None,sample_path,None,None,None,None,None,None,{})
    with pytest.raises(ValueError,match='metadata'):
        save_diagnostic_state(None,None,None,h.tmp/'new',None,None,None,None,None,None,{'checkpoint':{}})


@pytest.mark.parametrize('aux_value',[.1,.8])
def test_actual_probe_root_saves_all_boundaries_beyond_32_candidates(harness,monkeypatch,aux_value):
    """Run the real root_run body without its historical manifest bootstrap."""
    import dual_collect
    from curation_store import atomic_json,save_payload
    from dual_core import ActionContract,disagreement,fingerprint,require_compatible_contracts
    h=harness;h.aux_value=aux_value
    task=dict(h.task,language='synthetic placement task',initial_states={'0':{'sha256':'mock-init'}})
    root=dict(root_id='A_init0',task_label='A',init_index=0,env_seed=1)
    class Detector:
        def __init__(self):self.steps=[]
        def snapshot_state(self):return {'steps':list(self.steps)}
        def update(self,*a,**kw):
            self.steps.append(kw['step'])
            return SimpleNamespace(status=SimpleNamespace(value='normal'))
    class Adapter:
        def __init__(self,*a):pass
        def detector(self):return Detector()
        def truth(self,*a):return SimpleNamespace(task_success=False,as_dict=lambda:{'task_success':False})
    def make_env(**kwargs):
        env,obs,info=h.make_env(**kwargs);env.check_success=lambda:False
        return env,obs,info
    monkeypatch.setattr(sys.modules['libero_rollback_tool'],'make_libero_env',make_env)
    monkeypatch.setattr(sys.modules['libero_task_adapter'],'describe_task',lambda *a:task)
    monkeypatch.setattr(sys.modules['libero_task_adapter'],'TaskAdapter',Adapter)
    predicates=ModuleType('libero.libero.envs.predicates');predicates.eval_predicate_fn=lambda *a:False
    monkeypatch.setitem(sys.modules,predicates.__name__,predicates)
    module=ModuleType('probe_checkpoint_test')
    monkeypatch.setitem(sys.modules,module.__name__,module)
    manifest=dict(roots=[root],tasks={'A':task},action_contracts=h.cfg['action_contracts'],
        scales=h.cfg['scales'],d_min=h.cfg['d_min'],numeric_floor=h.cfg['numeric_floor'],
        run_id='mock_probe',not_evaluated=[])
    control=lambda *a:SimpleNamespace(authorize_worker=lambda *a:None,
        boundary=lambda *a:None,finish=lambda *a:None)
    module.__dict__.update(B=h.tmp,M=manifest,DEADLINE=time.time()+600,
        np=np,sys=sys,time=time,signal=signal,traceback=traceback,random=random,
        atomic_json=atomic_json,save_payload=save_payload,save_diagnostic_state=save_diagnostic_state,
        fingerprint=fingerprint,disagreement=disagreement,ActionContract=ActionContract,
        require_compatible_contracts=require_compatible_contracts,Control=control,
        stop_signal=lambda *a:None,guard=lambda:None,orientation=lambda *a:None,
        probe_gripper_direction=lambda *a:{'status':'PASS'},
        budget_step=lambda p,env,a,*rest:h.step(env,a),
        client=lambda role,out:dual_collect.VerifiedClient(h.cfg['models'][role],h.cfg['task_id'],
            ActionContract(**h.cfg['action_contracts'][role]),out,role),
        ask=lambda client,root,obs,rid,role:client.proposal(obs,task['language'],rid,1))
    source=Path(__file__).resolve().parents[1]/'code/probe_runner.py'
    tree=ast.parse(source.read_text())
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='root_run')
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),module.__dict__)
    module.root_run('A_init0')
    folder=h.tmp/'roots/A_init0'
    rows=[json.loads(p.read_text()) for p in sorted((folder/'comparisons').glob('*.json'))]
    assert len(rows)==104 and len(h.captures)==104
    assert all(r['checkpoint'] is not None for r in rows)
    for r in rows:
        sample=load_payload(folder/'states'/('%06d'%r['control_step']))
        assert sample['checkpoint']==r['checkpoint']
        assert sample['checkpoint']['fork_step']==r['env_step']==r['control_step']+10
    if aux_value==.1:assert not any(r['static_candidate'] for r in rows)
    else:assert sum(r['static_candidate'] for r in rows)>32
