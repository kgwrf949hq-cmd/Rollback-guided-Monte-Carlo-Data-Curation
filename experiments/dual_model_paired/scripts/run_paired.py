"""Run only the bounded paired pilot; owns and cleans up its model processes."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from curation_store import atomic_json

M=json.loads((BASE/'experiment_manifest.json').read_text())
E=BASE/'evidence';E.mkdir(exist_ok=True)
IDENTITIES=[]


def owned(info):
    try:
        p=Path('/proc')/str(info['pid'])
        return (p.stat().st_uid==os.getuid() and (p/'stat').read_text().split()[21]==info['start_ticks']
            and str(BASE/'code/dual_model_server.py').encode() in (p/'cmdline').read_bytes().split(b'\0'))
    except (FileNotFoundError,ProcessLookupError):return False


def launch(role,gpu):
    settings=M['models'][role]
    with socket.socket() as s:s.bind(('127.0.0.1',settings['port']))
    output=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(output.strip())<22000:raise RuntimeError('insufficient free GPU memory')
    command=['/home/chenxiaofu/openpi/.venv/bin/python',str(BASE/'code/dual_model_server.py'),
        '--config',settings['identity']['name'],'--weights',settings['identity']['weights_path'],
        '--identity',str(E/(role+'_model_identity.json')),'--port',str(settings['port']),'--task-id',M['run_id']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
        XLA_PYTHON_CLIENT_MEM_FRACTION='.40',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    with (E/(role+'_service.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(role=role,pid=process.pid,start_ticks=(Path('/proc')/str(process.pid)/'stat').read_text().split()[21],gpu=gpu,port=settings['port'],started_at=time.time())
    IDENTITIES.append(info);atomic_json(E/(role+'_process.json'),info)


def stop():
    for info in IDENTITIES:
        if owned(info):os.kill(info['pid'],signal.SIGTERM)
    until=time.time()+20
    while time.time()<until and any(owned(i) for i in IDENTITIES):time.sleep(.25)
    for info in IDENTITIES:
        if owned(info):os.kill(info['pid'],signal.SIGKILL)
        info.update(stopped_at=time.time(),alive_after_stop=owned(info))
        atomic_json(E/(info['role']+'_process.json'),info)


def worker(args,logname):
    with (E/logname).open('x') as log:
        result=subprocess.run(['/usr/bin/python3','-u',str(BASE/'scripts/paired_worker.py')]+args,
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MUJOCO_GL='egl',PYTHONPATH='/home/chenxiaofu/openpi/packages/openpi-client/src',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'),
            timeout=max(1,int(M['deadline_epoch']-time.time()-180)))
    if result.returncode:raise RuntimeError('worker failed: '+logname)


def run():
    acceptance=json.loads(Path(M['restoration_acceptance']).read_text())
    if acceptance['status']!='PASS' or not acceptance['saver_unchanged']:
        raise RuntimeError('final restoration acceptance must PASS before models start')
    summary=dict(status='RUNNING',restoration_status='PASS',candidates=[],branches=[],
        target_candidates=M['target_candidates'],replicates=M['replicates'],fresh_collection=True)
    atomic_json(BASE/'run_status.json',summary)
    atomic_json(E/'runtime_budget.json',dict(real_epoch=time.time(),max_real_wall_seconds=int(M['deadline_epoch']-time.time())))
    try:
        subprocess.run(['/usr/bin/python3','-c','import openpi_client.msgpack_numpy,openpi_client.image_tools,h5py,mujoco,PIL; from websockets.sync.client import connect'],
            env=dict(os.environ,PYTHONPATH='/home/chenxiaofu/openpi/packages/openpi-client/src'),check=True)
        launch('main',0);launch('aux',1)
        until=time.time()+900
        while True:
            ready=[]
            for role in ['main','aux']:
                p=E/(role+'_model_identity.json')
                ready.append(p.exists())
            if all(ready):break
            if time.time()>until or any(not owned(info) for info in IDENTITIES):raise RuntimeError('model startup failed or timed out')
            time.sleep(2)
        for role in ['main','aux']:
            actual=json.loads((E/(role+'_model_identity.json')).read_text())
            for field in ['weights_fingerprint','adapter_version','name']:
                if actual[field]!=M['models'][role]['identity'][field]:raise RuntimeError('model identity mismatch: '+role+' '+field)
        for root in M['roots']:
            worker(['collect',root['root_id']],root['root_id']+'_collection.log')
            result=json.loads((BASE/'collection'/root['root_id']/'result.json').read_text())
            if result['candidate']:summary['candidates'].append(result['candidate'])
            atomic_json(BASE/'run_status.json',summary)
            if len(summary['candidates'])>=M['target_candidates']:break
        for path in summary['candidates']:
            item=json.loads(Path(path).read_text());cid=item['candidate_id']
            for rep in range(M['replicates']):
                # Alternate execution order across replicates; common Main noise schedule.
                for arm in (['control','dual'] if rep%2==0 else ['dual','control']):
                    worker(['branch',path,'--rep',str(rep),'--arm',arm],cid+'_r%d_%s.log'%(rep,arm))
                    result_path=BASE/'pairs'/cid/('rep%d_%s'%(rep,arm))/'result.json'
                    summary['branches'].append(str(result_path));atomic_json(BASE/'run_status.json',summary)
        summary['status']='COMPLETED' if summary['candidates'] else 'NO_QUALIFIED_CANDIDATES'
    except Exception as error:
        summary.update(status='ERROR',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        stop();atomic_json(BASE/'run_status.json',summary)


if __name__=='__main__':run()
