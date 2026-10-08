"""Independent readback of saved starts, action roles, seeds, hashes and videos."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'code'))
from curation_store import atomic_json,load_payload
from verify_state_replay import compare

r=json.loads((BASE/'paired_results.json').read_text())
checks=[]
for candidate in r['candidates']:
    cp=candidate['checkpoint']
    for ext,key in [('.json','json_sha256'),('.npz','npz_sha256')]:
        assert hashlib.sha256(Path(cp['path']).with_suffix(ext).read_bytes()).hexdigest()==cp[key]
    for rep in range(2):
        rows={b['arm']:b for b in candidate['branches'] if b['replicate']==rep}
        directories={a:Path(v['trajectory']).parent for a,v in rows.items()}
        states={a:load_payload(d/'initial_state')['state'] for a,d in directories.items()}
        assert not compare(states['control'],states['dual'])
        seeds={}
        for arm,directory in directories.items():
            result=rows[arm]
            assert result['checkpoint']==cp
            restored=json.loads((directory/'restore.json').read_text())
            assert restored['success'] and restored['cross_process']
            with h5py.File(directory/'trajectory.h5') as h:
                n=len(h['executed_actions']);assert n==result['steps']
                assert len(h['observation_sha256'])==n+1
                decisions=[json.loads(x) for x in h['decisions'].asstr()[:]]
                roles=[x['action_source'] for x in decisions]
                expected=['main']*n if arm=='control' else ['aux']*min(5,n)+['main']*max(0,n-5)
                assert roles==expected
                seeds[arm]=[x['seed'] for x in decisions]
                assert hashlib.sha256((directory/'trajectory.h5').read_bytes()).hexdigest()==result['readback']['sha256']
                with np.load(directory/'key_states.npz') as k:
                    assert len(k['qpos'])==n+1 and np.array_equal(k['qpos'][1:],h['qpos'][:])
                    assert np.array_equal(k['qpos'][0],states[arm]['simulator']['qpos'])
                    assert np.array_equal(k['site_xpos'][0],states[arm]['simulator']['site_xpos'])
            meta=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(directory/'trajectory.mp4')]))['streams'][0]
            assert meta['width']==640 and meta['height']==360 and meta['r_frame_rate']=='20/1' and int(meta['nb_frames'])==n+1
            assert json.loads((directory/'video.json').read_text())['frames']==n+1
            checks.append(dict(candidate_id=candidate['candidate_id'],replicate=rep,arm=arm,steps=n,video_frames=n+1,exact_start=True,actual_action_sources_pass=True,checkpoint_hashes_pass=True,trajectory_hash_pass=True,video_pass=True))
        common=min(len(v) for v in seeds.values())
        assert seeds['control'][5:common]==seeds['dual'][5:common]
assert len(checks)==8 and not r['original_source_files_changed']
for role in ['main','aux']:
    info=json.loads((BASE/'evidence'/(role+'_process.json')).read_text())
    assert info['alive_after_stop'] is False
out=dict(status='PASS',branches=len(checks),pairs=4,checks=checks,
    same_checkpoint_verified=True,serialized_initial_states_exact=True,
    main_seeds_matched_after_handoff=True,original_source_unchanged=True,owned_services_stopped=True)
atomic_json(BASE/'evidence/artifact_verification.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
