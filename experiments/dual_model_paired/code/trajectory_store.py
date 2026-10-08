"""Lossless HDF5 trajectories; RGB uses XOR with the preceding observation."""
import hashlib,json,os
from pathlib import Path
import h5py
import numpy as np

def digest_obs(obs):
    h=hashlib.sha256()
    for k in sorted(obs):
        a=np.ascontiguousarray(obs[k]);h.update(k.encode());h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

class Trajectory:
    def __init__(self,path,obs):
        self.path=Path(path);self.h=h5py.File(path,'w');self.prev={};self.n=0
        self.h.attrs['schema']='lossless_observation_sequence_v1';self.h.attrs['image_encoding']='first frame raw, subsequent uint8 XOR previous decoded frame'
        self.obs=self.h.create_group('observations')
        for k,v in obs.items():
            a=np.asarray(v)
            ds=self.obs.create_dataset(k,shape=(0,)+a.shape,maxshape=(None,)+a.shape,dtype=a.dtype,chunks=(1,)+a.shape,compression='gzip',compression_opts=4)
            ds.attrs['xor_previous']=bool('image' in k and a.dtype==np.uint8)
        self.hashes=self.h.create_dataset('observation_sha256',shape=(0,),maxshape=(None,),dtype='S64')
        for k in ['executed_actions','policy_actions']:
            self.h.create_dataset(k,shape=(0,7),maxshape=(None,7),dtype=np.float32,chunks=True,compression='gzip')
        for k,dtype in [('env_steps','i8'),('rewards','f8'),('done','?')]:self.h.create_dataset(k,shape=(0,),maxshape=(None,),dtype=dtype,chunks=True)
        self.h.create_dataset('decisions',shape=(0,),maxshape=(None,),dtype=h5py.string_dtype('utf-8'))
        self.add_observation(obs)

    def add_observation(self,obs):
        i=len(self.hashes)
        for k,ds in self.obs.items():
            a=np.asarray(obs[k]);stored=np.bitwise_xor(a,self.prev[k]) if ds.attrs['xor_previous'] and k in self.prev else a
            ds.resize(i+1,axis=0);ds[i]=stored;self.prev[k]=a.copy()
        self.hashes.resize(i+1,axis=0);self.hashes[i]=digest_obs(obs).encode()

    def step(self,obs,action,policy_action,env_step,reward,done,decision,sim):
        values={'executed_actions':action,'policy_actions':policy_action,'env_steps':env_step,'rewards':reward,'done':done,'decisions':json.dumps(decision,allow_nan=False)}
        for k,a in [('qpos',sim.data.qpos),('qvel',sim.data.qvel)]:
            if k not in self.h:self.h.create_dataset(k,shape=(0,len(a)),maxshape=(None,len(a)),dtype='f8',chunks=True,compression='gzip')
            values[k]=a
        for k,v in values.items():ds=self.h[k];ds.resize(self.n+1,axis=0);ds[self.n]=v
        self.add_observation(obs);self.n+=1;self.h.flush()

    def close(self):
        self.h.flush();self.h.close()
        with self.path.open('rb') as f:os.fsync(f.fileno())

def readback(path,fork_step,initial_sha=None):
    with h5py.File(path,'r') as h:
        n=len(h['executed_actions']);assert n>0
        for key in ['executed_actions','policy_actions','env_steps','rewards','done','decisions','qpos','qvel']:assert len(h[key])==n,key
        assert h['executed_actions'].shape==(n,7) and h['policy_actions'].shape==(n,7)
        for key in ['executed_actions','policy_actions','rewards','qpos','qvel']:assert np.all(np.isfinite(h[key][:])),key
        for key,ds in h['observations'].items():assert len(ds)==n+1,key
        assert len(h['observation_sha256'])==n+1
        assert np.array_equal(h['env_steps'][:],np.arange(fork_step+1,fork_step+n+1))
        assert np.all(np.isfinite(h['executed_actions'][:])) and np.all(np.abs(h['executed_actions'][:])<=1)
        prev={}
        for i in range(n+1):
            obs={k:np.bitwise_xor(ds[i],prev[k]) if ds.attrs['xor_previous'] and i else ds[i] for k,ds in h['observations'].items()}
            sha=digest_obs(obs);assert sha==h['observation_sha256'][i].decode()
            if i==0 and initial_sha is not None:assert sha==initial_sha
            prev=obs
        return {'readback_passed':True,'control_steps':n,'observations':n+1,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'lossless_images':True}
