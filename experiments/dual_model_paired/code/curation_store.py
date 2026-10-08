"""Small atomic JSON + NPZ store. No pickle; rollback is a lineage edge."""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_payload(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def encode(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError('Object arrays are not a portable trajectory format')
            key = 'array_%06d' % len(arrays)
            arrays[key] = item
            return {'__npz__': key}
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, enum.Enum):
            return item.value
        if dataclasses.is_dataclass(item):
            return {f.name: encode(getattr(item, f.name)) for f in dataclasses.fields(item)}
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, tuple):
            return {'__tuple__': [encode(v) for v in item]}
        if isinstance(item, list):
            return [encode(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            return {'__nonfinite__': repr(item)}
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise TypeError('Unsupported payload type: ' + type(item).__name__)

    tree = encode(value)
    array_path = path.with_suffix('.npz')
    temporary = array_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, array_path)
    atomic_json(path.with_suffix('.json'), {
        'arrays': array_path.name,
        'arrays_sha256': hashlib.sha256(array_path.read_bytes()).hexdigest(),
        'payload': tree,
    })


def load_payload(path):
    path = Path(path).with_suffix('.json')
    document = json.loads(path.read_text())
    array_path = path.parent / document['arrays']
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != document['arrays_sha256']:
        raise ValueError('Payload checksum mismatch: ' + str(array_path))
    with np.load(array_path, allow_pickle=False) as arrays:
        def decode(item):
            if isinstance(item, dict):
                if set(item) == {'__tuple__'}:
                    return tuple(decode(v) for v in item['__tuple__'])
                if set(item) == {'__npz__'}:
                    return arrays[item['__npz__']].copy()
                if set(item) == {'__nonfinite__'}:
                    return float(item['__nonfinite__'])
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return [decode(v) for v in item]
            return item
        return decode(document['payload'])


def same_observation(left, right):
    return (isinstance(left, dict) and isinstance(right, dict) and
            left.keys() == right.keys() and
            all(np.array_equal(np.asarray(left[k]), np.asarray(right[k])) for k in left))


class CurationStore:
    def __init__(self, run_dir):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=False)
        self.saved_checkpoints = set()

    def step_data(self, branch, data):
        # This is written immediately after env.step, before feature processing.
        save_payload(self.path / 'branches' / branch.branch_id /
                     ('step_%06d' % data['branch_step']), data)

    def transition(self, branch, transition):
        atomic_json(self.path / 'branches' / branch.branch_id /
                    ('decision_%06d.json' % transition.branch_step), transition.to_summary())

    def branch_end(self, branch, checkpoints):
        directory = self.path / 'branches' / branch.branch_id
        save_payload(directory / 'initial', branch.initial_observation)
        for key, record in checkpoints.items():
            if key in self.saved_checkpoints:
                continue
            cp = record.checkpoint
            save_payload(self.path / 'checkpoints' / key, {
                'summary': record.to_summary(),
                'observation': getattr(cp, 'obs', None),
                'integration_state': getattr(cp, 'integration_state', None),
                'model_xml_sha256': getattr(cp, 'model_xml_sha256', None),
                'event_detector_state': record.event_detector_state,
                'failure_detector_state': record.failure_detector_state,
            })
            save_payload(self.path / 'full_checkpoints' / key, {name: value for name, value in cp.__dict__.items() if name != 'sim_state'})
            self.saved_checkpoints.add(key)
        atomic_json(directory / 'branch.json', branch.to_summary())
        print(json.dumps({'branch_saved': branch.branch_id, 'status': branch.status.value,
                          'steps': len(branch.transitions), 'fork_step': branch.fork_step}), flush=True)


def check_run(run_dir):
    """Rebuild every path from checkpoint owners, never from rollback jumps."""
    directory = Path(run_dir)
    branches = {p.parent.name: json.loads(p.read_text())
                for p in (directory / 'branches').glob('*/branch.json')}
    checkpoints = {p.stem: load_payload(p) for p in (directory / 'checkpoints').glob('*.json')}
    paths = {}
    visiting = set()

    def reconstruct(key):
        if key in paths:
            return paths[key]
        if key in visiting:
            raise ValueError('Cyclic lineage')
        visiting.add(key)
        branch = branches[key]
        initial = load_payload(directory / 'branches' / key / 'initial')
        prefix = []
        fork = branch['fork_checkpoint_id']
        if fork:
            assert branch['parent_branch_id'] in branches
            checkpoint = checkpoints[fork]
            owner = checkpoint['summary']['owner_branch_id']
            assert checkpoint['summary']['step'] == branch['fork_step']
            ancestor = branch['parent_branch_id']
            ancestry = []
            while ancestor is not None:
                ancestry.append(ancestor)
                ancestor = branches[ancestor]['parent_branch_id']
            assert owner in ancestry, 'Fork belongs to a sibling'
            prefix = [step for step in reconstruct(owner) if step['env_step'] <= branch['fork_step']]
            if branch['status'] != 'restore_failed':
                assert branch['restore_report']['success'] is True
                assert same_observation(initial, checkpoint['observation']), 'Fork observation differs'
                if prefix:
                    assert same_observation(prefix[-1]['next_observation'], initial)
        suffix = []
        previous = initial
        for i in range(1, branch['transition_count'] + 1):
            step = load_payload(directory / 'branches' / key / ('step_%06d' % i))
            assert step['branch_step'] == i
            assert step['env_step'] == branch['fork_step'] + i, 'Rollback encoded as transition'
            assert same_observation(previous, step['observation']), 'State/action alignment mismatch'
            decision = json.loads((directory / 'branches' / key / ('decision_%06d.json' % i)).read_text())
            assert np.array_equal(step['executed_action'], decision['executed_action'])
            assert step['reward'] == decision['reward'] and step['done'] == decision['done']
            previous = step['next_observation']
            suffix.append(step)
        if branch['status'] == 'success':
            assert branch['task_success_source'] == 'env.check_success'
        paths[key] = prefix + suffix
        visiting.remove(key)
        return paths[key]

    for key in branches:
        reconstruct(key)
    result = {'readback_passed': True, 'branch_count': len(branches),
              'path_lengths': {key: len(value) for key, value in paths.items()},
              'rollback_is_lineage_edge': True,
              'unprocessed_step_files': {key: len(list((directory / 'branches' / key).glob('step_*.json')))
                                         - branch['transition_count'] for key, branch in branches.items()}}
    atomic_json(directory / 'readback.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    print(json.dumps(check_run(parser.parse_args().run_dir), indent=2))
