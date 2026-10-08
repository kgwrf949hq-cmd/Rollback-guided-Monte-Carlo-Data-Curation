"""Validated continuation authorization, inherited accounting and durable run slots.

All continuation state is anchored beside the frozen authorization, never beside
the selected run or task id. Call reserve_run before the first real attempt;
retrying the same identity/configuration returns the already consumed slot.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

AUTHORIZATION_ID = 'calibration_resume_4h_v2'
MAX_ARTIFACT_BYTES = 20 * 1024**3
MIN_FREE_BYTES = 20 * 1024**3
STAGES = {'shadow': 'calibration_shadow', 'calibration_shadow': 'calibration_shadow',
          'calibration_shadow_extension_v1': 'calibration_shadow',
          'smoke': 'smoke_tree', 'smoke_tree': 'smoke_tree',
          'acceptance': 'acceptance_tree', 'acceptance_tree': 'acceptance_tree'}


class ResumeBudgetError(RuntimeError):
    pass


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ResumeBudgetError('invalid numeric authorization field: ' + field)
    return float(value)


def _same(left, right):
    return abs(left - right) <= 1e-5


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def read_authorization(cfg, *, _ancestors=()):
    """Return a verified record; no implicit extension for legacy configurations."""
    path = cfg.get('resume_authorization_path')
    if not path:
        return None
    path = Path(path).resolve()
    expected = cfg.get('resume_authorization_sha256')
    if str(path) in _ancestors or len(_ancestors) >= 8:
        raise ResumeBudgetError('cyclic or excessive authorization chain')
    raw = path.read_bytes()
    if not _sha(expected) or hashlib.sha256(raw).hexdigest() != expected:
        raise ResumeBudgetError('continuation authorization SHA256 mismatch')
    record = json.loads(raw)
    identity = record.get('authorization_id', '')
    is_sample_review = isinstance(identity, str) and re.fullmatch(r'sample_review_[0-9]{8}T[0-9]{6}Z', identity)
    if identity != AUTHORIZATION_ID and not is_sample_review:
        raise ResumeBudgetError('unknown continuation authorization')
    if record.get('record_status') != 'AUTHORIZED_HISTORY_INHERITED_NO_RESET':
        raise ResumeBudgetError('authorization does not retain original history')
    original = _finite(record['original_real_epoch'], 'original_real_epoch')
    authorized = _finite(record['T_auth_epoch'], 'T_auth_epoch')
    deadline = _finite(record['new_deadline_epoch'], 'new_deadline_epoch')
    maximum = _finite(record['max_real_wall_seconds'], 'max_real_wall_seconds')
    added = _finite(record['authorized_additional_seconds'], 'authorized_additional_seconds')
    previous_deadline = _finite(record['original_deadline'], 'original_deadline')
    if not (0 < original <= authorized < deadline and added == 14400
            and _same(deadline, authorized + added) and _same(maximum, deadline - original)):
        raise ResumeBudgetError('authorization deadline or inherited epoch is inconsistent')
    for name, value in [('T_auth', authorized), ('new_deadline', deadline)]:
        stamp = datetime.fromisoformat(record[name].replace('Z', '+00:00'))
        if stamp.tzinfo is None or not _same(stamp.timestamp(), value):
            raise ResumeBudgetError('authorization timestamp mismatch: ' + name)
    history = record['cumulative_resource_ledger']
    old_time = history['time']
    if not (_same(_finite(old_time['original_epoch'], 'history.original_epoch'), original)
            and _same(_finite(old_time['deadline_epoch'], 'history.deadline_epoch'), previous_deadline)
            and _same(previous_deadline, original + _finite(old_time['maximum_seconds'], 'history.maximum_seconds'))):
        raise ResumeBudgetError('historical epoch/deadline changed')
    storage = history['storage']
    if storage['maximum_bytes'] != MAX_ARTIFACT_BYTES or storage['minimum_free_bytes'] != MIN_FREE_BYTES:
        raise ResumeBudgetError('historical storage limits changed')
    _finite(storage['combined_conservative_upper_bytes'], 'historical storage')
    if storage['combined_conservative_upper_bytes'] < 0:
        raise ResumeBudgetError('negative historical storage')
    limits = history['frozen_limits']
    required = dict(inference_timeout_seconds=120, extra_same_transient_network_retries_max=2,
                    simulation_concurrency=1, max_artifacts_bytes=MAX_ARTIFACT_BYTES)
    if any(limits.get(k) != v for k, v in required.items()):
        raise ResumeBudgetError('historical resource limits changed')
    counters = history['new_run_counters']
    if any(counters.get(stage) not in (0, 1) for stage in set(STAGES.values())):
        raise ResumeBudgetError('invalid inherited run counters')
    if is_sample_review:
        # The new authorization adds time, while the hash-bound original bytes
        # retain the previous window, cumulative charges, failures and run slots.
        previous_path = record.get('previous_authorization_path')
        previous_sha = record.get('previous_authorization_sha256')
        if not previous_path or not _sha(previous_sha):
            raise ResumeBudgetError('sample review requires hash-bound previous authorization')
        previous = read_authorization(dict(resume_authorization_path=previous_path,
            resume_authorization_sha256=previous_sha), _ancestors=(*_ancestors, str(path)))
        if (record.get('previous_authorization_id') != previous['authorization_id']
                or original != previous['original_real_epoch']
                or authorized < previous['T_auth_epoch']
                or record.get('previous_deadline') != previous['new_deadline']):
            raise ResumeBudgetError('previous authorization identity or time changed')
        old = previous['cumulative_resource_ledger']
        if (history['storage']['combined_conservative_upper_bytes'] < old['storage']['combined_conservative_upper_bytes']
                or any(counters[k] < old['new_run_counters'][k] for k in set(STAGES.values()))
                or not set(old['time'].get('prior_failed_attempts_retained', [])).issubset(
                    set(old_time.get('prior_failed_attempts_retained', [])))):
            raise ResumeBudgetError('historical charges, failures or run slots were reset')
        prior_ledger_path = record.get('previous_resource_ledger_path')
        if prior_ledger_path:
            prior_raw = Path(prior_ledger_path).read_bytes()
            if hashlib.sha256(prior_raw).hexdigest() != record.get('previous_resource_ledger_sha256'):
                raise ResumeBudgetError('previous resource ledger SHA256 mismatch')
            prior_ledger = json.loads(prior_raw)
            _validate_ledger(dict(resume_authorization_sha256=previous_sha), previous, prior_ledger)
            if (history['storage']['combined_conservative_upper_bytes'] < prior_ledger['combined_conservative_upper_bytes']
                    or any(counters[k] < prior_ledger['new_run_counters'][k] for k in set(STAGES.values()))):
                raise ResumeBudgetError('previous persisted resource charges or run slots were reset')
            if record.get('inherited_runs', {}) != prior_ledger['runs']:
                raise ResumeBudgetError('inherited consumed run identities differ from previous ledger')
        elif record.get('inherited_runs'):
            raise ResumeBudgetError('inherited run identities require their previous resource ledger')
    return record


def _ledger_path(cfg):
    return Path(cfg['resume_authorization_path']).resolve().parent / 'evidence' / 'resume_resource_ledger.json'


def _validate_ledger(cfg, record, ledger):
    if (ledger.get('authorization_sha256') != cfg['resume_authorization_sha256']
            or ledger.get('authorization_id') != record['authorization_id']
            or ledger.get('original_real_epoch') != record['original_real_epoch']
            or ledger.get('new_deadline_epoch') != record['new_deadline_epoch']
            or ledger.get('historical_resource_ledger') != record['cumulative_resource_ledger']):
        raise ResumeBudgetError('persisted continuation ledger identity changed')
    for field in ['new_local_high_water_bytes', 'new_remote_high_water_bytes']:
        if type(ledger.get(field)) is not int or ledger[field] < 0:
            raise ResumeBudgetError('persisted storage accounting is invalid')
    for stage in set(STAGES.values()):
        count = ledger['new_run_counters'].get(stage)
        inherited = record['cumulative_resource_ledger']['new_run_counters'][stage]
        if type(count) is not int or count not in (0, 1) or count < inherited or (stage in ledger['runs'] and count != 1):
            raise ResumeBudgetError('persisted run counter is invalid')


def read_resource_ledger(cfg):
    """Read-only inspection for preflight; absence means no new slots consumed."""
    record = read_authorization(cfg)
    if record is None:
        raise ResumeBudgetError('continuation authorization is required')
    path = _ledger_path(cfg)
    if not path.exists():
        return None
    ledger = json.loads(path.read_text())
    _validate_ledger(cfg, record, ledger)
    return ledger


@contextmanager
def _ledger(cfg):
    record = read_authorization(cfg)
    if record is None:
        raise ResumeBudgetError('continuation authorization is required')
    path = _ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            ledger = json.loads(path.read_text())
            _validate_ledger(cfg, record, ledger)
        else:
            history = copy.deepcopy(record['cumulative_resource_ledger'])
            ledger = dict(schema=1, authorization_id=record['authorization_id'],
                          authorization_sha256=cfg['resume_authorization_sha256'],
                          original_real_epoch=record['original_real_epoch'],
                          new_deadline_epoch=record['new_deadline_epoch'],
                          max_real_wall_seconds=record['max_real_wall_seconds'],
                          historical_resource_ledger=history,
                          new_run_counters=copy.deepcopy(history['new_run_counters']),
                          runs=copy.deepcopy(record.get('inherited_runs', {})), admissions=[], new_local_high_water_bytes=0,
                          new_remote_high_water_bytes=0,
                          combined_conservative_upper_bytes=history['storage']['combined_conservative_upper_bytes'])
        yield record, ledger
        ledger.update(updated_at=time.time(), budget_reset=False)
        _atomic(path, ledger)


def ensure_runtime_budget(cfg):
    """Initialize continuation runtime evidence without changing its original epoch."""
    with _ledger(cfg) as (record, ledger):
        path = Path(cfg['task_directory']) / 'evidence' / 'runtime_budget.json'
        budget = json.loads(path.read_text()) if path.exists() else {}
        if budget and budget.get('real_epoch') != record['original_real_epoch']:
            raise ResumeBudgetError('runtime real_epoch differs from original epoch')
        budget.update(real_epoch=record['original_real_epoch'],
                      max_real_wall_seconds=record['max_real_wall_seconds'],
                      deadline_epoch=record['new_deadline_epoch'],
                      authorization_id=record['authorization_id'],
                      authorization_sha256=cfg['resume_authorization_sha256'],
                      accounting='continuous time since original real_epoch including history, failures and idle time',
                      historical_resource_ledger=copy.deepcopy(ledger['historical_resource_ledger']),
                      budget_reset=False)
        _atomic(path, budget)
    return budget


def remaining_seconds(cfg, *, now=None):
    record = read_authorization(cfg)
    if record is None:
        raise ResumeBudgetError('continuation authorization is required')
    path = Path(cfg['task_directory']) / 'evidence' / 'runtime_budget.json'
    if path.exists():
        budget = json.loads(path.read_text())
        if budget.get('real_epoch') != record['original_real_epoch']:
            raise ResumeBudgetError('runtime real_epoch differs from original epoch')
    return max(0., record['new_deadline_epoch'] - (time.time() if now is None else now))


def record_is_sample_review(cfg):
    record = read_authorization(cfg)
    return bool(record and record['authorization_id'].startswith('sample_review_'))


def check_stage_admission(cfg, kind, estimated_seconds, finalization_seconds=600, *, now=None):
    """Require enough time for the complete estimated phase and final shutdown."""
    estimate = _finite(estimated_seconds, 'stage estimate')
    reserve = _finite(finalization_seconds, 'finalization reserve')
    minimum_reserve = 600 if record_is_sample_review(cfg) else 60
    if estimate <= 0 or reserve < minimum_reserve:
        raise ResumeBudgetError('positive stage estimate and at least %ss finalization reserve required' % minimum_reserve)
    current = time.time() if now is None else now
    remaining = remaining_seconds(cfg, now=current)
    report = dict(kind=kind, checked_at=current, estimated_seconds=estimate,
                  finalization_reserve_seconds=reserve, remaining_seconds=remaining,
                  required_seconds=estimate + reserve,
                  estimate_basis=cfg.get('stage_estimate_basis', {}),
                  status='PASS' if remaining >= estimate + reserve else 'TIME_BUDGET_INSUFFICIENT_FOR_NEXT_STAGE')
    with _ledger(cfg) as (_, ledger):
        ledger['admissions'].append(report)
    if report['status'] != 'PASS':
        raise ResumeBudgetError(report['status'])
    return report


def reserve_run(cfg, kind, run_id, config_sha256):
    """Consume one durable slot, or resume its same immutable run and config."""
    stage = STAGES.get(kind)
    if stage is None or not isinstance(run_id, str) or not run_id or not _sha(config_sha256):
        raise ResumeBudgetError('valid authorized stage, stable run identity and config SHA256 required')
    if remaining_seconds(cfg) <= 0:
        raise ResumeBudgetError('task real wall budget exhausted')
    with _ledger(cfg) as (_, ledger):
        existing = ledger['runs'].get(stage)
        if existing:
            if existing['run_id'] != run_id or existing['config_sha256'] != config_sha256:
                raise ResumeBudgetError('run slot already consumed by another run or configuration: ' + stage)
            result = dict(existing, resumed=True)
        else:
            if ledger['new_run_counters'][stage] >= 1:
                raise ResumeBudgetError('inherited run slot already consumed: ' + stage)
            existing = dict(stage=stage, run_id=run_id, config_sha256=config_sha256,
                            reserved_at=time.time(), count=1)
            ledger['runs'][stage] = existing
            ledger['new_run_counters'][stage] += 1
            result = dict(existing, resumed=False)
    return result


def record_resources(cfg, *, new_local_bytes, new_remote_bytes, local_free_bytes,
                     remote_free_bytes, simulation_concurrency=0):
    """Conservatively add current-execution high-water storage to inherited usage."""
    supplied = dict(new_local_bytes=new_local_bytes, new_remote_bytes=new_remote_bytes,
                    local_free_bytes=local_free_bytes, remote_free_bytes=remote_free_bytes)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in supplied.values()):
        raise ResumeBudgetError('resource byte counts must be nonnegative integers')
    if isinstance(simulation_concurrency, bool) or not isinstance(simulation_concurrency, int) or simulation_concurrency < 0:
        raise ResumeBudgetError('simulation concurrency must be a nonnegative integer')
    with _ledger(cfg) as (_, ledger):
        ledger['new_local_high_water_bytes'] = max(ledger['new_local_high_water_bytes'], new_local_bytes)
        ledger['new_remote_high_water_bytes'] = max(ledger['new_remote_high_water_bytes'], new_remote_bytes)
        combined = (ledger['historical_resource_ledger']['storage']['combined_conservative_upper_bytes']
                    + ledger['new_local_high_water_bytes'] + ledger['new_remote_high_water_bytes'])
        reasons = []
        if combined > MAX_ARTIFACT_BYTES:
            reasons.append('ARTIFACT_BUDGET_EXHAUSTED')
        if min(local_free_bytes, remote_free_bytes) < MIN_FREE_BYTES:
            reasons.append('INSUFFICIENT_FREE_SPACE')
        if simulation_concurrency > 1:
            reasons.append('SIMULATION_CONCURRENCY_EXCEEDED')
        result = dict(supplied, combined_conservative_upper_bytes=combined,
                      maximum_bytes=MAX_ARTIFACT_BYTES, minimum_free_bytes=MIN_FREE_BYTES,
                      simulation_concurrency=simulation_concurrency,
                      status='PASS' if not reasons else 'BLOCKED', reasons=reasons, checked_at=time.time())
        ledger['combined_conservative_upper_bytes'] = combined
        ledger['latest_resource_check'] = result
    if reasons:
        raise ResumeBudgetError(','.join(reasons))
    return result
