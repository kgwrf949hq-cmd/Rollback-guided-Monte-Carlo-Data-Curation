"""Durable control for an existing live worker; never resets or starts a root."""
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

TERMINAL = {'COMPLETED', 'FAILED', 'EXPIRED'}
WAITING = {'WAITING_REVIEW', 'REVIEW_READY', 'PAUSED'}

class ControlError(RuntimeError):
    pass

class ReviewRejected(ControlError):
    pass

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def process_identity(pid=None):
    """Linux PID reuse protection: boot ID, uid and /proc start ticks."""
    pid = os.getpid() if pid is None else int(pid)
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19], 'uid': proc.stat().st_uid,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (OSError, IndexError):
        return None

def same_process(saved, reader=process_identity):
    if not saved or 'pid' not in saved or 'start_ticks' not in saved:
        return False
    live = reader(saved['pid'])
    return bool(live and all(live.get(k) == saved[k]
                           for k in ('pid', 'start_ticks', 'uid', 'boot_id') if k in saved))

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary.exists():
            temporary.unlink()

class Control:
    def __init__(self, base, now=time.time, reader=process_identity):
        self.base = Path(base).resolve()
        self.now, self.reader = now, reader
        self.manifest_path = self.base / 'experiment_manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_sha = digest(self.manifest_path)
        self.deadline = datetime.datetime.fromisoformat(
            self.manifest['authorization']['deadline_utc'].replace('Z', '+00:00')).timestamp()
        self.roots = {r['root_id']: r for r in self.manifest['roots']}

    def directory(self, rid):
        if rid not in self.roots or not re.fullmatch(r'[A-Za-z0-9_.-]+', rid):
            raise ControlError('unknown_root')
        return self.base / 'control' / 'roots' / rid

    @contextlib.contextmanager
    def locked(self, rid, create=False):
        directory = self.directory(rid)
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        elif not (directory / 'state.json').exists():
            raise ControlError('no_live_control_record_no_restart')
        with (directory / 'control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield directory

    def read(self, rid):
        return json.loads((self.directory(rid) / 'state.json').read_text())

    def save(self, rid, state, event=None):
        state['updated_at'] = self.now()
        atomic(self.directory(rid) / 'state.json', state)
        if event:
            with (self.directory(rid) / 'events.jsonl').open('a') as f:
                f.write(json.dumps({'event': event, 'at': self.now(), 'phase': state['phase'],
                                    'binding_id': state['binding_id'], 'pause_id': state.get('pause_id')}) + '\n')
                f.flush()
                os.fsync(f.fileno())

    def check_deadline(self):
        if digest(self.manifest_path) != self.manifest_sha:
            raise ControlError('manifest_changed')
        if self.now() >= self.deadline - 180:
            raise ControlError('deadline_finalization_reserve')

    def check_live(self, state):
        self.check_deadline()
        if state['manifest_sha256'] != self.manifest_sha:
            raise ControlError('manifest_binding_mismatch')
        if state['phase'] in TERMINAL:
            raise ControlError('root_is_terminal_no_restart')
        if not same_process(state['owner'], self.reader):
            raise ControlError('worker_not_alive_exact_restore_not_validated')
        if not same_process(state['supervisor'], self.reader):
            raise ControlError('supervisor_not_alive_no_automatic_restart')

    def supervisor(self):
        s = json.loads((self.base / 'evidence/scheduler_started.json').read_text())
        if s['manifest_sha256'] != self.manifest_sha or not same_process(s, self.reader):
            raise ControlError('scheduler_identity_mismatch')
        return s

    def authorize_worker(self, rid):
        self.check_deadline()
        self.directory(rid)
        slot = json.loads((self.base / 'root_slots' / rid).read_text())
        if slot.get('root_id') != rid or slot.get('manifest_sha256') != self.manifest_sha or slot.get('consumed') is not True:
            raise ControlError('root_slot_binding_mismatch')
        return self.supervisor()

    def begin_review(self, rid, observation_sha256, image_path, control_step=0):
        supervisor = self.authorize_worker(rid)
        owner = self.reader(os.getpid())
        if not owner:
            raise ControlError('cannot_identify_worker')
        expected_path = self.base / 'roots' / rid / 'orientation.png'
        if Path(image_path).resolve() != expected_path:
            raise ControlError('unexpected_review_image_path')
        with self.locked(rid, create=True) as d:
            if (d / 'state.json').exists():
                raise ControlError('review_already_registered_no_reentry')
            root = self.roots[rid]
            binding = {'run_id': self.manifest['run_id'], 'manifest_sha256': self.manifest_sha,
                       'root_id': rid, 'task_name': root['task_name'], 'init_index': root['init_index'],
                       'request_id': uuid.uuid4().hex, 'observation_sha256': observation_sha256,
                       'image_sha256': digest(image_path), 'owner': owner, 'supervisor': supervisor}
            binding_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            state = dict(binding, binding_id=binding_id, phase='WAITING_REVIEW',
                         pause_id=binding['request_id'], control_step=control_step,
                         image_path=str(image_path), deadline_utc=self.manifest['authorization']['deadline_utc'])
            atomic(d / 'review_request.json', dict(binding, binding_id=binding_id))
            self.save(rid, state, 'review_requested')
            return state

    def review(self, rid, request_id, image_sha256, decision, note):
        if decision not in ('PASS', 'REJECT') or not note.strip():
            raise ControlError('explicit_decision_and_review_note_required')
        with self.locked(rid) as d:
            s = self.read(rid)
            self.check_live(s)
            if s['phase'] != 'WAITING_REVIEW' or request_id != s['request_id']:
                raise ControlError('not_the_current_review_request')
            if image_sha256 != s['image_sha256'] or digest(s['image_path']) != image_sha256:
                raise ControlError('review_image_hash_mismatch')
            path = d / 'review_response.json'
            if path.exists():
                raise ControlError('review_already_submitted')
            response = {'binding_id': s['binding_id'], 'request_id': request_id,
                        'root_id': rid, 'image_sha256': image_sha256, 'manifest_sha256': self.manifest_sha,
                        'decision': decision, 'note': note, 'at': self.now()}
            atomic(path, response)
            s['phase'] = 'REVIEW_READY'
            self.save(rid, s, 'review_submitted')
            return response

    def valid_review(self, rid, state):
        path = self.directory(rid) / 'review_response.json'
        if not path.exists():
            return None
        response = json.loads(path.read_text())
        for key in ('binding_id', 'request_id', 'root_id', 'image_sha256', 'manifest_sha256'):
            if response.get(key) != state[key]:
                raise ControlError('review_response_binding_mismatch:' + key)
        if digest(state['image_path']) != state['image_sha256']:
            raise ControlError('review_image_changed')
        if response.get('decision') not in ('PASS', 'REJECT'):
            raise ControlError('invalid_review_decision')
        return response

    def continue_root(self, rid, pause_id):
        with self.locked(rid) as d:
            s = self.read(rid)
            self.check_live(s)
            if s['phase'] not in WAITING or s['pause_id'] != pause_id:
                raise ControlError('not_the_current_pause')
            if s['phase'] in ('WAITING_REVIEW', 'REVIEW_READY'):
                response = self.valid_review(rid, s)
                if not response or response['decision'] != 'PASS':
                    raise ControlError('image_review_required_before_continue')
            path = d / ('continue_' + pause_id + '.json')
            command = {'binding_id': s['binding_id'], 'pause_id': pause_id,
                       'owner': s['owner'], 'requested_at': self.now()}
            if path.exists():
                old = json.loads(path.read_text())
                if any(old[k] != command[k] for k in ('binding_id', 'pause_id', 'owner')):
                    raise ControlError('conflicting_continue')
                return dict(old, idempotent=True)
            atomic(path, command)
            self.save(rid, s, 'continue_requested')
            return command

    def request_pause(self, rid):
        with self.locked(rid) as d:
            s = self.read(rid)
            self.check_live(s)
            if s['phase'] != 'RUNNING':
                raise ControlError('worker_is_not_running')
            path = d / 'pause_request.json'
            if path.exists():
                return json.loads(path.read_text())
            request = {'binding_id': s['binding_id'], 'pause_id': uuid.uuid4().hex,
                       'owner': s['owner'], 'requested_at': self.now()}
            atomic(path, request)
            self.save(rid, s, 'pause_requested')
            return request

    def poll_release(self, rid, guard):
        with self.locked(rid) as d:
            s = self.read(rid)
            self.check_live(s)
            if s['phase'] not in WAITING:
                raise ControlError('worker_is_not_waiting')
            if s['phase'] in ('WAITING_REVIEW', 'REVIEW_READY'):
                response = self.valid_review(rid, s)
                if response and response['decision'] == 'REJECT':
                    raise ReviewRejected('orientation_review_rejected')
                if not response:
                    return False
            path = d / ('continue_' + s['pause_id'] + '.json')
            if not path.exists():
                return False
            command = json.loads(path.read_text())
            if any(command.get(k) != s[k] for k in ('binding_id', 'pause_id', 'owner')):
                raise ControlError('continue_binding_mismatch')
            try:
                guard()
            except RuntimeError as e:
                if str(e) != 'local_resource_snapshot_stale':
                    raise
                if s.get('blocked_reason') != str(e):
                    s['blocked_reason'] = str(e)
                    self.save(rid, s, 'continue_waiting_for_resources')
                return False
            s['phase'] = 'RUNNING'
            s.pop('blocked_reason', None)
            self.save(rid, s, 'continue_acknowledged')
            return True

    def wait(self, rid, guard, sleep=time.sleep):
        # No per-review 600-second expiry. Absolute deadline and owner identity still apply.
        while not self.poll_release(rid, guard):
            sleep(1)
        return self.read(rid)

    def boundary(self, rid, step, guard):
        with self.locked(rid) as d:
            s = self.read(rid)
            self.check_live(s)
            if s['phase'] != 'RUNNING':
                raise ControlError('invalid_running_phase')
            reason = None
            try:
                guard()
            except RuntimeError as e:
                if str(e) != 'local_resource_snapshot_stale':
                    raise
                reason = str(e)
            path = d / 'pause_request.json'
            request = json.loads(path.read_text()) if path.exists() else None
            if request and (request.get('binding_id') != s['binding_id'] or request.get('owner') != s['owner']):
                raise ControlError('pause_binding_mismatch')
            s['control_step'] = step
            if request or reason:
                s.update(phase='PAUSED', pause_id=request['pause_id'] if request else uuid.uuid4().hex,
                         pause_reason=reason or 'terminal_pause')
                self.save(rid, s, 'paused_at_replan_boundary')
                if request:
                    os.replace(path, d / ('consumed_pause_' + request['pause_id'] + '.json'))
            else:
                self.save(rid, s)
                return
        self.wait(rid, guard)

    def finish(self, rid, phase, reason=None):
        if phase not in TERMINAL:
            raise ValueError('terminal phase required')
        if not (self.directory(rid) / 'state.json').exists():
            return
        with self.locked(rid):
            s = self.read(rid)
            s.update(phase=phase, terminal_reason=reason)
            self.save(rid, s, 'root_terminal')

    def status(self):
        roots = {}
        for rid in self.roots:
            path = self.directory(rid) / 'state.json'
            if path.exists():
                s = json.loads(path.read_text())
                roots[rid] = dict(s, owner_alive=same_process(s['owner'], self.reader),
                                 supervisor_alive=same_process(s['supervisor'], self.reader))
            else:
                roots[rid] = {'phase': 'NO_LIVE_CONTINUATION_RECORD',
                              'slot_consumed': (self.base / 'root_slots' / rid).exists(),
                              'legacy_or_not_started': True}
        return {'run_id': self.manifest['run_id'], 'manifest_sha256': self.manifest_sha,
                'deadline_utc': self.manifest['authorization']['deadline_utc'],
                'deadline_expired': self.now() >= self.deadline - 180, 'roots': roots,
                'run_status': json.loads((self.base / 'run_status.json').read_text())
                    if (self.base / 'run_status.json').exists() else None,
                'terminal_session': json.loads((self.base / 'evidence/terminal_session.json').read_text())
                    if (self.base / 'evidence/terminal_session.json').exists() else None,
                'resume_semantics': 'continue same live worker only; never reset/restart root'}
