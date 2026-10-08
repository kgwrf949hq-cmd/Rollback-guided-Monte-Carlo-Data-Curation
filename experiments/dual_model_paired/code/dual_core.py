"""Pure, fail-closed primitives for disagreement-guided tree collection.

This module does not import a model client, simulator, or legacy sampler. Action
scales are in *verified environment command coordinates*, not raw model units.
The caller must verify controller, frame, frequency and gripper conventions from
the actual deployed adapters/controller before setting ActionContract.verified.
Synthetic calibration can exercise the implementation but cannot unlock active.
"""
from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
import math
from typing import Mapping

import numpy as np

K = 5
FORMULA = '(RMS(delta_translation)/translation_scale + RMS(delta_rotation)/rotation_scale + mean(abs(delta_gripper))/gripper_scale)/3'


def _plain(value):
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'data': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, 'as_dict'):
        return _plain(value.as_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def fingerprint(value):
    """Stable content identity; NaN and non-serializable state fail closed."""
    payload = json.dumps(_plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _weight_identity(identity):
    for key in ('weights_fingerprint', 'weight_fingerprint', 'weights_sha256', 'weight_sha256'):
        if identity.get(key):
            return str(identity[key])
    raise ValueError('model identity requires a verified weight fingerprint')


def validate_independent_models(main_identity, aux_identity):
    for identity in (main_identity, aux_identity):
        if not isinstance(identity, Mapping) or not identity.get('name'):
            raise ValueError('model identity requires a name and weight fingerprint')
    if _weight_identity(main_identity) == _weight_identity(aux_identity):
        raise ValueError('auxiliary weights must differ from main weights; seed/noise is not independence')
    return True


def validate_config(config, main_identity=None, aux_identity=None):
    """Validate supported combinations without touching aux in legacy mode."""
    cfg = dict(config)
    cfg.setdefault('fork_strategy', 'legacy')
    cfg.setdefault('branch_action', 'legacy_sampler')
    cfg.setdefault('mode', 'shadow')
    cfg.setdefault('intervention_steps', K)
    cfg.setdefault('allow_legacy_fallback', False)
    cfg.setdefault('target_nonroot_branches', 16)
    cfg.setdefault('max_depth', 3)
    cfg.setdefault('max_attempts', 32)
    cfg.setdefault('candidate_spacing', 2 * K)
    cfg.setdefault('suffix_cooldown', 2 * K)
    cfg.setdefault('max_candidates_per_parent', 32)
    if (cfg['fork_strategy'], cfg['branch_action']) not in {
        ('legacy', 'legacy_sampler'), ('dual_disagreement', 'aux_short_then_main')
    }:
        raise ValueError('unsupported fork_strategy/branch_action combination')
    if cfg['mode'] not in ('shadow', 'active'):
        raise ValueError('mode must be shadow or active')
    if cfg['fork_strategy'] == 'legacy':
        return cfg
    if cfg['intervention_steps'] != K:
        raise ValueError('this configuration version freezes K=5; changing K requires new calibration/version')
    if cfg['allow_legacy_fallback'] is not False:
        raise ValueError('dual_disagreement never fills candidates from legacy')
    for key, low, high in (('target_nonroot_branches', 1, 16), ('max_depth', 1, 3),
                           ('max_attempts', 1, 32), ('max_candidates_per_parent', 1, 32)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError('%s outside supported task budget' % key)
    for key in ('candidate_spacing', 'suffix_cooldown'):
        if type(cfg[key]) is not int or cfg[key] < 2 * K:
            raise ValueError('%s must be at least 2K' % key)
    if main_identity is not None or aux_identity is not None:
        validate_independent_models(main_identity, aux_identity)
    elif cfg['mode'] == 'active':
        raise ValueError('active requires verified main and auxiliary identities')
    return cfg


@dataclass(frozen=True)
class ActionContract:
    """Explicit adapter-to-environment affine mapping followed by clipping.

    Rotation is a common-frame OSC rotation-vector delta, never target Euler
    angles. Default unit scales below describe command units only and carry no
    claim about deployed metric or angular ranges. Evidence must cover these.
    """
    verified: bool = False
    controller: str = 'OSC_POSE'
    semantics: str = 'delta'
    coordinate_frame: str = 'world'
    rotation_representation: str = 'rotation_vector'
    units: str = 'normalized_environment_command'
    control_frequency_hz: float = 20.0
    gripper_positive: str = 'close'
    action_scale: tuple = (1., 1., 1., 1., 1., 1., 1.)
    action_offset: tuple = (0., 0., 0., 0., 0., 0., 0.)
    gripper_sign: int = 1
    evidence: tuple = ()

    def validate(self):
        if not self.verified or not self.evidence:
            raise ValueError('unverified action contract: actual adapter/controller evidence required')
        if self.controller != 'OSC_POSE' or self.semantics != 'delta' or self.rotation_representation != 'rotation_vector':
            raise ValueError('only verified 7D OSC delta/rotation-vector commands are supported')
        if not self.coordinate_frame or self.units != 'normalized_environment_command':
            raise ValueError('common environment coordinate frame/units must be specified')
        if not math.isfinite(self.control_frequency_hz) or self.control_frequency_hz <= 0:
            raise ValueError('invalid control frequency')
        if self.gripper_sign not in (-1, 1) or self.gripper_positive not in ('open', 'close'):
            raise ValueError('invalid gripper convention')
        for name, values in (('action_scale', self.action_scale), ('action_offset', self.action_offset)):
            a = np.asarray(values, dtype=np.float64)
            if a.shape != (7,) or not np.isfinite(a).all():
                raise ValueError('invalid ' + name)
        if np.any(np.asarray(self.action_scale) == 0):
            raise ValueError('adapter scales cannot delete an action dimension')
        return self

    def environment_signature(self):
        return (self.controller, self.semantics, self.coordinate_frame, self.rotation_representation,
                self.units, self.control_frequency_hz, self.gripper_positive)

    def as_dict(self):
        return asdict(self)


def require_compatible_contracts(main, aux):
    main.validate(); aux.validate()
    if main.environment_signature() != aux.environment_signature():
        raise ValueError('main and aux do not map to the same environment action semantics')
    return True


def _actions(value, minimum=1, canonical=False):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 7 or len(a) < minimum or not np.isfinite(a).all():
        raise ValueError('expected finite [steps, 7] actions with at least %d steps' % minimum)
    if canonical and np.any(np.abs(a) > 1):
        raise ValueError('canonical action outside environment [-1,1]')
    return a


def canonicalize_actions(raw_actions, contract, minimum_steps=K):
    contract.validate()
    raw = _actions(raw_actions, minimum_steps)
    transformed = raw * np.asarray(contract.action_scale) + np.asarray(contract.action_offset)
    transformed[:, 6] *= contract.gripper_sign
    # The actual collector sends float32; scoring must see exactly those values.
    return np.clip(transformed, -1, 1).astype(np.float32)


@dataclass(frozen=True)
class DisagreementScales:
    translation: float = 1.0
    rotation: float = 1.0
    gripper: float = 1.0
    source: str = ''

    def validate(self):
        if not self.source or not all(math.isfinite(x) and x > 0 for x in
                                      (self.translation, self.rotation, self.gripper)):
            raise ValueError('positive frozen action scales and their verified source are required')
        return self

    def as_dict(self):
        return asdict(self)


def disagreement(main_actions, aux_actions, scales, steps=None):
    """RMS is over all m*3 translation or rotation entries; grip uses m entries."""
    if isinstance(scales, Mapping):
        scales = DisagreementScales(**scales)
    scales.validate()
    a = _actions(main_actions, canonical=True); b = _actions(aux_actions, canonical=True)
    m = min(len(a), len(b)) if steps is None else steps
    if type(m) is not int or m <= 0 or m > min(len(a), len(b)):
        raise ValueError('invalid comparable prefix length')
    d = a[:m] - b[:m]
    translation = float(np.sqrt(np.mean(d[:, :3] ** 2)) / scales.translation)
    rotation = float(np.sqrt(np.mean(d[:, 3:6] ** 2)) / scales.rotation)
    gripper = float(np.mean(np.abs(d[:, 6])) / scales.gripper)
    return {'D': (translation + rotation + gripper) / 3, 'translation': translation,
            'rotation': rotation, 'gripper': gripper, 'm': m, 'formula': FORMULA}


def executed_disagreement(parent_executed, child_aux_executed, scales, d_min, numeric_floor):
    """Recompute from actually sent commands, including an early terminal prefix."""
    if not math.isfinite(d_min) or not math.isfinite(numeric_floor) or numeric_floor <= 0 or d_min < numeric_floor:
        raise ValueError('invalid frozen execution thresholds')
    if len(parent_executed) == 0 or len(child_aux_executed) == 0:
        return {'m': 0, 'D': None, 'valid': False, 'reason': 'no_common_executed_prefix'}
    m = min(K, len(parent_executed), len(child_aux_executed))
    score = disagreement(parent_executed, child_aux_executed, scales, m)
    score['valid'] = score['D'] >= d_min and score['D'] > numeric_floor
    return score


@dataclass(frozen=True)
class Proposal:
    raw_actions: tuple
    canonical_actions: tuple
    identity_json: str
    adapter_version: str
    inference_json: str
    raw_obs_fingerprint: str
    input_fingerprint: str
    context_fingerprint: str
    request_id: str
    contract_fingerprint: str
    cache_key: str
    proposal_fingerprint: str
    cache_hit: bool = False

    @classmethod
    def create(cls, raw_actions, canonical_actions, model_identity, adapter_version,
               inference_settings, raw_obs_fingerprint, input_fingerprint,
               context_fingerprint, request_id, contract_fingerprint, cache_hit=False):
        raw = _actions(raw_actions, K); canonical = _actions(canonical_actions, K, True)
        if raw.shape != canonical.shape:
            raise ValueError('raw and canonical action shape mismatch')
        _weight_identity(model_identity)
        if not model_identity.get('name'):
            raise ValueError('missing model name')
        fields = (adapter_version, raw_obs_fingerprint, input_fingerprint,
                  context_fingerprint, request_id, contract_fingerprint)
        if not all(isinstance(x, str) and x for x in fields):
            raise ValueError('proposal fingerprints, adapter and request identity must be explicit')
        identity_json = json.dumps(_plain(model_identity), sort_keys=True, allow_nan=False)
        inference_json = json.dumps(_plain(inference_settings), sort_keys=True, allow_nan=False)
        cache_key = fingerprint(dict(model=model_identity, adapter=adapter_version,
            inference=inference_settings, observation=raw_obs_fingerprint, input=input_fingerprint,
            context=context_fingerprint, request=request_id, contract=contract_fingerprint))
        raw_t = tuple(tuple(float(v) for v in row) for row in raw)
        canonical_t = tuple(tuple(float(v) for v in row) for row in canonical)
        proposal_fp = fingerprint(dict(cache_key=cache_key, raw=raw_t, canonical=canonical_t))
        return cls(raw_t, canonical_t, identity_json, adapter_version, inference_json,
                   raw_obs_fingerprint, input_fingerprint, context_fingerprint, request_id,
                   contract_fingerprint, cache_key, proposal_fp, bool(cache_hit))

    @property
    def model_identity(self):
        return json.loads(self.identity_json)

    def as_dict(self):
        result = asdict(self)
        result['model_identity'] = result.pop('identity_json')
        result['model_identity'] = json.loads(result['model_identity'])
        result['inference_settings'] = json.loads(result.pop('inference_json'))
        return result

    @classmethod
    def from_dict(cls, data):
        p = cls.create(**{k: data[k] for k in ('raw_actions', 'canonical_actions', 'model_identity',
            'adapter_version', 'inference_settings', 'raw_obs_fingerprint', 'input_fingerprint',
            'context_fingerprint', 'request_id', 'contract_fingerprint')}, cache_hit=data.get('cache_hit', False))
        if p.cache_key != data.get('cache_key') or p.proposal_fingerprint != data.get('proposal_fingerprint'):
            raise ValueError('cached proposal fingerprint mismatch')
        return p


@dataclass(frozen=True)
class Candidate:
    branch_id: str
    depth: int
    step: int
    branch_fork_step: int
    state_fingerprint: str
    checkpoint: dict
    main: Proposal
    aux: Proposal
    score: dict
    candidate_id: str

    def as_dict(self):
        return dict(branch_id=self.branch_id, depth=self.depth, step=self.step,
            branch_fork_step=self.branch_fork_step, state_fingerprint=self.state_fingerprint,
            checkpoint=_plain(self.checkpoint), main=self.main.as_dict(), aux=self.aux.as_dict(),
            score=_plain(self.score), candidate_id=self.candidate_id)

    @classmethod
    def from_dict(cls, data):
        value = dict(data); value['main'] = Proposal.from_dict(value['main']); value['aux'] = Proposal.from_dict(value['aux'])
        c = cls(**value)
        expected = fingerprint(dict(branch=c.branch_id, step=c.step, state=c.state_fingerprint,
            main=c.main.proposal_fingerprint, aux=c.aux.proposal_fingerprint))
        if c.candidate_id != expected:
            raise ValueError('persisted candidate identity mismatch')
        if c.main.raw_obs_fingerprint != c.aux.raw_obs_fingerprint:
            raise ValueError('persisted candidate proposals do not share an observation')
        cp = c.checkpoint
        if (cp.get('fork_step', cp.get('step')) != c.step or
                cp.get('owner_branch_id', cp.get('branch_id')) != c.branch_id or
                cp.get('observation_sha256', c.main.raw_obs_fingerprint) != c.main.raw_obs_fingerprint):
            raise ValueError('persisted candidate checkpoint binding mismatch')
        validate_independent_models(c.main.model_identity, c.aux.model_identity)
        return c


class DisagreementForkSelector:
    """Bounded deterministic pool. No outcome-based score changes or fallback.

    consider() must run at a real replan boundary before any environment step.
    checkpoint is either a mapping or zero-argument exact-state capture callback;
    rejected candidates never invoke that callback. All state is serializable.
    """
    def __init__(self, *, d_min, numeric_floor, scales, max_depth=3, spacing=10,
                 cooldown=10, max_per_parent=32, mode='active'):
        if not math.isfinite(d_min) or not math.isfinite(numeric_floor) or numeric_floor <= 0 or d_min < numeric_floor:
            raise ValueError('D_min must be finite and at least the positive numerical floor')
        if (any(type(v) is not int for v in (spacing, cooldown, max_per_parent, max_depth)) or
                spacing < 10 or cooldown < 10 or not 1 <= max_per_parent <= 32 or not 1 <= max_depth <= 3):
            raise ValueError('invalid frozen candidate bounds')
        if isinstance(scales, Mapping):
            scales = DisagreementScales(**scales)
        scales.validate()
        if mode not in ('shadow', 'active'):
            raise ValueError('invalid mode')
        self.d_min, self.numeric_floor, self.scales = d_min, numeric_floor, scales
        self.max_depth, self.spacing, self.cooldown = max_depth, spacing, cooldown
        self.max_per_parent, self.mode = max_per_parent, mode
        self.pending = {}; self.used = set(); self.admitted_steps = {}; self.seen_states = set()
        self.candidate_count = 0

    def consider(self, *, branch_id, depth, step, branch_fork_step, horizon,
                 state_fingerprint, main, aux, checkpoint, replan_boundary=True,
                 terminal=False, confirmed_failed=False, recoverable=True):
        def rejected(reason):
            return {'admitted': False, 'reason': reason, 'candidate': None}
        if (any(type(v) is not int for v in (depth, step, branch_fork_step, horizon)) or
                depth < 0 or step < 0 or branch_fork_step < 0 or horizon < 0):
            raise ValueError('branch depth and absolute state steps must be nonnegative integers')
        if not replan_boundary:
            return rejected('not_main_replan_boundary')
        if terminal or confirmed_failed or not recoverable:
            return rejected('terminal_failed_or_unrecoverable')
        if depth >= self.max_depth:
            return rejected('depth_limit')
        if step < branch_fork_step or (depth > 0 and step - branch_fork_step < self.cooldown):
            return rejected('outside_new_suffix_or_cooldown')
        if horizon - step < 2 * K:
            return rejected('insufficient_shared_horizon')
        if main.raw_obs_fingerprint != aux.raw_obs_fingerprint:
            raise ValueError('proposal observations do not describe the same frozen state')
        validate_independent_models(main.model_identity, aux.model_identity)
        if not state_fingerprint or not branch_id:
            raise ValueError('state and branch identity required')
        score = disagreement(main.canonical_actions, aux.canonical_actions, self.scales, K)
        if not score['D'] >= self.d_min or not score['D'] > self.numeric_floor:
            return rejected('below_disagreement_threshold')
        cid = fingerprint(dict(branch=branch_id, step=step, state=state_fingerprint,
                               main=main.proposal_fingerprint, aux=aux.proposal_fingerprint))
        if cid in self.pending or cid in self.used or state_fingerprint in self.seen_states:
            return rejected('duplicate_state_or_proposal')
        if any(abs(step - old) < self.spacing for old in self.admitted_steps.get(branch_id, [])):
            return rejected('parent_candidate_spacing')
        if sum(c.branch_id == branch_id for c in self.pending.values()) >= self.max_per_parent:
            return rejected('parent_pool_full')
        cp = checkpoint() if callable(checkpoint) else dict(checkpoint)
        if cp.get('fork_step', cp.get('step')) != step or cp.get('owner_branch_id', cp.get('branch_id')) != branch_id:
            raise ValueError('checkpoint must capture the exact candidate state and owner')
        if cp.get('observation_sha256', main.raw_obs_fingerprint) != main.raw_obs_fingerprint:
            raise ValueError('checkpoint observation does not match proposals')
        candidate = Candidate(branch_id, depth, step, branch_fork_step, state_fingerprint,
                              cp, main, aux, score, cid)
        self.pending[cid] = candidate; self.seen_states.add(state_fingerprint)
        self.admitted_steps.setdefault(branch_id, []).append(step); self.candidate_count += 1
        return {'admitted': True, 'reason': 'disagreement_and_all_hard_gates', 'candidate': candidate}

    def choose(self, actual_max_depth, remaining_valid_budget=16, remaining_attempts=32, *, consume=True):
        if self.mode == 'shadow' or remaining_valid_budget <= 0 or remaining_attempts <= 0:
            return None
        options = [c for c in self.pending.values() if c.candidate_id not in self.used and c.depth < self.max_depth]
        if not options:
            return None
        options.sort(key=lambda c: (0 if c.depth + 1 > actual_max_depth else 1,
                                    -c.score['D'], c.branch_id, c.step, c.candidate_id))
        candidate = options[0]
        reason = ('unachieved_depth_then_D' if candidate.depth + 1 > actual_max_depth else 'D_then_branch_id_step')
        if consume:
            self.pending.pop(candidate.candidate_id); self.used.add(candidate.candidate_id)
        return {'candidate': candidate, 'reason': reason, 'child_depth': candidate.depth + 1}

    def as_dict(self):
        scales = self.scales.as_dict() if hasattr(self.scales, 'as_dict') else self.scales
        return dict(parameters=dict(d_min=self.d_min, numeric_floor=self.numeric_floor,
            scales=scales, max_depth=self.max_depth, spacing=self.spacing, cooldown=self.cooldown,
            max_per_parent=self.max_per_parent, mode=self.mode), pending=[c.as_dict() for c in self.pending.values()],
            used=sorted(self.used), admitted_steps=_plain(self.admitted_steps), seen_states=sorted(self.seen_states),
            candidate_count=self.candidate_count)

    @classmethod
    def from_dict(cls, data):
        obj = cls(**data['parameters'])
        for raw in data['pending']:
            c = Candidate.from_dict(raw)
            expected = disagreement(c.main.canonical_actions, c.aux.canonical_actions, obj.scales, K)
            if fingerprint(expected) != fingerprint(c.score):
                raise ValueError('persisted candidate score mismatch')
            if (c.candidate_id in obj.pending or not c.score['D'] >= obj.d_min or
                    not c.score['D'] > obj.numeric_floor or c.depth < 0 or c.depth >= obj.max_depth or
                    c.step < c.branch_fork_step or
                    (c.depth > 0 and c.step - c.branch_fork_step < obj.cooldown)):
                raise ValueError('persisted candidate is no longer eligible')
            obj.pending[c.candidate_id] = c
        obj.used = set(data['used']); obj.admitted_steps = data['admitted_steps']
        obj.seen_states = set(data['seen_states']); obj.candidate_count = data['candidate_count']
        if set(obj.pending) & obj.used or obj.candidate_count != len(obj.pending) + len(obj.used):
            raise ValueError('persisted candidate accounting mismatch')
        for branch, steps in obj.admitted_steps.items():
            if any(abs(a - b) < obj.spacing for a, b in combinations(steps, 2)):
                raise ValueError('persisted candidate spacing mismatch')
            if sum(c.branch_id == branch for c in obj.pending.values()) > obj.max_per_parent:
                raise ValueError('persisted candidate pool exceeds bound')
        for c in obj.pending.values():
            if c.state_fingerprint not in obj.seen_states or c.step not in obj.admitted_steps.get(c.branch_id, []):
                raise ValueError('persisted candidate deduplication state mismatch')
        return obj


class AuxThenMainExecutor:
    """Consumes cached aux commands, never calls a model or resamples a proposal.

    fork_step names state s_t. Each record_transition must name the new state:
    t+1..t+K. After K, caller reads that new observation and invokes main once.
    A next_action must be paired with record_transition before another action.
    """
    def __init__(self, proposal, *, fork_step, intervention_steps=K):
        if intervention_steps != K:
            raise ValueError('K is frozen at 5')
        if type(fork_step) is not int or fork_step < 0:
            raise ValueError('fork state step must be a nonnegative integer')
        _actions(proposal.canonical_actions, K, True)
        self.proposal = proposal; self.fork_step = fork_step
        self.actual_k = 0; self.terminal = False; self.terminal_evidence = None
        self.handoff_occurred = False; self.handoff_observation_fingerprint = None
        self._pending = False

    @property
    def needs_main_replan(self):
        return self.actual_k == K and not self.terminal and not self.handoff_occurred

    def next_action(self):
        if self._pending:
            raise RuntimeError('previous action has not been recorded')
        if self.terminal or self.actual_k >= K:
            return None
        self._pending = True
        return np.asarray(self.proposal.canonical_actions[self.actual_k], dtype=np.float32).copy()

    def record_transition(self, state_step, *, terminal=False, success=False, evidence=None):
        if not self._pending or state_step != self.fork_step + self.actual_k + 1:
            raise ValueError('transition must advance exactly one state from the fork')
        if (terminal or success) and not evidence:
            raise ValueError('early end requires real terminal/success evidence')
        self.actual_k += 1; self._pending = False
        if terminal or success:
            self.terminal = True; self.terminal_evidence = evidence

    def mark_handoff(self, state_step, observation_fingerprint):
        if not self.needs_main_replan or self._pending or state_step != self.fork_step + K or not observation_fingerprint:
            raise ValueError('main handoff requires fresh observation at s_(t+K)')
        self.handoff_occurred = True; self.handoff_observation_fingerprint = observation_fingerprint

    def report(self):
        return dict(planned_K=K, actual_K=self.actual_k, fork_state_step=self.fork_step,
            first_transition_end_step=self.fork_step + 1, intervention_end_state_step=self.fork_step + self.actual_k,
            handoff_occurred=self.handoff_occurred, handoff_state_step=self.fork_step + K if self.handoff_occurred else None,
            handoff_observation_fingerprint=self.handoff_observation_fingerprint, terminal=self.terminal,
            terminal_evidence=self.terminal_evidence, proposal_fingerprint=self.proposal.proposal_fingerprint,
            aux_cache_key=self.proposal.cache_key, action_source='aux', pending_uncommitted_action=self._pending)


def uniformly_select_states(states, maximum=32):
    """Caller supplies eligible nonterminal snapshots ordered by episode step."""
    if not 1 <= maximum <= 32:
        raise ValueError('calibration maximum must be <=32')
    if len(states) <= maximum:
        return list(states)
    return [states[int(i)] for i in np.linspace(0, len(states) - 1, maximum, dtype=int)]


def calibrate_threshold(states, *, scales, numeric_floor, numeric_floor_source,
                        evidence_kind, isolated_contexts, randomness_statement,
                        configuration_version):
    """Each state = {state_fingerprint, main:[3 Proposals], aux:[3 Proposals]}.

    Every pair of repeats contributes one within-model distance: six values per
    state. Distinct request IDs and cache misses are mandatory. Isolation is an
    explicit verified caller assertion; false is reported insufficient, not fixed
    through additional calls. Returns a report; makes no model requests itself.
    """
    if not math.isfinite(numeric_floor) or numeric_floor <= 0 or not numeric_floor_source:
        raise ValueError('predeclared positive numeric floor and precision source required')
    if evidence_kind not in ('mock', 'real_shadow') or not configuration_version or not randomness_statement:
        raise ValueError('explicit evidence type, version and randomness limits required')
    if len(states) > 32:
        raise ValueError('calibration exceeds 32-state bound')
    distributions = {'main': [], 'aux': []}; seen_requests = set(); seen_states = set(); frozen_ids = {}
    state_reports = []
    for state in states:
        sid = state['state_fingerprint']
        if not sid or sid in seen_states:
            raise ValueError('duplicate calibration state')
        seen_states.add(sid); obs_ids = set(); within = {}
        for role in ('main', 'aux'):
            repeats = state[role]
            if len(repeats) != 3:
                raise ValueError('exactly three fresh repeated requests per model/state required')
            contexts = set(); input_ids = set(); adapters = set(); contracts = set()
            for p in repeats:
                if p.cache_hit or p.request_id in seen_requests:
                    raise ValueError('calibration repeats must be new requests, never cache hits')
                seen_requests.add(p.request_id); obs_ids.add(p.raw_obs_fingerprint)
                contexts.add(p.context_fingerprint)
                input_ids.add(p.input_fingerprint); adapters.add(p.adapter_version); contracts.add(p.contract_fingerprint)
                identity = fingerprint(p.model_identity)
                if role in frozen_ids and frozen_ids[role] != identity:
                    raise ValueError('model identity changed during calibration')
                frozen_ids[role] = identity
            if any(len(values) != 1 for values in (contexts, input_ids, adapters, contracts)):
                raise ValueError('repeated inference must restore the same input, adapter, contract and pre-request context')
            values = [disagreement(a.canonical_actions, b.canonical_actions, scales, K)['D'] for a, b in combinations(repeats, 2)]
            distributions[role].extend(values); within[role] = values
        if len(obs_ids) != 1:
            raise ValueError('calibration proposals do not share the same raw state observation')
        validate_independent_models(state['main'][0].model_identity, state['aux'][0].model_identity)
        state_reports.append(dict(state_fingerprint=sid, observation_fingerprint=next(iter(obs_ids)), within_model=within,
            request_ids={role: [p.request_id for p in state[role]] for role in ('main', 'aux')}))
    merged = distributions['main'] + distributions['aux']
    p95 = float(np.quantile(merged, .95, method='linear')) if merged else None
    threshold = max(numeric_floor, p95) if p95 is not None else None
    sufficient = len(states) >= 16 and bool(isolated_contexts)
    active_eligible = sufficient and evidence_kind == 'real_shadow'
    return dict(schema='dual_disagreement_calibration_v1', configuration_version=configuration_version,
        status='PASS' if active_eligible else ('MOCK_ONLY' if sufficient else 'INSUFFICIENT'),
        active_eligible=active_eligible, state_count=len(states), repeats_per_model=3,
        pairwise_count=len(merged), within_model=distributions, merged_p95=p95, D_min=threshold,
        numeric_floor=numeric_floor, numeric_floor_source=numeric_floor_source,
        scales=scales.as_dict() if hasattr(scales, 'as_dict') else scales, weights=[1/3, 1/3, 1/3],
        K=K, formula=FORMULA, candidate_spacing=10, suffix_cooldown=10, max_candidates_per_parent=32,
        evidence_kind=evidence_kind, isolated_contexts=bool(isolated_contexts),
        randomness_statement=randomness_statement, model_identity_fingerprints=frozen_ids,
        states=state_reports, interpretation='engineering noise floor only; not statistical significance or recovery success')
