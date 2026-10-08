"""BDDL-bound task families; original coordinator/selector/store remain in use."""
from __future__ import annotations
import argparse
import copy
import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from curation_store import atomic_json
from semantic_checkpoint import SemanticKeyframeSelector, SemanticKeyframeConfig
from pick_place_failure import FailureDecision, FailureStatus
from pick_place_checkpoint import PickPlaceTaskSpec, _find_libero_task_owner

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520, 400)))


def describe_task(suite, task_id, init_state_id=0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    if suite not in SUITES:
        raise ValueError('Unknown suite: ' + suite)
    bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    if not 0 <= task_id < bench.n_tasks:
        raise ValueError('task_id outside official suite')
    task = bench.get_task(task_id)
    bddl = Path(bench.get_task_bddl_file_path(task_id))
    init = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
    if not bddl.is_file() or not init.is_file():
        raise FileNotFoundError('Missing official BDDL/init file: %s / %s' % (bddl, init))
    states = bench.get_task_init_states(task_id)
    if not 0 <= init_state_id < len(states):
        raise ValueError('init_state_id outside official initial states')
    parsed = robosuite_parse_problem(str(bddl))
    goals = parsed['goal_state']
    if len(goals) == 1 and goals[0][0] in ('on', 'in'):
        family, support = 'placement', 'semantic'
    elif len(goals) == 1 and goals[0][0] in ('open', 'close', 'turnon', 'turnoff'):
        family, support = 'articulation', 'semantic'
    else:
        family, support = 'composite', 'generic'
    return dict(suite=suite, task_id=task_id, init_state_id=init_state_id,
                task_order_index=0, name=task.name, language=task.language,
                goals=goals, family=family, support=support,
                support_scope='family rules; runtime binding required; efficacy unmeasured',
                limitations=('No ordered subtask detector; physical stationarity only' if support == 'generic'
                             else 'Rule cutoffs are heuristic, not confirmed natural task failures'),
                bddl=str(bddl), bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                init_file=str(init), init_file_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),
                init_state_count=len(states), default_horizon=HORIZONS[suite])


@dataclasses.dataclass
class TaskTruth:
    task_success: bool
    sources: dict
    predicates: list
    metrics: dict
    stage: str
    support: str

    def as_dict(self):
        return dataclasses.asdict(self)


class TaskAdapter:
    def __init__(self, description, env):
        self.description, self.env = description, env
        self.core = _find_libero_task_owner(env)
        if self.core is None:
            raise ValueError('LIBERO task owner unavailable')
        self.goals = description['goals']
        if self.core.parsed_problem['goal_state'] != self.goals:
            raise ValueError('Loaded BDDL goal mismatch')
        for goal in self.goals:
            for name in goal[1:]:
                if name not in self.core.object_states_dict:
                    raise ValueError('Unbound BDDL object/site: ' + name)
        self.spec = PickPlaceTaskSpec(target_object=self.goals[0][1],
                                      goal_object=self.goals[0][-1])
        self.truth(env, self.spec)  # fail before policy if a metric cannot bind

    def truth(self, env, spec=None, info=None):
        from libero.libero.envs.predicates import eval_predicate_fn
        states = self.core.object_states_dict
        predicates = [bool(eval_predicate_fn(g[0], *[states[n] for n in g[1:]])) for g in self.goals]
        # Only this environment truth can set SUCCESS in the coordinator.
        success = bool(env.check_success())
        eef = np.asarray(self.core.sim.data.site_xpos[self.core.robots[0].eef_site_id])
        metrics = {'predicate_count': int(sum(predicates)), 'eef': eef.tolist()}
        family = self.description['family']
        if family == 'placement':
            g = self.goals[0]
            target, goal = states[g[1]], states[g[2]]
            pos = np.asarray(target.get_geom_state()['pos'])
            goal_pos = np.asarray(goal.get_geom_state()['pos'])
            grasped = bool(self.core._check_grasp(self.core.robots[0].gripper, self.core.get_object(g[1])))
            stage = 'goal_satisfied' if predicates[0] else ('transport' if grasped else 'reach_and_grasp')
            metrics.update(target_position=pos.tolist(), goal_position=goal_pos.tolist(), grasped=grasped,
                           eef_target_distance=float(np.linalg.norm(eef-pos)),
                           target_goal_distance=float(np.linalg.norm(pos-goal_pos)))
        elif family == 'articulation':
            target = states[self.goals[0][1]]
            obj = (self.core.object_sites_dict[target.object_name] if target.object_state_type == 'site'
                   else self.core.get_object(target.object_name))
            joints = list(obj.joints)
            if not joints:
                raise ValueError('No articulation joints for actual BDDL target')
            values = [float(self.core.sim.data.get_joint_qpos(j)) for j in joints]
            pos = np.asarray(target.get_geom_state()['pos'])
            distance = float(np.linalg.norm(eef-pos))
            stage = 'goal_satisfied' if predicates[0] else ('operate_joint' if distance < .18 else 'reach_articulation')
            metrics.update(joints=joints, joint_values=values, eef_target_distance=distance)
        else:
            stage = 'generic_physical_monitor'
            metrics['qpos'] = np.asarray(self.core.sim.data.qpos).tolist()
        return TaskTruth(success, {'task_success': 'env.check_success',
                         'predicates': 'libero.eval_predicate_fn'}, predicates, metrics, stage,
                         self.description['support'])

    def manager(self):
        return TaskCheckpointManager(self)

    def detector(self):
        return TaskFailureDetector(self.description['family'])


class PredicateEvents:
    def __init__(self):
        self.completed_events = ()

    def snapshot_state(self):
        return {'completed_events': list(self.completed_events)}

    def restore_state(self, state, obs=None):
        self.completed_events = tuple(state['completed_events'])


class TaskCheckpointManager:
    def __init__(self, adapter):
        self.adapter = adapter
        self.detector = PredicateEvents()
        self.selector = SemanticKeyframeSelector(config=SemanticKeyframeConfig(min_interval=10, max_interval=30))

    def observe(self, env, step, obs, reward=0., done=False, info=None, **kwargs):
        truth = self.adapter.truth(env)
        before = self.detector.completed_events
        self.detector.completed_events = tuple('bddl_goal_%d' % i for i, x in enumerate(truth.predicates) if x)
        metadata = dict(kwargs.pop('metadata', {}) or {})
        metadata.update(task_family=self.adapter.description['family'], detector_support=truth.support,
                        task_truth=truth.as_dict(), checkpoint_source='predicate_or_temporal')
        if before != self.detector.completed_events:
            self.selector.observe_only(env, step, obs, reward, done, info)
            keyframe = self.selector.force_capture(env, step, obs, labels=('bddl_predicate_change',),
                score=5., features={'predicate_count': float(sum(truth.predicates))}, reward=reward,
                done=done, metadata=metadata, **kwargs)
        else:
            keyframe = self.selector.maybe_capture(env, step, obs, reward, done, info,
                                                   metadata=metadata, **kwargs)
        return SimpleNamespace(keyframe=keyframe)


class TaskFailureDetector:
    """Conservative family stall rules. No fixed step budget is a failure rule.

    Placement: goal distance only while grasped, EEF approach otherwise.
    Articulation: actual target joint movement near the target; no placement rule.
    Composite: explicitly generic stationarity, no semantic completeness claim.
    """
    def __init__(self, family):
        self.family = family
        self.history = []
        self.stage = None
        self.stage_start = 0
        self.warning = None
        self.count = 0

    def snapshot_state(self):
        return copy.deepcopy(self.__dict__)

    def restore_state(self, state):
        self.__dict__.update(copy.deepcopy(state))

    def update(self, obs, *, step, task_truth, done=False, **kwargs):
        t, reason = task_truth, None
        if t.stage != self.stage:
            self.stage, self.stage_start, self.history = t.stage, step, []
            self.warning, self.count = None, 0
        m = t.metrics
        self.history.append(copy.deepcopy(m))
        self.history = self.history[-60:]
        evidence = dict(support=t.support, metrics=m, rule_cutoff=True,
                        natural_failure_confirmed=False, history_steps=len(self.history))
        if not t.task_success and len(self.history) == 60 and step-self.stage_start >= 80:
            if self.family == 'placement':
                key = 'target_goal_distance' if t.stage == 'transport' else 'eef_target_distance'
                values = [x[key] for x in self.history]
                progress = values[0] - min(values[1:])
                evidence.update(progress_metric=key, best_improvement=progress)
                if progress < .003:
                    reason = 'placement_%s_stalled' % t.stage
            elif self.family == 'articulation':
                if t.stage == 'operate_joint':
                    spread = float(np.max(np.ptp(np.asarray([x['joint_values'] for x in self.history]), axis=0)))
                    evidence.update(progress_metric='target_joint_motion', joint_span=spread)
                    if spread < .003:
                        reason = 'articulation_joint_stalled'
                else:
                    values = [x['eef_target_distance'] for x in self.history]
                    improvement = values[0] - min(values[1:])
                    evidence.update(progress_metric='eef_to_articulation_target', best_improvement=improvement)
                    if improvement < .003:
                        reason = 'articulation_approach_stalled'
            else:
                qspan = float(np.max(np.ptp(np.asarray([x['qpos'] for x in self.history]), axis=0)))
                espan = float(np.max(np.ptp(np.asarray([x['eef'] for x in self.history]), axis=0)))
                evidence.update(progress_metric='generic_physical_stationarity', qpos_span=qspan, eef_span=espan)
                if qspan < .003 and espan < .003:
                    reason = 'generic_physical_stationarity'
        if reason:
            self.count += 1
            if self.warning is None:
                self.warning = step
        else:
            self.count, self.warning = 0, None
        status = (FailureStatus.RECOVERABLE_FAILURE if self.count >= 3 else
                  FailureStatus.WARNING if reason else FailureStatus.NORMAL)
        if done and not t.task_success:
            status, reason = FailureStatus.TERMINAL_FAILURE, 'environment_terminated_without_success'
        return FailureDecision(step, status, .8 if reason else 0., reason, t.stage,
                               self.count, bool(done), self.warning, evidence)


def inventory(output):
    from libero.libero import benchmark
    rows = []
    for suite in SUITES:
        n = benchmark.get_benchmark_dict()[suite]().n_tasks
        for task_id in range(n):
            try:
                rows.append(describe_task(suite, task_id))
            except Exception as exc:
                rows.append(dict(suite=suite, task_id=task_id, support='blocked', error=repr(exc)))
    atomic_json(output, rows)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    inventory(p.parse_args().output)
