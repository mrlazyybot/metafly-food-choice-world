"""Preference-reversal experiments for an artificial connectome-inspired learner.

The model is an engineering abstraction, not a biological fly simulation.  Its
"fast" and "slow" traces are explicit hypotheses used to make retention and
suppression measurable; they are not mapped to specific neurons.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import csv, json, math, random

A, B = "banana", "cucumber"

@dataclass(frozen=True)
class ReversalConfig:
    seed: int = 0
    acquisition_trials: int = 180
    reversal_trials: int = 220
    extinction_trials: int = 120
    rest_steps: int = 180
    reacquisition_trials: int = 140
    epsilon: float = 0.10
    reward_hi: float = 0.90
    reward_lo: float = 0.10
    reward_sd: float = 0.08
    fast_alpha: float = 0.22
    slow_alpha: float = 0.035
    fast_decay: float = 0.006
    slow_decay: float = 0.00015
    slow_weight: float = 0.65
    criterion_window: int = 40
    criterion_rate: float = 0.80

class Agent:
    name = "agent"
    def choose(self, rng: random.Random) -> str: raise NotImplementedError
    def update(self, choice: str, reward: float) -> None: raise NotImplementedError
    def rest(self) -> None: pass
    def values(self) -> Dict[str, float]: return {A: 0.0, B: 0.0}

class RandomAgent(Agent):
    name = "random"
    def choose(self, rng): return rng.choice((A, B))
    def update(self, choice, reward): pass

class QAgent(Agent):
    name = "tabular_q"
    def __init__(self, alpha=.14, epsilon=.10):
        self.q = {A: 0.0, B: 0.0}; self.alpha=alpha; self.epsilon=epsilon
    def choose(self, rng):
        if rng.random() < self.epsilon: return rng.choice((A,B))
        d=self.q[A]-self.q[B]
        if abs(d)<1e-12: return rng.choice((A,B))
        return A if d>0 else B
    def update(self, choice, reward): self.q[choice] += self.alpha*(reward-self.q[choice])
    def values(self): return dict(self.q)

class DualTraceAgent(Agent):
    """Two-timescale prediction-error learner with explicit latent traces."""
    name = "metafly_dual_trace"
    def __init__(self, c: ReversalConfig, use_fast=True, use_slow=True):
        self.c=c; self.use_fast=use_fast; self.use_slow=use_slow
        self.fast={A:0.0,B:0.0}; self.slow={A:0.0,B:0.0}
        if not use_slow: self.name="ablation_no_slow"
        if not use_fast: self.name="ablation_no_fast"
    def values(self):
        fw=1.0 if self.use_fast else 0.0; sw=self.c.slow_weight if self.use_slow else 0.0
        return {x:fw*self.fast[x]+sw*self.slow[x] for x in (A,B)}
    def choose(self, rng):
        if rng.random()<self.c.epsilon: return rng.choice((A,B))
        v=self.values(); d=v[A]-v[B]
        if abs(d)<1e-12: return rng.choice((A,B))
        return A if d>0 else B
    def _decay(self):
        if self.use_fast:
            for x in (A,B): self.fast[x] *= 1-self.c.fast_decay
        if self.use_slow:
            for x in (A,B): self.slow[x] *= 1-self.c.slow_decay
    def update(self, choice, reward):
        self._decay(); pred=self.values()[choice]; pe=reward-pred
        if self.use_fast: self.fast[choice] += self.c.fast_alpha*pe
        if self.use_slow: self.slow[choice] += self.c.slow_alpha*pe
    def rest(self): self._decay()
    def traces(self): return {"fast_A":self.fast[A],"fast_B":self.fast[B],"slow_A":self.slow[A],"slow_B":self.slow[B]}

def make_agents(c):
    return [RandomAgent(), QAgent(epsilon=c.epsilon), DualTraceAgent(c),
            DualTraceAgent(c,use_slow=False), DualTraceAgent(c,use_fast=False)]

def _reward(rng, mean, sd): return min(1.0,max(0.0,rng.gauss(mean,sd)))

def _criterion(choices: List[str], target: str, window: int, rate: float):
    need=math.ceil(window*rate)
    for end in range(window,len(choices)+1):
        if sum(x==target for x in choices[end-window:end]) >= need: return end
    return None

def run_one(agent: Agent, c: ReversalConfig):
    # Separate reward and policy streams make agents comparable without sharing actions.
    rng=random.Random(c.seed*1009 + sum(ord(x) for x in agent.name))
    rows=[]; checkpoints={}
    phases=[("acquisition",c.acquisition_trials,A,c.reward_hi,c.reward_lo),
            ("reversal",c.reversal_trials,B,c.reward_hi,c.reward_lo),
            ("extinction",c.extinction_trials,None,0.0,0.0)]
    global_trial=0
    for phase,n,target,hi,lo in phases:
        for t in range(1,n+1):
            global_trial+=1; before=agent.values(); choice=agent.choose(rng)
            mean=0.0 if target is None else (hi if choice==target else lo)
            reward=_reward(rng,mean,c.reward_sd) if target is not None else 0.0
            agent.update(choice,reward); after=agent.values()
            row={"seed":c.seed,"agent":agent.name,"phase":phase,"phase_trial":t,
                 "global_trial":global_trial,"choice":choice,"target":target or "none",
                 "reward":reward,"value_A":after[A],"value_B":after[B],
                 "value_delta_A_minus_B":after[A]-after[B]}
            if isinstance(agent,DualTraceAgent): row.update(agent.traces())
            else: row.update({k:"" for k in ("fast_A","fast_B","slow_A","slow_B")})
            rows.append(row)
        checkpoints[f"end_{phase}"]={**agent.values(), **(agent.traces() if isinstance(agent,DualTraceAgent) else {})}
    # Rest has no choices or reward. It tests differential trace decay, not a claim about sleep.
    for _ in range(c.rest_steps): agent.rest()
    checkpoints["post_rest"]={**agent.values(), **(agent.traces() if isinstance(agent,DualTraceAgent) else {})}
    # Twenty reward-free probes quantify spontaneous recovery without further learning.
    probe=[]
    for t in range(1,21):
        global_trial+=1; choice=agent.choose(rng); probe.append(choice); v=agent.values()
        rows.append({"seed":c.seed,"agent":agent.name,"phase":"recovery_probe","phase_trial":t,
                     "global_trial":global_trial,"choice":choice,"target":"none","reward":0.0,
                     "value_A":v[A],"value_B":v[B],"value_delta_A_minus_B":v[A]-v[B],
                     **(agent.traces() if isinstance(agent,DualTraceAgent) else {k:"" for k in ("fast_A","fast_B","slow_A","slow_B")})})
    reacq=[]
    for t in range(1,c.reacquisition_trials+1):
        global_trial+=1; choice=agent.choose(rng); reacq.append(choice)
        reward=_reward(rng,c.reward_hi if choice==A else c.reward_lo,c.reward_sd)
        agent.update(choice,reward); v=agent.values()
        rows.append({"seed":c.seed,"agent":agent.name,"phase":"reacquisition","phase_trial":t,
                     "global_trial":global_trial,"choice":choice,"target":A,"reward":reward,
                     "value_A":v[A],"value_B":v[B],"value_delta_A_minus_B":v[A]-v[B],
                     **(agent.traces() if isinstance(agent,DualTraceAgent) else {k:"" for k in ("fast_A","fast_B","slow_A","slow_B")})})
    byphase={p:[r for r in rows if r["phase"]==p] for p in ("acquisition","reversal","extinction")}
    acq_choices=[r["choice"] for r in byphase["acquisition"]]
    rev_choices=[r["choice"] for r in byphase["reversal"]]
    ext_choices=[r["choice"] for r in byphase["extinction"]]
    acqcrit=_criterion(acq_choices,A,c.criterion_window,c.criterion_rate)
    revcrit=_criterion(rev_choices,B,c.criterion_window,c.criterion_rate)
    reacqcrit=_criterion(reacq,A,c.criterion_window,c.criterion_rate)
    old_end_acq=checkpoints["end_acquisition"][A]; old_end_rev=checkpoints["end_reversal"][A]
    retained=(old_end_rev/old_end_acq) if old_end_acq>1e-9 else 0.0
    rev_late=sum(x==B for x in rev_choices[-40:])/40
    old_slow_acq=checkpoints["end_acquisition"].get("slow_A",0.0)
    old_slow_rev=checkpoints["end_reversal"].get("slow_A",0.0)
    slow_retained=(old_slow_rev/old_slow_acq) if old_slow_acq>1e-9 else 0.0
    suppressed=float(rev_late>=.8 and slow_retained>=.5)
    metrics={"seed":c.seed,"agent":agent.name,
      "acquisition_criterion":acqcrit,"reversal_criterion":revcrit,
      "reacquisition_criterion":reacqcrit,
      "reacquisition_savings":(acqcrit-reacqcrit) if acqcrit and reacqcrit else None,
      "early_reversal_perseveration":sum(x==A for x in rev_choices[:30])/30,
      "late_reversal_accuracy":rev_late,
      "old_value_end_acquisition":old_end_acq,"old_value_end_reversal":old_end_rev,
      "old_value_retention_ratio":retained,
      "old_slow_trace_retention_ratio":slow_retained,
      "discard_fraction":1-min(1.0,max(0.0,retained)),
      "suppressed_not_discarded":suppressed,
      "late_extinction_old_choice":sum(x==A for x in ext_choices[-30:])/30,
      "post_rest_old_choice":sum(x==A for x in probe)/len(probe),
      "spontaneous_recovery":sum(x==A for x in probe)/len(probe)-sum(x==A for x in ext_choices[-30:])/30}
    return rows, metrics, checkpoints

def _quantile(xs,p):
    ys=sorted(xs); k=(len(ys)-1)*p; lo=int(k); hi=min(lo+1,len(ys)-1); f=k-lo
    return ys[lo]*(1-f)+ys[hi]*f

def summarize(metrics):
    fields=[k for k in metrics[0] if k not in ("seed","agent")]
    out={}
    for agent in sorted({m["agent"] for m in metrics}):
        ms=[m for m in metrics if m["agent"]==agent]; out[agent]={}
        for field in fields:
            vals=[float(m[field]) for m in ms if m[field] is not None]
            out[agent][field]={"n":len(vals),"mean":sum(vals)/len(vals) if vals else None,
                "sd":(sum((x-sum(vals)/len(vals))**2 for x in vals)/(len(vals)-1))**.5 if len(vals)>1 else None,
                "ci95":[_quantile(vals,.025),_quantile(vals,.975)] if vals else [None,None]}
    return out

def run_experiment(outdir: Path|str, seeds: Iterable[int]=range(200), config_overrides=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); rows=[]; metrics=[]; checks=[]
    for seed in seeds:
        c=ReversalConfig(seed=seed,**(config_overrides or {}))
        for agent in make_agents(c):
            rr,mm,cc=run_one(agent,c); rows.extend(rr); metrics.append(mm)
            checks.append({"seed":seed,"agent":agent.name,"checkpoints":cc})
    with (out/"reversal_trials.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with (out/"reversal_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=metrics[0]); w.writeheader(); w.writerows(metrics)
    summary={"scientific_boundary":"Artificial connectome-inspired learner; not evidence about real fly memory, strategy, or consciousness.",
             "config":asdict(ReversalConfig()),"seed_count":len(set(m["seed"] for m in metrics)),
             "definitions":{"retained":"old option latent value at end reversal / end acquisition",
               "suppressed_not_discarded":"late reversal choice accuracy >=0.8 while >=0.5 of the old slow trace remains",
               "spontaneous_recovery":"old-option choice rate after rest minus final 30 extinction trials",
               "reacquisition_savings":"trials to initial criterion minus trials to reacquisition criterion"},
             "agents":summarize(metrics)}
    (out/"reversal_summary.json").write_text(json.dumps(summary,indent=2))
    (out/"checkpoints.json").write_text(json.dumps(checks,indent=2))
    return rows,metrics,summary
