"""Context-dependent reversal in an artificial connectome-inspired learner.

Contexts, traces and food rewards are engineering variables. They are not mapped
memories, neurons, or evidence about real flies or consciousness.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json, math, random, statistics

A,B="banana","cucumber"; CX,CY="context_X","context_Y"; MASK="no_context"

@dataclass(frozen=True)
class ContextConfig:
    seed:int=0; cue_reliability:float=1.0
    acquisition_trials:int=160; reversal_trials:int=180
    switch_blocks:int=12; switch_block_trials:int=12; reacquisition_trials:int=100
    epsilon:float=.10; reward_hi:float=.90; reward_lo:float=.10; reward_sd:float=.08
    fast_alpha:float=.22; slow_alpha:float=.035; fast_decay:float=.006
    slow_decay:float=.00015; slow_weight:float=.65
    criterion_window:int=30; criterion_rate:float=.80

class ContextAgent:
    name="context_dual_trace"
    def __init__(self,c,use_context=True,use_slow=True,shuffle_cues=False):
        self.c=c; self.use_context=use_context; self.use_slow=use_slow; self.shuffle_cues=shuffle_cues
        cues=(CX,CY) if use_context else (MASK,)
        self.fast={z:{A:0.,B:0.} for z in cues}; self.slow={z:{A:0.,B:0.} for z in cues}
        if not use_context:self.name="ablation_no_context"
        elif not use_slow:self.name="ablation_fast_only"
        elif shuffle_cues:self.name="control_shuffled_cue"
    def key(self,cue): return cue if self.use_context else MASK
    def values(self,cue):
        k=self.key(cue); return {x:self.fast[k][x]+(self.c.slow_weight*self.slow[k][x] if self.use_slow else 0.) for x in (A,B)}
    def choose(self,cue,rng,explore=True):
        if explore and rng.random()<self.c.epsilon:return rng.choice((A,B))
        v=self.values(cue); d=v[A]-v[B]
        return rng.choice((A,B)) if abs(d)<1e-12 else (A if d>0 else B)
    def update(self,cue,choice,reward):
        # All traces decay, but only the observed context receives prediction error.
        for k in self.fast:
            for x in (A,B):
                self.fast[k][x]*=1-self.c.fast_decay
                if self.use_slow:self.slow[k][x]*=1-self.c.slow_decay
        k=self.key(cue); pe=reward-self.values(cue)[choice]
        self.fast[k][choice]+=self.c.fast_alpha*pe
        if self.use_slow:self.slow[k][choice]+=self.c.slow_alpha*pe
    def contrasts(self):
        return {z:self.values(z)[A]-self.values(z)[B] for z in (CX,CY)}

def agents(c): return [ContextAgent(c),ContextAgent(c,use_context=False),ContextAgent(c,use_slow=False),ContextAgent(c,shuffle_cues=True)]
def cue_for(rng,latent,reliability): return latent if rng.random()<reliability else (CY if latent==CX else CX)
def reward(rng,choice,target,c): return min(1.,max(0.,rng.gauss(c.reward_hi if choice==target else c.reward_lo,c.reward_sd)))
def criterion(xs,target,w,rate):
    need=math.ceil(w*rate)
    for end in range(w,len(xs)+1):
        if sum(x==target for x in xs[end-w:end])>=need:return end
    return None

def run_one(agent,c):
    # Separate streams prevent reward draws from changing cue schedules and choices.
    base=c.seed*10007+sum(map(ord,agent.name))+int(c.cue_reliability*1000)
    cue_rng=random.Random(base+1); choice_rng=random.Random(base+2); reward_rng=random.Random(base+3)
    rows=[]; phase_choices={}; global_t=0
    def train(phase,n,latent,target,block=0):
        nonlocal global_t
        xs=[]
        for t in range(1,n+1):
            global_t+=1; cue=cue_for(cue_rng,latent,c.cue_reliability)
            if agent.shuffle_cues: cue=choice_rng.choice((CX,CY))
            choice=agent.choose(cue,choice_rng); rew=reward(reward_rng,choice,target,c); agent.update(cue,choice,rew); xs.append(choice)
            if t==1 or t==n or t%10==0:
                co=agent.contrasts(); rows.append(dict(seed=c.seed,agent=agent.name,reliability=c.cue_reliability,phase=phase,block=block,phase_trial=t,global_trial=global_t,latent_context=latent,observed_cue=cue,target=target,accuracy=int(choice==target),contrast_X=co[CX],contrast_Y=co[CY]))
        phase_choices.setdefault(phase,[]).extend(xs); return xs
    acq=train("acquisition",c.acquisition_trials,CX,A)
    end_acq=agent.contrasts(); rev=train("reversal",c.reversal_trials,CY,B); end_rev=agent.contrasts()
    # Frozen probes isolate stored cue-specific policies from new learning.
    def probe(latent,cue,target,n=80):
        pr=random.Random(base+500+sum(map(ord,latent+cue+target)))
        return sum(agent.choose(cue,pr,explore=False)==target for _ in range(n))/n
    correct_X=probe(CX,CX,A); wrong_X=probe(CX,CY,A); correct_Y=probe(CY,CY,B); wrong_Y=probe(CY,CX,B)
    switch_acc=[]; first_acc=[]
    for block in range(c.switch_blocks):
        latent,target=(CX,A) if block%2==0 else (CY,B)
        xs=train("context_switch",c.switch_block_trials,latent,target,block+1)
        switch_acc.extend(x==target for x in xs); first_acc.extend(x==target for x in xs[:3])
    reacq=train("reacquisition",c.reacquisition_trials,CX,A)
    ac=criterion(acq,A,c.criterion_window,c.criterion_rate); rc=criterion(reacq,A,c.criterion_window,c.criterion_rate)
    storage_X=float(end_rev[CX]>0); storage_Y=float(end_rev[CY]<0)
    metrics=dict(seed=c.seed,agent=agent.name,reliability=c.cue_reliability,
      acquisition_criterion=ac,reacquisition_criterion=rc,reacquisition_savings=(ac-rc if ac and rc else None),
      old_strategy_contrast_end_acquisition=end_acq[CX],old_strategy_contrast_after_reversal=end_rev[CX],
      old_strategy_retention_ratio=(end_rev[CX]/end_acq[CX] if end_acq[CX]>1e-9 else None),
      new_strategy_contrast_after_reversal=-end_rev[CY],both_strategies_stored=float(storage_X and storage_Y),
      correct_cue_retrieval_accuracy=(correct_X+correct_Y)/2,wrong_cue_accuracy=(wrong_X+wrong_Y)/2,
      cue_retrieval_advantage=((correct_X+correct_Y)-(wrong_X+wrong_Y))/2,
      context_switch_accuracy=sum(switch_acc)/len(switch_acc),switch_first3_accuracy=sum(first_acc)/len(first_acc),
      context_switch_cost=(sum(switch_acc)/len(switch_acc)-sum(first_acc)/len(first_acc)))
    return rows,metrics

def quantile(xs,p):
    ys=sorted(xs); k=(len(ys)-1)*p; lo=int(k); hi=min(lo+1,len(ys)-1); f=k-lo
    return ys[lo]*(1-f)+ys[hi]*f

def summarize(ms):
    fields=[k for k in ms[0] if k not in ("seed","agent","reliability")]; out={}
    for a in sorted({m['agent'] for m in ms}):
      out[a]={}
      for r in sorted({m['reliability'] for m in ms}):
        group=[m for m in ms if m['agent']==a and m['reliability']==r]; out[a][str(r)]={}
        for f in fields:
          xs=[float(m[f]) for m in group if m[f] is not None]
          out[a][str(r)][f]={"n":len(xs),"mean":statistics.fmean(xs) if xs else None,"sd":statistics.stdev(xs) if len(xs)>1 else None,"seed_interval_95":[quantile(xs,.025),quantile(xs,.975)] if xs else [None,None]}
    return out

def paired_effects(ms,baseline="ablation_no_context"):
    out={}; focal="context_dual_trace"
    for r in sorted({m['reliability'] for m in ms}):
      by={(m['seed'],m['agent']):m for m in ms if m['reliability']==r}; out[str(r)]={}
      for f in ("correct_cue_retrieval_accuracy","cue_retrieval_advantage","context_switch_accuracy","reacquisition_savings"):
        ds=[]
        for seed in sorted({m['seed'] for m in ms}):
          x=by.get((seed,focal),{}).get(f); y=by.get((seed,baseline),{}).get(f)
          if x is not None and y is not None:ds.append(float(x)-float(y))
        out[str(r)][f]={"comparison":f"{focal} - {baseline}","n":len(ds),"mean_difference":statistics.fmean(ds) if ds else None,"seed_interval_95":[quantile(ds,.025),quantile(ds,.975)] if ds else [None,None]}
    return out

def run_experiment(outdir,seeds=range(200),reliabilities=(1.0,.9,.75,.5),config_overrides=None):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True); curves=[]; metrics=[]
    for reliability in reliabilities:
      for seed in seeds:
        c=ContextConfig(seed=seed,cue_reliability=reliability,**(config_overrides or {}))
        for a in agents(c):
          rr,mm=run_one(a,c); curves.extend(rr); metrics.append(mm)
    with (out/'context_metrics.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=metrics[0]); w.writeheader(); w.writerows(metrics)
    with (out/'learning_curves.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=curves[0]); w.writeheader(); w.writerows(curves)
    summary={"scientific_boundary":"Artificial connectome-inspired model; not biological evidence about real fly memory, behavior, or consciousness.","config":asdict(ContextConfig()),"seed_count":len(set(m['seed'] for m in metrics)),"cue_reliabilities":list(reliabilities),
      "hypotheses":{"H1":"With informative context cues, the contextual dual-trace agent stores opposite policies and retrieves the policy matching the cue.","H2":"Retrieval advantage and switch performance fall as cue reliability approaches chance.","H3":"Removing context or shuffling cues removes the retrieval advantage; removing the slow trace weakens retention and reacquisition savings."},
      "definitions":{"storage":"positive banana-minus-cucumber contrast for context X and negative contrast for context Y after reversal","retrieval":"frozen-policy accuracy under correct versus counterfactual wrong cue","switch_cost":"overall alternating-block accuracy minus accuracy on the first three trials after each switch","reacquisition_savings":"initial acquisition trials-to-criterion minus context-X reacquisition trials-to-criterion"},
      "summaries":summarize(metrics),"paired_effects":paired_effects(metrics)}
    (out/'context_summary.json').write_text(json.dumps(summary,indent=2))
    return curves,metrics,summary
