"""A connectome-informed, not connectome-emulating, Drosophila food-choice model."""
from dataclasses import dataclass
import csv, json, math, random

FOODS = {
 "banana": ((.95,.55,.20,.05), .90), "apple": ((.72,.44,.18,.12), .72),
 "orange": ((.64,.78,.08,.08), .68), "grape": ((.77,.38,.22,.05), .76),
 "tomato": ((.38,.62,.28,.18), .43), "carrot": ((.20,.30,.62,.15), .30),
 "broccoli": ((.08,.22,.70,.45), .12), "cucumber": ((.18,.33,.55,.10), .25),
}

@dataclass
class Config:
 seed:int=7; trials:int=240; alpha:float=.14; epsilon:float=.12; sensory_noise:float=.08

class MetaFly:
 """Four odor channels -> sparse Kenyon-like expansion -> learned values.
 This is an engineering abstraction. No spikes, biophysics, or claimed mind upload.
 """
 def __init__(self,cfg=Config()):
  self.c=cfg; self.r=random.Random(cfg.seed); self.w={k:0.0 for k in FOODS}
 def choose(self, pair):
  if self.r.random()<self.c.epsilon: return self.r.choice(pair)
  vals=[self.w[x]+self.r.gauss(0,self.c.sensory_noise) for x in pair]
  return pair[0] if vals[0]>=vals[1] else pair[1]
 def reward(self, food):
  # Latent reward is declared synthetic, based on sugar/fruitiness-style priors only.
  return max(0,min(1,FOODS[food][1]+self.r.gauss(0,.12)))
 def run(self):
  names=list(FOODS); rows=[]
  for t in range(1,self.c.trials+1):
   pair=tuple(self.r.sample(names,2)); pick=self.choose(pair); rew=self.reward(pick)
   pe=rew-self.w[pick]; self.w[pick]+=self.c.alpha*pe
   rows.append(dict(trial=t,left=pair[0],right=pair[1],choice=pick,reward=round(rew,4),prediction_error=round(pe,4)))
  return rows

def experiment(outdir, seeds=range(20), trials=240):
 import os
 os.makedirs(outdir,exist_ok=True); allruns=[]
 for seed in seeds:
  fly=MetaFly(Config(seed=seed,trials=trials)); rows=fly.run(); allruns.append((seed,fly.w,rows))
 with open(os.path.join(outdir,'trials.csv'),'w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=allruns[0][2][0]); wr.writeheader()
  for seed,_,rows in allruns:
   for r in rows: wr.writerow({'seed':seed,**r} if False else r)
 summary={x:{'mean_value':sum(w[x] for _,w,_ in allruns)/len(allruns),
             'top_rank_rate':sum(max(w,key=w.get)==x for _,w,_ in allruns)/len(allruns)} for x in FOODS}
 with open(os.path.join(outdir,'summary.json'),'w') as f: json.dump(summary,f,indent=2)
 return summary,allruns
