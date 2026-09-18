from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'src'))
from metafly.reversal import run_experiment
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).parent/'results'/'preference_reversal'
rows,metrics,summary=run_experiment(out)
agents=['metafly_dual_trace','tabular_q','ablation_no_slow','ablation_no_fast','random']
labels=['MetaFly dual trace','Tabular Q','No slow trace','No fast trace','Random']
colors=['#6842a6','#278a8a','#ef8a47','#5975c1','#8b8b8b']
fig,axs=plt.subplots(2,2,figsize=(13,9),facecolor='#fbfaf6')
for ax in axs.flat: ax.set_facecolor('#fffdf8'); ax.grid(axis='y',alpha=.22)
# Learning curves, 20-trial bins
phases=['acquisition','reversal','extinction','recovery_probe','reacquisition']; targets={'acquisition':'banana','reversal':'cucumber','reacquisition':'banana'}
for agent,label,color in zip(agents,labels,colors):
    rs=[r for r in rows if r['agent']==agent]; xs=[]; ys=[]; cursor=0
    for p in phases:
        pr=[r for r in rs if r['phase']==p]; n=max(r['phase_trial'] for r in pr)
        for b in range(1,n+1,20):
            br=[r for r in pr if b<=r['phase_trial']<b+20]
            # Plot probability of the currently adaptive choice; extinction/probe show old-choice probability.
            target=targets.get(p,'banana'); xs.append(cursor+b+10); ys.append(sum(r['choice']==target for r in br)/len(br))
        cursor+=n
    axs[0,0].plot(xs,ys,label=label,color=color,lw=2)
axs[0,0].set(title='Behavior across acquisition, reversal, extinction and reacquisition',ylabel='Adaptive choice rate\n(old choice during extinction/probe)',xlabel='Training/probe trials'); axs[0,0].set_ylim(0,1); axs[0,0].legend(fontsize=8)
for x,t in [(180,'reverse'),(400,'extinguish'),(520,'probe'),(540,'reacquire')]: axs[0,0].axvline(x,color='black',lw=.7,alpha=.5); axs[0,0].text(x+.5,.02,t,rotation=90,fontsize=8)
def vals(field): return [[m[field] for m in metrics if m['agent']==a and m[field] is not None] for a in agents]
for ax,field,title,ylabel in [
 (axs[0,1],'reversal_criterion','Adaptation speed after reversal','Trials to 80% criterion (lower is faster)'),
 (axs[1,0],'old_slow_trace_retention_ratio','Old slow trace retained after reversal','Retention ratio'),
 (axs[1,1],'spontaneous_recovery','Recovery of old behavior after extinction + rest','Change in old-choice rate')]:
    data=vals(field); means=[np.mean(x) if x else np.nan for x in data]; cis=[np.percentile(x,[2.5,97.5]) if x else [np.nan,np.nan] for x in data]
    err=np.array([[m-c[0] for m,c in zip(means,cis)],[c[1]-m for m,c in zip(means,cis)]])
    ax.bar(range(len(agents)),means,color=colors); ax.errorbar(range(len(agents)),means,yerr=err,fmt='none',ecolor='black',capsize=3)
    ax.set_xticks(range(len(agents)),labels,rotation=25,ha='right'); ax.set(title=title,ylabel=ylabel)
fig.suptitle('MetaFly preference reversal: retained value, suppressed behavior, and recovery',fontsize=16,weight='bold')
fig.text(.5,.005,'Artificial learning model. Error bars are seed-level 2.5th–97.5th percentiles across 200 deterministic seeds; not biological evidence.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.025,1,.96)); fig.savefig(out/'preference_reversal.png',dpi=180)
print(out/'preference_reversal.png')
