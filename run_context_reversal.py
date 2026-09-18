from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'src'))
from metafly.context_reversal import run_experiment
import matplotlib.pyplot as plt
import numpy as np
out=Path(__file__).parent/'results'/'context_reversal'
rows,metrics,summary=run_experiment(out)
agents=['context_dual_trace','ablation_no_context','ablation_fast_only','control_shuffled_cue']
labels=['Context dual trace','No context','Fast only','Shuffled cue']; colors=['#5b3f95','#db7b2b','#3976a8','#888888']; rels=[1.,.9,.75,.5]
fig,axs=plt.subplots(2,2,figsize=(13,9),facecolor='#f8f7f2')
for ax in axs.flat: ax.set_facecolor('#fffefd'); ax.grid(axis='y',alpha=.2)
def means(field,a): return [np.mean([m[field] for m in metrics if m['agent']==a and m['reliability']==r and m[field] is not None]) for r in rels]
for a,l,c in zip(agents,labels,colors): axs[0,0].plot(rels,means('correct_cue_retrieval_accuracy',a),'-o',label=l,color=c); axs[0,1].plot(rels,means('cue_retrieval_advantage',a),'-o',label=l,color=c); axs[1,0].plot(rels,means('context_switch_accuracy',a),'-o',label=l,color=c); axs[1,1].plot(rels,means('old_strategy_retention_ratio',a),'-o',label=l,color=c)
for ax in axs.flat: ax.invert_xaxis(); ax.set_xlabel('Cue reliability (1.0 perfect; 0.5 chance)')
axs[0,0].set(title='Correct-cue retrieval',ylabel='Frozen-probe accuracy',ylim=(0,1.03)); axs[0,0].axhline(.5,color='black',ls=':',lw=1); axs[0,0].legend(fontsize=8)
axs[0,1].set(title='Causal cue retrieval advantage',ylabel='Correct-cue minus wrong-cue accuracy',ylim=(-.05,1.03)); axs[0,1].axhline(0,color='black',ls=':',lw=1)
axs[1,0].set(title='Alternating-context performance',ylabel='Choice accuracy',ylim=(0,1.03)); axs[1,0].axhline(.5,color='black',ls=':',lw=1)
axs[1,1].set(title='Old strategy retained after reversal',ylabel='Context-X contrast retention ratio'); axs[1,1].axhline(0,color='black',ls=':',lw=1)
fig.suptitle('MetaFly context-dependent reversal: separating storage from retrieval',fontsize=16,weight='bold')
fig.text(.5,.008,'Means across 200 deterministic seeds. Synthetic contexts and rewards in an artificial learner; not biological evidence.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.03,1,.96)); fig.savefig(out/'context_reversal.png',dpi=180); fig.savefig(out/'context_reversal.svg')
print(out/'context_reversal.png')
