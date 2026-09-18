import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from metafly.core import MetaFly,Config,FOODS,experiment

def test_reproducible():
 a=MetaFly(Config(seed=3,trials=30)).run(); b=MetaFly(Config(seed=3,trials=30)).run(); assert a==b

def test_values_bounded():
 f=MetaFly(Config(seed=1,trials=500)); f.run(); assert all(0<=v<=1 for v in f.w.values())

def test_learning_signal():
 f=MetaFly(Config(seed=2,trials=2000,epsilon=.2)); f.run(); assert f.w['banana']>f.w['broccoli']

def test_food_set_has_both_groups():
 assert {'banana','apple','orange','grape'}<=FOODS.keys(); assert {'carrot','broccoli','cucumber','tomato'}<=FOODS.keys()

from metafly.reversal import ReversalConfig,DualTraceAgent,run_one,run_experiment

def test_reversal_reproducible(tmp_path):
 a=run_one(DualTraceAgent(ReversalConfig(seed=9)),ReversalConfig(seed=9))
 b=run_one(DualTraceAgent(ReversalConfig(seed=9)),ReversalConfig(seed=9))
 assert a==b

def test_reversal_phases_and_metrics():
 c=ReversalConfig(seed=4); rows,m,_=run_one(DualTraceAgent(c),c)
 assert {r['phase'] for r in rows}=={'acquisition','reversal','extinction','recovery_probe','reacquisition'}
 assert 0<=m['old_value_retention_ratio']
 assert -1<=m['spontaneous_recovery']<=1

def test_no_slow_ablation_has_zero_slow_trace():
 c=ReversalConfig(seed=2); a=DualTraceAgent(c,use_slow=False); run_one(a,c)
 assert a.slow['banana']==a.slow['cucumber']==0

def test_small_experiment_writes_outputs(tmp_path):
 _,metrics,summary=run_experiment(tmp_path,seeds=range(3),config_overrides={'acquisition_trials':40,'reversal_trials':40,'extinction_trials':30,'reacquisition_trials':40})
 assert len(metrics)==15 and summary['seed_count']==3
 assert (tmp_path/'reversal_summary.json').exists() and (tmp_path/'reversal_trials.csv').exists()

from metafly.context_reversal import ContextConfig,ContextAgent,run_one as run_context_one,run_experiment as run_context_experiment

def test_context_reversal_reproducible():
 c=ContextConfig(seed=11,cue_reliability=.9)
 assert run_context_one(ContextAgent(c),c)==run_context_one(ContextAgent(c),c)

def test_context_storage_and_retrieval_with_perfect_cues():
 c=ContextConfig(seed=3,cue_reliability=1.0)
 _,m=run_context_one(ContextAgent(c),c)
 assert m['both_strategies_stored']==1
 assert m['correct_cue_retrieval_accuracy']>m['wrong_cue_accuracy']

def test_no_context_has_no_cue_advantage():
 c=ContextConfig(seed=2,cue_reliability=1.0)
 _,m=run_context_one(ContextAgent(c,use_context=False),c)
 assert m['cue_retrieval_advantage']==0

def test_context_small_experiment_outputs(tmp_path):
 _,ms,s=run_context_experiment(tmp_path,seeds=range(2),reliabilities=(1.,.5),config_overrides={'acquisition_trials':40,'reversal_trials':40,'switch_blocks':2,'switch_block_trials':6,'reacquisition_trials':40})
 assert len(ms)==16 and s['seed_count']==2
 assert (tmp_path/'context_metrics.csv').exists() and (tmp_path/'context_summary.json').exists()
