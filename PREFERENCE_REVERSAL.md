# Preference reversal: retention, suppression, and strategy loss

## Question and operational definitions

This experiment asks what an artificial learner keeps when reward contingencies reverse. "Behavior" means choosing one of two food labels. "Strategy" is used narrowly: a learned value contrast that drives exploitation. It does not mean planning, understanding, memory in a real fly, or consciousness.

The same two options are presented throughout. Banana is rewarded during acquisition; cucumber becomes rewarded during reversal; rewards are removed during extinction; an update-free rest interval is followed by reward-free probes; then the original banana contingency returns for reacquisition.

We separate observable choice from latent learned value:

- **Adaptation speed:** first reversal trial ending a 40-trial window with at least 80% choices of the new option.
- **Perseveration:** old-option choice rate in the first 30 reversal trials.
- **Retention:** old-option latent value after reversal divided by its value after acquisition.
- **Suppressed, not discarded:** at least 80% new-option choices late in reversal while at least 50% of the old slow trace remains.
- **Extinction:** reduction in learned value and choice bias while rewards are absent.
- **Spontaneous recovery:** old-option choice rate in 20 reward-free probes after rest minus its rate in the final 30 extinction trials.
- **Reacquisition savings:** trials to the original 80% acquisition criterion minus trials to the same criterion when the original contingency returns.

These thresholds were declared in code before reading the full 200-seed output. They are descriptive model diagnostics, not inferential tests on animals.

## Agents and ablations

1. **MetaFly dual trace:** fast and slow prediction-error traces. Fast learning adapts quickly and decays faster; slow learning is retained longer.
2. **Tabular Q:** one learned value per option, a simpler reinforcement-learning control.
3. **No-slow ablation:** tests whether retained values and recovery depend on the slow trace.
4. **No-fast ablation:** tests whether fast adaptation depends on the fast trace.
5. **Random:** chance-choice negative control.

All agents get the same phase lengths, reward distributions, and exploration rate where applicable. Each condition runs across 200 deterministic seeds. Random streams are reproducible and isolated by seed and agent. The code exports every trial, per-seed metrics, checkpoints, summary statistics, and a four-panel figure.

## How to run

```bash
python run_reversal_experiment.py
pytest -q
```

Outputs are written to `results/preference_reversal/`:

- `preference_reversal.png`: behavioral and metric overview
- `reversal_trials.csv`: trial-level choices, rewards, values, and trace states
- `reversal_metrics.csv`: one metric row per seed and agent
- `reversal_summary.json`: means, standard deviations, and seed-level 2.5th–97.5th percentiles
- `checkpoints.json`: latent values/traces at phase boundaries

## Reading the result responsibly

The dual-trace result can show a clean dissociation: current choices may favor the new option while an older slow value remains. In this model, that supports the precise claim that the old learned value was **suppressed rather than numerically erased**. Recovery after rest and faster reacquisition are converging software diagnostics of dormancy. They do not prove a biological memory mechanism.

A value retained in a variable is not by itself evidence of a retained real-world strategy. Conversely, a chance-level choice rate does not prove erasure because competing learned values can mask one another. The ablations make those alternatives visible within the declared model.

## Scientific boundary

The architecture is connectome-inspired only at a high level. Fast and slow traces are explicit engineering hypotheses, not neuron-level reconstructions. Food labels and rewards are synthetic. The experiment does not establish how real fruit flies remember, forget, strategize, or behave, and it says nothing about consciousness. Biological claims would require measured stimuli and behavior, circuit-specific perturbations, preregistration, held-out animals, and fits against simpler models.
