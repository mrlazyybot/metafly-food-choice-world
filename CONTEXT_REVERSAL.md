# Context-dependent reversal: storage versus retrieval

## Question and preregistered hypotheses

Can an artificial learner preserve two opposing food-choice policies and select between them with context cues? This experiment tests retrieval, not just persistence of a number.

- **H1:** With informative cues, the contextual dual-trace agent will store a banana policy in context X and a cucumber policy in context Y, then retrieve the matching policy in frozen probes.
- **H2:** Retrieval advantage and alternating-context performance will fall as cue reliability approaches chance (1.00, 0.90, 0.75, 0.50).
- **H3:** Removing context or shuffling cues will remove cue-specific retrieval. Removing the slow trace tests whether durable retention and reacquisition savings depend on the slow component.

## Design

Each of 200 deterministic seeds runs every agent and reliability condition. Acquisition rewards banana in context X. Reversal rewards cucumber in context Y. Frozen, balanced probes present the correct cue and a counterfactual wrong cue without learning. Twelve alternating context blocks then measure immediate switch cost. Finally, context X returns for reacquisition.

Controls and ablations:

1. `context_dual_trace`: cue-indexed fast and slow traces.
2. `ablation_no_context`: the same learner with one shared state.
3. `ablation_fast_only`: cue-indexed learning without the slow trace.
4. `control_shuffled_cue`: context architecture given cues independent of the true context.

Reward and choice random streams are isolated. All conditions use the same declared synthetic reward distribution and exploration rate. Summary intervals are empirical 2.5th to 97.5th percentiles across seeds. Paired seed-level differences against the no-context baseline are included. They describe this simulation and are not animal-level confidence intervals.

## Metrics that separate storage from retrieval

- **Storage:** after reversal, the context-X value contrast still favors banana and the context-Y contrast favors cucumber.
- **Correct-cue retrieval:** frozen-policy accuracy when cue and latent context match.
- **Wrong-cue accuracy:** a counterfactual probe using the other cue.
- **Cue retrieval advantage:** correct-cue minus wrong-cue accuracy. A stored variable alone cannot pass this test.
- **Context-switch cost:** overall accuracy minus first-three-trial accuracy after each context change.
- **Reacquisition savings:** initial trials-to-criterion minus trials-to-criterion when context X returns.

## Reproduce

```bash
python run_context_reversal.py
pytest -q
```

Outputs in `results/context_reversal/`:

- `context_reversal.png` and `.svg`: publication-ready overview
- `context_metrics.csv`: one row per seed, agent and cue reliability
- `learning_curves.csv`: compact 10-trial checkpoints, rather than a huge opaque trial dump
- `context_summary.json`: definitions, configuration, empirical intervals and paired effects

## Scientific boundary

This is an artificial connectome-inspired reinforcement-learning model. Its contexts, traces, food labels and rewards are engineering variables. It does not reconstruct the MaleCNS, identify a real memory mechanism, predict real fruit-fly behavior, or provide evidence about consciousness. Biological tests would require measured cues and behavior, circuit mapping, perturbations, held-out animals and comparison with simpler fitted models.

## Results from the committed 200-seed run

At perfect cue reliability, the contextual agent stored both policies in every seed, reached 1.00 correct-cue frozen-probe accuracy, a 1.00 counterfactual cue advantage, and 0.95 alternating-context accuracy. The no-context and shuffled-cue controls stayed near chance and had no cue advantage. As reliability fell, the contextual agent's retrieval advantage declined to 0.415 at 0.90, 0.070 at 0.75, and approximately zero at chance reliability. Alternating-context accuracy likewise fell from 0.950 to 0.460.

The slow-trace ablation closely matched the full contextual model on the main retrieval measures. Thus this implementation supports context-indexed retrieval, but does **not** support a claim that its slow trace is necessary under these phase lengths. No-context and shuffled-cue controls show that the effect depends on informative cue indexing rather than generic reversal learning. Cue corruption also reveals a limit: noisy cues update the wrong context state, so reliable dual-policy storage deteriorates rapidly. These are properties of this declared artificial model, not findings about real flies.
