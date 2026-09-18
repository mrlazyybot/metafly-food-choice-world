# MetaFly food-choice prototype

A reproducible toy "meta world" for testing preference learning across fruits and vegetables. The likely dataset in the request is **MaleCNS v1.0**, released June 8, 2026, with Google Research as a collaborator. It is an adult male **fruit fly** (*Drosophila melanogaster*), not a bee. The primary data are from HHMI Janelia and collaborators.

## Scientific boundary

A connectome is a structural wiring diagram. It is not a mind upload and does not contain a functioning animal's membrane dynamics, receptor responses, internal state, learned weights, memories, or consciousness. This prototype therefore does **not** claim to emulate the MaleCNS or predict real fruit-fly food preferences. It uses a small, declared reinforcement-learning abstraction inspired by known fly organization: odor-like channels, a sparse associative stage, prediction-error learning, and action choice. The fruit/vegetable reward values are synthetic hypotheses for testing the experiment pipeline.

## What the experiment measures

Across 20 deterministic seeds and 240 two-choice trials per run:
- learned value for banana, apple, orange, grape, tomato, carrot, broccoli, and cucumber;
- how often each item finishes first;
- trial-by-trial choice, reward, and prediction error.

The included result should be read as a software check: under the declared synthetic reward assumptions, does the learner acquire their ranking? It is not biological evidence.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py
pytest -q
```

Outputs go to `results/`. Seeds and parameters live in `src/metafly/core.py`.

## How to turn this into research

1. Replace synthetic odor vectors and rewards with measured headspace chemistry, concentration, hunger state, sex, age, and controlled behavioral assays.
2. Pre-register a balanced two-choice design, randomize left/right and presentation order, add odorless and caloric controls, and separate innate preference from post-ingestive reinforcement.
3. Use the MaleCNS/FlyWire data only for explicitly mapped sensory, mushroom-body, dopamine, and descending pathways. State every unmapped transfer function.
4. Fit parameters on one assay set and test on held-out animals. Compare against simpler non-connectome baselines.
5. Report uncertainty and failures. Do not call model behavior consciousness, preference of the source specimen, or proof of biological fidelity.

## Provenance and licensing

- Dataset home/download: https://male-cns.janelia.org/download/ (MaleCNS v1.0, CC-BY)
- Google Research release note: https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
- Project code/site repository: https://github.com/janelia-flyem/male-cns (GPL-3.0)
- Analysis access package: https://github.com/natverse/malecns (GPL-3.0)
- Derived paper products: https://github.com/flyconnectome/2025malecns (no repository-level license file observed; use primary CC-BY data and check individual artifact terms)
- Female FlyWire comparison and access tutorial: https://github.com/seung-lab/FlyConnectome
- Female FlyWire v783 connectivity archive: https://zenodo.org/records/10676866 (CC-BY-4.0)
- Connectome definition/dataset catalog: https://codex.flywire.ai/faq
- Olfactory circuit evidence: https://pmc.ncbi.nlm.nih.gov/articles/PMC10366338/

This prototype's original code is MIT-licensed. It does not redistribute MaleCNS data.

## Preference-reversal extension

`run_reversal_experiment.py` adds a 200-seed comparison of a dual-timescale MetaFly learner, tabular Q-learning, random choice, and fast/slow-trace ablations. It measures adaptation, perseveration, latent-value retention, behavioral suppression versus numerical discarding, extinction, spontaneous recovery, and reacquisition savings. See [`PREFERENCE_REVERSAL.md`](PREFERENCE_REVERSAL.md) for definitions and interpretation limits.

```bash
python run_reversal_experiment.py
```
