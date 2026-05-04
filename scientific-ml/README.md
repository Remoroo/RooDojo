# Scientific ML

Tabular + scientific-data workflows where there's a published reference
result and we want to push the *constrained* version of the same
benchmark.

## Workflows

| Status | Workflow | Headline | Constraint |
|---|---|---|---|
| iterating | [`higgs-boost`](./higgs-boost) | ROC AUC ≥ 0.733 (shallow Baldi) → 0.880 (deep Baldi) on canonical Baldi 2014 test split | 1 CPU core · 4 GB RAM · 5 min · seed locked |
| iterating | [`variant-triage`](./variant-triage) | ROC AUC ≥ 0.85 (REVEL floor) → 0.92 (AlphaMissense class) on ClinVar 2024+ time-holdout | 1 CPU core · 4 GB RAM · 10 min · seed locked |

The point isn't to claim a new SOTA on either dataset. The published
references both ran on much bigger compute:

- **Baldi 2014's** deep-net + handcrafted features (HIGGS AUC 0.880)
  took hours on a multi-GPU cluster.
- **AlphaMissense (Cheng 2023, *Science*)** — zero-shot pathogenicity
  scores for 71 M possible human missense variants — was trained on
  Google's TPU pods as a protein-language-model derivative.

The point is: **how close can the engine get on one Mac core in a
handful of minutes?** That ratio — `published_SOTA / constrained_run`
— is the interesting metric for an autonomous research engine.
Closing it means the engine found engineering ideas (feature
interactions, calibration, hyperparameter choices) that matter
regardless of the compute budget.
