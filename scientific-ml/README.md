# Scientific ML

Tabular + scientific-data workflows where there's a published reference
result and we want to push the *constrained* version of the same
benchmark.

## Workflows

| Status | Workflow | Headline | Constraint |
|---|---|---|---|
| iterating | [`higgs-boost`](./higgs-boost) | ROC AUC ≥ 0.733 (shallow Baldi) → 0.880 (deep Baldi) on canonical Baldi 2014 test split | 1 CPU core · 4 GB RAM · 5 min · seed locked |

The point isn't to claim a new SOTA on the HIGGS dataset — Baldi 2014's
deep-net+features result of 0.880 took hours on a multi-GPU cluster.
The point is: **how close can you get on one Mac core in five
minutes?** That ratio is the interesting metric for an autonomous
research engine.
