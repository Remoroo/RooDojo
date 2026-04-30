# Vision

Constrained-vision workflows. Where the published lineage cares about
*how cheap* you can make a strong model.

## Workflows

| Status | Workflow | Headline | Constraint |
|---|---|---|---|
| iterating | [`cifar10-speedrun`](./cifar10-speedrun) | ≥ 95.0 % CIFAR-10 top-1 | ≤ 1 M params · ≤ 15 min on Apple Silicon · seed locked |

The interesting Pareto point isn't "best CIFAR-10 ever" — it's
**best model under a strict budget**. Public lineage to beat:

- DAWNBench (Stanford) — fastest-to-94 % leaderboard.
- David Page — *How to train your ResNet*.
- tysam-code — *hlb-CIFAR10*.

We deliberately picked tighter constraints than any of those (param cap
*and* a 15-minute Mac budget) so the headline is "what's the best you
can do on a laptop in the time it takes to brew coffee?".
