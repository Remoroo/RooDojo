# RooDojo

> The training dojo for Remoroo. Seven workflow benchmarks, one universal
> contract, every commit a logged experiment.

[Remoroo](https://www.remoroo.com) is an autonomous coding agent that reads a
codebase, proposes a change, runs it, and keeps what improves the headline
metric. **RooDojo** is the curated set of workflow benchmarks the engine
iterates on in public. Every workflow uses the same harness shape so the
engine can pick any of them up with no special-casing — and so a third party
can audit progress with one glance.

This is **not** a software-engineering test suite. For the SWE benchmark
(47 tasks across four difficulty tiers, 95.2% pass rate), see
[`Remoroo/remoroo_benchmark`](https://github.com/Remoroo/remoroo_benchmark).
RooDojo is for the workflows where the loop is **research-shaped**:
real metric, real eval set, real iteration log.

## The seven workflows

| # | Workflow | Domain | Headline metric | Status | Best |
|---|---|---|---|---|---|
| 1 | [Eye-in-Hand Calibration](./robotics/eye-in-hand-calibration) | Robotics · Perception | `trans_std_mm` (target < 1 mm) | **solved** | **0.17 mm** |
| 2 | [PPO · BipedalWalkerHardcore](./reinforcement-learning/ppo-bipedal-hardcore) | RL · Control | s2 avg reward (target ≥ 300) | iterating | 166.58 |
| 3 | [Quadruped (dog_run)](./reinforcement-learning/dog-run-locomotion) | RL · Locomotion | s2 avg reward (target ≥ 700) | iterating | 169.34 (baseline) |
| 4 | [CIFAR-10 Speedrun](./vision/cifar10-speedrun) | Vision · Constrained | top-1 acc on locked CIFAR-10 test (target ≥ 95%) | iterating | — |
| 5 | [Higgs Boost](./scientific-ml/higgs-boost) | Scientific ML · Tabular | ROC AUC on Baldi 2014 test split (target ≥ 0.733 → 0.880) | iterating | — |
| 6 | [Neural Voice Synthesis (TTS)](./speech/tts-neural-voice) | Speech · TTS | `mel_recon_loss` | open frontier | — |
| 7 | [Speech Recognition (STT)](./speech/asr-speech-recognition) | Speech · STT | WER (target ≤ 5%) | open frontier | — |

Status taxonomy:

- **Solved** — the engine reliably hits the target metric.
- **Iterating** — engine is in the loop, with logged experiments and partial
  progress. Some entries here have *documented plateaus* (eye-in-hand spent
  27 experiments stuck at ~47 mm before the breakthrough) — the failure log
  is part of the receipt, not a bug to hide.
- **Open frontier** — harness locked, eval set defined, baseline pending.

## Universal contract

Every workflow follows the same five rules. They are what makes "the engine
got better at X" mean something across wildly different domains.

1. **One entry point.** `python run.py` — or whatever the workflow's
   `program.md` declares. No multi-script orchestration.
2. **Locked harness.** `program.md` lists the files the agent may edit
   (algorithm, hyperparameters, model architecture, sampling strategy) and
   the files it may not (validation set, sensor / data pipeline, metrics,
   scene). Tampering with locked files invalidates cross-commit comparisons.
3. **Locked validation set.** Each workflow has a `VAL_*` block (poses,
   utterances, episodes, image splits, row indices) committed at a known
   seed. Editing it is a benchmark-breaking change.
4. **Append-only `results.tsv`.** Every run — keep, regress, neutral, crash,
   budget_exceeded — appends one row, with the commit hash and a one-line
   description. Missing rows are bugs.
5. **One headline metric.** Plus optional cross-checks that never inform the
   optimiser. Anti-gaming by construction.

For the constrained workflows (CIFAR-10 Speedrun, Higgs Boost) the contract
also includes **explicit budgets** — wall-clock, parameter count, CPU /
memory caps — enforced by the harness, not by trust. Cheating the budget
shows up as a `time_exceeded` / `params_exceeded` / `mem_exceeded` row in
the trace.

## Repo layout

```
RooDojo/
├── README.md                              ← this file
├── .gitignore
├── reinforcement-learning/
│   ├── README.md
│   ├── ppo-bipedal-hardcore/              ← Stage-1 → Stage-2 PPO on Box2D
│   └── dog-run-locomotion/                ← dm_control 38-DoF quadruped
├── robotics/
│   ├── README.md
│   └── eye-in-hand-calibration/           ← MuJoCo hand-eye calibration
├── vision/
│   ├── README.md
│   └── cifar10-speedrun/                  ← ≤ 1 M params · 15 min on Mac
├── scientific-ml/
│   ├── README.md
│   └── higgs-boost/                       ← 1 CPU · 4 GB · 5 min · Baldi 2014
└── speech/
    ├── README.md
    ├── tts-neural-voice/                  ← scaffold (open frontier)
    └── asr-speech-recognition/            ← scaffold (open frontier)
```

Each workflow folder contains, at minimum:

- `program.md` — the contract (entry point, headline metric, locked files,
  budgets where applicable).
- `results.tsv` — append-only experiment trace.
- `harness.py` (constrained workflows) — enforces budgets, never edited.
- The code the agent edits (`*.py`, `*.xml`).

Heavy artifacts — `__pycache__/`, `.venv/`, raw datasets, multi-megabyte
checkpoints, `renders/`, `artifacts/` — are excluded by `.gitignore`. The
repo stays small enough to clone in seconds; checkpoints and CIFAR-10 /
HIGGS data can be regenerated from a clean run.

## Running a workflow

Each subdirectory is self-contained. From the workflow folder:

```bash
cd reinforcement-learning/ppo-bipedal-hardcore
python ppo_agent.py        # or whatever `program.md` declares
```

For the constrained workflows the entry point is uniform:

```bash
cd vision/cifar10-speedrun        # or scientific-ml/higgs-boost
pip install -r requirements.txt
python run.py
```

The constrained workflows target Apple Silicon by default — CIFAR-10 uses
PyTorch MPS, Higgs runs on a single CPU core via pinned BLAS / OpenMP
threads.

## Submodules + auto-bump

Each workflow lives in its own GitHub repo (independent history,
independent `baseline` tag). RooDojo tracks them as submodules — that
way the engine's per-experiment commit/reset loop operates inside one
workflow without touching siblings.

The submodule pins in this repo are kept current automatically. Each
workflow repo carries a `.github/workflows/bump-roodojo.yml` that
fires on every push to its `main` branch and updates the gitlink in
RooDojo. Result: pushes from the engine show up here within ~30
seconds — no manual `cd .. && git submodule update && git push` ever
again.

One-time setup (per maintainer who wants auto-bump enabled):

```bash
# 1. Create a fine-grained PAT at github.com/settings/personal-access-tokens
#    - Repository access: Remoroo/RooDojo
#    - Permissions: Contents (read+write)
# 2. Distribute it as ROODOJO_TOKEN to each workflow repo:
./scripts/setup-auto-bump.sh <pat-token>
```

Cloning RooDojo with all workflows checked out:

```bash
git clone --recurse-submodules https://github.com/Remoroo/RooDojo.git
# or, on an existing clone:
git submodule update --init --recursive
```

To pull the latest pinned state of every workflow:

```bash
git pull && git submodule update --recursive
```

## How to read a workflow

Open the workflow folder. Read three files in order:

1. `program.md` — what the task is, what the metric is, what's locked,
   what the budget is.
2. `results.tsv` — the experiment trace, every keep / regress / discard
   the engine logged with its commit hash.
3. The latest `keep` row's commit (in this repo's git history) for the
   actual code change.

The story of any workflow is the diff between the baseline and the latest
keep row, with the trace in between explaining how the engine got there.

## Adding a new workflow

A new workflow earns its place when it satisfies all five contract rules.
The shortest credible path:

1. Pick a problem with a single scalar metric.
2. Lock a validation / test set at a known seed.
3. Write a 1–2 page `program.md` declaring the entry point, headline metric,
   editable / locked files, and any explicit budgets.
4. Implement `harness.py` if budgets exist — they have to be enforced in
   code, not in prose.
5. Commit an empty `results.tsv` with the column header.
6. Open a PR. The engine will pick it up.

## License

Code: MIT. Datasets and trained checkpoints under their respective upstream
licenses (Box2D BipedalWalker, dm_control, CIFAR-10, UCI HIGGS, public
TTS / ASR corpora).

## See also

- [www.remoroo.com/benchmarks](https://www.remoroo.com/benchmarks) — the
  showcase, with status badges, progress bars and the full SWE catalog.
- [www.remoroo.com/try](https://www.remoroo.com/try) — watch the engine
  iterate on a fresh PPO run live.
