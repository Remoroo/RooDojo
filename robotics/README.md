# Robotics · RooDojo

Robotics & perception workflows.

| Workflow | Domain | Headline metric | Status | Best |
|---|---|---|---|---|
| [`eye-in-hand-calibration`](./eye-in-hand-calibration) | hand-eye calibration in MuJoCo | `trans_std_mm` on locked val set, target < 1 mm | **solved** | **0.17 mm** at commit `9581ef5` (35 logged experiments) |

## The eye-in-hand story

This workflow is the page's clearest example of *honest iteration*. The
engine spent **27 experiments plateaued around 47 mm** — many of them logged
as `regress` or `discard` — before breaking through:

- Baseline (locked val set): `55.66 mm`
- Plateau period: 27 experiments at ~47–55 mm, all logged with their
  hypothesis and outcome
- Breakthrough at `2d563e2`: `9.54 mm` (IPPE multi-sol + depth filter)
- Final solve at `9581ef5`: **`0.17 mm`** (depth-corrected PnP +
  geometric prior + global bundle adjustment)

The full trace is in `eye-in-hand-calibration/results.tsv`.

## Universal contract

Same shape as the RL workflows: locked harness, locked validation pose set
(`VAL_*` constants in `run.py`), append-only `results.tsv`, one commit per
experiment. The agent may edit solvers, ensembling, outlier rejection,
samplers and adaptive collection; it may not edit the sensor pipeline,
metrics, scene, or the `VAL_*` poses.
