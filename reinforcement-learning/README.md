# Reinforcement Learning · RooDojo

Two RL workflows the Remoroo engine iterates on.

| Workflow | Env | Headline metric | Status | Best |
|---|---|---|---|---|
| [`ppo-bipedal-hardcore`](./ppo-bipedal-hardcore) | BipedalWalkerHardcore-v3 (Box2D) | `s2_avg` reward, target ≥ 300 | iterating | **166.58** at commit `84f212b` (Stage-1 nailed at 291.15) |
| [`dog-run-locomotion`](./dog-run-locomotion) | dm_control / dog-run-v0 (MuJoCo) | `s2_avg` reward, target ≥ 700 | iterating | **169.34** at commit `bfab11b` (baseline PPO) |

## Universal contract

Every RL workflow in RooDojo follows the same shape so the engine can pick up
any of them with no special-casing:

1. **One entry point.** `python run.py` (or `python ppo_agent.py`) reads
   the `program.md` contract, runs the experiment, writes artifacts.
2. **Locked harness.** `program.md` declares which files the agent may edit
   (algorithm, hyperparameters, curriculum) and which it may not (env wrappers,
   reward function, validation schedule). Tampering with locked files invalidates
   cross-commit comparisons.
3. **Append-only `results.tsv`.** Every run — keep, regress, neutral, crash —
   appends one row. Missing rows are bugs. The history is the receipt.
4. **One headline metric per workflow.** Plus disclosed cross-checks that
   never inform the optimiser. Anti-gaming by construction.

The two workflows here use the same shape but solve different problems:
`ppo-bipedal-hardcore` is a **continuous-control** problem in 2D Box2D physics;
`dog-run-locomotion` is a **high-DoF locomotion** problem in MuJoCo.
