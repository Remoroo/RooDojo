# PPO Research Loop — dm_control Dog Run

Autonomous research loop for improving a PPO agent on **dm_control/dog-run-v0** via a **two-stage curriculum**: first learn basic locomotion on the easier **dog-walk-v0**, then fine-tune for speed on the harder **dog-run-v0** — all from a single command. **Training targets Apple Silicon using PyTorch MPS.**

## Environment

The [dm_control dog](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/dog.py) is a fully articulated quadruped simulated in MuJoCo — 4 legs, spine, head, tail — with 3D shadows, textured ground, and proper lighting out of the box.

| Property | Value |
|----------|-------|
| Observation | Dict → **223-dim** flat (joint angles 73, joint velocities 73, actuator state 38, inertial sensors 9, z-projection 9, foot forces 12, touch sensors 4, torso pelvis height 2, torso com velocity 3) |
| Action | **38-dim** continuous [-1, 1] (joint torques) |
| Episode length | 1000 steps (25 s at 40 Hz control) |
| Reward per step | [0, 1] — higher = faster horizontal velocity toward target |
| Max episode return | ~1000 |
| Random policy baseline | ~5 (dog-run), ~13 (dog-walk), ~16 (dog-stand) |

## Configuration

Single source of truth for runtime parameters. Edit here; all code and docs read from this file.

| Parameter | Value | Description |
|-----------|-------|-------------|
| STAGE1_ENV | dm_control/dog-walk-v0 | Easier env for basic locomotion |
| STAGE2_ENV | dm_control/dog-run-v0 | Final evaluation env |
| STAGE1_STEPS | 2000000 | Environment steps for Stage 1 |
| STAGE2_STEPS | 8000000 | Environment steps for Stage 2 |
| STAGE1_BUDGET | 1800 | Wall-clock seconds for Stage 1 (30 minutes) |
| STAGE2_BUDGET | 7200 | Wall-clock seconds for Stage 2 (2 hours) |
| STAGE1_GOAL | 600 | Avg reward (last 100 eps) to early-exit Stage 1 |
| STAGE2_GOAL | 700 | Avg reward (last 100 eps) to consider Run solved |
| HIDDEN_DIM | 512 | Network hidden layer width |
| LR | 3e-4 | Initial learning rate (linear decay per stage) | 
| N_ENVS | 8 | Parallel vectorized environments per stage |

```
STAGE1_ENV=dm_control/dog-walk-v0
STAGE2_ENV=dm_control/dog-run-v0
STAGE1_STEPS=2000000
STAGE2_STEPS=8000000
STAGE1_BUDGET=1800
STAGE2_BUDGET=7200
STAGE1_GOAL=600
STAGE2_GOAL=700
HIDDEN_DIM=512
LR=3e-4
N_ENVS=8
```
<!-- Edit the block above to change runtime parameters. -->

## Two-stage curriculum rationale

**dog-run-v0** rewards fast horizontal velocity. Training from scratch is wasteful because the agent first needs to learn to stand and produce a coordinated gait — skills the easier **dog-walk-v0** (which rewards moderate forward progress) teaches much faster.

1. **Stage 1 — dog-walk-v0** (~2M steps, 30 min): Learn a stable walking gait. Target avg reward ≥ 600. This gives the policy a working motor prior and populates the observation normalizer with meaningful statistics.

2. **Stage 2 — dog-run-v0** (~8M steps, 2 hours): Fine-tune for speed. The agent already walks, so it can focus on accelerating. Target avg reward ≥ 700.

**What transfers**: model weights, optimizer state, and the running observation normalizer. The LR schedule resets per stage so Stage 2 starts with full learning rate.

**What to watch**: Stage 2 may show an initial dip as the policy tries to increase speed beyond its walking gait. If reward drops below 200 and does not recover within ~500k steps, the curriculum may need a gentler transition (try dog-stand-v0 → dog-walk-v0 → dog-run-v0 three-stage).

## Dependencies

```
torch
gymnasium
numpy
shimmy[dm-control]
```

## Experimentation

Launch with:

```bash
uv run ppo_agent.py > run.log 2>&1
```

The script runs **both stages sequentially** and prints labeled output:


A checkpoint (`checkpoint.pt`) is saved between stages so you can resume Stage 2 independently if needed.

Extract results:

```bash
tail -5 run.log | grep -E "FINAL|Stage|Goal reached"
grep "Avg Reward" run.log | grep "Stage 1" | awk -F': ' '{print $2}' | awk -F',' '{print $1}' | sort -n | tail -1
grep "Avg Reward" run.log | grep "Stage 2" | awk -F': ' '{print $2}' | awk -F',' '{print $1}' | sort -n | tail -1
```

**What you CAN modify** in `ppo_agent.py`:
- Network architecture, hyperparameters, optimizer, rollout config, exploration, reward shaping, observation handling, stage split, stage budgets — everything.
- Change agent to somthing much more powerful. 


## Logging results

Log to `results.tsv` (tab-separated):

```
commit	s1_avg	s2_avg	steps_k	status	description
```

1. `commit` — git hash (short, 7 chars)
2. `s1_avg` — best avg reward in Stage 1 (dog-walk)
3. `s2_avg` — best avg reward in Stage 2 (dog-run) — **this is the primary metric**
4. `steps_k` — total steps (both stages) in thousands
5. `status` — `keep`, `discard`, or `crash`
6. `description` — what was tried

Example:

```
commit	s1_avg	s2_avg	steps_k	status	description
a1b2c3d	180.10	55.20	10000	keep	baseline two-stage curriculum
c3d4e5f	80.00	20.00	10000	discard	double hidden_dim to 1024
d4e5f6g	0.00	0.00	0	crash	MPS error on LayerNorm
```

## The experiment loop

LOOP FOREVER:

1. Check git state and branch.
2. Read `results.tsv` — what's been tried, current best `s2_avg`.
3. **Formulate hypothesis**: what change do you expect to improve Stage 2 avg reward, and why? Consider whether the change should target Stage 1 learning, Stage 2 adaptation, or both.
4. Edit `ppo_agent.py`.
5. `git commit`.
6. Run: `uv run ppo_agent.py > run.log 2>&1`.
7. Extract: `tail -10 run.log; grep "Avg Reward" run.log | tail -5`.
8. If crash, check `tail -n 50 run.log`. Fix MPS issues with narrow fallbacks. Try up to 2 fixes before moving on.
9. Record in `results.tsv`.
10. If `s2_avg` improved → keep. Otherwise → `git reset --hard` to previous best.

**Stage-specific tuning**: You can change the step split (e.g. 1M/9M), budgets, goals, or hyperparameters independently per stage. If Stage 1 already converges quickly, shift more steps to Stage 2.

**Timeout**: Total wall time should stay under `STAGE1_BUDGET + STAGE2_BUDGET` (2.5 hours). Kill if > 1.5× that.

**NEVER STOP**: Run indefinitely until manually stopped.

## Convergence recipe

### 1) Diagnose before changing

After each run, check `run.log` for:
- **Stage 1 not converging**: If Stage 1 avg stays below 50, basic walking isn't learned — fix Stage 1 first (LR, network, rollout).
- **Stage 2 initial collapse**: Dip at Stage 2 start is normal; if it never recovers after 1M steps, the policy may be too specialized. Try shorter Stage 1 or higher entropy in Stage 2.
- **Entropy collapse**: Policy becomes deterministic too early in either stage. With 38 action dims, entropy collapse is a major risk.
- **Value instability**: Loss spikes, reward cliffs — common with large action spaces.
