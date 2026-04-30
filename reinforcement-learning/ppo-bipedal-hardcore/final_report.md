# Remoroo v2 Run Report

**Run ID:** c751b963
**Verdict:** fail
**Actions:** 65
**Tokens:** 1,800,701
**User Cost:** Pending billing finalization

## Evidence
Run completed with verdict: fail. Best Stage 2 result was 4.92 avg reward (commit 544da2b), far below the 300 target. Multiple approaches tried:
1. Action distribution fixes (tanh squashing, raw actions) - helped Stage 1 but not Stage 2
2. Reward normalization - no significant improvement 
3. Constant LR/clip (no decay) - prevented further learning decay but didn't help convergence
4. Various entropy coefficients, batch sizes, n_envs configs
5. Different stage budgets and step allocations

Key findings:
- Stage 1 (BipedalWalker-v3) can reach 243 avg with 2M steps and LR decay
- Stage 2 (BipedalWalkerHardcore-v3) never exceeded ~5 avg reward across all experiments
- The agent cannot learn to navigate hardcore obstacles with this architecture/approach
- ~2000 steps/s throughput on MPS limits total training to ~20M steps in available time
- BipedalWalkerHardcore typically needs 10M+ steps with proper reward normalization and potentially gSDE or other exploration mechanisms not implemented here

*Trace: .remoroo/runs/c751b963/trace.jsonl*