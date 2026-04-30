import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import os
import time


class RunningMeanStd:
    """Welford online mean/variance tracker."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Separate actor and critic networks with orthogonal init
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Per-action-dimension learnable log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        actor_layers = [m for m in self.actor_net if isinstance(m, nn.Linear)]
        critic_layers = [m for m in self.critic_net if isinstance(m, nn.Linear)]
        nn.init.orthogonal_(actor_layers[-1].weight, gain=0.01)
        nn.init.orthogonal_(critic_layers[-1].weight, gain=1.0)
        # Initialize log_std to give std ~ 0.5
        nn.init.constant_(self.log_std, np.log(0.5))

    def forward(self, state):
        action_mean = self.actor_net(state)
        value = self.critic_net(state)
        return action_mean, self.log_std, value

    def get_action_and_value(self, state):
        action_mean, log_std, value = self.forward(state)
        std = torch.exp(log_std.clamp(-5, 2))
        dist = torch.distributions.Normal(action_mean, std)
        action_raw = dist.rsample()
        # Clamp for env but keep raw for log_prob consistency
        action_env = action_raw.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        return action_raw, action_env, log_prob, value

    def evaluate(self, state, action):
        """Evaluate actions stored in buffer."""
        action_mean, log_std, value = self.forward(state)
        std = torch.exp(log_std.clamp(-5, 2))
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy


class RewardNormalizer:
    """Normalize rewards using running return variance (like VecNormalize)."""
    def __init__(self, n_envs, gamma=0.99, epsilon=1e-8, clip=10.0):
        self.n_envs = n_envs
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.ret = np.zeros(n_envs, dtype=np.float64)
        self.ret_rms = RunningMeanStd(shape=())

    def normalize(self, rewards, dones):
        """Normalize a batch of rewards (n_envs,)."""
        self.ret = self.ret * self.gamma + rewards
        self.ret_rms.update(self.ret)
        normalized = rewards / (np.sqrt(self.ret_rms.var) + self.epsilon)
        normalized = np.clip(normalized, -self.clip, self.clip)
        # Reset return for done envs
        self.ret *= (1.0 - dones)
        return normalized.astype(np.float32)


class PPOAgent:
    def __init__(self, state_dim, action_dim, device='cpu', lr=3e-4,
                 hidden_dim=256, total_steps=500000):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_lr = lr
        self.total_steps = total_steps
        self.initial_clip_ratio = 0.2

        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        self.obs_normalizer = RunningMeanStd(shape=(state_dim,))

        self.clip_ratio = 0.2
        self.entropy_coef = 0.001
        self.value_coef = 0.5
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5
        self.global_step = 0
        self.obs_clip = 10.0  # Clip normalized observations

    def normalize_obs(self, obs):
        normed = (obs - self.obs_normalizer.mean) / np.sqrt(self.obs_normalizer.var + 1e-8)
        return np.clip(normed, -self.obs_clip, self.obs_clip)

    def normalize_obs_batch(self, obs_batch):
        normed = (obs_batch - self.obs_normalizer.mean) / np.sqrt(self.obs_normalizer.var + 1e-8)
        return np.clip(normed, -self.obs_clip, self.obs_clip)

    def get_actions_batch(self, states_np):
        """Batched action selection for vectorized envs.
        Returns: (raw_actions, env_actions, log_probs, values)
        """
        with torch.no_grad():
            norm = self.normalize_obs_batch(states_np)
            t = torch.FloatTensor(norm).to(self.device)
            raw, env, log_probs, values = self.actor_critic.get_action_and_value(t)
            return raw.cpu().numpy(), env.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def get_values_batch(self, states_np):
        """Batched value estimation."""
        with torch.no_grad():
            norm = self.normalize_obs_batch(states_np)
            t = torch.FloatTensor(norm).to(self.device)
            _, _, values = self.actor_critic.forward(t)
            return values.squeeze(-1).cpu().numpy()

    def compute_gae_vectorized(self, rewards, values, dones, next_values):
        """Compute GAE for multiple envs stored as (n_steps, n_envs) arrays."""
        n_steps, n_envs = rewards.shape
        advantages = np.zeros_like(rewards)
        gae = np.zeros(n_envs)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update_lr(self, step_within_stage, stage_steps):
        """Linear LR decay."""
        frac = 1.0 - step_within_stage / max(stage_steps, 1)
        frac = max(frac, 0.0)
        lr = self.initial_lr * frac
        lr = max(lr, 1e-6)  # Floor
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_clip_ratio(self, step_within_stage, stage_steps):
        """Linear clip ratio decay."""
        frac = 1.0 - step_within_stage / max(stage_steps, 1)
        frac = max(frac, 0.0)
        return self.initial_clip_ratio * frac + 0.02  # Floor at 0.02

    def update(self, states, actions, old_log_probs, advantages, returns, old_values=None,
               num_epochs=10, batch_size=64, clip_ratio=None):
        """PPO update from flat arrays of shape (total_samples, ...)."""
        self.obs_normalizer.update(states)
        norm_states = self.normalize_obs_batch(states)

        states_t = torch.FloatTensor(norm_states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        old_val_t = torch.FloatTensor(old_values).to(self.device) if old_values is not None else None

        # Normalize advantages per-batch
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        if clip_ratio is None:
            clip_ratio = self.clip_ratio

        n = len(states_t)
        total_loss = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            indices = np.random.permutation(n)
            for i in range(0, n, batch_size):
                bi = indices[i:i + batch_size]
                b_s = states_t[bi]
                b_a = actions_t[bi]
                b_olp = old_lp_t[bi]
                b_adv = adv_t[bi]
                b_ret = ret_t[bi]

                log_probs, vals, entropy = self.actor_critic.evaluate(b_s, b_a)

                ratio = torch.exp(log_probs - b_olp)
                # Clamp ratio for stability
                ratio = ratio.clamp(0.0, 10.0)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                if old_val_t is not None:
                    b_old_val = old_val_t[bi]
                    v_clipped = b_old_val + torch.clamp(vals - b_old_val, -clip_ratio * 50, clip_ratio * 50)
                    v_loss1 = (vals - b_ret).pow(2)
                    v_loss2 = (v_clipped - b_ret).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * (b_ret - vals).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

    def save(self, path):
        torch.save({
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_mean': self.obs_normalizer.mean,
            'obs_var': self.obs_normalizer.var,
            'obs_count': self.obs_normalizer.count,
            'global_step': self.global_step,
        }, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.obs_normalizer.mean = state['obs_mean']
        self.obs_normalizer.var = state['obs_var']
        self.obs_normalizer.count = state['obs_count']
        self.global_step = state.get('global_step', 0)


# ---------------------------------------------------------------------------
# Two-stage vectorized training
# ---------------------------------------------------------------------------

def run_stage(agent, env_name, num_steps, time_budget, stage_label,
              n_envs=32, goal_reward=None, rollout_steps=2048,
              num_epochs=10, batch_size=64, use_reward_norm=False):
    """
    Run one training stage with n_envs parallel environments.
    rollout_steps is per-env; total data per update = rollout_steps * n_envs.
    Returns (best_avg_reward, steps_used, timed_out).
    """
    env = SyncVectorEnv([lambda e=env_name: gym.make(e) for _ in range(n_envs)])

    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    # Reward normalizer (optional)
    reward_normalizer = RewardNormalizer(n_envs, gamma=agent.gamma) if use_reward_norm else None

    # Per-env episode tracking
    ep_reward = np.zeros(n_envs)
    ep_length = np.zeros(n_envs, dtype=int)

    # Rollout buffers: (rollout_steps, n_envs, ...)
    buf_states = np.zeros((rollout_steps, n_envs, agent.state_dim), dtype=np.float32)
    buf_actions = np.zeros((rollout_steps, n_envs, agent.action_dim), dtype=np.float32)
    buf_log_probs = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    buf_rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    buf_dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    buf_values = np.zeros((rollout_steps, n_envs), dtype=np.float32)

    state, _ = env.reset()
    step = 0
    update_step = 0
    best_avg = -float('inf')
    start_time = time.time()
    steps_per_sec_ema = 0.0

    total_rollout_per_update = rollout_steps * n_envs

    print(f"\n{'='*60}")
    print(f"  {stage_label}: {env_name}")
    print(f"  {num_steps} steps | budget {time_budget}s | {n_envs} envs")
    print(f"  rollout: {rollout_steps}/env = {total_rollout_per_update} total/update")
    print(f"  reward_norm: {use_reward_norm} | entropy: {agent.entropy_coef}")
    print(f"  gamma: {agent.gamma} | gae_lambda: {agent.gae_lambda}")
    print(f"{'='*60}\n")

    while step < num_steps:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            print(f"[{stage_label}] Time budget exhausted ({elapsed:.0f}s)")
            break

        rollout_start = time.time()

        # --- Collect rollout across all envs ---
        for t in range(rollout_steps):
            raw_actions_np, env_actions_np, log_probs_np, values_np = agent.get_actions_batch(state)

            next_state, reward, terminated, truncated, _ = env.step(env_actions_np)
            done = terminated | truncated

            buf_states[t] = state
            buf_actions[t] = raw_actions_np  # Store raw for consistent evaluation
            buf_log_probs[t] = log_probs_np
            buf_dones[t] = done.astype(np.float32)
            buf_values[t] = values_np

            # Track raw rewards for logging
            ep_reward += reward
            ep_length += 1

            # Normalize or use raw rewards
            if reward_normalizer is not None:
                buf_rewards[t] = reward_normalizer.normalize(reward, done.astype(np.float64))
            else:
                buf_rewards[t] = reward

            step += n_envs
            agent.global_step += n_envs

            for i in range(n_envs):
                if done[i]:
                    episode_rewards.append(ep_reward[i])
                    episode_lengths.append(ep_length[i])
                    ep_reward[i] = 0.0
                    ep_length[i] = 0

            state = next_state

            if step >= num_steps:
                break

        rollout_time = time.time() - rollout_start
        if rollout_time > 0:
            cur_sps = (rollout_steps * n_envs) / rollout_time
            steps_per_sec_ema = 0.9 * steps_per_sec_ema + 0.1 * cur_sps if steps_per_sec_ema > 0 else cur_sps

        # --- Compute GAE ---
        next_values = agent.get_values_batch(state)
        advantages, returns = agent.compute_gae_vectorized(
            buf_rewards, buf_values, buf_dones, next_values
        )

        # Flatten (rollout_steps, n_envs, ...) → (rollout_steps * n_envs, ...)
        n_collected = buf_states.shape[0]
        flat_states = buf_states.reshape(n_collected * n_envs, -1)
        flat_actions = buf_actions.reshape(n_collected * n_envs, -1)
        flat_log_probs = buf_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = buf_values.reshape(-1)

        # --- PPO update ---
        current_lr = agent.update_lr(step, num_steps)
        current_clip_ratio = agent.get_clip_ratio(step, num_steps)
        loss = agent.update(flat_states, flat_actions, flat_log_probs,
                            flat_advantages, flat_returns, flat_values,
                            num_epochs=num_epochs, batch_size=batch_size,
                            clip_ratio=current_clip_ratio)
        update_step += 1

        # --- Logging ---
        if len(episode_rewards) > 0:
            avg = np.mean(episode_rewards)
            best_avg = max(best_avg, avg)
            elapsed = time.time() - start_time
            print(
                f"[{stage_label}] Step {step:>8d} | Avg: {avg:>8.2f} | "
                f"Best: {best_avg:>8.2f} | Loss: {loss:.4f} | "
                f"LR: {current_lr:.2e} | Clip: {current_clip_ratio:.3f} | "
                f"Steps/s: {steps_per_sec_ema:.0f} | {elapsed:.0f}s"
            )

            if goal_reward is not None and avg >= goal_reward:
                print(f"\n✓ [{stage_label}] Goal reached! Avg reward: {avg:.2f}")
                env.close()
                return best_avg, step, False

    env.close()
    final_avg = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0
    best_avg = max(best_avg, final_avg)
    timed_out = (time.time() - start_time) >= time_budget
    total_time = time.time() - start_time
    print(f"\n[{stage_label}] Done in {total_time:.0f}s. Best avg: {best_avg:.2f}, "
          f"Final avg: {final_avg:.2f}, Steps: {step}, Steps/s: {steps_per_sec_ema:.0f}")
    return best_avg, step, timed_out


def train_two_stage(device='cpu',
                    stage1_steps=10_000_000,
                    stage2_steps=100_000_000,
                    stage1_budget=3600,
                    stage2_budget=16000,
                    hidden_dim=256,
                    lr=3e-4,
                    n_envs=32,
                    stage1_goal=350,
                    stage2_goal=350,
                    checkpoint_path='checkpoint.pt'):
    """Two-stage curriculum: BipedalWalker-v3 then BipedalWalkerHardcore-v3."""

    total_steps = stage1_steps + stage2_steps
    print(f"Device: {device}")
    print(f"n_envs: {n_envs}")
    print(f"Stage 1: BipedalWalker-v3         | {stage1_steps} steps | budget {stage1_budget}s | goal {stage1_goal}")
    print(f"Stage 2: BipedalWalkerHardcore-v3  | {stage2_steps} steps | budget {stage2_budget}s | goal {stage2_goal}")
    print(f"Total steps: {total_steps}")

    probe_env = gym.make('BipedalWalker-v3')
    state_dim = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    agent = PPOAgent(state_dim, action_dim, device=device, lr=lr,
                     hidden_dim=hidden_dim, total_steps=total_steps)

    # Stage 1: standard entropy
    agent.entropy_coef = 0.001
    agent.gamma = 0.99
    agent.gae_lambda = 0.95

    # --- Stage 1: learn basic locomotion ---
    best1, steps1, _ = run_stage(
        agent, env_name='BipedalWalker-v3',
        num_steps=stage1_steps, time_budget=stage1_budget,
        stage_label='Stage 1', n_envs=n_envs,
        goal_reward=stage1_goal,
        rollout_steps=2048, num_epochs=4, batch_size=256,
        use_reward_norm=False,
    )
    print(f"\nStage 1 result: best_avg={best1:.2f}, steps={steps1}")
    agent.save(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # --- Stage 2 preparation ---
    print("\n--- Stage 2 preparation ---")
    print(f"  log_std before reset: {agent.actor_critic.log_std.data.cpu().numpy()}")
    # Slightly boost exploration for obstacles
    agent.actor_critic.log_std.data.clamp_(min=np.log(0.5))
    print(f"  log_std after reset:  {agent.actor_critic.log_std.data.cpu().numpy()}")

    # Stage 2 hyperparams
    agent.entropy_coef = 0.005  # More exploration than Stage 1 but not too much
    agent.gamma = 0.995  # Higher gamma for longer hardcore episodes
    agent.gae_lambda = 0.95
    agent.initial_lr = lr  # Reset LR schedule for Stage 2
    agent.initial_clip_ratio = 0.2
    # Fresh optimizer so LR schedule starts clean
    agent.optimizer = optim.Adam(agent.actor_critic.parameters(), lr=lr, eps=1e-5)
    print(f"  entropy_coef: {agent.entropy_coef}, gamma: {agent.gamma}, lr: {lr}")

    # --- Stage 2: fine-tune on hardcore ---
    best2, steps2, _ = run_stage(
        agent, env_name='BipedalWalkerHardcore-v3',
        num_steps=stage2_steps, time_budget=stage2_budget,
        stage_label='Stage 2', n_envs=n_envs,
        goal_reward=stage2_goal,
        rollout_steps=2048, num_epochs=4, batch_size=256,
        use_reward_norm=True,  # Reward normalization helps on Hardcore
    )
    print(f"\nStage 2 result: best_avg={best2:.2f}, steps={steps2}")
    agent.save(checkpoint_path)

    print(f"\n{'='*60}")
    print(f"  FINAL: Stage1 best={best1:.2f}  Stage2 best={best2:.2f}")
    print(f"  Total steps: {steps1 + steps2}")
    print(f"{'='*60}")
    return agent, best1, best2


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    if torch.backends.mps.is_available():
        pass
        #device = 'mps'
        #print("Using MPS device")
    else:
        device = 'cpu'
        print("Using CPU device")
    device = 'cpu'

    agent, s1_reward, s2_reward = train_two_stage(
        device=device,
    )
    print(f"\nFinal Stage 1 (BipedalWalker-v3) Avg Reward: {s1_reward:.2f}")
    print(f"Final Stage 2 (Hardcore-v3) Avg Reward: {s2_reward:.2f}")
