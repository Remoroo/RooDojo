#!/usr/bin/env python3
"""DroQ-SAC for dm_control dog-run. Optimized for CPU with limited steps.
H=256, UTD=5, dropout=0.01, LayerNorm in critics, obs normalization, reward scaling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import shimmy  # noqa: F401
import time
import subprocess
from collections import deque

def get_short_hash():
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD']).decode().strip()
    except: return "unknown"

def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32).flatten() for k in sorted(obs.keys())])
    return np.asarray(obs, dtype=np.float32).flatten()

DEVICE = torch.device("cpu")

class RunningNorm:
    def __init__(self, n):
        self.mu = np.zeros(n, np.float64)
        self.var = np.ones(n, np.float64)
        self.cnt = 0
    def update(self, x):
        self.cnt += 1
        d = x - self.mu
        self.mu += d / self.cnt
        self.var += (d * (x - self.mu) - self.var) / self.cnt
    def __call__(self, x):
        return ((x - self.mu) / np.sqrt(np.maximum(self.var, 1e-6))).astype(np.float32)
    def state_dict(self):
        return dict(mu=self.mu.copy(), var=self.var.copy(), cnt=self.cnt)
    def load_state_dict(self, d):
        self.mu, self.var, self.cnt = d['mu'], d['var'], d['cnt']

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class Critic(nn.Module):
    def __init__(self, sd, ad, h=256, drop=0.01):
        super().__init__()
        inp = sd + ad
        def make_q():
            return nn.Sequential(
                nn.Linear(inp, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(h, 1))
        self.q1, self.q2 = make_q(), make_q()
    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], -1)).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, sd, ad, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.mu = nn.Linear(h, ad)
        self.ls = nn.Linear(h, ad)
    def forward(self, s):
        h = self.net(s)
        return self.mu(h), self.ls(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
    def sample(self, s):
        mu, ls = self.forward(s)
        std = ls.exp()
        n = torch.randn_like(mu)
        xt = mu + std * n
        a = torch.tanh(xt)
        lp = (-0.5*n.pow(2) - ls - 0.9189385332).sum(-1)
        lp -= (2*(0.6931472 - xt - F.softplus(-2*xt))).sum(-1)
        return a, lp
    @torch.no_grad()
    def act(self, s_np, det=False):
        s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
        mu, ls = self.forward(s)
        if det: return torch.tanh(mu).squeeze(0).numpy()
        return torch.tanh(mu + ls.exp() * torch.randn_like(mu)).squeeze(0).numpy()

class Buf:
    def __init__(self, sd, ad, cap=1_000_000):
        self.cap, self.p, self.n = cap, 0, 0
        self.s = np.zeros((cap, sd), np.float32)
        self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.ns = np.zeros((cap, sd), np.float32)
        self.d = np.zeros(cap, np.float32)
    def add(self, s, a, r, ns, d):
        i = self.p
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, d
        self.p = (i+1) % self.cap
        if self.n < self.cap: self.n += 1
    def sample(self, b):
        i = np.random.randint(0, self.n, b)
        return (torch.from_numpy(self.s[i]), torch.from_numpy(self.a[i]),
                torch.from_numpy(self.r[i]), torch.from_numpy(self.ns[i]),
                torch.from_numpy(self.d[i]))

class Agent:
    def __init__(self, sd, ad, lr=3e-4, g=0.99, tau=0.005, h=256, drop=0.01, rs=0.1):
        self.g, self.tau, self.rs = g, tau, rs
        self.norm = RunningNorm(sd)
        self.pi = Actor(sd, ad, h)
        self.q = Critic(sd, ad, h, drop)
        self.qt = Critic(sd, ad, h, drop)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        for p in self.qt.parameters(): p.requires_grad = False
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=lr)
        self.q_opt = optim.Adam(self.q.parameters(), lr=lr)
        self.tent = -float(ad)
        self.la = torch.zeros(1, requires_grad=True)
        self.a_opt = optim.Adam([self.la], lr=lr)
        self.buf = Buf(sd, ad)

    @property
    def alpha(self): return self.la.exp().item()

    def act(self, raw, det=False):
        return self.pi.act(self.norm(raw), det)

    def store(self, s, a, r, ns, term):
        self.buf.add(self.norm(s), a, r*self.rs, self.norm(ns), float(term))

    def update(self, bs=256):
        s, a, r, ns, d = self.buf.sample(bs)
        al = self.la.exp().detach()
        with torch.no_grad():
            na, nlp = self.pi.sample(ns)
            q1t, q2t = self.qt(ns, na)
            tgt = r + self.g*(1-d)*(torch.min(q1t,q2t) - al*nlp)
        self.q.train()
        q1, q2 = self.q(s, a)
        ql = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
        self.q_opt.zero_grad(); ql.backward(); self.q_opt.step()

        self.q.eval()
        na2, lp2 = self.pi.sample(s)
        q1p, q2p = self.q(s, na2)
        pl = (al*lp2 - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(); pl.backward(); self.pi_opt.step()

        al_l = -(self.la*(lp2.detach()+self.tent)).mean()
        self.a_opt.zero_grad(); al_l.backward(); self.a_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.qt.parameters()):
                tp.data.lerp_(p.data, self.tau)
        return ql.item(), pl.item()

    def save(self, path):
        torch.save({
            'pi': self.pi.state_dict(), 'q': self.q.state_dict(),
            'qt': self.qt.state_dict(), 'la': self.la.data,
            'po': self.pi_opt.state_dict(), 'qo': self.q_opt.state_dict(),
            'ao': self.a_opt.state_dict(), 'norm': self.norm.state_dict(),
        }, path)
    def load(self, path):
        c = torch.load(path, map_location='cpu', weights_only=False)
        self.pi.load_state_dict(c['pi']); self.q.load_state_dict(c['q'])
        self.qt.load_state_dict(c['qt']); self.la.data.copy_(c['la'])
        self.pi_opt.load_state_dict(c['po']); self.q_opt.load_state_dict(c['qo'])
        self.a_opt.load_state_dict(c['ao'])
        if 'norm' in c: self.norm.load_state_dict(c['norm'])

def evaluate(agent, eid, n=10):
    env = gym.make(eid)
    rews = []
    for _ in range(n):
        od, _ = env.reset(); o = flatten_obs(od)
        tot, dn = 0.0, False
        while not dn:
            a = agent.act(o, det=True)
            od, r, tm, tr, _ = env.step(a); o = flatten_obs(od)
            tot += r; dn = tm or tr
        rews.append(tot)
    env.close()
    return float(np.mean(rews)), float(np.std(rews))

def main():
    E = "dm_control/dog-run-v0"
    SEED, UTD, H, BS = 42, 5, 256, 256
    WARMUP = 5000
    EVAL_EVERY = 10000
    TLIMIT = 8100

    np.random.seed(SEED); torch.manual_seed(SEED)
    env = gym.make(E)
    od, _ = env.reset(seed=SEED); obs = flatten_obs(od)
    sd, ad = obs.shape[0], env.action_space.shape[0]

    agent = Agent(sd, ad, h=H)
    np_ = sum(p.numel() for p in agent.pi.parameters()) + sum(p.numel() for p in agent.q.parameters())
    print(f"s={sd} a={ad} p={np_:,} UTD={UTD} H={H}", flush=True)

    best, epr, er, ec = -1e9, deque(maxlen=100), 0.0, 0
    t0 = time.time()

    for step in range(1, 5_000_001):
        agent.norm.update(obs)
        action = env.action_space.sample() if step <= WARMUP else agent.act(obs)
        nd, rew, tm, tr, _ = env.step(action)
        nobs = flatten_obs(nd)
        agent.store(obs, action, rew, nobs, tm)
        er += rew
        if tm or tr:
            epr.append(er); ec += 1; er = 0.0
            od, _ = env.reset(); obs = flatten_obs(od)
        else:
            obs = nobs

        if step > WARMUP:
            for _ in range(UTD):
                agent.update(BS)

        if step % EVAL_EVERY == 0:
            el = time.time() - t0
            a100 = float(np.mean(epr)) if epr else 0
            em, es = evaluate(agent, E, 10)
            print(f"[{step:>7d}] ep={ec} a100={a100:.1f} eval={em:.1f}+/-{es:.1f} "
                  f"a={agent.alpha:.4f} SPS={step/el:.0f} t={el/60:.1f}m", flush=True)
            if em > best:
                best = em; agent.save("checkpoint_best.pt")
                print(f"  ** best={best:.1f}", flush=True)
            if em >= 700:
                agent.save("checkpoint.pt"); write_results(em); env.close(); return

        if step % 50000 == 0: agent.save("checkpoint.pt")
        if (time.time() - t0) > TLIMIT:
            print("TIME"); break

    em, es = evaluate(agent, E, 10)
    print(f"Final: {em:.1f}+/-{es:.1f}")
    agent.save("checkpoint.pt")
    write_results(max(em, best))
    env.close()

def write_results(avg):
    with open("results.tsv","w") as f:
        f.write("commit\taverage_reward\n")
        f.write(f"{get_short_hash()}\t{avg:.2f}\n")
    print(f"results.tsv: {avg:.2f}")

if __name__ == "__main__":
    main()
