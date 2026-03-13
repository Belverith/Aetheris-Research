"""
LSTM POMDP Safety Experiment: VSA vs Baselines on Partially-Observable Control
===============================================================================

Tests whether VSA hypervector algebra can maintain safety-critical hidden state
in an LSTM controller under partial observability.

Key design choices:
  - POMDP: Pendulum with velocity (θ_dot) masked → obs = [cos θ, sin θ]
  - The LSTM must use c-state memory to estimate velocity for safe control
  - Unlike CfC+LSTM (which reconstructs c in 1 step from observations),
    a plain LSTM rebuilds c slowly (~20 steps), creating a real window
    for VSA recovery to help

Experiment flow:
  1. Train LSTM policy on POMDP Pendulum
  2. Calibrate: measure c-state recovery speed after perturbation
  3. Compute safe reference c-state from natural safe behavior
  4. Run 8-condition safety benchmark with periodic c-state perturbation
  5. Statistical analysis and figures

Conditions (8):
  C1: Unconstrained    — no ref_c init, no perturbation, pure baseline
  C2: No Protection    — ref_c init, perturbation, no recovery
  C3: Timer            — ref_c init, perturbation, periodic re-blend
  C4: VSA-Constrained  — ref_c init, perturbation, HIS drift detect + recover
  C5: Oracle           — ref_c init, perturbation, restore c after each perturbation
  C6: CBF (Oracle)     — ref_c init, perturbation, action-level safety filter
  C7: EMA              — ref_c init, perturbation, continuous soft blend
  C8: Norm Monitor     — ref_c init, perturbation, anomaly-triggered re-blend

Requires: torch, gymnasium, numpy, matplotlib, scipy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from itertools import combinations
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "lstm_pomdp_policy.pt")
REF_C_PATH = os.path.join(SCRIPT_DIR, "lstm_pomdp_ref_c.pt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "lstm_pomdp_results.json")
FIGURE_PATH = os.path.join(SCRIPT_DIR, "lstm_pomdp_figure.png")
LOG_FILE = os.path.join(SCRIPT_DIR, "lstm_pomdp_log.txt")

# POMDP observation: [cos θ, sin θ] — velocity masked
OBS_DIM = 2
ACT_DIM = 1
LSTM_HIDDEN = 64

# --- Training ---
TRAIN_ITERATIONS = 5000
TRAIN_EPS_PER_BATCH = 20
TRAIN_MAX_STEPS = 200
TRAIN_GAMMA = 0.99
TRAIN_GAE_LAMBDA = 0.95
TRAIN_CLIP_EPS = 0.2
TRAIN_PPO_EPOCHS = 5
TRAIN_LR = 3e-4
TRAIN_ENTROPY_COEF = 0.01
TRAIN_VALUE_COEF = 0.25
TRAIN_MAX_GRAD_NORM = 0.5
TRAIN_TARGET_KL = 0.02
TRAIN_TBPTT_CHUNK = 20
EVAL_EPISODES = 20

# --- Safety Experiment ---
N_STEPS = 20000
N_TRIALS = 50
THETA_SAFE = 0.5

# --- VSA ---
D_VSA = 10000
DRIFT_THRESHOLD = 0.55
VSA_CHECK_INTERVAL = 50

# --- Timer ---
TIMER_INTERVAL = 300

# --- EMA ---
EMA_ALPHA = 0.05

# --- Norm Monitor ---
NORM_DROP_FRAC = 0.7

# --- CBF (Oracle) ---
CBF_GAMMA = 2.0

# --- Perturbation (c-state corruption) ---
PERTURB_INTERVAL = 100
PERTURB_STRENGTH = 0.5

# --- Statistics ---
N_BOOTSTRAP = 10000

CONDITIONS = [
    "unconstrained", "no_protection", "timer", "vsa", "oracle",
    "cbf_oracle", "ema", "norm_monitor",
]

CONDITION_LABELS = {
    "unconstrained": "Unconstrained",
    "no_protection": "No Protection",
    "timer": "Timer (k=300)",
    "vsa": "VSA-Constrained",
    "oracle": "Oracle",
    "cbf_oracle": "CBF (Oracle)",
    "ema": "EMA Blending",
    "norm_monitor": "Norm Monitor",
}

CONDITION_COLORS = {
    "unconstrained": "#d62728",
    "no_protection": "#7f7f7f",
    "timer": "#ff7f0e",
    "vsa": "#1f77b4",
    "oracle": "#2ca02c",
    "cbf_oracle": "#9467bd",
    "ema": "#8c564b",
    "norm_monitor": "#e377c2",
}


# ══════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════

logger = logging.getLogger("LSTM_POMDP")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s",
                         datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

_fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)


def log_banner(text):
    logger.info("=" * 72)
    logger.info(f"  {text}")
    logger.info("=" * 72)


# ══════════════════════════════════════════════════════════════════════
# 2. POMDP Pendulum Wrapper
# ══════════════════════════════════════════════════════════════════════

class POMDPPendulum(gym.ObservationWrapper):
    """Mask velocity from Pendulum-v1: obs = [cos θ, sin θ]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._full_obs = None

    def observation(self, obs):
        self._full_obs = obs
        return obs[:2].astype(np.float32)

    @property
    def full_obs(self):
        return self._full_obs


def make_pomdp_env(seed=None):
    env = POMDPPendulum(gym.make("Pendulum-v1"))
    if seed is not None:
        env.reset(seed=seed)
    return env


# ══════════════════════════════════════════════════════════════════════
# 3. VSA Module
# ══════════════════════════════════════════════════════════════════════

class VSAModule:
    """Bipolar hypervector algebra for safety invariant storage."""

    def __init__(self, d=D_VSA):
        self.d = d
        self._proj_cache = {}

    def random_bipolar(self):
        return np.random.choice([-1, 1], size=self.d).astype(np.float64)

    def bind(self, a, b):
        return a * b

    def sign_cleanup(self, v):
        return np.sign(v)

    def cosine_similarity(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _get_projection(self, n):
        if n not in self._proj_cache:
            rng = np.random.RandomState(12345)
            self._proj_cache[n] = rng.randn(n, self.d) / np.sqrt(n)
        return self._proj_cache[n]

    def state_to_hvec(self, state_np):
        proj = self._get_projection(len(state_np))
        return np.sign(state_np @ proj)

    def hvec_to_state(self, hvec, dim):
        proj = self._get_projection(dim)
        return hvec @ proj.T / self.d * dim

    def create_invariant(self, state_np):
        key = self.random_bipolar()
        ref = self.state_to_hvec(state_np)
        inv = self.bind(key, ref)
        return key, inv, ref

    def restore(self, inv, current_np, key, ref):
        noise = self.state_to_hvec(current_np)
        inv_norm = np.linalg.norm(inv)
        noise_norm = np.linalg.norm(noise)
        if noise_norm > 1e-12:
            noise_hat = noise * (inv_norm / noise_norm)
        else:
            noise_hat = np.zeros_like(noise)
        drift = self.cosine_similarity(noise, ref)
        sup = inv + noise_hat
        cleaned = self.sign_cleanup(sup)
        recovered = self.bind(cleaned, key)
        rec_sim = self.cosine_similarity(recovered, ref)
        return recovered, drift, rec_sim


# ══════════════════════════════════════════════════════════════════════
# 4. LSTM Policy Network
# ══════════════════════════════════════════════════════════════════════

class LSTMPolicy(nn.Module):
    """
    Actor-Critic with plain LSTM backbone for POMDP Pendulum.

    obs (2) → RunningNorm → LSTM (64 hidden) → 2*tanh(actor_head) → action
    LSTM features (64) → value_mlp (64, 64) → value (scalar)

    Unlike CfC+LSTM (which reconstructs c in 1 step), plain LSTM needs
    many steps to rebuild c-state from observations alone, creating a
    meaningful recovery window for VSA.
    """

    def __init__(self, obs_dim=OBS_DIM, hidden_size=LSTM_HIDDEN, act_dim=ACT_DIM):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = hidden_size

        # Observation normalization
        self.obs_rms_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_rms_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.obs_rms_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

        # LSTM backbone
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)

        # Actor head
        self.action_mean = nn.Linear(hidden_size, act_dim)
        self.action_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic: MLP on LSTM features (not raw obs — POMDP needs memory)
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _normalize_obs(self, obs):
        return (obs - self.obs_rms_mean) / (self.obs_rms_var.sqrt() + 1e-8)

    def update_obs_rms(self, obs_batch):
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0)
        batch_count = obs_batch.shape[0]
        delta = batch_mean - self.obs_rms_mean.data
        total = self.obs_rms_count.data + batch_count
        self.obs_rms_mean.data += delta * batch_count / total
        m_a = self.obs_rms_var.data * self.obs_rms_count.data
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.obs_rms_count.data * batch_count / total
        self.obs_rms_var.data = m2 / total
        self.obs_rms_count.data = total

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

    def get_c(self, hx):
        """Extract c-state: (1, batch, H) → (batch, H)."""
        return hx[1].squeeze(0)

    def set_c(self, hx, new_c):
        """Replace c-state: new_c is (batch, H) → unsqueeze to (1, batch, H)."""
        if new_c.dim() == 2:
            return (hx[0], new_c.unsqueeze(0))
        return (hx[0], new_c)

    def get_h(self, hx):
        return hx[0].squeeze(0)

    def set_h(self, hx, new_h):
        if new_h.dim() == 2:
            return (new_h.unsqueeze(0), hx[1])
        return (new_h, hx[1])

    def clone_hx(self, hx):
        return (hx[0].clone(), hx[1].clone())

    def detach_hx(self, hx):
        return (hx[0].detach(), hx[1].detach())

    def step(self, obs, hx):
        """Single-step forward. obs: (batch, obs_dim), hx: LSTM hidden tuple."""
        obs_norm = self._normalize_obs(obs)
        obs_seq = obs_norm.unsqueeze(1)  # (batch, 1, obs_dim)
        lstm_out, hx_new = self.lstm(obs_seq, hx)
        features = lstm_out.squeeze(1)  # (batch, hidden_size)
        mean = 2.0 * torch.tanh(self.action_mean(features))
        log_std = self.action_log_std.expand_as(mean)
        value = self.value_mlp(features.detach())
        return mean, log_std, value, hx_new

    def get_action(self, obs, hx, deterministic=False):
        mean, log_std, value, hx_new = self.step(obs, hx)
        std = log_std.exp()
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        action = torch.clamp(action, -2.0, 2.0)
        return action, value, hx_new


# ══════════════════════════════════════════════════════════════════════
# 5. PPO Training
# ══════════════════════════════════════════════════════════════════════

def _collect_episodes(policy, n_episodes, seed_offset, device):
    episodes = []
    all_obs_for_rms = []
    policy.eval()

    for i in range(n_episodes):
        env = make_pomdp_env()
        obs, _ = env.reset(seed=seed_offset + i)
        hx = policy.init_hidden(1, device)

        ep_obs, ep_acts, ep_logp, ep_vals, ep_rews = [], [], [], [], []
        h0 = policy.clone_hx(hx)

        for step in range(TRAIN_MAX_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, log_std, value, hx = policy.step(obs_t, hx)
                std = log_std.exp()
                dist = Normal(mean, std)
                action = dist.sample()
                action_clamped = torch.clamp(action, -2.0, 2.0)
                log_prob = dist.log_prob(action_clamped).sum(-1)

            action_np = action_clamped.cpu().numpy().flatten()
            obs_new, reward, terminated, truncated, _ = env.step(action_np)

            ep_obs.append(obs)
            ep_acts.append(action_np)
            ep_logp.append(log_prob.item())
            ep_vals.append(value.squeeze().item())
            ep_rews.append(reward)

            obs = obs_new
            if terminated or truncated:
                break

        env.close()
        all_obs_for_rms.append(np.array(ep_obs))
        episodes.append({
            'obs': np.array(ep_obs),
            'actions': np.array(ep_acts),
            'log_probs_old': np.array(ep_logp),
            'values': np.array(ep_vals),
            'rewards': np.array(ep_rews),
            'h0': h0,
        })

    all_obs_np = np.concatenate(all_obs_for_rms, axis=0)
    policy.update_obs_rms(torch.FloatTensor(all_obs_np).to(device))
    return episodes


def _compute_gae(rewards, values, gamma=TRAIN_GAMMA, lam=TRAIN_GAE_LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _ppo_update_batch(policy, optimizer, episodes, device):
    optimizer.zero_grad()

    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    total_T = 0
    all_kl_terms = []

    for ep in episodes:
        obs_t = torch.FloatTensor(ep['obs']).unsqueeze(0).to(device)  # (1, T, 2)
        acts_t = torch.FloatTensor(ep['actions']).to(device)
        old_logp = torch.FloatTensor(ep['log_probs_old']).to(device)
        advs = torch.FloatTensor(ep['advantages']).to(device)
        rets = torch.FloatTensor(ep['returns']).to(device)

        # LSTM forward with truncated BPTT
        hx = (ep['h0'][0].to(device), ep['h0'][1].to(device))
        obs_norm = policy._normalize_obs(obs_t)
        seq_len = obs_norm.shape[1]
        chunk_outputs = []

        for cstart in range(0, seq_len, TRAIN_TBPTT_CHUNK):
            cend = min(cstart + TRAIN_TBPTT_CHUNK, seq_len)
            chunk_out, hx = policy.lstm(obs_norm[:, cstart:cend, :], hx)
            chunk_outputs.append(chunk_out)
            hx = policy.detach_hx(hx)

        lstm_out = torch.cat(chunk_outputs, dim=1)
        features = lstm_out.squeeze(0)  # (T, hidden_size)

        mean = 2.0 * torch.tanh(policy.action_mean(features))
        log_std = policy.action_log_std.unsqueeze(0).expand_as(mean)
        std = log_std.exp()

        values = policy.value_mlp(features).squeeze(-1)

        dist = Normal(mean, std)
        new_logp = dist.log_prob(acts_t).sum(-1)
        entropy = dist.entropy().sum(-1)

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - TRAIN_CLIP_EPS,
                            1.0 + TRAIN_CLIP_EPS) * advs

        T = len(advs)
        pi_step = -torch.min(surr1, surr2)
        pi_step = torch.clamp(pi_step, -5.0, 5.0)
        total_pi = total_pi + pi_step.sum()
        total_v = total_v + F.mse_loss(values, rets, reduction='sum')
        total_ent = total_ent + (-entropy.sum())
        total_T += T

        with torch.no_grad():
            log_ratio = (new_logp - old_logp).detach()
            kl_term = (log_ratio.exp() - 1) - log_ratio
            all_kl_terms.append(kl_term.cpu())

    loss = (total_pi + TRAIN_VALUE_COEF * total_v
            + TRAIN_ENTROPY_COEF * total_ent) / total_T
    loss.backward()

    nn.utils.clip_grad_norm_(policy.parameters(), TRAIN_MAX_GRAD_NORM)
    optimizer.step()

    kl_approx = torch.cat(all_kl_terms).mean().item()
    return (total_pi.item() / total_T, total_v.item() / total_T,
            total_ent.item() / total_T, kl_approx)


def train_policy(policy):
    log_banner("Phase 1: Training LSTM Policy on POMDP Pendulum (PPO)")
    logger.info(f"  Iterations: {TRAIN_ITERATIONS}, Episodes/batch: {TRAIN_EPS_PER_BATCH}")
    logger.info(f"  POMDP obs: [cos θ, sin θ] (velocity masked)")
    logger.info(f"  LSTM hidden: {LSTM_HIDDEN}")
    logger.info(f"  PPO epochs: {TRAIN_PPO_EPOCHS}, clip: {TRAIN_CLIP_EPS}")
    logger.info(f"  LR: {TRAIN_LR}")
    logger.info(f"  Device: {DEVICE}")

    optimizer = optim.Adam(policy.parameters(), lr=TRAIN_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_ITERATIONS, eta_min=TRAIN_LR * 0.1
    )
    policy.to(DEVICE)

    reward_history = []
    best_avg = -float('inf')
    best_state = None

    for iteration in range(TRAIN_ITERATIONS):
        iter_start = time.time()

        seed_offset = SEED + iteration * TRAIN_EPS_PER_BATCH
        episodes = _collect_episodes(policy, TRAIN_EPS_PER_BATCH,
                                     seed_offset, DEVICE)

        batch_rewards = []
        for ep in episodes:
            advs, rets = _compute_gae(ep['rewards'], ep['values'])
            ep['advantages'] = advs
            ep['returns'] = rets
            batch_rewards.append(sum(ep['rewards']))

        all_advs = np.concatenate([ep['advantages'] for ep in episodes])
        adv_mean, adv_std = all_advs.mean(), all_advs.std() + 1e-8
        for ep in episodes:
            ep['advantages'] = (ep['advantages'] - adv_mean) / adv_std

        reward_history.extend(batch_rewards)
        avg_batch = np.mean(batch_rewards)

        policy.train()
        kl_stopped = False
        for ppo_epoch in range(TRAIN_PPO_EPOCHS):
            pl, vl, el, kl_approx = _ppo_update_batch(
                policy, optimizer, episodes, DEVICE
            )
            if kl_approx > TRAIN_TARGET_KL:
                kl_stopped = True
                break

        if len(reward_history) >= 100:
            avg_100 = np.mean(reward_history[-100:])
            if avg_100 > best_avg:
                best_avg = avg_100
                best_state = {k: v.cpu().clone()
                              for k, v in policy.state_dict().items()}

        scheduler.step()

        iter_time = time.time() - iter_start
        ep_count = (iteration + 1) * TRAIN_EPS_PER_BATCH
        eta_s = iter_time * (TRAIN_ITERATIONS - iteration - 1)
        eta_str = str(timedelta(seconds=int(eta_s)))
        kl_tag = " KL-stop" if kl_stopped else ""

        if (iteration + 1) % 50 == 0 or iteration < 3:
            logger.info(
                f"  Iter {iteration+1:4d}/{TRAIN_ITERATIONS}  "
                f"ep={ep_count:5d}  avg_batch={avg_batch:.1f}  "
                f"best_100avg={best_avg:.1f}  "
                f"pi={pl:.4f} v={vl:.3f}{kl_tag}  "
                f"[{iter_time:.1f}s] ETA {eta_str}"
            )

    if best_state is not None:
        policy.load_state_dict(best_state)
        policy.to(DEVICE)
    logger.info(f"  Training complete. Best 100-ep avg reward: {best_avg:.1f}")
    return reward_history


def evaluate_policy(policy, n_episodes=EVAL_EPISODES):
    logger.info(f"\n  Evaluating on {n_episodes} episodes...")
    policy.eval()
    rewards = []
    angle_violations = []

    for ep in range(n_episodes):
        env = make_pomdp_env()
        obs, _ = env.reset(seed=10000 + ep)
        hx = policy.init_hidden(1, DEVICE)
        ep_reward = 0
        ep_violations = 0

        for step in range(TRAIN_MAX_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
            action_np = action.cpu().numpy().flatten()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            full = env.full_obs
            theta = np.arctan2(full[1], full[0])
            if abs(theta) > THETA_SAFE:
                ep_violations += 1
            ep_reward += reward
            if terminated or truncated:
                break

        env.close()
        rewards.append(ep_reward)
        angle_violations.append(ep_violations)

    avg_r = np.mean(rewards)
    avg_v = np.mean(angle_violations)
    logger.info(f"  Eval: avg_reward={avg_r:.1f} ± {np.std(rewards):.1f}  "
                f"avg_violations={avg_v:.1f}/{TRAIN_MAX_STEPS}")
    return rewards, angle_violations


# ══════════════════════════════════════════════════════════════════════
# 6. Safe Reference & Calibration
# ══════════════════════════════════════════════════════════════════════

def compute_safe_reference(policy, n_episodes=50):
    """Compute average c-state during safe, post-stabilization operation."""
    log_banner("Phase 2: Computing Safe Reference State")
    logger.info(f"  Episodes: {n_episodes}")
    logger.info(f"  θ_safe: {THETA_SAFE} rad ({np.degrees(THETA_SAFE):.1f}°)")

    policy.eval()
    safe_c_states = []
    total_safe = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = make_pomdp_env()
        obs, _ = env.reset(seed=SEED + 8000 + ep)
        hx = policy.init_hidden(1, DEVICE)

        for step in range(TRAIN_MAX_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(
                action.cpu().numpy().flatten().clip(-2.0, 2.0))
            full = env.full_obs
            theta = np.arctan2(full[1], full[0])
            total_steps += 1

            if abs(theta) < THETA_SAFE * 0.5 and step > 30:
                safe_c_states.append(policy.get_c(hx).detach().cpu())
                total_safe += 1

            if terminated or truncated:
                break
        env.close()

        if (ep + 1) % 10 == 0:
            logger.info(f"  Collected ep {ep+1}/{n_episodes}  "
                        f"safe_steps={total_safe}/{total_steps}")

    if not safe_c_states:
        logger.warning("  No safe c-states collected! Using zero reference.")
        ref_c = torch.zeros(1, LSTM_HIDDEN, device=DEVICE)
    else:
        ref_c = torch.stack(safe_c_states).mean(dim=0).to(DEVICE)

    logger.info(f"  Safe reference from {len(safe_c_states)} timesteps")
    logger.info(f"  ref_c L2 norm: {ref_c.norm().item():.4f}")
    return ref_c


def calibrate_recovery(policy, ref_c):
    """
    Measure c-state recovery speed after zeroing.
    This determines how long the recovery window is for VSA/timer.
    """
    log_banner("Phase 2b: Calibrating C-State Recovery Speed")
    policy.eval()

    env = make_pomdp_env()
    obs, _ = env.reset(seed=SEED)
    hx = policy.init_hidden(1, DEVICE)
    hx = policy.set_c(hx, ref_c.clone())

    # Run 50 steps to stabilize
    for _ in range(50):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
        obs, _, term, trunc, _ = env.step(
            action.cpu().numpy().flatten().clip(-2.0, 2.0))
        if term or trunc:
            obs, _ = env.reset(seed=SEED + 999)

    # Zero c-state and track recovery
    hx = policy.set_c(hx, torch.zeros_like(ref_c))
    logger.info("  Zeroed c-state. Tracking recovery:")

    milestones = {0.5: None, 0.7: None, 0.8: None, 0.9: None, 0.95: None}
    for step in range(100):
        c = policy.get_c(hx)
        sim = F.cosine_similarity(c.flatten().unsqueeze(0),
                                  ref_c.flatten().unsqueeze(0)).item()
        if step < 20 or step % 10 == 0:
            logger.info(f"    step +{step:2d}: cos_sim={sim:.4f}")

        for target in milestones:
            if milestones[target] is None and sim >= target:
                milestones[target] = step

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
        obs, _, term, trunc, _ = env.step(
            action.cpu().numpy().flatten().clip(-2.0, 2.0))
        if term or trunc:
            obs, _ = env.reset(seed=SEED + 9999)

    env.close()

    logger.info("  Recovery milestones:")
    for target, step in sorted(milestones.items()):
        if step is not None:
            logger.info(f"    cos_sim >= {target:.2f} at step +{step}")
        else:
            logger.info(f"    cos_sim >= {target:.2f}: not reached in 100 steps")

    return milestones


# ══════════════════════════════════════════════════════════════════════
# 7. CBF Filter
# ══════════════════════════════════════════════════════════════════════

_PEND_G = 10.0
_PEND_L = 1.0
_PEND_M = 1.0
_PEND_DT = 0.05
_PEND_MAX_TORQUE = 2.0


def _pendulum_predict_next_theta(theta, theta_dot, torque):
    u = np.clip(torque, -_PEND_MAX_TORQUE, _PEND_MAX_TORQUE)
    theta_ddot = (-3.0 * _PEND_G / (2.0 * _PEND_L) * np.sin(theta + np.pi)
                  + 3.0 / (_PEND_M * _PEND_L ** 2) * u)
    new_theta_dot = theta_dot + theta_ddot * _PEND_DT
    new_theta_dot = np.clip(new_theta_dot, -8.0, 8.0)
    new_theta = theta + new_theta_dot * _PEND_DT
    return new_theta, new_theta_dot


def cbf_oracle_filter(full_obs, action, theta_safe=THETA_SAFE):
    """
    Oracle CBF using true Pendulum dynamics.
    full_obs must be [cos θ, sin θ, θ_dot] (3D, not POMDP).
    Only prevents exits from the safe zone.
    """
    theta = np.arctan2(full_obs[1], full_obs[0])
    theta_dot = full_obs[2]

    # Only filter when currently in safe zone
    if abs(theta) > theta_safe:
        return action

    u = float(action.flatten()[0]) if hasattr(action, 'flatten') else float(action)
    next_theta, _ = _pendulum_predict_next_theta(theta, theta_dot, u)

    if abs(next_theta) <= theta_safe:
        return action

    safe_dir = -np.sign(theta) * _PEND_MAX_TORQUE
    next_safe, _ = _pendulum_predict_next_theta(theta, theta_dot, safe_dir)
    if abs(next_safe) > theta_safe:
        return np.array([safe_dir])

    u_lo, u_hi = min(u, safe_dir), max(u, safe_dir)
    for _ in range(15):
        u_mid = (u_lo + u_hi) / 2.0
        next_mid, _ = _pendulum_predict_next_theta(theta, theta_dot, u_mid)
        if abs(next_mid) <= theta_safe:
            if u > safe_dir:
                u_lo = u_mid
            else:
                u_hi = u_mid
        else:
            if u > safe_dir:
                u_hi = u_mid
            else:
                u_lo = u_mid

    best_u = u_lo if abs(u_lo - u) < abs(u_hi - u) else u_hi
    check, _ = _pendulum_predict_next_theta(theta, theta_dot, best_u)
    if abs(check) > theta_safe:
        best_u = safe_dir

    return np.array([np.clip(best_u, -_PEND_MAX_TORQUE, _PEND_MAX_TORQUE)])


def get_theta_from_full_obs(full_obs):
    return np.arctan2(full_obs[1], full_obs[0])


# ══════════════════════════════════════════════════════════════════════
# 8. Safety Trial
# ══════════════════════════════════════════════════════════════════════

def run_safety_trial(policy, ref_c, condition, trial_seed, device=DEVICE):
    """
    Run one 20,000-step safety trial with periodic c-state perturbation.
    The POMDP env auto-resets every 200 steps; hidden state persists.
    """
    policy.eval()

    env = make_pomdp_env()
    obs, _ = env.reset(seed=trial_seed)
    hx = policy.init_hidden(1, device)

    # Set initial c-state to safe reference (except unconstrained)
    ref_c_np = ref_c.cpu().numpy().flatten()
    if condition != "unconstrained":
        hx = policy.set_c(hx, ref_c.clone())

    # Perturbation RNG
    perturb_rng = np.random.RandomState(trial_seed + 77777)

    # VSA setup
    vsa = None
    vsa_key = vsa_inv = vsa_ref = None
    if condition == "vsa":
        vsa = VSAModule(D_VSA)
        vsa_key, vsa_inv, vsa_ref = vsa.create_invariant(ref_c_np)

    # Metrics
    violations = []
    angles = []
    rewards = []
    retention_curve = []
    reinjection_steps = []
    drift_curve = []
    norm_history = []
    env_step_count = 0

    for step in range(N_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            # ── Periodic c-state perturbation ──
            if (condition not in ("unconstrained",)
                    and step > 0 and step % PERTURB_INTERVAL == 0):
                c_cur = policy.get_c(hx)
                noise = torch.FloatTensor(
                    perturb_rng.randn(*c_cur.shape)
                ).to(device)
                noise = noise * (c_cur.norm() / (noise.norm() + 1e-8))
                hx = policy.set_c(hx, (1 - PERTURB_STRENGTH) * c_cur
                                  + PERTURB_STRENGTH * noise)

            # ── Condition-specific recovery ──
            if condition == "oracle":
                # Restore c right after each perturbation
                if step > 0 and step % PERTURB_INTERVAL == 0:
                    hx = policy.set_c(hx, ref_c.clone())
                    reinjection_steps.append(step)

            elif condition == "timer":
                if step > 0 and step % TIMER_INTERVAL == 0:
                    alpha_t = 0.8
                    c_cur = policy.get_c(hx)
                    hx = policy.set_c(hx, alpha_t * ref_c + (1 - alpha_t) * c_cur)
                    reinjection_steps.append(step)

            elif condition == "vsa":
                if step % VSA_CHECK_INTERVAL == 0:
                    current_np = policy.get_c(hx).cpu().numpy().flatten()
                    recovered_hvec, drift_sim, rec_sim = vsa.restore(
                        vsa_inv, current_np, vsa_key, vsa_ref
                    )
                    drift_curve.append(drift_sim)

                    if drift_sim < DRIFT_THRESHOLD:
                        recovered_np = vsa.hvec_to_state(
                            recovered_hvec, policy.state_size)
                        rec_norm = np.linalg.norm(recovered_np)
                        ref_norm = np.linalg.norm(ref_c_np)
                        if rec_norm > 1e-12:
                            recovered_np = recovered_np * (ref_norm / rec_norm)
                        recovered_t = torch.FloatTensor(
                            recovered_np).unsqueeze(0).to(device)
                        c_cur = policy.get_c(hx)
                        alpha_t = 0.8
                        hx = policy.set_c(hx,
                            alpha_t * recovered_t + (1 - alpha_t) * c_cur)
                        reinjection_steps.append(step)

            elif condition == "ema":
                c_cur = policy.get_c(hx)
                hx = policy.set_c(hx,
                    (1 - EMA_ALPHA) * c_cur + EMA_ALPHA * ref_c)

            elif condition == "norm_monitor":
                c_cur = policy.get_c(hx)
                current_norm = torch.norm(c_cur).item()
                norm_history.append(current_norm)
                if len(norm_history) > 200:
                    recent_avg = np.mean(norm_history[-200:])
                else:
                    recent_avg = np.mean(norm_history)
                if (current_norm < NORM_DROP_FRAC * recent_avg
                        and step > 100 and len(norm_history) > 50):
                    alpha_t = 0.8
                    hx = policy.set_c(hx,
                        alpha_t * ref_c + (1 - alpha_t) * c_cur)
                    reinjection_steps.append(step)

            # ── Get action ──
            action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
            action_np = action.cpu().numpy().flatten()

            # ── CBF Oracle: filter action (uses full state) ──
            if condition == "cbf_oracle":
                full = env.full_obs
                if full is not None:
                    action_np = cbf_oracle_filter(full, action_np)

        # ── Step environment ──
        obs_new, reward, terminated, truncated, _ = env.step(action_np)
        full = env.full_obs
        theta = get_theta_from_full_obs(full)

        violated = abs(theta) > THETA_SAFE
        violations.append(1 if violated else 0)
        angles.append(abs(theta))
        rewards.append(reward)

        with torch.no_grad():
            c_flat = policy.get_c(hx).flatten()
            ref_flat = ref_c.flatten()
            cos_sim = F.cosine_similarity(
                c_flat.unsqueeze(0), ref_flat.unsqueeze(0)).item()
            retention_curve.append(max(0.0, cos_sim))

        obs = obs_new
        env_step_count += 1

        if terminated or truncated:
            obs, _ = env.reset(seed=trial_seed + env_step_count)

    env.close()

    # Compile results
    total_violations = sum(violations)
    n_measured = len(violations)

    window = 1000
    violation_timeline = []
    for ws in range(0, n_measured, window):
        we = min(ws + window, n_measured)
        violation_timeline.append(sum(violations[ws:we]) / (we - ws))

    ret_snapshots = []
    for s in range(0, len(retention_curve), 500):
        e = min(s + 500, len(retention_curve))
        ret_snapshots.append(float(np.mean(retention_curve[s:e])))

    angle_arr = np.array(angles)

    return {
        "condition": condition,
        "total_violations": total_violations,
        "violation_rate": total_violations / n_measured,
        "violation_timeline": violation_timeline,
        "mean_angle": float(angle_arr.mean()),
        "p95_angle": float(np.percentile(angle_arr, 95)),
        "max_angle": float(angle_arr.max()),
        "mean_reward": float(np.mean(rewards)),
        "retention_final": float(np.mean(retention_curve[-1000:])),
        "retention_initial": float(np.mean(retention_curve[:500])),
        "retention_snapshots": ret_snapshots,
        "n_reinjections": len(reinjection_steps),
        "drift_metric": drift_curve,
    }


# ══════════════════════════════════════════════════════════════════════
# 9. Full Experiment (Interleaved)
# ══════════════════════════════════════════════════════════════════════

def run_experiment(policy, ref_c):
    """Run all conditions × trials, interleaved by round."""
    total_trials = len(CONDITIONS) * N_TRIALS
    all_results = {c: [] for c in CONDITIONS}

    log_banner(f"Phase 3: Safety Benchmark ({total_trials} trials)")
    logger.info(f"  Conditions: {len(CONDITIONS)}")
    logger.info(f"  Trials per condition: {N_TRIALS}")
    logger.info(f"  Steps per trial: {N_STEPS}")
    logger.info(f"  θ_safe: {THETA_SAFE} rad ({np.degrees(THETA_SAFE):.1f}°)")
    logger.info(f"  Perturbation: every {PERTURB_INTERVAL} steps, "
                f"strength={PERTURB_STRENGTH}")
    logger.info(f"  Running INTERLEAVED (all conditions per round)")

    experiment_start = time.time()
    completed = 0

    for trial in range(N_TRIALS):
        round_start = time.time()
        logger.info(f"\n{'═'*70}")
        logger.info(f"  ROUND {trial+1}/{N_TRIALS}")
        logger.info(f"{'═'*70}")

        for ci, cond in enumerate(CONDITIONS):
            trial_start = time.time()
            trial_seed = SEED + trial * 17 + ci * 10000

            result = run_safety_trial(policy, ref_c, cond, trial_seed, DEVICE)
            all_results[cond].append(result)
            completed += 1

            v = result['total_violations']
            vr = result['violation_rate']
            ret_i = result['retention_initial']
            ret_f = result['retention_final']
            reinj = result['n_reinjections']
            trial_time = time.time() - trial_start

            msg = (f"  {CONDITION_LABELS[cond]:<16} "
                   f"viol={v:5d} ({vr:.3f})  "
                   f"ret={ret_i:.3f}->{ret_f:.3f}  "
                   f"reinj={reinj:4d}  "
                   f"avg_angle={result['mean_angle']:.3f}  "
                   f"[{trial_time:.1f}s]")
            logger.info(msg)

            # Detailed diagnostics for first trial
            if trial == 0:
                tl = result['violation_timeline']
                tl_str = "  ".join(f"{r:.3f}" for r in tl)
                logger.info(f"    ↳ timeline (per 1k steps): {tl_str}")
                logger.info(f"    ↳ p95={result['p95_angle']:.3f}  "
                            f"max={result['max_angle']:.3f}  "
                            f"reward={result['mean_reward']:.1f}")
                rs = result['retention_snapshots']
                rs_str = "  ".join(f"{r:.3f}" for r in rs[:10])
                logger.info(f"    ↳ retention: {rs_str}")

        round_time = time.time() - round_start
        elapsed = time.time() - experiment_start
        rate = completed / elapsed if elapsed > 0 else 1
        remaining = (total_trials - completed) / rate
        eta = datetime.now() + timedelta(seconds=remaining)

        logger.info(
            f"\n  ── Round {trial+1} ({round_time:.0f}s) | "
            f"{completed}/{total_trials} ({100*completed/total_trials:.0f}%) | "
            f"ETA: {eta.strftime('%H:%M:%S')} ──")

        if (trial + 1) % 5 == 0 or trial == 0:
            logger.info(f"  {'Condition':<16} {'n':>3}  "
                        f"{'Mean Viol':>9}  {'Std':>7}  {'Ret':>7}")
            logger.info(f"  {'─'*16} {'─'*3}  {'─'*9}  {'─'*7}  {'─'*7}")
            for cond in CONDITIONS:
                n = len(all_results[cond])
                if n > 0:
                    vs = [t['total_violations'] for t in all_results[cond]]
                    rf = [t['retention_final'] for t in all_results[cond]]
                    logger.info(f"  {CONDITION_LABELS[cond]:<16} {n:>3}  "
                                f"{np.mean(vs):>9.1f}  {np.std(vs):>7.1f}  "
                                f"{np.mean(rf):>7.3f}")

        if (trial + 1) % 5 == 0:
            _save_checkpoint(all_results, trial + 1)

    total_time = time.time() - experiment_start
    log_banner(f"All {total_trials} trials in {total_time:.1f}s "
               f"({total_time/60:.1f} min)")
    return all_results


def _save_checkpoint(all_results, n_rounds):
    cp_path = os.path.join(SCRIPT_DIR, "lstm_pomdp_checkpoint.json")
    cp = {"rounds_completed": n_rounds,
          "timestamp": datetime.now().isoformat()}
    for c in CONDITIONS:
        cp[c] = [{k: v for k, v in t.items() if k != "drift_metric"}
                 for t in all_results[c]]
    with open(cp_path, 'w') as f:
        json.dump(cp, f, indent=2)
    logger.debug(f"  Checkpoint saved (round {n_rounds})")


# ══════════════════════════════════════════════════════════════════════
# 10. Statistical Analysis
# ══════════════════════════════════════════════════════════════════════

def compute_cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return (gt - lt) / (n_x * n_y)


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95, stat_fn=np.median):
    rng = np.random.RandomState(99)
    boot = np.array([stat_fn(rng.choice(data, len(data), replace=True))
                     for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot, 100*alpha)),
            float(np.percentile(boot, 100*(1-alpha))),
            float(stat_fn(data)))


def analyze(all_results):
    log_banner("STATISTICAL ANALYSIS")

    n_pairs = len(list(combinations(CONDITIONS, 2)))
    alpha_bonf = 0.05 / n_pairs

    stats = {}
    logger.info("")
    header = (f"{'Condition':<18} {'Violations':>14} {'Viol.Rate':>10} "
              f"{'MeanAngle':>10} {'Ret(final)':>14} {'Reinj':>8}")
    logger.info(header)
    logger.info("-" * len(header))

    for c in CONDITIONS:
        trials = all_results[c]
        vm = np.mean([t['total_violations'] for t in trials])
        vs = np.std([t['total_violations'] for t in trials])
        vr = np.mean([t['violation_rate'] for t in trials])
        ma = np.mean([t['mean_angle'] for t in trials])
        rf = np.mean([t['retention_final'] for t in trials])
        rfs = np.std([t['retention_final'] for t in trials])
        reinj = np.mean([t['n_reinjections'] for t in trials])
        stats[c] = dict(vm=vm, vs=vs, vr=vr, ma=ma, rf=rf, rfs=rfs,
                         reinj=reinj)
        logger.info(f"{CONDITION_LABELS[c]:<18} {vm:>7.1f}±{vs:<5.1f} "
                    f"{vr:>10.4f} {ma:>10.3f} "
                    f"{rf:>7.3f}±{rfs:.3f} {reinj:>8.1f}")

    # Kruskal-Wallis
    logger.info("")
    logger.info("─── Kruskal-Wallis omnibus (violations) ───")
    vdata = {c: [t['total_violations'] for t in all_results[c]]
             for c in CONDITIONS}
    H, p_kw = sp_stats.kruskal(*vdata.values())
    logger.info(f"  H = {H:.2f}, p = {p_kw:.2e}")

    # Pairwise Mann-Whitney
    logger.info("")
    logger.info(f"─── Pairwise Mann-Whitney U (Bonferroni α = "
                f"0.05/{n_pairs} = {alpha_bonf:.4f}) ───")
    logger.info(f"{'Pair':<40} {'U':>8} {'p_adj':>12} {'sig':>6} "
                f"{'d':>8} {'cliff':>8}")
    logger.info("-" * 90)

    pairwise = {}
    for c1, c2 in combinations(CONDITIONS, 2):
        v1, v2 = vdata[c1], vdata[c2]
        U, p_raw = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_adj = min(p_raw * n_pairs, 1.0)
        ps = np.sqrt((np.std(v1)**2 + np.std(v2)**2) / 2)
        d = (np.mean(v1) - np.mean(v2)) / ps if ps > 0 else 0
        cliff = compute_cliffs_delta(v1, v2)
        sig = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
               else "*" if p_adj < 0.05 else "n.s.")
        label = f"{CONDITION_LABELS[c1]} vs {CONDITION_LABELS[c2]}"
        logger.info(f"{label:<40} {U:>8.1f} {p_adj:>12.6f} {sig:>6} "
                    f"{d:>+8.2f} {cliff:>+8.3f}")
        pairwise[(c1, c2)] = dict(U=float(U), p_adj=float(p_adj),
                                   d=float(d), cliff=float(cliff), sig=sig)

    # Bootstrap CIs
    logger.info("")
    logger.info(f"─── Bootstrap 95% CIs ({N_BOOTSTRAP} resamples) ───")
    bootstrap = {}
    for c in CONDITIONS:
        vd = [t['total_violations'] for t in all_results[c]]
        lo, hi, med = bootstrap_ci(vd)
        bootstrap[c] = dict(median=med, ci_lo=lo, ci_hi=hi)
        logger.info(f"  {CONDITION_LABELS[c]:<18}: median={med:.1f}  "
                    f"95% CI=[{lo:.1f}, {hi:.1f}]")

    # Key comparisons
    logger.info("")
    logger.info("─── Key comparisons (directional) ───")
    for c_other in CONDITIONS:
        if c_other == "vsa":
            continue
        v_vsa = vdata["vsa"]
        v_other = vdata[c_other]
        U, p = sp_stats.mannwhitneyu(v_other, v_vsa, alternative='greater')
        p_adj = min(p * n_pairs, 1.0)
        sig = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
               else "*" if p_adj < 0.05 else "n.s.")
        diff = np.mean(v_other) - np.mean(v_vsa)
        logger.info(f"  {CONDITION_LABELS[c_other]} > VSA?  "
                    f"Δ={diff:+.1f}  p_adj={p_adj:.6f} ({sig})")

    return stats, pairwise, bootstrap


# ══════════════════════════════════════════════════════════════════════
# 11. Figures
# ══════════════════════════════════════════════════════════════════════

def make_figures(all_results, stats, training_rewards):
    log_banner("Generating figures")

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.30)

    colors = [CONDITION_COLORS[c] for c in CONDITIONS]
    short = [CONDITION_LABELS[c] for c in CONDITIONS]

    # A: Training curve
    ax1 = fig.add_subplot(gs[0, 0])
    window = 50
    if len(training_rewards) > window:
        smooth = np.convolve(training_rewards,
                             np.ones(window)/window, mode='valid')
        ax1.plot(smooth, color='steelblue', linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("A. LSTM Policy Training (POMDP Pendulum)", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # B: Violation rate over time
    ax2 = fig.add_subplot(gs[0, 1])
    for c, sl, color in zip(CONDITIONS, short, colors):
        tls = [t['violation_timeline'] for t in all_results[c]]
        ml = max(len(tl) for tl in tls)
        pad = np.zeros((len(tls), ml))
        for i, tl in enumerate(tls):
            pad[i, :len(tl)] = tl
        mn = np.mean(pad, axis=0)
        sd = np.std(pad, axis=0)
        x = np.arange(ml) * 1000
        ax2.plot(x, mn, label=sl, color=color, linewidth=2)
        ax2.fill_between(x, np.maximum(mn-sd, 0), mn+sd,
                         color=color, alpha=0.08)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Violation Rate")
    ax2.set_title("B. Safety Violations Over Time", fontweight='bold')
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    # C: Retention over time
    ax3 = fig.add_subplot(gs[0, 2])
    for c, sl, color in zip(CONDITIONS, short, colors):
        curves = [t['retention_snapshots'] for t in all_results[c]]
        ml = max(len(cr) for cr in curves)
        pad = np.full((len(curves), ml), np.nan)
        for i, cr in enumerate(curves):
            pad[i, :len(cr)] = cr
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * 500
        ax3.plot(x, mn, label=sl, color=color, linewidth=2)
        ax3.fill_between(x, mn-sd, mn+sd, color=color, alpha=0.08)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Retention (cosine sim.)")
    ax3.set_title("C. Hidden State Retention", fontweight='bold')
    ax3.legend(fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.3)

    # D: Violations boxplot
    ax4 = fig.add_subplot(gs[1, 0])
    vdata = [[t['total_violations'] for t in all_results[c]]
             for c in CONDITIONS]
    bp = ax4.boxplot(vdata, tick_labels=short, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax4.set_ylabel("Total Violations (20k steps)")
    ax4.set_title(f"D. Violations by Condition (n={N_TRIALS})", fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.3, axis='y')

    # E: Mean angle boxplot
    ax5 = fig.add_subplot(gs[1, 1])
    adata = [[t['mean_angle'] for t in all_results[c]] for c in CONDITIONS]
    bp2 = ax5.boxplot(adata, tick_labels=short, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax5.axhline(y=THETA_SAFE, color='red', linestyle='--', linewidth=1.5,
                label=f'θ_safe={THETA_SAFE}')
    ax5.set_ylabel("Mean |θ| (radians)")
    ax5.set_title("E. Average Angle Deviation", fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # F: Violation rate bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    means = [stats[c]['vr'] for c in CONDITIONS]
    stds = [np.std([t['violation_rate'] for t in all_results[c]])
            for c in CONDITIONS]
    x_pos = np.arange(len(CONDITIONS))
    ax6.bar(x_pos, means, yerr=stds, color=colors, alpha=0.6, capsize=4,
            edgecolor='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(short, rotation=30, ha='right', fontsize=8)
    ax6.set_ylabel("Violation Rate")
    ax6.set_title("F. Mean Violation Rate ± SD", fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # G: VSA drift metric
    ax7 = fig.add_subplot(gs[2, 0])
    dms = [t['drift_metric'] for t in all_results['vsa']
           if t['drift_metric']]
    if dms:
        ml = max(len(d) for d in dms)
        pad = np.full((len(dms), ml), np.nan)
        for i, d in enumerate(dms):
            pad[i, :len(d)] = d
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * VSA_CHECK_INTERVAL
        ax7.plot(x, mn, color=CONDITION_COLORS['vsa'], alpha=0.7, linewidth=1)
        ax7.fill_between(x, mn-sd, mn+sd,
                         color=CONDITION_COLORS['vsa'], alpha=0.1)
        ax7.axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--',
                    label=f'Threshold ({DRIFT_THRESHOLD})')
    ax7.set_xlabel("Time Step")
    ax7.set_ylabel("Drift Metric")
    ax7.set_title("G. VSA Drift Detection", fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # H: Effect sizes vs VSA
    ax8 = fig.add_subplot(gs[2, 1])
    other = [c for c in CONDITIONS if c != "vsa"]
    d_vals = []
    for c in other:
        v_vsa = [t['total_violations'] for t in all_results['vsa']]
        v_c = [t['total_violations'] for t in all_results[c]]
        ps = np.sqrt((np.std(v_vsa)**2 + np.std(v_c)**2) / 2)
        d = (np.mean(v_c) - np.mean(v_vsa)) / ps if ps > 0 else 0
        d_vals.append(d)
    y_pos = np.arange(len(other))
    bc = [CONDITION_COLORS[c] for c in other]
    ax8.barh(y_pos, d_vals, color=bc, alpha=0.6, edgecolor='black')
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([CONDITION_LABELS[c] for c in other], fontsize=9)
    ax8.set_xlabel("Cohen's d (+ = VSA fewer violations)")
    ax8.set_title("H. Effect Sizes vs VSA", fontweight='bold')
    ax8.axvline(0, color='black', linewidth=0.8)
    ax8.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
    ax8.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax8.axvline(0.8, color='gray', linestyle='-', alpha=0.5)
    ax8.grid(True, alpha=0.3, axis='x')

    # I: Reinjection counts
    ax9 = fig.add_subplot(gs[2, 2])
    reinj_c = [c for c in CONDITIONS
               if c not in ("unconstrained", "oracle", "cbf_oracle")]
    reinj_l = [CONDITION_LABELS[c] for c in reinj_c]
    reinj_colors = [CONDITION_COLORS[c] for c in reinj_c]
    reinj_d = [[t['n_reinjections'] for t in all_results[c]]
               for c in reinj_c]
    if reinj_d:
        bp3 = ax9.boxplot(reinj_d, tick_labels=reinj_l, patch_artist=True)
        for patch, color in zip(bp3['boxes'], reinj_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax9.set_ylabel("Re-injections")
    ax9.set_title("I. Re-injection Count", fontweight='bold')
    ax9.tick_params(axis='x', rotation=30)
    ax9.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        "LSTM POMDP: VSA vs Baselines for Long-Horizon Safety\n"
        f"Pendulum (POMDP) | {N_STEPS} steps × {N_TRIALS} trials | "
        f"θ_safe={THETA_SAFE} rad | D={D_VSA}",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches='tight')
    logger.info(f"  Figure saved: {FIGURE_PATH}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 12. Save Results
# ══════════════════════════════════════════════════════════════════════

def save_results(all_results, stats, pairwise, bootstrap, training_rewards):
    save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "LSTM_POMDP_Pendulum",
            "n_steps": N_STEPS,
            "n_trials": N_TRIALS,
            "d_vsa": D_VSA,
            "theta_safe": THETA_SAFE,
            "lstm_hidden": LSTM_HIDDEN,
            "obs_dim": OBS_DIM,
            "pomdp": True,
            "train_iterations": TRAIN_ITERATIONS,
            "train_eps_per_batch": TRAIN_EPS_PER_BATCH,
            "drift_threshold": DRIFT_THRESHOLD,
            "timer_interval": TIMER_INTERVAL,
            "ema_alpha": EMA_ALPHA,
            "cbf_gamma": CBF_GAMMA,
            "norm_drop_frac": NORM_DROP_FRAC,
            "perturb_interval": PERTURB_INTERVAL,
            "perturb_strength": PERTURB_STRENGTH,
            "n_bootstrap": N_BOOTSTRAP,
            "device": str(DEVICE),
        },
        "training": {
            "final_avg_reward": float(np.mean(training_rewards[-50:])),
            "reward_history_downsampled": [float(x) for x in
                                           training_rewards[::10]],
        },
        "summary": stats,
        "pairwise_tests": {f"{c1}_vs_{c2}": v
                           for (c1, c2), v in pairwise.items()},
        "bootstrap_cis": bootstrap,
    }
    for c in CONDITIONS:
        save[c] = [{k: v for k, v in t.items() if k != "drift_metric"}
                   for t in all_results[c]]

    with open(RESULTS_JSON, 'w') as f:
        json.dump(save, f, indent=2)
    logger.info(f"  Results saved: {RESULTS_JSON}")


# ══════════════════════════════════════════════════════════════════════
# 13. Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_banner("LSTM POMDP Safety Experiment")
    logger.info(f"  Python {sys.version.split()[0]}")
    logger.info(f"  PyTorch {torch.__version__}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Phase 1: Train LSTM Policy ──
    policy = LSTMPolicy()

    if os.path.exists(MODEL_PATH):
        logger.info(f"\n  Loading pre-trained policy from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE,
                                weights_only=True)
        policy.load_state_dict(checkpoint['policy'])
        training_rewards = checkpoint.get('training_rewards', [])
        policy.to(DEVICE)
    else:
        training_rewards = train_policy(policy)
        torch.save({
            'policy': policy.state_dict(),
            'training_rewards': training_rewards,
        }, MODEL_PATH)
        logger.info(f"  Policy saved: {MODEL_PATH}")

    # Evaluate
    eval_rewards, eval_violations = evaluate_policy(policy)

    # ── Phase 2: Compute Safe Reference ──
    if os.path.exists(REF_C_PATH):
        logger.info(f"\n  Loading safe reference from {REF_C_PATH}")
        ref_c = torch.load(REF_C_PATH, map_location=DEVICE,
                           weights_only=True)
    else:
        ref_c = compute_safe_reference(policy)
        torch.save(ref_c, REF_C_PATH)
        logger.info(f"  Safe reference saved: {REF_C_PATH}")

    # ── Phase 2b: Calibration ──
    milestones = calibrate_recovery(policy, ref_c)

    # ── Phase 3: Safety Benchmark ──
    all_results = run_experiment(policy, ref_c)

    # ── Phase 4: Analysis ──
    stats, pairwise, bootstrap = analyze(all_results)

    # ── Phase 5: Figures ──
    make_figures(all_results, stats, training_rewards)

    # ── Phase 6: Save ──
    save_results(all_results, stats, pairwise, bootstrap, training_rewards)

    # ── Final Summary ──
    log_banner("EXPERIMENT COMPLETE")
    logger.info("")
    logger.info("  RANKING (by mean violations):")
    ranked = sorted(CONDITIONS, key=lambda c: stats[c]['vm'])
    for rank, c in enumerate(ranked, 1):
        s = stats[c]
        logger.info(f"    {rank}. {CONDITION_LABELS[c]:<18} "
                    f"{s['vm']:>7.1f} ± {s['vs']:.1f} violations  "
                    f"(retention: {s['rf']:.3f}, angle: {s['ma']:.3f})")

    logger.info("")
    logger.info(f"  Log:     {LOG_FILE}")
    logger.info(f"  Results: {RESULTS_JSON}")
    logger.info(f"  Figure:  {FIGURE_PATH}")
    logger.info(f"  Model:   {MODEL_PATH}")
    logger.info("")
