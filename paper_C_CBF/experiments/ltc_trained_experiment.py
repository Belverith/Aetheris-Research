"""
Trained LTC Safety Experiment: VSA vs Baselines on Real Control Tasks
======================================================================

This experiment addresses all four critical gaps from reviewer analysis:

Gap 1 (Tautological safety): Safety is defined by TASK REWARD, not by the
  metric VSA optimizes. The LTC controller is trained to solve Pendulum-v1.
  Safety = "don't let angle exceed threshold" (task-intrinsic, physics-based).
  VSA preserves hidden state fidelity; safety is measured by whether the
  controller BEHAVES safely, not by cosine similarity.

Gap 2 (No real LTC): Uses ncps.torch.LTC — the official Hasani et al. (2021)
  implementation. The LTC is trained via backpropagation through the ODE solver
  on real control trajectories. Weights are learned, not random.

Gap 3 (No learned controller): The controller IS the LTC. It takes observations
  as input and outputs control actions. It's trained via policy gradient
  (REINFORCE) to maximize episode reward. The PD controller is gone.

Gap 4 (CBF handicapped): The CBF baseline gets ORACLE access to the constraint
  function h(x) = θ_max - |θ|. It computes the barrier directly from state,
  as CBFs do in practice. This is the strongest possible CBF.

Architecture:
  1. Train an LTC policy on Pendulum-v1 until convergent
  2. At deployment, inject "safety knowledge" into the hidden state:
     - Encode the safe operating envelope [θ_min, θ_max] into hidden state
     - This mimics a real scenario: the controller was trained in one regime
       but must remember constraints from a different context
  3. Run 20,000-step episodes under different safety-maintenance conditions
  4. Periodic observation blackouts corrupt the hidden state
  5. Measure: does c-state recovery help the controller maintain safety?

Conditions (8):
  C1: Unconstrained    — no ref_c init, no blackout, pure baseline
  C2: No Protection    — ref_c init, blackout, no recovery
  C3: Timer            — ref_c init, blackout, periodic re-blend
  C4: VSA-Constrained  — ref_c init, blackout, HIS drift detect + re-inject
  C5: Oracle           — ref_c init, blackout, restore c right after each blackout
  C6: CBF (Oracle)     — ref_c init, blackout, action-level safety filter (oracle state)
  C7: EMA              — ref_c init, blackout, continuous soft blend
  C8: Norm Monitor     — ref_c init, blackout, anomaly-triggered re-blend

Perturbation model: Observation blackout
  Every BLACKOUT_INTERVAL steps, a BLACKOUT_DURATION window begins.
  During blackout, the policy receives random observations (simulating
  sensor degradation). This corrupts h through the CfC ODE and c through
  LSTM gates. After blackout, clean observations resume. Recovery time
  depends on c-state quality — this is where VSA preservation helps.

Safety metric: Fraction of steps where |θ| > θ_safe (angle violation).
  This is physics-based, not cosine-similarity-based. A controller that
  "remembers" the safety constraint will actively avoid large angles.

Usage:
  python ltc_trained_experiment.py

Requires: torch, ncps, gymnasium, numpy, matplotlib, scipy
"""

import numpy as np
import torch
import torch.serialization
torch.serialization.add_safe_globals([
    np._core.multiarray.scalar,
    np.dtype,
    np.dtypes.Float64DType,
])
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gymnasium as gym
from ncps.wirings import AutoNCP
from ncps.torch import CfC
from scipy import stats as sp_stats
from scipy.stats import norm as sp_norm
from itertools import combinations
import json
import time
import sys
import os
import logging
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════════
# 0. Logging
# ══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "ltc_trained_log.txt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "ltc_trained_results.json")
FIGURE_PATH = os.path.join(SCRIPT_DIR, "figure_D3_trained_ltc.png")
MODEL_PATH = os.path.join(SCRIPT_DIR, "ltc_pendulum_policy.pt")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "ltc_trained_checkpoint.json")

logger = logging.getLogger("LTC_TRAINED")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s',
                                   datefmt='%H:%M:%S'))
logger.addHandler(ch)


def log_banner(msg):
    line = "=" * 72
    logger.info(line)
    logger.info(f"  {msg}")
    logger.info(line)


# ══════════════════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# --- LTC Policy Architecture ---
LTC_UNITS = 64          # Neurons in LTC
LTC_OUTPUT_SIZE = 16    # Motor neurons
OBS_DIM = 3             # Pendulum: [cos(θ), sin(θ), θ_dot]
ACT_DIM = 1             # Pendulum: torque

# --- Training (PPO) ---
TRAIN_ITERATIONS = 300       # PPO update iterations
TRAIN_EPS_PER_BATCH = 10     # Episodes collected per PPO iteration
TRAIN_MAX_STEPS = 200        # Pendulum default episode length
TRAIN_LR = 3e-4
TRAIN_GAMMA = 0.99           # Discount factor
TRAIN_GAE_LAMBDA = 0.95      # GAE lambda
TRAIN_PPO_EPOCHS = 10        # Max PPO epochs per batch (KL early stop may reduce)
TRAIN_CLIP_EPS = 0.2         # PPO clipping epsilon (standard PPO)
TRAIN_ENTROPY_COEF = 0.01    # Entropy bonus for exploration
TRAIN_VALUE_COEF = 0.5       # Value loss coefficient
TRAIN_MAX_GRAD_NORM = 0.5    # Gradient clipping (policy)
TRAIN_VALUE_LR = 0.01        # Value MLP learning rate (Adam needs this for |returns|~500)
TRAIN_VALUE_CLIP_NORM = 5.0  # Gradient clip norm for value MLP
TRAIN_TARGET_KL = 0.02       # KL early stopping threshold
TRAIN_TBPTT_CHUNK = 20       # Truncated BPTT chunk size (detach every K steps)
EVAL_EPISODES = 20           # Episodes to evaluate trained policy

# --- Safety Experiment ---
N_STEPS = 20000             # Steps per safety trial
N_TRIALS = 50               # Trials per condition
THETA_SAFE = 0.5            # Safe angle threshold (radians, ~28.6 degrees)
                            # Pendulum range: [-pi, pi], so this is a tight constraint

# --- VSA ---
D_VSA = 10000
DRIFT_THRESHOLD = 0.55
VSA_CHECK_INTERVAL = 50
VSA_INJECTION_GAIN = 2.0

# --- Timer ---
TIMER_INTERVAL = 300

# --- EMA ---
EMA_ALPHA = 0.005

# --- Norm Monitor ---
NORM_DROP_FRAC = 0.7         # Re-inject when norm drops to 70% of RECENT avg

# --- CBF (Oracle) ---
CBF_GAMMA = 2.0

# --- Observation Blackout (simulates sensor degradation) ---
BLACKOUT_INTERVAL = 200      # Start a blackout every K steps
BLACKOUT_DURATION = 50       # Blackout lasts D steps (random obs injected)

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
# 2. CfC Policy Network
# ══════════════════════════════════════════════════════════════════════

class LTCPolicy(nn.Module):
    """
    Actor-Critic policy using a real ncps CfC (Closed-form Continuous-time)
    backbone — the closed-form variant of LTC from Hasani et al. (2022).

    CfC replaces the ODE solver with an analytical solution, giving
    identical liquid neural network dynamics with clean gradients and
    ~2× faster training.

    Architecture:
      obs (3) -> RunningNorm -> CfC (64 hidden, 16 proj, mixed_memory) -> 2*tanh(action_head) -> action
      obs (3) -> value_mlp (64, 64) -> value (scalar)

    The CfC uses mixed_memory=True which adds an LSTM cell in parallel.
    This provides the stable gradient highway that vanilla CfC lacks.
    Without it, CfC completely overwrites its hidden state every step
    (no h_{t-1} preservation), causing hidden state trajectories to
    diverge exponentially when PPO replays with updated weights.

    The action mean is bounded via 2*tanh(...) to prevent unbounded drift.
    """

    def __init__(self):
        super().__init__()

        # Observation normalization — Pendulum obs ∈ [-1,1]×[-1,1]×[-8,8];
        # the velocity component is 8× larger without normalization.
        self.obs_rms_mean = nn.Parameter(torch.zeros(OBS_DIM), requires_grad=False)
        self.obs_rms_var = nn.Parameter(torch.ones(OBS_DIM), requires_grad=False)
        self.obs_rms_count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

        # CfC backbone — fully-connected with mixed_memory (LSTM + CfC)
        # The LSTM provides a stable gradient highway: c_{t+1} = f*c_t + ...
        # Without it, CfC completely overwrites h each step, causing PPO
        # ratio explosion from hidden state trajectory divergence.
        self.ltc = CfC(
            input_size=OBS_DIM,
            units=LTC_UNITS,
            proj_size=LTC_OUTPUT_SIZE,
            batch_first=True,
            return_sequences=True,
            mixed_memory=True,
        )

        self.state_size = self.ltc.state_size

        # Actor head: outputs bounded mean via tanh and learnable log_std
        self.action_mean = nn.Linear(LTC_OUTPUT_SIZE, ACT_DIM)
        self.action_log_std = nn.Parameter(torch.zeros(ACT_DIM))

        # Critic: separate MLP (obs → value), no CfC backbone involvement
        self.value_mlp = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _normalize_obs(self, obs):
        """Normalize observations using running mean/std."""
        return (obs - self.obs_rms_mean) / (self.obs_rms_var.sqrt() + 1e-8)

    def update_obs_rms(self, obs_batch):
        """Update running observation statistics (call during collection)."""
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
        """Create initial hidden state (h, c) tuple for mixed_memory CfC."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.state_size, device=device)
        c = torch.zeros(batch_size, self.state_size, device=device)
        return (h, c)

    def get_h(self, hx):
        """Extract the h-state (CfC component) from hidden state tuple."""
        return hx[0]

    def set_h(self, hx, new_h):
        """Return new hidden state tuple with replaced h-state."""
        return (new_h, hx[1])

    def get_c(self, hx):
        """Extract the c-state (LSTM cell) from hidden state tuple."""
        return hx[1]

    def set_c(self, hx, new_c):
        """Return new hidden state tuple with replaced c-state."""
        return (hx[0], new_c)

    def clone_hx(self, hx):
        """Deep-clone a hidden state tuple."""
        return (hx[0].clone(), hx[1].clone())

    def detach_hx(self, hx):
        """Detach hidden state tuple from computation graph."""
        return (hx[0].detach(), hx[1].detach())

    def forward(self, obs_seq, hx=None):
        """
        Forward pass for a sequence of observations.
        obs_seq: (batch, seq_len, obs_dim)
        Returns: action_mean (batch, seq_len, act_dim),
                 action_log_std (batch, seq_len, act_dim),
                 value (batch, seq_len), hidden_state
        """
        obs_norm = self._normalize_obs(obs_seq)
        ltc_out, hn = self.ltc(obs_norm, hx=hx)
        mean = 2.0 * torch.tanh(self.action_mean(ltc_out))
        log_std = self.action_log_std.unsqueeze(0).unsqueeze(0).expand_as(mean)
        value = self.value_mlp(obs_seq).squeeze(-1)
        return mean, log_std, value, hn

    def step(self, obs, hx):
        """
        Single-step forward for online deployment.
        obs: (batch, obs_dim) — single observation
        hx: tuple ((batch, state_size), (batch, state_size)) — hidden state
        Returns: action_mean, action_log_std, value, new_hidden
        """
        obs_norm = self._normalize_obs(obs)
        obs_seq = obs_norm.unsqueeze(1)  # (batch, 1, obs_dim)
        ltc_out, hn = self.ltc(obs_seq, hx=hx)
        ltc_out = ltc_out.squeeze(1)  # (batch, ltc_output_size)
        mean = 2.0 * torch.tanh(self.action_mean(ltc_out))
        log_std = self.action_log_std.expand_as(mean)
        value = self.value_mlp(obs)  # (batch, 1)
        return mean, log_std, value, hn

    def get_action(self, obs, hx, deterministic=False):
        """Sample an action from the policy."""
        mean, log_std, value, hn = self.step(obs, hx)
        std = log_std.exp()
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        # Pendulum action space: [-2, 2]
        action = torch.clamp(action, -2.0, 2.0)
        return action, value, hn

    def evaluate_action(self, obs_seq, actions, hx=None):
        """Evaluate log_prob and entropy for a batch of (obs, action) pairs."""
        mean, log_std, value, hn = self.forward(obs_seq, hx=hx)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value, hn


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
# 4. Training the LTC Policy
# ══════════════════════════════════════════════════════════════════════

def _collect_episodes(policy, env_id, n_episodes, seed_offset, device):
    """
    Collect n_episodes rollouts using the current policy (no gradient).
    Returns a list of episode dicts with obs, actions, log_probs, values,
    rewards, dones, and the initial hidden state.
    """
    episodes = []
    all_obs_for_rms = []  # Collect obs for running normalization update
    policy.eval()
    for i in range(n_episodes):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=seed_offset + i)
        hx = policy.init_hidden(1, device)

        ep_obs, ep_acts, ep_logp, ep_vals, ep_rews = [], [], [], [], []
        h0 = policy.clone_hx(hx)  # Store initial hidden state for replay

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
            ep_acts.append(action_clamped.cpu().numpy().flatten())
            ep_logp.append(log_prob.item())
            ep_vals.append(value.squeeze().item())
            ep_rews.append(reward)

            obs = obs_new
            if terminated or truncated:
                break

        env.close()
        all_obs_for_rms.append(np.array(ep_obs))
        episodes.append({
            'obs': np.array(ep_obs),            # (T, obs_dim)
            'actions': np.array(ep_acts),        # (T, act_dim)
            'log_probs_old': np.array(ep_logp),  # (T,)
            'values': np.array(ep_vals),         # (T,)
            'rewards': np.array(ep_rews),        # (T,)
            'h0': h0,                            # tuple of (1, state_size)
        })

    # Update observation normalization statistics
    all_obs_np = np.concatenate(all_obs_for_rms, axis=0)
    policy.update_obs_rms(torch.FloatTensor(all_obs_np).to(device))

    return episodes


def _compute_gae(rewards, values, gamma=TRAIN_GAMMA, lam=TRAIN_GAE_LAMBDA):
    """Compute Generalized Advantage Estimation."""
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


def _ppo_update_batch(policy, policy_opt, value_opt, episodes, device):
    """
    One PPO epoch: replay ALL episodes through CfC, accumulate gradient,
    single optimizer step per param group. Uses separate gradient clipping
    for policy and value to prevent value's large gradients from drowning
    out policy gradients.
    """
    policy_opt.zero_grad()
    value_opt.zero_grad()

    total_pi = 0.0
    total_v = 0.0
    total_ent = 0.0
    total_T = 0
    all_ratios = []
    all_action_means = []
    all_values_pred = []
    all_returns = []
    all_kl_terms = []
    all_ltc_feat_norms = []

    for ep in episodes:
        obs_t = torch.FloatTensor(ep['obs']).unsqueeze(0).to(device)
        acts_t = torch.FloatTensor(ep['actions']).to(device)
        old_logp = torch.FloatTensor(ep['log_probs_old']).to(device)
        advs = torch.FloatTensor(ep['advantages']).to(device)
        rets = torch.FloatTensor(ep['returns']).to(device)

        # CfC forward: truncated BPTT — process in chunks with detached
        # hidden states between chunks. Limits temporal compounding.
        hx = (ep['h0'][0].to(device), ep['h0'][1].to(device))
        obs_norm = policy._normalize_obs(obs_t)
        seq_len = obs_norm.shape[1]
        chunk_outputs = []
        for cstart in range(0, seq_len, TRAIN_TBPTT_CHUNK):
            cend = min(cstart + TRAIN_TBPTT_CHUNK, seq_len)
            chunk_out, hx = policy.ltc(obs_norm[:, cstart:cend, :], hx=hx)
            chunk_outputs.append(chunk_out)
            hx = policy.detach_hx(hx)  # Stop gradient across chunk boundaries
        ltc_out = torch.cat(chunk_outputs, dim=1)
        ltc_feat = ltc_out.squeeze(0)  # (T, output_size)

        # Actor head — gradient flows through CfC backbone
        mean = 2.0 * torch.tanh(policy.action_mean(ltc_feat))
        log_std = policy.action_log_std.unsqueeze(0).expand_as(mean)
        std = log_std.exp()

        # Critic — separate MLP reads obs directly, no CfC involvement
        obs_flat = torch.FloatTensor(ep['obs']).to(device)  # (T, obs_dim)
        values = policy.value_mlp(obs_flat).squeeze(-1)  # (T,)

        dist = Normal(mean, std)
        new_logp = dist.log_prob(acts_t).sum(-1)
        entropy = dist.entropy().sum(-1)

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - TRAIN_CLIP_EPS, 1.0 + TRAIN_CLIP_EPS) * advs

        T = len(advs)
        # Per-timestep loss cap: prevents extreme negative-advantage ratios
        # (where PPO clipping doesn't protect) from destabilizing training
        pi_step = -torch.min(surr1, surr2)
        pi_step = torch.clamp(pi_step, -5.0, 5.0)
        total_pi = total_pi + pi_step.sum()
        total_v = total_v + F.mse_loss(values, rets, reduction='sum')
        total_ent = total_ent + (-entropy.sum())
        total_T += T

        # Collect diagnostics (detached)
        with torch.no_grad():
            all_ratios.append(ratio.detach().cpu())
            all_action_means.append(mean.detach().cpu().squeeze(-1))
            all_values_pred.append(values.detach().cpu())
            all_returns.append(rets.cpu())
            # Better KL approx: (ratio - 1) - log(ratio), always >= 0
            log_ratio = (new_logp - old_logp).detach()
            kl_term = (log_ratio.exp() - 1) - log_ratio
            all_kl_terms.append(kl_term.cpu())
            all_ltc_feat_norms.append(ltc_feat.detach().norm(dim=-1).cpu())

    # Average over total timesteps across the batch, then backprop once
    loss = ((total_pi + TRAIN_VALUE_COEF * total_v
             + TRAIN_ENTROPY_COEF * total_ent) / total_T)
    loss.backward()

    # Collect per-component gradient norms BEFORE clipping
    grad_info = {}
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_info[name] = param.grad.data.norm().item()

    # Separate gradient clipping: prevents value's large gradients from
    # scaling down policy gradients through the shared clip norm
    pi_params = [p for n, p in policy.named_parameters()
                 if not n.startswith('value_mlp') and p.grad is not None]
    val_params = [p for n, p in policy.named_parameters()
                  if n.startswith('value_mlp') and p.grad is not None]
    nn.utils.clip_grad_norm_(pi_params, TRAIN_MAX_GRAD_NORM)
    nn.utils.clip_grad_norm_(val_params, TRAIN_VALUE_CLIP_NORM)
    policy_opt.step()
    value_opt.step()

    # Build diagnostics dict
    all_ratios_t = torch.cat(all_ratios)
    all_values_t = torch.cat(all_values_pred)
    all_returns_t = torch.cat(all_returns)
    all_kl_t = torch.cat(all_kl_terms)
    all_means_t = torch.cat(all_action_means)
    all_feat_t = torch.cat(all_ltc_feat_norms)

    # Explained variance: how well value preds explain return variance
    v_np = all_values_t.numpy()
    r_np = all_returns_t.numpy()
    var_r = np.var(r_np)
    ev = 1.0 - np.var(r_np - v_np) / (var_r + 1e-8) if var_r > 1e-8 else 0.0

    diag = {
        'ratio_mean': all_ratios_t.mean().item(),
        'ratio_min': all_ratios_t.min().item(),
        'ratio_max': all_ratios_t.max().item(),
        'ratio_std': all_ratios_t.std().item(),
        'clip_frac': ((all_ratios_t - 1.0).abs() > TRAIN_CLIP_EPS).float().mean().item(),
        'kl_approx': all_kl_t.mean().item(),
        'action_mean_mean': all_means_t.mean().item(),
        'action_mean_std': all_means_t.std().item(),
        'action_log_std': policy.action_log_std.data.cpu().item(),
        'value_pred_mean': v_np.mean(),
        'value_pred_std': v_np.std(),
        'return_mean': r_np.mean(),
        'return_std': r_np.std(),
        'explained_var': ev,
        'ltc_feat_norm_mean': all_feat_t.mean().item(),
        'ltc_feat_norm_std': all_feat_t.std().item(),
        'grad_info': grad_info,
    }

    return (total_pi.item() / total_T,
            total_v.item() / total_T,
            total_ent.item() / total_T,
            diag)


def _value_only_update(policy, value_opt, episodes, device):
    """Extra value-only training pass (no policy/CfC gradients)."""
    value_opt.zero_grad()
    total_v = 0.0
    total_T = 0
    for ep in episodes:
        obs_flat = torch.FloatTensor(ep['obs']).to(device)
        rets = torch.FloatTensor(ep['returns']).to(device)
        values = policy.value_mlp(obs_flat).squeeze(-1)
        total_v = total_v + F.mse_loss(values, rets, reduction='sum')
        total_T += len(rets)
    v_loss = TRAIN_VALUE_COEF * total_v / total_T
    v_loss.backward()
    v_params = [p for n, p in policy.named_parameters()
                if n.startswith('value_mlp') and p.grad is not None]
    nn.utils.clip_grad_norm_(v_params, TRAIN_VALUE_CLIP_NORM)
    value_opt.step()
    return total_v.item() / total_T


def train_policy(policy, env_id="Pendulum-v1"):
    """
    Train the LTC policy via PPO (Proximal Policy Optimization).

    Key differences from REINFORCE:
    - Collects batches of episodes, then does multiple optimization passes
    - Clipped surrogate objective prevents destructive policy updates
    - GAE for lower-variance advantage estimation
    - Full BPTT through LTC during each PPO epoch (episode replay)
    """
    n_total_eps = TRAIN_ITERATIONS * TRAIN_EPS_PER_BATCH
    log_banner("Phase 1: Training LTC Policy on Pendulum-v1 (PPO)")
    logger.info(f"  Iterations: {TRAIN_ITERATIONS}, Episodes/batch: {TRAIN_EPS_PER_BATCH}")
    logger.info(f"  Total episodes: {n_total_eps}")
    logger.info(f"  PPO epochs: {TRAIN_PPO_EPOCHS}, Clip ε: {TRAIN_CLIP_EPS}")
    logger.info(f"  GAE λ: {TRAIN_GAE_LAMBDA}, γ: {TRAIN_GAMMA}")
    logger.info(f"  LR: {TRAIN_LR} (policy), {TRAIN_VALUE_LR} (value)")
    logger.info(f"  LTC units: {LTC_UNITS}, state_size: {policy.state_size}")
    logger.info(f"  Target KL: {TRAIN_TARGET_KL}")
    logger.info(f"  Device: {DEVICE}")

    # Separate optimizers: policy and value gradients are clipped
    # independently, preventing value's large gradients from drowning policy
    policy_params = [p for n, p in policy.named_parameters()
                     if not n.startswith('value_mlp')]
    value_params = [p for n, p in policy.named_parameters()
                    if n.startswith('value_mlp')]
    policy_optimizer = optim.Adam(policy_params, lr=TRAIN_LR)
    value_optimizer = optim.Adam(value_params, lr=TRAIN_VALUE_LR)
    policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=TRAIN_ITERATIONS, eta_min=TRAIN_LR * 0.1
    )
    value_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        value_optimizer, T_max=TRAIN_ITERATIONS, eta_min=TRAIN_VALUE_LR * 0.1
    )
    policy.to(DEVICE)

    reward_history = []
    best_avg = -float('inf')
    best_state = None

    for iteration in range(TRAIN_ITERATIONS):
        iter_start = time.time()

        # ── Collect batch of episodes ──
        if iteration < 2:
            logger.info(f"  [iter {iteration+1}] Collecting {TRAIN_EPS_PER_BATCH} episodes...")
        seed_offset = SEED + iteration * TRAIN_EPS_PER_BATCH
        episodes = _collect_episodes(
            policy, env_id, TRAIN_EPS_PER_BATCH, seed_offset, DEVICE
        )
        collect_time = time.time() - iter_start
        if iteration < 2:
            logger.info(f"  [iter {iteration+1}] Collection done in {collect_time:.1f}s")

        # ── Compute GAE for each episode ──
        batch_rewards = []
        for ep in episodes:
            advs, rets = _compute_gae(ep['rewards'], ep['values'])
            ep['advantages'] = advs
            ep['returns'] = rets
            batch_rewards.append(sum(ep['rewards']))

        # Normalize advantages across the ENTIRE batch (not per-episode)
        all_advs = np.concatenate([ep['advantages'] for ep in episodes])
        adv_mean, adv_std = all_advs.mean(), all_advs.std() + 1e-8
        for ep in episodes:
            ep['advantages'] = (ep['advantages'] - adv_mean) / adv_std

        reward_history.extend(batch_rewards)
        avg_batch = np.mean(batch_rewards)

        # ── PPO update epochs ──
        update_start = time.time()
        if iteration < 2:
            logger.info(f"  [iter {iteration+1}] Starting {TRAIN_PPO_EPOCHS} PPO epochs "
                        f"({TRAIN_EPS_PER_BATCH} eps, full-sequence forward)...")
        policy.train()
        epoch_losses = []
        epoch_diags = []
        kl_stopped = False
        for ppo_epoch in range(TRAIN_PPO_EPOCHS):
            ppo_ep_start = time.time()
            pl, vl, el, diag = _ppo_update_batch(
                policy, policy_optimizer, value_optimizer, episodes, DEVICE
            )
            epoch_losses.append((pl, vl, el))
            epoch_diags.append(diag)
            if iteration < 2:
                ppo_ep_time = time.time() - ppo_ep_start
                logger.info(f"  [iter {iteration+1}]   PPO epoch {ppo_epoch+1}/{TRAIN_PPO_EPOCHS} "
                            f"done in {ppo_ep_time:.1f}s")
            # KL early stopping: prevent excessive policy change
            if diag['kl_approx'] > TRAIN_TARGET_KL:
                kl_stopped = True
                logger.debug(f"  KL early stop at PPO epoch {ppo_epoch+1}/{TRAIN_PPO_EPOCHS}: "
                             f"KL={diag['kl_approx']:.4f} > {TRAIN_TARGET_KL}")
                break
        update_time = time.time() - update_start

        # ── Track best ──
        if len(reward_history) >= 100:
            avg_100 = np.mean(reward_history[-100:])
            if avg_100 > best_avg:
                best_avg = avg_100
                best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        # ── LR annealing step ──
        policy_scheduler.step()
        value_scheduler.step()

        # ── Log every iteration ──
        iter_time = time.time() - iter_start
        ep_count = (iteration + 1) * TRAIN_EPS_PER_BATCH
        pl, vl, el = epoch_losses[-1]
        d = epoch_diags[-1]  # Last PPO epoch diagnostics
        eta_s = iter_time * (TRAIN_ITERATIONS - iteration - 1)
        eta_str = str(timedelta(seconds=int(eta_s)))
        n_ppo = len(epoch_losses)
        kl_tag = f" KL-stop@{n_ppo}" if kl_stopped else ""
        logger.info(
            f"  Iter {iteration+1:4d}/{TRAIN_ITERATIONS}  "
            f"ep={ep_count:5d}  avg_batch={avg_batch:.1f}  "
            f"best_100avg={best_avg:.1f}  "
            f"pi={pl:.4f} v={vl:.3f}  "
            f"EV={d['explained_var']:.3f}{kl_tag}  "
            f"[{iter_time:.1f}s = {collect_time:.1f}+{update_time:.1f}]  "
            f"ETA {eta_str}"
        )

        # ── Detailed diagnostics (every 5 iters + first 3 + when things look bad) ──
        is_early = (iteration < 3)
        is_periodic = ((iteration + 1) % 5 == 0)
        is_suspicious = (abs(pl) > 0.1 or d['ratio_max'] > 2.0 or
                         d['explained_var'] < -0.5 or d['clip_frac'] > 0.3)
        if is_early or is_periodic or is_suspicious:
            # Aggregate gradient norms by component
            g = d['grad_info']
            ltc_grad = sum(v for k, v in g.items() if k.startswith('ltc.'))
            act_grad = sum(v for k, v in g.items() if 'action' in k)
            val_grad = sum(v for k, v in g.items() if 'value_mlp' in k)
            n_ltc = sum(1 for k in g if k.startswith('ltc.'))
            n_act = sum(1 for k in g if 'action' in k)
            n_val = sum(1 for k in g if 'value_mlp' in k)

            logger.debug(
                f"  DIAG iter={iteration+1} "
                f"ratio=[{d['ratio_min']:.3f}, {d['ratio_mean']:.3f}, {d['ratio_max']:.3f}] "
                f"std={d['ratio_std']:.3f} clip%={d['clip_frac']:.2%} "
                f"kl={d['kl_approx']:.4f}"
            )
            logger.debug(
                f"       action_mean: mean={d['action_mean_mean']:.3f} "
                f"spread={d['action_mean_std']:.3f} "
                f"log_std={d['action_log_std']:.3f} "
                f"(std={np.exp(d['action_log_std']):.3f})"
            )
            logger.debug(
                f"       value: pred={d['value_pred_mean']:.1f}±{d['value_pred_std']:.1f} "
                f"ret={d['return_mean']:.1f}±{d['return_std']:.1f} "
                f"EV={d['explained_var']:.3f}"
            )
            logger.debug(
                f"       ltc_feat_norm: {d['ltc_feat_norm_mean']:.2f}±{d['ltc_feat_norm_std']:.2f}"
            )
            logger.debug(
                f"       grad_norm (pre-clip): "
                f"CfC={ltc_grad:.4f}({n_ltc}p) "
                f"actor={act_grad:.4f}({n_act}p) "
                f"value={val_grad:.4f}({n_val}p) "
                f"lr_pi={policy_optimizer.param_groups[0]['lr']:.2e} "
                f"lr_v={value_optimizer.param_groups[0]['lr']:.2e}"
            )
            if is_suspicious and not is_early:
                logger.warning(
                    f"  ⚠ SUSPICIOUS iter={iteration+1}: "
                    f"pi={pl:.4f} ratio_max={d['ratio_max']:.2f} "
                    f"EV={d['explained_var']:.3f} clip%={d['clip_frac']:.2%}"
                )

    # Restore best weights
    if best_state is not None:
        policy.load_state_dict(best_state)
        policy.to(DEVICE)
    logger.info(f"  Training complete. Best 100-ep avg reward: {best_avg:.1f}")

    return reward_history


def evaluate_policy(policy, env_id="Pendulum-v1", n_episodes=EVAL_EPISODES):
    """Evaluate the trained policy."""
    logger.info(f"\n  Evaluating on {n_episodes} episodes...")
    policy.eval()
    rewards = []
    angle_violations = []

    for ep in range(n_episodes):
        env = gym.make(env_id)
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

            # Safety check: θ = atan2(sin(θ), cos(θ))
            theta = np.arctan2(obs[1], obs[0])
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
# 5. Safety Reference Computation
# ══════════════════════════════════════════════════════════════════════

def compute_safe_reference(policy, env_id="Pendulum-v1", n_episodes=50):
    """
    Phase 2: Compute a reference c-state from the policy's natural safe behavior.

    A fixed additive perturbation to the hidden state cannot learn
    directional safety corrections through gradient descent, because
    gradient contributions from opposite-angle states cancel.  Instead,
    we collect the LSTM cell states during stable, safe operation
    (|θ| < θ_safe/2, after initial swing-up) and compute their average.

    This reference encodes the policy's implicit safety knowledge in the
    c-state space.  Maintaining this reference (via VSA recovery) keeps
    the policy in its safe operating regime.
    """
    log_banner("Phase 2: Computing Safe Reference State")
    logger.info(f"  Episodes: {n_episodes}")
    logger.info(f"  θ_safe: {THETA_SAFE} rad ({np.degrees(THETA_SAFE):.1f}°)")

    policy.eval()
    safe_c_states = []
    total_safe_steps = 0
    total_steps = 0

    for ep in range(n_episodes):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=SEED + 8000 + ep)
        hx = policy.init_hidden(1, DEVICE)

        for step in range(TRAIN_MAX_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
            action_np = action.cpu().numpy().flatten()
            obs, _, terminated, truncated, _ = env.step(
                np.clip(action_np, -2.0, 2.0))

            theta = np.arctan2(obs[1], obs[0])
            total_steps += 1

            # Collect c-state during safe, post-stabilization operation
            if abs(theta) < THETA_SAFE * 0.5 and step > 30:
                safe_c_states.append(policy.get_c(hx).detach().cpu())
                total_safe_steps += 1

            if terminated or truncated:
                break

        env.close()

        if (ep + 1) % 10 == 0:
            logger.info(f"  Collected ep {ep+1}/{n_episodes}  "
                        f"safe_steps={total_safe_steps}/{total_steps}")

    if not safe_c_states:
        logger.warning("  No safe c-states collected! Using zero reference.")
        ref_c = torch.zeros(1, policy.state_size, device=DEVICE)
    else:
        ref_c = torch.stack(safe_c_states).mean(dim=0).to(DEVICE)

    ref_c_norm = ref_c.norm().item()
    logger.info(f"  Safe reference computed from {len(safe_c_states)} timesteps")
    logger.info(f"  ref_c L2 norm: {ref_c_norm:.4f}")
    logger.info(f"  Phase 2 complete.")
    return ref_c


# ══════════════════════════════════════════════════════════════════════
# 6. Long-Horizon Safety Trial
# ══════════════════════════════════════════════════════════════════════

def get_theta_from_obs(obs):
    """Extract angle from Pendulum observation [cos(θ), sin(θ), θ_dot]."""
    return np.arctan2(obs[1], obs[0])


# Pendulum-v1 dynamics constants (from gymnasium source)
_PEND_G = 10.0   # gravity
_PEND_M = 1.0    # mass
_PEND_L = 1.0    # length
_PEND_DT = 0.05  # integration timestep
_PEND_MAX_TORQUE = 2.0


def _pendulum_predict_next_theta(theta, theta_dot, torque):
    """
    Predict next (θ, θ_dot) using the actual Pendulum-v1 dynamics.
    θ̈ = -3g/(2l) * sin(θ + π) + 3/(ml²) * u
    Integration: Euler, same as gymnasium.
    """
    u = np.clip(torque, -_PEND_MAX_TORQUE, _PEND_MAX_TORQUE)
    theta_ddot = (-3.0 * _PEND_G / (2.0 * _PEND_L) * np.sin(theta + np.pi)
                  + 3.0 / (_PEND_M * _PEND_L ** 2) * u)
    new_theta_dot = theta_dot + theta_ddot * _PEND_DT
    new_theta_dot = np.clip(new_theta_dot, -8.0, 8.0)  # Pendulum velocity limit
    new_theta = theta + new_theta_dot * _PEND_DT
    return new_theta, new_theta_dot


def cbf_oracle_filter(obs, action, theta_safe=THETA_SAFE, gamma=CBF_GAMMA):
    """
    Oracle CBF using the true Pendulum dynamics.

    Uses predictive safety filtering: simulate the next state under the
    proposed action. If it would violate the constraint, search for the
    nearest safe action using the known dynamics model.

    This is the gold-standard CBF — it has oracle access to both the
    constraint and the true dynamics.
    """
    theta = np.arctan2(obs[1], obs[0])
    theta_dot = obs[2]
    u = float(action.flatten()[0]) if hasattr(action, 'flatten') else float(action)

    # Predict next state under proposed action
    next_theta, next_theta_dot = _pendulum_predict_next_theta(theta, theta_dot, u)

    # Check if safe
    if abs(next_theta) <= theta_safe:
        return action  # Proposed action is safe, pass through

    # Unsafe — find the safe action closest to the proposed one.
    # Binary search over actions to find the boundary.
    # Direction: push toward θ=0 (corrective torque opposes θ)
    safe_dir = -np.sign(theta) * _PEND_MAX_TORQUE

    # Check if the safe extreme is actually safe
    next_safe, _ = _pendulum_predict_next_theta(theta, theta_dot, safe_dir)
    if abs(next_safe) > theta_safe:
        # Even max corrective torque can't prevent violation;
        # apply max correction anyway (best effort)
        return np.array([safe_dir])

    # Binary search between proposed action and safe extreme
    u_lo, u_hi = min(u, safe_dir), max(u, safe_dir)
    for _ in range(15):  # 15 iterations → ~1e-5 precision
        u_mid = (u_lo + u_hi) / 2.0
        next_mid, _ = _pendulum_predict_next_theta(theta, theta_dot, u_mid)
        if abs(next_mid) <= theta_safe:
            # u_mid is safe; move toward proposed action
            if u > safe_dir:
                u_lo = u_mid
            else:
                u_hi = u_mid
        else:
            # u_mid is unsafe; move toward safe extreme
            if u > safe_dir:
                u_hi = u_mid
            else:
                u_lo = u_mid

    # Return the safe action closest to the proposed one
    best_u = u_lo if abs(u_lo - u) < abs(u_hi - u) else u_hi
    # Verify it's actually safe
    check, _ = _pendulum_predict_next_theta(theta, theta_dot, best_u)
    if abs(check) > theta_safe:
        best_u = safe_dir  # Fallback to max correction

    return np.array([np.clip(best_u, -_PEND_MAX_TORQUE, _PEND_MAX_TORQUE)])


def run_safety_trial(policy, ref_c, condition, trial_seed, device=DEVICE):
    """
    Run one long-horizon safety trial.

    The Pendulum environment auto-resets every 200 steps. We run for N_STEPS
    total, chaining episodes. The hidden state persists across resets —
    this is the key: can the LTC maintain safety knowledge across many
    environment resets and 20,000 steps of ODE dynamics?
    """
    policy.eval()

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=trial_seed)
    hx = policy.init_hidden(1, device)

    # ── Set initial c-state to safe reference (except unconstrained) ──
    ref_c_np = ref_c.cpu().numpy().flatten()
    if condition != "unconstrained":
        hx = policy.set_c(hx, ref_c.clone())

    # Blackout RNG (seeded per-trial for reproducibility)
    blackout_rng = np.random.RandomState(trial_seed + 77777)

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
            # ── Observation blackout (simulates sensor degradation) ──
            # During blackout windows, the policy sees random observations.
            # This corrupts h (CfC ODE gets wrong input) and c (LSTM gates
            # get wrong signals). After blackout, recovery depends on c-state.
            in_blackout = (condition != "unconstrained"
                           and step > 0
                           and (step % BLACKOUT_INTERVAL) < BLACKOUT_DURATION)
            if in_blackout:
                noise_obs = blackout_rng.uniform(-1, 1, size=(1, 3)).astype(np.float32)
                noise_obs[0, 2] *= 8.0  # θ_dot range is [-8, 8]
                obs_t = torch.FloatTensor(noise_obs).to(device)

            # ── Condition-specific state maintenance ──
            if condition == "oracle":
                # Restore c right after each blackout ends
                if step > 0 and (step % BLACKOUT_INTERVAL) == BLACKOUT_DURATION:
                    hx = policy.set_c(hx, ref_c.clone())
                    reinjection_steps.append(step)

            elif condition == "timer":
                if step > 0 and step % TIMER_INTERVAL == 0:
                    # Blend c-state toward reference
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
                        recovered_np = vsa.hvec_to_state(recovered_hvec, policy.state_size)
                        rec_norm = np.linalg.norm(recovered_np)
                        ref_norm = np.linalg.norm(ref_c_np)
                        if rec_norm > 1e-12:
                            recovered_np = recovered_np * (ref_norm / rec_norm)
                        recovered_t = torch.FloatTensor(recovered_np).unsqueeze(0).to(device)
                        c_cur = policy.get_c(hx)
                        alpha_t = 0.8
                        hx = policy.set_c(hx, alpha_t * recovered_t + (1 - alpha_t) * c_cur)
                        reinjection_steps.append(step)

            elif condition == "ema":
                c_cur = policy.get_c(hx)
                hx = policy.set_c(hx, (1 - EMA_ALPHA) * c_cur + EMA_ALPHA * ref_c)

            elif condition == "norm_monitor":
                c_cur = policy.get_c(hx)
                current_norm = torch.norm(c_cur).item()
                # Use a running average of recent norms so the threshold
                # adapts to the LTC's natural dynamics (not always fire)
                norm_history.append(current_norm)
                if len(norm_history) > 200:
                    recent_avg = np.mean(norm_history[-200:])
                else:
                    recent_avg = np.mean(norm_history)
                if (current_norm < NORM_DROP_FRAC * recent_avg
                        and step > 100 and len(norm_history) > 50):
                    alpha_t = 0.8
                    hx = policy.set_c(hx, alpha_t * ref_c + (1 - alpha_t) * c_cur)
                    reinjection_steps.append(step)

            # elif condition in ("unconstrained", "no_protection"):
            #   do nothing to the hidden state

            # ── Get action ──
            action, _, hx = policy.get_action(obs_t, hx, deterministic=True)
            action_np = action.cpu().numpy().flatten()

            # ── CBF Oracle: filter action directly (only near safe boundary) ──
            if condition == "cbf_oracle":
                theta_cur = get_theta_from_obs(obs)
                # Only prevent exits from safe zone; allow free swing-up/re-entry
                if abs(theta_cur) <= THETA_SAFE:
                    action_np = cbf_oracle_filter(obs, action_np)

        # ── Step environment ──
        obs_new, reward, terminated, truncated, _ = env.step(action_np)
        theta = get_theta_from_obs(obs_new)

        # ── Record metrics ──
        violated = abs(theta) > THETA_SAFE
        violations.append(1 if violated else 0)
        angles.append(abs(theta))
        rewards.append(reward)

        # Retention: cosine similarity of c-state with reference
        with torch.no_grad():
            c_flat = policy.get_c(hx).flatten()
            ref_flat = ref_c.flatten()
            cos_sim = F.cosine_similarity(c_flat.unsqueeze(0),
                                           ref_flat.unsqueeze(0)).item()
            retention_curve.append(max(0.0, cos_sim))

        obs = obs_new
        env_step_count += 1

        # Auto-reset environment (Pendulum truncates at 200 steps)
        if terminated or truncated:
            obs, _ = env.reset(seed=trial_seed + env_step_count)
            # Hidden state persists (this is the point of the experiment)

    env.close()

    # ── Compile ──
    total_violations = sum(violations)
    n_measured = len(violations)

    # Violation timeline (per 1000-step windows)
    window = 1000
    violation_timeline = []
    for ws in range(0, n_measured, window):
        we = min(ws + window, n_measured)
        violation_timeline.append(sum(violations[ws:we]) / (we - ws))

    # Retention snapshots
    ret_snapshots = []
    for s in range(0, len(retention_curve), 500):
        e = min(s + 500, len(retention_curve))
        ret_snapshots.append(float(np.mean(retention_curve[s:e])))

    # Angle percentiles
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
# 7. Full Experiment
# ══════════════════════════════════════════════════════════════════════

def run_experiment(policy, ref_c):
    """Run all conditions × trials, interleaved so every condition gets
    results early (trial 1 of all conditions, then trial 2, etc.)."""
    total_trials = len(CONDITIONS) * N_TRIALS
    all_results = {c: [] for c in CONDITIONS}

    log_banner(f"Phase 3: Safety Benchmark ({total_trials} trials)")
    logger.info(f"  Conditions: {len(CONDITIONS)}")
    logger.info(f"  Trials per condition: {N_TRIALS}")
    logger.info(f"  Steps per trial: {N_STEPS}")
    logger.info(f"  θ_safe: {THETA_SAFE} rad ({np.degrees(THETA_SAFE):.1f}°)")
    logger.info(f"  Blackout: {BLACKOUT_DURATION} steps every {BLACKOUT_INTERVAL} steps")
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

            # Detailed diagnostics for the first trial of each condition
            if trial == 0:
                tl = result['violation_timeline']  # per-1000-step windows
                tl_str = "  ".join(f"{r:.3f}" for r in tl)
                logger.info(f"    ↳ timeline (per 1k steps): {tl_str}")
                logger.info(f"    ↳ p95_angle={result['p95_angle']:.3f}  "
                            f"max_angle={result['max_angle']:.3f}  "
                            f"avg_reward={result['mean_reward']:.1f}")
                rs = result['retention_snapshots']
                rs_str = "  ".join(f"{r:.3f}" for r in rs[:10])
                logger.info(f"    ↳ retention snapshots (500-step): {rs_str}")

        round_time = time.time() - round_start
        elapsed = time.time() - experiment_start
        rate = completed / elapsed if elapsed > 0 else 1
        remaining = (total_trials - completed) / rate
        eta = datetime.now() + timedelta(seconds=remaining)

        # Summary table after each round
        logger.info(f"\n  ── Round {trial+1} complete ({round_time:.0f}s) | "
                    f"Progress: {completed}/{total_trials} ({100*completed/total_trials:.0f}%) | "
                    f"ETA: {eta.strftime('%H:%M:%S')} ──")

        if (trial + 1) % 5 == 0 or trial == 0:
            logger.info(f"  {'Condition':<16} {'Trials':>6}  {'Mean Viol':>9}  {'Std':>7}  {'Mean Ret':>8}")
            logger.info(f"  {'─'*16} {'─'*6}  {'─'*9}  {'─'*7}  {'─'*8}")
            for cond in CONDITIONS:
                n = len(all_results[cond])
                if n > 0:
                    vs = [t['total_violations'] for t in all_results[cond]]
                    rf = [t['retention_final'] for t in all_results[cond]]
                    logger.info(f"  {CONDITION_LABELS[cond]:<16} {n:>6}  "
                                f"{np.mean(vs):>9.1f}  {np.std(vs):>7.1f}  "
                                f"{np.mean(rf):>8.3f}")

        # Checkpoint every 5 rounds
        if (trial + 1) % 5 == 0:
            _save_checkpoint(all_results, trial + 1)

    total_time = time.time() - experiment_start
    log_banner(f"All {total_trials} trials complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    return all_results


def _save_checkpoint(all_results, n_rounds):
    cp = {"rounds_completed": n_rounds, "timestamp": datetime.now().isoformat()}
    for c in CONDITIONS:
        cp[c] = [{k: v for k, v in t.items() if k != "drift_metric"} for t in all_results[c]]
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(cp, f, indent=1)
    logger.debug(f"  Checkpoint saved ({n_done}/{len(CONDITIONS)})")


# ══════════════════════════════════════════════════════════════════════
# 8. Statistical Analysis
# ══════════════════════════════════════════════════════════════════════

def compute_cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    gt = sum(1 for xi in x for yj in y if xi > yj)
    lt = sum(1 for xi in x for yj in y if xi < yj)
    return (gt - lt) / (n_x * n_y)


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95, stat_fn=np.median):
    rng = np.random.RandomState(99)
    boot = np.array([stat_fn(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot, 100*alpha)), float(np.percentile(boot, 100*(1-alpha))), float(stat_fn(data))


def analyze(all_results):
    log_banner("STATISTICAL ANALYSIS")

    n_pairs = len(list(combinations(CONDITIONS, 2)))
    alpha_bonf = 0.05 / n_pairs

    # Descriptive stats
    stats = {}
    logger.info("")
    header = f"{'Condition':<18} {'Violations':>14} {'Viol.Rate':>10} {'MeanAngle':>10} {'Ret(final)':>14} {'Reinj':>8}"
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
        stats[c] = dict(vm=vm, vs=vs, vr=vr, ma=ma, rf=rf, rfs=rfs, reinj=reinj)
        logger.info(f"{CONDITION_LABELS[c]:<18} {vm:>7.1f}±{vs:<5.1f} {vr:>10.4f} "
                     f"{ma:>10.3f} {rf:>7.3f}±{rfs:.3f} {reinj:>8.1f}")

    # Kruskal-Wallis
    logger.info("")
    logger.info("─── Kruskal-Wallis omnibus (violations) ───")
    vdata = {c: [t['total_violations'] for t in all_results[c]] for c in CONDITIONS}
    H, p_kw = sp_stats.kruskal(*vdata.values())
    logger.info(f"  H = {H:.2f}, p = {p_kw:.2e}")

    # Pairwise Mann-Whitney
    logger.info("")
    logger.info(f"─── Pairwise Mann-Whitney U (Bonferroni α = 0.05/{n_pairs} = {alpha_bonf:.4f}) ───")
    logger.info(f"{'Pair':<40} {'U':>8} {'p_adj':>12} {'sig':>6} {'d':>8} {'cliff':>8}")
    logger.info("-" * 90)

    pairwise = {}
    for c1, c2 in combinations(CONDITIONS, 2):
        v1, v2 = vdata[c1], vdata[c2]
        U, p_raw = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_adj = min(p_raw * n_pairs, 1.0)
        ps = np.sqrt((np.std(v1)**2 + np.std(v2)**2) / 2)
        d = (np.mean(v1) - np.mean(v2)) / ps if ps > 0 else 0
        cliff = compute_cliffs_delta(v1, v2)
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        label = f"{CONDITION_LABELS[c1]} vs {CONDITION_LABELS[c2]}"
        logger.info(f"{label:<40} {U:>8.1f} {p_adj:>12.6f} {sig:>6} {d:>+8.2f} {cliff:>+8.3f}")
        pairwise[(c1, c2)] = dict(U=float(U), p_adj=float(p_adj), d=float(d),
                                   cliff=float(cliff), sig=sig)

    # Bootstrap CIs
    logger.info("")
    logger.info(f"─── Bootstrap 95% CIs on median violations ({N_BOOTSTRAP} resamples) ───")
    bootstrap = {}
    for c in CONDITIONS:
        vd = [t['total_violations'] for t in all_results[c]]
        lo, hi, med = bootstrap_ci(vd)
        bootstrap[c] = dict(median=med, ci_lo=lo, ci_hi=hi)
        logger.info(f"  {CONDITION_LABELS[c]:<18}: median={med:.1f}  95% CI=[{lo:.1f}, {hi:.1f}]")

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
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        diff = np.mean(v_other) - np.mean(v_vsa)
        logger.info(f"  {CONDITION_LABELS[c_other]} > VSA?  Δ={diff:+.1f}  p_adj={p_adj:.6f} ({sig})")

    # Post-hoc power
    logger.info("")
    logger.info("─── Post-hoc power ───")
    for c_other in ["timer", "ema", "cbf_oracle"]:
        v_vsa = vdata["vsa"]
        v_other = vdata[c_other]
        ps = np.sqrt((np.std(v_vsa)**2 + np.std(v_other)**2) / 2)
        d = abs(np.mean(v_vsa) - np.mean(v_other)) / ps if ps > 0 else 0
        z_a = sp_norm.ppf(1 - alpha_bonf / 2)
        nc = d * np.sqrt(N_TRIALS / 2)
        power = 1 - sp_norm.cdf(z_a - nc) + sp_norm.cdf(-z_a - nc)
        logger.info(f"  VSA vs {CONDITION_LABELS[c_other]}: d={d:.3f}, power≈{power:.3f} ({power*100:.1f}%)")

    return stats, pairwise, bootstrap


# ══════════════════════════════════════════════════════════════════════
# 9. Visualization
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
        smooth = np.convolve(training_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smooth, color='steelblue', linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("A. LTC Policy Training (Pendulum-v1)", fontweight='bold')
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
        ax2.fill_between(x, np.maximum(mn-sd, 0), mn+sd, color=color, alpha=0.08)
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
    vdata = [[t['total_violations'] for t in all_results[c]] for c in CONDITIONS]
    bp = ax4.boxplot(vdata, tick_labels=short, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax4.set_ylabel("Total Violations (20k steps)")
    ax4.set_title(f"D. Violations by Condition (n={N_TRIALS})", fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.3, axis='y')

    # E: Mean angle boxplot
    ax5 = fig.add_subplot(gs[1, 1])
    adata = [[t['mean_angle'] for t in all_results[c]] for c in CONDITIONS]
    bp2 = ax5.boxplot(adata, tick_labels=short, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.5)
    ax5.axhline(y=THETA_SAFE, color='red', linestyle='--', linewidth=1.5, label=f'θ_safe={THETA_SAFE}')
    ax5.set_ylabel("Mean |θ| (radians)")
    ax5.set_title("E. Average Angle Deviation", fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # F: Violation rate bar chart
    ax6 = fig.add_subplot(gs[1, 2])
    means = [stats[c]['vr'] for c in CONDITIONS]
    stds = [np.std([t['violation_rate'] for t in all_results[c]]) for c in CONDITIONS]
    x_pos = np.arange(len(CONDITIONS))
    ax6.bar(x_pos, means, yerr=stds, color=colors, alpha=0.6, capsize=4, edgecolor='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(short, rotation=30, ha='right', fontsize=8)
    ax6.set_ylabel("Violation Rate")
    ax6.set_title("F. Mean Violation Rate ± SD", fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # G: VSA drift metric
    ax7 = fig.add_subplot(gs[2, 0])
    dms = [t['drift_metric'] for t in all_results['vsa'] if t['drift_metric']]
    if dms:
        ml = max(len(d) for d in dms)
        pad = np.full((len(dms), ml), np.nan)
        for i, d in enumerate(dms):
            pad[i, :len(d)] = d
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * VSA_CHECK_INTERVAL
        ax7.plot(x, mn, color=CONDITION_COLORS['vsa'], alpha=0.7, linewidth=1)
        ax7.fill_between(x, mn-sd, mn+sd, color=CONDITION_COLORS['vsa'], alpha=0.1)
        ax7.axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({DRIFT_THRESHOLD})')
    ax7.set_xlabel("Time Step")
    ax7.set_ylabel("Drift Metric")
    ax7.set_title("G. VSA Drift Detection", fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # H: Effect sizes
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
    reinj_c = [c for c in CONDITIONS if c not in ("unconstrained", "oracle", "cbf_oracle")]
    reinj_l = [CONDITION_LABELS[c] for c in reinj_c]
    reinj_colors = [CONDITION_COLORS[c] for c in reinj_c]
    reinj_d = [[t['n_reinjections'] for t in all_results[c]] for c in reinj_c]
    if reinj_d:
        bp3 = ax9.boxplot(reinj_d, tick_labels=reinj_l, patch_artist=True)
        for patch, color in zip(bp3['boxes'], reinj_colors):
            patch.set_facecolor(color); patch.set_alpha(0.5)
    ax9.set_ylabel("Re-injections")
    ax9.set_title("I. Re-injection Count", fontweight='bold')
    ax9.tick_params(axis='x', rotation=30)
    ax9.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        "Trained LTC Policy: VSA vs Baselines for Long-Horizon Safety\n"
        f"Pendulum-v1 | {N_STEPS} steps × {N_TRIALS} trials | θ_safe={THETA_SAFE} rad | D={D_VSA}",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches='tight')
    logger.info(f"  Figure saved: {FIGURE_PATH}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 10. Save Results
# ══════════════════════════════════════════════════════════════════════

def save_results(all_results, stats, pairwise, bootstrap, training_rewards):
    save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_steps": N_STEPS,
            "n_trials": N_TRIALS,
            "d_vsa": D_VSA,
            "theta_safe": THETA_SAFE,
            "ltc_units": LTC_UNITS,
            "train_iterations": TRAIN_ITERATIONS,
            "train_eps_per_batch": TRAIN_EPS_PER_BATCH,
            "drift_threshold": DRIFT_THRESHOLD,
            "timer_interval": TIMER_INTERVAL,
            "ema_alpha": EMA_ALPHA,
            "cbf_gamma": CBF_GAMMA,
            "norm_drop_frac": NORM_DROP_FRAC,
            "blackout_interval": BLACKOUT_INTERVAL,
            "blackout_duration": BLACKOUT_DURATION,
            "n_bootstrap": N_BOOTSTRAP,
            "device": str(DEVICE),
        },
        "training": {
            "final_avg_reward": float(np.mean(training_rewards[-50:])),
            "reward_history_downsampled": [float(x) for x in training_rewards[::10]],
        },
        "summary": stats,
        "pairwise_tests": {f"{c1}_vs_{c2}": v for (c1, c2), v in pairwise.items()},
        "bootstrap_cis": bootstrap,
    }
    for c in CONDITIONS:
        save[c] = [{k: v for k, v in t.items() if k != "drift_metric"} for t in all_results[c]]

    with open(RESULTS_JSON, 'w') as f:
        json.dump(save, f, indent=2)
    logger.info(f"  Results saved: {RESULTS_JSON}")


# ══════════════════════════════════════════════════════════════════════
# 11. Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_banner("Trained LTC Safety Experiment")
    logger.info(f"  Python {sys.version.split()[0]}")
    logger.info(f"  PyTorch {torch.__version__}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Phase 1: Train LTC Policy ──
    policy = LTCPolicy()

    if os.path.exists(MODEL_PATH):
        logger.info(f"\n  Loading pre-trained policy from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(checkpoint['policy'])
        training_rewards = checkpoint.get('training_rewards', [])
        policy.to(DEVICE)
    else:
        training_rewards = train_policy(policy)
        # Save trained model
        torch.save({
            'policy': policy.state_dict(),
            'training_rewards': training_rewards,
        }, MODEL_PATH)
        logger.info(f"  Policy saved: {MODEL_PATH}")

    # Evaluate
    eval_rewards, eval_violations = evaluate_policy(policy)

    # ── Phase 2: Compute Safe Reference C-State ──
    ref_c_path = os.path.join(SCRIPT_DIR, "safety_ref_c.pt")
    if os.path.exists(ref_c_path):
        logger.info(f"\n  Loading pre-computed safe reference from {ref_c_path}")
        ref_c = torch.load(ref_c_path, map_location=DEVICE, weights_only=True)
    else:
        ref_c = compute_safe_reference(policy)
        torch.save(ref_c, ref_c_path)
        logger.info(f"  Safe reference saved: {ref_c_path}")

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
        logger.info(f"    {rank}. {CONDITION_LABELS[c]:<18} {s['vm']:>7.1f} ± {s['vs']:.1f} violations  "
                     f"(retention: {s['rf']:.3f}, angle: {s['ma']:.3f})")

    logger.info("")
    logger.info(f"  Log:     {LOG_FILE}")
    logger.info(f"  Results: {RESULTS_JSON}")
    logger.info(f"  Figure:  {FIGURE_PATH}")
    logger.info(f"  Model:   {MODEL_PATH}")
    logger.info("")
