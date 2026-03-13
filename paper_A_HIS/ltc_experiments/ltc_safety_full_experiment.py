"""
Full Benchmark: VSA-Constrained Liquid Neural Networks for Long-Horizon Safety
================================================================================

7-condition experiment comparing safety-maintenance strategies in Liquid
Time-Constant (LTC) neural networks over 20,000 time steps.

Conditions:
  C1: Unconstrained       — safety encoded once at t=0, no re-injection
  C2: Timer re-injection  — safety re-encoded every K steps (fixed interval)
  C3: VSA-constrained     — HIS protocol recovers + re-injects when drift detected
  C4: Oracle              — perfect obstacle knowledge at every step (lower bound)
  C5: CBF (Control Barrier Function)  — QP-based safety filter on control
  C6: EMA (Exponential Moving Average) — continuous soft blend toward reference
  C7: Norm-monitoring     — re-inject when state norm drops below threshold

Power analysis:
  From pilot (n=25): VSA vs Timer had d=0.67 (the hardest comparison).
  For 7 conditions with Bonferroni-corrected alpha = 0.05/21 ≈ 0.0024 and
  80% power at d=0.67, we need n≈48. We use n=50 for margin.

Statistical battery:
  - Kruskal-Wallis omnibus test
  - Pairwise Mann-Whitney U with Bonferroni correction (21 pairs)
  - Bootstrap 95% CIs on median violations (10,000 resamples)
  - Cohen's d effect sizes
  - Cliff's delta (non-parametric effect size)

Benchmark: 2D point-mass navigation with forbidden zones.
  20,000 time steps (~200 seconds at dt=0.01), 50 trials per condition.

Usage:
  python ltc_safety_full_experiment.py

Requires: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import json
import time
import os
import sys
import logging
from datetime import datetime, timedelta
from itertools import combinations

# ══════════════════════════════════════════════════════════════════════
# 0. Logging Setup
# ══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "ltc_full_experiment_log.txt")
RESULTS_JSON = os.path.join(SCRIPT_DIR, "ltc_full_results.json")
FIGURE_PATH = os.path.join(SCRIPT_DIR, "figure_D2_ltc_safety_full.png")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "ltc_full_checkpoint.json")

# Configure dual logging: console (INFO) + file (DEBUG)
logger = logging.getLogger("LTC_FULL")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(ch)


def log_banner(msg):
    """Print a visible banner to both console and log."""
    line = "=" * 72
    logger.info(line)
    logger.info(f"  {msg}")
    logger.info(line)


# ══════════════════════════════════════════════════════════════════════
# 1. Configuration
# ══════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# Simulation
DT = 0.01
N_STEPS = 20000
WARMUP_STEPS = 100

# Agent dynamics
MASS = 1.0
DAMPING = 2.0
MAX_FORCE = 8.0

# PD Controller gains
KP = 3.0
KD = 2.0

# LTC Safety Memory Network
N_SAFETY_NEURONS = 64
TAU_SAFETY_MIN = 2.0
TAU_SAFETY_MAX = 8.0
SAFETY_DRIVE_SCALE = 0.3

# VSA / HIS
D_VSA = 10000
DRIFT_THRESHOLD = 0.55
VSA_INJECTION_GAIN = 1.5
VSA_CHECK_INTERVAL = 50  # Check drift every N steps

# Timer baseline
TIMER_INTERVAL = 300

# EMA baseline
EMA_ALPHA = 0.005  # Blending coefficient per step: x = (1-α)x + α·x_ref

# Norm-monitoring baseline
NORM_THRESHOLD_FRAC = 0.5  # Re-inject when ||x|| drops below this fraction of ||x_ref||

# CBF baseline
CBF_GAMMA = 1.0  # CBF decay rate: ḣ(x) + γ·h(x) ≥ 0

# Safety environment
N_OBSTACLES = 6
OBSTACLE_RADIUS = 1.2
ARENA_SIZE = 12.0
REPULSION_RANGE = 3.0
REPULSION_STRENGTH = 12.0

# Waypoint generation
WAYPOINT_HOLD_STEPS = 250

# Experiment
N_TRIALS = 50
N_BOOTSTRAP = 10000  # Bootstrap resamples for CIs

# All conditions
CONDITIONS = [
    "unconstrained",
    "timer",
    "vsa",
    "oracle",
    "cbf",
    "ema",
    "norm_monitor",
]

CONDITION_LABELS = {
    "unconstrained": "Unconstrained",
    "timer": "Timer (k=300)",
    "vsa": "VSA-Constrained",
    "oracle": "Oracle",
    "cbf": "CBF",
    "ema": "EMA Blending",
    "norm_monitor": "Norm Monitor",
}

CONDITION_COLORS = {
    "unconstrained": "#d62728",
    "timer": "#ff7f0e",
    "vsa": "#1f77b4",
    "oracle": "#2ca02c",
    "cbf": "#9467bd",
    "ema": "#8c564b",
    "norm_monitor": "#e377c2",
}


# ══════════════════════════════════════════════════════════════════════
# 2. VSA Module (Holographic Invariant Storage)
# ══════════════════════════════════════════════════════════════════════

class VSAModule:
    """Bipolar hypervector algebra for safety invariant storage and recovery."""

    def __init__(self, d=D_VSA):
        self.d = d
        self._proj_cache = {}

    def random_bipolar(self):
        return np.random.choice([-1, 1], size=self.d).astype(np.float64)

    def bind(self, a, b):
        return a * b

    def unbind(self, composite, key):
        return composite * key

    def sign_cleanup(self, v):
        return np.sign(v)

    def cosine_similarity(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return np.dot(a, b) / (na * nb)

    def _get_projection(self, n):
        if n not in self._proj_cache:
            rng = np.random.RandomState(12345)
            self._proj_cache[n] = rng.randn(n, self.d) / np.sqrt(n)
        return self._proj_cache[n]

    def state_to_hypervector(self, state_vec):
        proj = self._get_projection(len(state_vec))
        projected = state_vec @ proj
        return np.sign(projected)

    def hypervector_to_state(self, hvec, state_dim):
        proj = self._get_projection(state_dim)
        reconstructed = hvec @ proj.T / self.d * state_dim
        return reconstructed

    def create_invariant(self, safety_state):
        key = self.random_bipolar()
        ref_hvec = self.state_to_hypervector(safety_state)
        invariant = self.bind(key, ref_hvec)
        return key, invariant, ref_hvec

    def restore(self, invariant, current_state, key, ref_hvec):
        noise_hvec = self.state_to_hypervector(current_state)
        inv_norm = np.linalg.norm(invariant)
        noise_norm = np.linalg.norm(noise_hvec)
        if noise_norm > 1e-12:
            noise_hat = noise_hvec * (inv_norm / noise_norm)
        else:
            noise_hat = np.zeros_like(noise_hvec)

        drift_metric = self.cosine_similarity(noise_hvec, ref_hvec)

        superimposed = invariant + noise_hat
        cleaned = self.sign_cleanup(superimposed)
        recovered = self.unbind(cleaned, key)

        recovery_sim = self.cosine_similarity(recovered, ref_hvec)
        return recovered, drift_metric, recovery_sim


# ══════════════════════════════════════════════════════════════════════
# 3. LTC Safety Memory Network
# ══════════════════════════════════════════════════════════════════════

class LTCSafetyMemory:
    """
    Liquid Time-Constant network for safety memory.
    ODE: dx_i/dt = -(1/tau_i)*x_i + sigma(W_in@input + W_rec@x + b)*drive_scale
    The -(1/tau)*x term causes exponential decay of encoded safety information.
    """

    def __init__(self, n_obstacles, n_neurons=N_SAFETY_NEURONS, seed=0):
        self.n_obstacles = n_obstacles
        self.n_neurons = n_neurons

        rng = np.random.RandomState(seed)
        self.neurons_per_obstacle = n_neurons // n_obstacles
        self.tau = rng.uniform(TAU_SAFETY_MIN, TAU_SAFETY_MAX, size=n_neurons)
        self.W_in = rng.randn(2, n_neurons) * 0.3
        self.W_rec = np.zeros((n_neurons, n_neurons))
        for k in range(n_obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            block = rng.randn(self.neurons_per_obstacle, self.neurons_per_obstacle) * 0.1
            sn = np.linalg.norm(block, ord=2)
            if sn > 0.8:
                block *= 0.8 / sn
            self.W_rec[start:end, start:end] = block

        self.W_out = np.zeros((n_neurons, n_obstacles))
        for k in range(n_obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            self.W_out[start:end, k] = rng.randn(self.neurons_per_obstacle) * 0.5

        self.b = rng.randn(n_neurons) * 0.05
        self.x = np.zeros(n_neurons)

    def reset(self):
        self.x = np.zeros(self.n_neurons)

    def encode_safety(self, obstacles):
        for k, (ox, oy, r) in enumerate(obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            phases = np.linspace(0, 2 * np.pi, self.neurons_per_obstacle)
            pattern = (
                np.sin(phases * ox) * r +
                np.cos(phases * oy) * r +
                np.sin(phases * (ox + oy)) * 0.5
            )
            norm = np.linalg.norm(pattern)
            if norm > 1e-12:
                pattern = pattern / norm
            self.x[start:end] = pattern

    def get_reference_state(self, obstacles):
        ref = np.zeros(self.n_neurons)
        for k, (ox, oy, r) in enumerate(obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            phases = np.linspace(0, 2 * np.pi, self.neurons_per_obstacle)
            pattern = (
                np.sin(phases * ox) * r +
                np.cos(phases * oy) * r +
                np.sin(phases * (ox + oy)) * 0.5
            )
            norm = np.linalg.norm(pattern)
            if norm > 1e-12:
                pattern = pattern / norm
            ref[start:end] = pattern
        return ref

    def step(self, agent_pos, dt=DT, forcing=None):
        z_in = agent_pos @ self.W_in * SAFETY_DRIVE_SCALE
        z_rec = self.x @ self.W_rec
        drive = np.tanh(z_in + z_rec + self.b) * SAFETY_DRIVE_SCALE
        dxdt = -(1.0 / self.tau) * self.x + drive
        if forcing is not None:
            dxdt += forcing
        self.x = self.x + dt * dxdt
        self.x = np.clip(self.x, -5, 5)

        raw_scores = np.tanh(self.x) @ self.W_out
        awareness = 1.0 / (1.0 + np.exp(-raw_scores * 3))
        return awareness

    def get_overall_retention(self, obstacles):
        ref = self.get_reference_state(obstacles)
        nx = np.linalg.norm(self.x)
        nr = np.linalg.norm(ref)
        if nx < 1e-12 or nr < 1e-12:
            return 0.0
        return np.dot(self.x, ref) / (nx * nr)


# ══════════════════════════════════════════════════════════════════════
# 4. Navigation Environment
# ══════════════════════════════════════════════════════════════════════

class SafeNavigationEnv:
    """2D point-mass navigation with circular forbidden zones."""

    def __init__(self, obstacles=None):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.obstacles = obstacles if obstacles is not None else self._generate_obstacles()
        self.waypoints = []

    def _generate_obstacles(self):
        obstacles = []
        angles = np.linspace(0, 2 * np.pi, N_OBSTACLES, endpoint=False)
        ring_radius = ARENA_SIZE * 0.45
        for angle in angles:
            x = ring_radius * np.cos(angle) + np.random.uniform(-1, 1)
            y = ring_radius * np.sin(angle) + np.random.uniform(-1, 1)
            r = OBSTACLE_RADIUS + np.random.uniform(-0.2, 0.2)
            obstacles.append((x, y, r))
        return obstacles

    def generate_waypoints(self, n_steps=N_STEPS):
        self.waypoints = []
        n_waypoints = n_steps // WAYPOINT_HOLD_STEPS + 1
        for i in range(n_waypoints):
            if i % 2 == 0 and len(self.obstacles) > 0:
                obs_idx = i % len(self.obstacles)
                ox, oy, r = self.obstacles[obs_idx]
                angle_to_center = np.arctan2(-oy, -ox)
                target_angle = angle_to_center + np.random.uniform(-0.5, 0.5)
                target_dist = r + np.random.uniform(0.3, 1.0)
                wx = ox + target_dist * np.cos(target_angle)
                wy = oy + target_dist * np.sin(target_angle)
            else:
                wx = np.random.uniform(-ARENA_SIZE * 0.7, ARENA_SIZE * 0.7)
                wy = np.random.uniform(-ARENA_SIZE * 0.7, ARENA_SIZE * 0.7)
            self.waypoints.append((np.clip(wx, -ARENA_SIZE, ARENA_SIZE),
                                   np.clip(wy, -ARENA_SIZE, ARENA_SIZE)))

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        self.vel = np.array([0.0, 0.0])

    def get_target(self, step):
        idx = min(step // WAYPOINT_HOLD_STEPS, len(self.waypoints) - 1)
        return np.array(self.waypoints[idx])

    def step(self, control, dt=DT):
        control = np.clip(control, -MAX_FORCE, MAX_FORCE)
        acc = (control - DAMPING * self.vel) / MASS
        self.vel += dt * acc
        self.pos += dt * self.vel
        self.pos = np.clip(self.pos, -ARENA_SIZE, ARENA_SIZE)

        violated = False
        min_dist = float('inf')
        for ox, oy, r in self.obstacles:
            dist = np.sqrt((self.pos[0] - ox)**2 + (self.pos[1] - oy)**2) - r
            min_dist = min(min_dist, dist)
            if dist < 0:
                violated = True
        return violated, min_dist


def compute_repulsion(pos, obstacles, awareness_scores):
    """Compute safety repulsion force based on per-obstacle awareness."""
    force = np.zeros(2)
    for k, (ox, oy, r) in enumerate(obstacles):
        dx = pos[0] - ox
        dy = pos[1] - oy
        dist = np.sqrt(dx**2 + dy**2)
        if dist < r + REPULSION_RANGE and dist > 0.01:
            gap = max(dist - r, 0.01)
            strength = REPULSION_STRENGTH * awareness_scores[k] / (gap ** 1.5)
            strength = min(strength, MAX_FORCE * 2)
            force[0] += strength * dx / dist
            force[1] += strength * dy / dist
    return force


# ══════════════════════════════════════════════════════════════════════
# 5. CBF Safety Filter
# ══════════════════════════════════════════════════════════════════════

def cbf_filter(pos, vel, desired_force, obstacles, awareness_scores, gamma=CBF_GAMMA):
    """
    Control Barrier Function safety filter.

    For each obstacle k with awareness_scores[k], define barrier:
      h_k(x) = ||x - o_k||^2 - r_k^2

    CBF constraint: ḣ_k + γ·h_k ≥ 0
      => 2(x - o_k)·v̇ + 2v·v + γ(||x - o_k||^2 - r_k^2) ≥ 0

    If the desired control violates ANY active constraint, project it to the
    nearest safe control. This is solved as a simple 1D projection per obstacle
    (not a full QP, since we only modify the component toward each obstacle).

    Critically: awareness_scores scale the barrier. If awareness → 0 (memory
    decayed), the CBF is blind and cannot enforce the constraint.
    """
    u = desired_force.copy()

    for k, (ox, oy, r) in enumerate(obstacles):
        a_k = awareness_scores[k]
        if a_k < 0.01:
            continue  # No awareness = no barrier knowledge

        diff = pos - np.array([ox, oy])
        dist_sq = np.dot(diff, diff)
        dist = np.sqrt(dist_sq)

        # Barrier: h = dist^2 - r^2 (positive = safe)
        h = dist_sq - r * r

        # Time derivative under current control:
        # ḣ = 2 * diff · vel  (position part)
        # Plus acceleration contribution: 2 * diff · acc
        # acc = (u - DAMPING * vel) / MASS
        acc = (u - DAMPING * vel) / MASS
        hdot = 2.0 * np.dot(diff, vel) + 2.0 * np.dot(diff, acc) * DT

        # CBF condition: hdot + gamma * h >= 0
        # Scale barrier by awareness (degraded memory → weaker barrier)
        constraint_value = hdot + gamma * h * a_k

        if constraint_value < 0:
            # Violation — project u along the gradient of h
            # Gradient of h w.r.t. u: ∂ḣ/∂u = 2·diff / MASS * DT
            grad = 2.0 * diff / MASS * DT
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq > 1e-12:
                # Minimum correction to satisfy constraint
                correction = (-constraint_value / grad_norm_sq) * grad
                u = u + correction

    return np.clip(u, -MAX_FORCE, MAX_FORCE)


# ══════════════════════════════════════════════════════════════════════
# 6. Trial Runner
# ══════════════════════════════════════════════════════════════════════

def run_trial(condition, env, obstacles, trial_seed=0):
    """
    Run one trial under the given condition.
    Returns a dict with all metrics.
    """
    safety_mem = LTCSafetyMemory(len(obstacles), N_SAFETY_NEURONS, seed=trial_seed)
    safety_mem.encode_safety(obstacles)
    ref_state = safety_mem.get_reference_state(obstacles)
    ref_norm = np.linalg.norm(ref_state)

    # VSA setup
    vsa = None
    vsa_key = vsa_inv = vsa_ref = None
    if condition == "vsa":
        vsa = VSAModule(D_VSA)
        vsa_key, vsa_inv, vsa_ref = vsa.create_invariant(ref_state)

    env.reset()

    # Metrics
    violations = []
    retention_curve = []
    tracking_errors = []
    min_distances = []
    positions = []
    reinjection_steps = []
    drift_metric_curve = []

    for step in range(N_STEPS):
        target = env.get_target(step)

        # ── PD Controller ──
        error = target - env.pos
        pd_force = KP * error - KD * env.vel

        # ── Safety mechanism (condition-dependent) ──
        forcing = None

        if condition == "oracle":
            safety_mem.x = ref_state.copy()
            safety_mem.step(env.pos, dt=DT)
            retention = 1.0

        elif condition == "timer":
            if step > 0 and step % TIMER_INTERVAL == 0:
                forcing = (ref_state - safety_mem.x) * VSA_INJECTION_GAIN / DT
                reinjection_steps.append(step)
            safety_mem.step(env.pos, dt=DT, forcing=forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        elif condition == "vsa":
            if step % VSA_CHECK_INTERVAL == 0:
                current_state = safety_mem.x.copy()
                recovered_hvec, drift_sim, recovery_sim = vsa.restore(
                    vsa_inv, current_state, vsa_key, vsa_ref
                )
                drift_metric_curve.append(drift_sim)

                if drift_sim < DRIFT_THRESHOLD:
                    recovered_state = vsa.hypervector_to_state(recovered_hvec, N_SAFETY_NEURONS)
                    rec_norm = np.linalg.norm(recovered_state)
                    if rec_norm > 1e-12:
                        recovered_state = recovered_state * (ref_norm / rec_norm)
                    forcing = (recovered_state - safety_mem.x) * VSA_INJECTION_GAIN / DT
                    reinjection_steps.append(step)

            safety_mem.step(env.pos, dt=DT, forcing=forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        elif condition == "ema":
            # Continuous soft blend: every step, nudge state toward reference
            ema_forcing = EMA_ALPHA * (ref_state - safety_mem.x) / DT
            safety_mem.step(env.pos, dt=DT, forcing=ema_forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))
            # Count "effective reinjections" as # of steps where blend was active
            # (always active, so we track cumulative blend magnitude instead)

        elif condition == "norm_monitor":
            # Monitor ||x|| and reinject when it drops too low
            current_norm = np.linalg.norm(safety_mem.x)
            if current_norm < NORM_THRESHOLD_FRAC * ref_norm and step > 0:
                forcing = (ref_state - safety_mem.x) * VSA_INJECTION_GAIN / DT
                reinjection_steps.append(step)
            safety_mem.step(env.pos, dt=DT, forcing=forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        elif condition == "cbf":
            # CBF doesn't modify the LTC state — it filters the control output
            # The LTC decays naturally (like unconstrained), and the CBF
            # tries to keep the agent safe by modifying forces
            safety_mem.step(env.pos, dt=DT)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        else:  # unconstrained
            safety_mem.step(env.pos, dt=DT)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        # ── Safety repulsion ──
        awareness = np.full(len(obstacles), retention)

        if condition == "cbf":
            # CBF: use awareness for BOTH repulsion AND the barrier filter
            safety_force = compute_repulsion(env.pos, obstacles, awareness)
            total_force = pd_force + safety_force
            # Apply CBF filter on top
            total_force = cbf_filter(env.pos, env.vel, total_force, obstacles,
                                     awareness, gamma=CBF_GAMMA)
        else:
            safety_force = compute_repulsion(env.pos, obstacles, awareness)
            total_force = pd_force + safety_force

        # ── Step environment ──
        violated, min_dist = env.step(total_force, dt=DT)

        # ── Record ──
        if step >= WARMUP_STEPS:
            violations.append(1 if violated else 0)
            min_distances.append(min_dist)
            tracking_errors.append(np.linalg.norm(env.pos - target))
            retention_curve.append(retention)

        positions.append(env.pos.copy())

    # ── Compile results ──
    positions = np.array(positions)
    total_violations = sum(violations)
    n_measured = max(len(violations), 1)

    # Violation timeline (per 1000-step windows)
    window = 1000
    violation_timeline = []
    for ws in range(0, len(violations), window):
        we = min(ws + window, len(violations))
        violation_timeline.append(sum(violations[ws:we]) / (we - ws))

    # Retention snapshots
    ret_snapshots = []
    snapshot_interval = 500
    for s in range(0, len(retention_curve), snapshot_interval):
        e = min(s + snapshot_interval, len(retention_curve))
        ret_snapshots.append(float(np.mean(retention_curve[s:e])))

    return {
        "condition": condition,
        "total_violations": total_violations,
        "violation_rate": total_violations / n_measured,
        "violation_timeline": violation_timeline,
        "mean_tracking_error": float(np.mean(tracking_errors)),
        "mean_min_distance": float(np.mean(min_distances)),
        "retention_snapshots": ret_snapshots,
        "retention_final": float(np.mean(retention_curve[-1000:])) if len(retention_curve) >= 1000 else 0.0,
        "retention_initial": float(np.mean(retention_curve[:500])) if len(retention_curve) >= 500 else 0.0,
        "n_reinjections": len(reinjection_steps),
        "reinjection_steps": reinjection_steps[:200],
        "positions": positions,  # kept as ndarray, converted to list at save time
        "drift_metric": drift_metric_curve if drift_metric_curve else [],
    }


# ══════════════════════════════════════════════════════════════════════
# 7. Main Experiment Loop
# ══════════════════════════════════════════════════════════════════════

def run_experiment():
    """Run all 7 conditions x 50 trials with progress logging."""
    total_trials = len(CONDITIONS) * N_TRIALS
    all_results = {c: [] for c in CONDITIONS}

    log_banner(f"VSA-Constrained LTC Safety: Full 7-Condition Benchmark")
    logger.info(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Conditions: {len(CONDITIONS)} ({', '.join(CONDITIONS)})")
    logger.info(f"  Trials:     {N_TRIALS} per condition ({total_trials} total)")
    logger.info(f"  Steps:      {N_STEPS} per trial")
    logger.info(f"  VSA dim:    {D_VSA}")
    logger.info(f"  Seed:       {RANDOM_SEED}")

    # Generate shared environment
    logger.info("")
    logger.info("[SETUP] Generating environment...")
    np.random.seed(RANDOM_SEED)
    env = SafeNavigationEnv()
    obstacles = env.obstacles
    env.generate_waypoints()
    waypoints = env.waypoints

    for i, (ox, oy, r) in enumerate(obstacles):
        logger.info(f"  Obstacle {i+1}: ({ox:.2f}, {oy:.2f}), r={r:.2f}")
    logger.info(f"  Waypoints: {len(waypoints)}")
    logger.info(f"  Every other waypoint forces an obstacle encounter")

    logger.info("")
    log_banner(f"Running {total_trials} trials")

    experiment_start = time.time()
    completed_trials = 0

    for ci, cond in enumerate(CONDITIONS):
        cond_start = time.time()
        logger.info("")
        logger.info(f"━━━ [{cond.upper()}] Condition {ci+1}/{len(CONDITIONS)} ━━━")

        cond_violations = []
        cond_retentions = []
        cond_reinjections = []

        for trial in range(N_TRIALS):
            trial_start = time.time()

            env_t = SafeNavigationEnv(obstacles=obstacles)
            env_t.waypoints = waypoints

            result = run_trial(cond, env_t, obstacles,
                               trial_seed=RANDOM_SEED + trial * 13 + ci * 10000)
            all_results[cond].append(result)
            completed_trials += 1

            v = result['total_violations']
            ret_i = result['retention_initial']
            ret_f = result['retention_final']
            reinj = result['n_reinjections']
            trial_time = time.time() - trial_start

            cond_violations.append(v)
            cond_retentions.append(ret_f)
            cond_reinjections.append(reinj)

            # Per-trial log (DEBUG level = file only, except every 10th = INFO)
            msg = (f"  Trial {trial+1:3d}/{N_TRIALS}  "
                   f"viol={v:5d}  ret={ret_i:.3f}->{ret_f:.3f}  "
                   f"reinj={reinj:4d}  err={result['mean_tracking_error']:.1f}  "
                   f"[{trial_time:.1f}s]")

            if (trial + 1) % 10 == 0 or trial == 0:
                logger.info(msg)
            else:
                logger.debug(msg)

            # Progress estimate every 10 trials
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - experiment_start
                rate = completed_trials / elapsed
                remaining = (total_trials - completed_trials) / rate
                eta = datetime.now() + timedelta(seconds=remaining)
                logger.info(
                    f"  ── Progress: {completed_trials}/{total_trials} trials "
                    f"({100*completed_trials/total_trials:.0f}%) | "
                    f"ETA: {eta.strftime('%H:%M:%S')} "
                    f"(~{remaining:.0f}s remaining) ──"
                )

        # Condition summary
        cond_time = time.time() - cond_start
        logger.info(f"  [{cond.upper()}] Complete in {cond_time:.1f}s")
        logger.info(f"    Violations: {np.mean(cond_violations):.1f} ± {np.std(cond_violations):.1f}")
        logger.info(f"    Retention:  {np.mean(cond_retentions):.3f} ± {np.std(cond_retentions):.3f}")
        logger.info(f"    Reinjections: {np.mean(cond_reinjections):.1f}")

        # Save checkpoint after each condition
        _save_checkpoint(all_results, obstacles, waypoints, ci + 1)

    total_time = time.time() - experiment_start
    logger.info("")
    log_banner(f"All {total_trials} trials complete in {total_time:.1f}s ({total_time/60:.1f} min)")

    return all_results, obstacles, waypoints


def _save_checkpoint(all_results, obstacles, waypoints, conditions_done):
    """Save partial results as checkpoint (no positions to keep file small)."""
    cp = {
        "conditions_completed": conditions_done,
        "timestamp": datetime.now().isoformat(),
    }
    for c in CONDITIONS[:conditions_done]:
        cp[c] = [{k: v for k, v in t.items() if k not in ("positions",)}
                  for t in all_results[c]]
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(cp, f, indent=1)
    logger.debug(f"  Checkpoint saved ({conditions_done}/{len(CONDITIONS)} conditions)")


# ══════════════════════════════════════════════════════════════════════
# 8. Statistical Analysis
# ══════════════════════════════════════════════════════════════════════

def compute_cliffs_delta(x, y):
    """
    Cliff's delta: non-parametric effect size.
    δ = (#{x_i > y_j} - #{x_i < y_j}) / (n_x · n_y)
    Range: [-1, 1]. |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    n_x, n_y = len(x), len(y)
    count_greater = sum(1 for xi in x for yj in y if xi > yj)
    count_less = sum(1 for xi in x for yj in y if xi < yj)
    return (count_greater - count_less) / (n_x * n_y)


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95, statistic=np.median):
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(99)
    boot_stats = np.array([
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    return float(lo), float(hi), float(statistic(data))


def analyze(all_results, obstacles, waypoints):
    """Full statistical analysis with Bonferroni correction and bootstrap CIs."""
    from scipy import stats as sp_stats

    logger.info("")
    log_banner("STATISTICAL ANALYSIS")

    # ── 1. Descriptive statistics ──
    stats = {}
    logger.info("")
    header = (f"{'Condition':<18} {'Violations':>14} {'Viol.Rate':>10} "
              f"{'Ret(init)':>10} {'Ret(final)':>14} {'Reinjections':>12} "
              f"{'TrackErr':>10}")
    logger.info(header)
    logger.info("-" * len(header))

    for c in CONDITIONS:
        trials = all_results[c]
        vm = np.mean([t['total_violations'] for t in trials])
        vs = np.std([t['total_violations'] for t in trials])
        vr = np.mean([t['violation_rate'] for t in trials])
        ri = np.mean([t['retention_initial'] for t in trials])
        rf = np.mean([t['retention_final'] for t in trials])
        rfs = np.std([t['retention_final'] for t in trials])
        te = np.mean([t['mean_tracking_error'] for t in trials])
        reinj = np.mean([t['n_reinjections'] for t in trials])
        stats[c] = dict(vm=vm, vs=vs, vr=vr, ri=ri, rf=rf, rfs=rfs, te=te, reinj=reinj)
        logger.info(f"{CONDITION_LABELS[c]:<18} {vm:>7.1f}±{vs:<5.1f} {vr:>10.4f} "
                     f"{ri:>10.3f} {rf:>7.3f}±{rfs:.3f} {reinj:>12.1f} {te:>10.1f}")

    # ── 2. Kruskal-Wallis omnibus test ──
    logger.info("")
    logger.info("─── Kruskal-Wallis omnibus test (violations) ───")
    violation_data = {c: [t['total_violations'] for t in all_results[c]] for c in CONDITIONS}
    H, p_kw = sp_stats.kruskal(*violation_data.values())
    logger.info(f"  H = {H:.2f},  p = {p_kw:.2e}  {'(***) SIGNIFICANT' if p_kw < 0.001 else ''}")

    # ── 3. Pairwise Mann-Whitney U with Bonferroni correction ──
    n_pairs = len(list(combinations(CONDITIONS, 2)))  # 21 pairs
    alpha_bonf = 0.05 / n_pairs

    logger.info("")
    logger.info(f"─── Pairwise Mann-Whitney U (Bonferroni α = 0.05/{n_pairs} = {alpha_bonf:.4f}) ───")
    logger.info(f"{'Pair':<38} {'U':>8} {'p-raw':>12} {'p-adj':>12} {'sig':>6} {'Cohen d':>8} {'Cliff δ':>8}")
    logger.info("-" * 98)

    pairwise_results = {}
    for c1, c2 in combinations(CONDITIONS, 2):
        v1 = violation_data[c1]
        v2 = violation_data[c2]
        U, p_raw = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_adj = min(p_raw * n_pairs, 1.0)  # Bonferroni adjustment

        # Cohen's d
        pooled_std = np.sqrt((np.std(v1)**2 + np.std(v2)**2) / 2)
        d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0

        # Cliff's delta
        cliff = compute_cliffs_delta(v1, v2)

        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        pair_label = f"{CONDITION_LABELS[c1]} vs {CONDITION_LABELS[c2]}"
        logger.info(f"{pair_label:<38} {U:>8.1f} {p_raw:>12.6f} {p_adj:>12.6f} {sig:>6} {d:>+8.2f} {cliff:>+8.3f}")

        pairwise_results[(c1, c2)] = {
            "U": float(U), "p_raw": float(p_raw), "p_adj": float(p_adj),
            "cohens_d": float(d), "cliffs_delta": float(cliff), "sig": sig
        }

    # ── 4. Bootstrap CIs on median violations ──
    logger.info("")
    logger.info(f"─── Bootstrap 95% CIs on median violations ({N_BOOTSTRAP} resamples) ───")
    bootstrap_results = {}
    for c in CONDITIONS:
        vdata = [t['total_violations'] for t in all_results[c]]
        lo, hi, med = bootstrap_ci(vdata)
        bootstrap_results[c] = {"median": med, "ci_lo": lo, "ci_hi": hi}
        logger.info(f"  {CONDITION_LABELS[c]:<18}: median = {med:.1f}  95% CI = [{lo:.1f}, {hi:.1f}]")

    # ── 5. Key head-to-head comparisons ──
    logger.info("")
    logger.info("─── Key head-to-head comparisons ───")

    key_pairs = [
        ("vsa", "unconstrained", "greater"),
        ("vsa", "timer", "greater"),
        ("vsa", "cbf", "greater"),
        ("vsa", "ema", "greater"),
        ("vsa", "norm_monitor", "greater"),
    ]
    for c_worse, c_better, alt in key_pairs:
        v_worse = violation_data[c_worse]
        v_better = violation_data[c_better]
        U, p = sp_stats.mannwhitneyu(v_worse, v_better, alternative=alt)
        p_adj = min(p * n_pairs, 1.0)
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        diff = np.mean(v_worse) - np.mean(v_better)
        logger.info(f"  {CONDITION_LABELS[c_better]} > {CONDITION_LABELS[c_worse]}? "
                     f"  Δ = {diff:+.1f} violations  p_adj={p_adj:.6f} ({sig})")

    # ── 6. Retention tests ──
    logger.info("")
    logger.info("─── Final retention: VSA vs each condition (Mann-Whitney, Bonferroni) ───")
    for c in CONDITIONS:
        if c == "vsa":
            continue
        vsa_r = [t['retention_final'] for t in all_results['vsa']]
        c_r = [t['retention_final'] for t in all_results[c]]
        U, p = sp_stats.mannwhitneyu(vsa_r, c_r, alternative='two-sided')
        p_adj = min(p * n_pairs, 1.0)
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."
        logger.info(f"  VSA vs {CONDITION_LABELS[c]:<18}: U={U:>8.1f}  p_adj={p_adj:.6f} ({sig})  "
                     f"VSA={np.mean(vsa_r):.3f}  {CONDITION_LABELS[c]}={np.mean(c_r):.3f}")

    # ── 7. Power analysis ──
    logger.info("")
    logger.info("─── Post-hoc power analysis ───")
    # For the VSA vs Timer comparison (expected to be the tightest)
    vsa_v = violation_data['vsa']
    tmr_v = violation_data['timer']
    d_vt = abs(np.mean(vsa_v) - np.mean(tmr_v)) / np.sqrt((np.std(vsa_v)**2 + np.std(tmr_v)**2) / 2)
    logger.info(f"  VSA vs Timer: observed d = {d_vt:.3f}, n = {N_TRIALS}")
    logger.info(f"  At α = {alpha_bonf:.4f} (Bonferroni), d = {d_vt:.3f}:")
    # Approximate power from normal approximation
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha_bonf / 2)
    noncentrality = d_vt * np.sqrt(N_TRIALS / 2)
    power = 1 - norm.cdf(z_alpha - noncentrality) + norm.cdf(-z_alpha - noncentrality)
    logger.info(f"  Approximate power ≈ {power:.3f} ({power*100:.1f}%)")

    return stats, pairwise_results, bootstrap_results


# ══════════════════════════════════════════════════════════════════════
# 9. Visualization
# ══════════════════════════════════════════════════════════════════════

def make_figures(all_results, obstacles, waypoints, stats):
    """Generate comprehensive multi-panel figure."""
    logger.info("")
    log_banner("Generating figures")

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.38, wspace=0.30)

    conds = CONDITIONS
    labels = [CONDITION_LABELS[c] for c in conds]
    colors = [CONDITION_COLORS[c] for c in conds]
    short_labels = [CONDITION_LABELS[c].split('\n')[0] for c in conds]

    # ── A: Violation rate over time ──
    ax1 = fig.add_subplot(gs[0, 0])
    for c, sl, color in zip(conds, short_labels, colors):
        tls = [t['violation_timeline'] for t in all_results[c]]
        ml = max(len(tl) for tl in tls)
        pad = np.zeros((len(tls), ml))
        for i, tl in enumerate(tls):
            pad[i, :len(tl)] = tl
        mn = np.mean(pad, axis=0)
        sd = np.std(pad, axis=0)
        x = np.arange(ml) * 1000 + WARMUP_STEPS
        ax1.plot(x, mn, label=sl, color=color, linewidth=2)
        ax1.fill_between(x, np.maximum(mn - sd, 0), mn + sd, color=color, alpha=0.10)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Violation Rate (per 1000-step window)")
    ax1.set_title("A. Safety Violations Over Time", fontweight='bold')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── B: Safety retention over time ──
    ax2 = fig.add_subplot(gs[0, 1])
    for c, sl, color in zip(conds, short_labels, colors):
        curves = [t['retention_snapshots'] for t in all_results[c]]
        ml = max(len(cr) for cr in curves)
        pad = np.full((len(curves), ml), np.nan)
        for i, cr in enumerate(curves):
            pad[i, :len(cr)] = cr
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * 500 + WARMUP_STEPS
        ax2.plot(x, mn, label=sl, color=color, linewidth=2)
        ax2.fill_between(x, mn - sd, mn + sd, color=color, alpha=0.10)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Safety Retention (cosine sim.)")
    ax2.set_title("B. Safety Memory Retention Over Time", fontweight='bold')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── C: Violations boxplot ──
    ax3 = fig.add_subplot(gs[0, 2])
    vdata = [[t['total_violations'] for t in all_results[c]] for c in conds]
    bp = ax3.boxplot(vdata, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax3.set_ylabel("Total Violations (20k steps)")
    ax3.set_title(f"C. Safety Violations by Condition (n={N_TRIALS})", fontweight='bold')
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.3, axis='y')

    # ── D: Final retention boxplot ──
    ax4 = fig.add_subplot(gs[1, 0])
    rdata = [[t['retention_final'] for t in all_results[c]] for c in conds]
    bp2 = ax4.boxplot(rdata, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax4.set_ylabel("Final Retention (last 1k steps)")
    ax4.set_title("D. Safety Memory Retention at End", fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.3, axis='y')

    # ── E: Reinjection counts ──
    ax5 = fig.add_subplot(gs[1, 1])
    reinj_conds = [c for c in conds if c not in ("unconstrained", "oracle")]
    reinj_labels = [CONDITION_LABELS[c].split('\n')[0] for c in reinj_conds]
    reinj_colors = [CONDITION_COLORS[c] for c in reinj_conds]
    reinj_data = [[t['n_reinjections'] for t in all_results[c]] for c in reinj_conds]
    if reinj_data:
        bp3 = ax5.boxplot(reinj_data, tick_labels=reinj_labels, patch_artist=True)
        for patch, color in zip(bp3['boxes'], reinj_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax5.set_ylabel("Number of Re-injections")
    ax5.set_title("E. Re-injection Count by Method", fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)
    ax5.grid(True, alpha=0.3, axis='y')

    # ── F: Violation rate bar chart with bootstrap CIs ──
    ax6 = fig.add_subplot(gs[1, 2])
    means = [np.mean([t['violation_rate'] for t in all_results[c]]) for c in conds]
    stds = [np.std([t['violation_rate'] for t in all_results[c]]) for c in conds]
    x_pos = np.arange(len(conds))
    bars = ax6.bar(x_pos, means, yerr=stds, color=colors, alpha=0.6, capsize=4, edgecolor='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
    ax6.set_ylabel("Violation Rate")
    ax6.set_title("F. Mean Violation Rate ± SD", fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # ── G: Example trajectories (unconstrained vs VSA) ──
    ax7 = fig.add_subplot(gs[2, 0])
    for c, sl, color in [("unconstrained", "Unconstrained", CONDITION_COLORS["unconstrained"]),
                          ("vsa", "VSA", CONDITION_COLORS["vsa"])]:
        vlist = [t['total_violations'] for t in all_results[c]]
        med_idx = np.argsort(vlist)[len(vlist)//2]
        pos = np.array(all_results[c][med_idx]['positions'])
        ax7.plot(pos[:, 0], pos[:, 1], color=color, alpha=0.5, linewidth=0.3, label=sl)
    for ox, oy, r in obstacles:
        ax7.add_patch(Circle((ox, oy), r, color='red', alpha=0.25, fill=True))
        ax7.add_patch(Circle((ox, oy), r, color='red', alpha=0.7, fill=False, linewidth=2))
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_title("G. Trajectories: Unconstrained vs VSA (median)", fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.set_aspect('equal')
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-ARENA_SIZE * 1.1, ARENA_SIZE * 1.1)
    ax7.set_ylim(-ARENA_SIZE * 1.1, ARENA_SIZE * 1.1)

    # ── H: VSA drift metric ──
    ax8 = fig.add_subplot(gs[2, 1])
    dms = [t['drift_metric'] for t in all_results['vsa'] if t['drift_metric']]
    if dms:
        ml = max(len(d) for d in dms)
        pad = np.full((len(dms), ml), np.nan)
        for i, d in enumerate(dms):
            pad[i, :len(d)] = d
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * VSA_CHECK_INTERVAL
        ax8.plot(x, mn, color=CONDITION_COLORS['vsa'], alpha=0.7, linewidth=1, label='Mean drift')
        ax8.fill_between(x, mn - sd, mn + sd, color=CONDITION_COLORS['vsa'], alpha=0.1)
        if len(mn) > 20:
            sm = np.convolve(mn, np.ones(20)/20, mode='valid')
            ax8.plot(x[10:10+len(sm)], sm, color=CONDITION_COLORS['vsa'],
                     linewidth=2.5, label='Smoothed (20-pt)')
        ax8.axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                     label=f'Threshold ({DRIFT_THRESHOLD})')
    ax8.set_xlabel("Time Step")
    ax8.set_ylabel("Drift Metric (cosine sim.)")
    ax8.set_title("H. VSA Drift Detection Signal", fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # ── I: Effect size heatmap (Cohen's d, VSA vs all) ──
    ax9 = fig.add_subplot(gs[2, 2])
    other_conds = [c for c in conds if c != "vsa"]
    d_values = []
    for c in other_conds:
        v_vsa = [t['total_violations'] for t in all_results['vsa']]
        v_c = [t['total_violations'] for t in all_results[c]]
        ps = np.sqrt((np.std(v_vsa)**2 + np.std(v_c)**2) / 2)
        d = (np.mean(v_c) - np.mean(v_vsa)) / ps if ps > 0 else 0
        d_values.append(d)

    y_pos = np.arange(len(other_conds))
    bar_colors = [CONDITION_COLORS[c] for c in other_conds]
    bars = ax9.barh(y_pos, d_values, color=bar_colors, alpha=0.6, edgecolor='black')
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels([CONDITION_LABELS[c].split('\n')[0] for c in other_conds], fontsize=9)
    ax9.set_xlabel("Cohen's d (positive = VSA has fewer violations)")
    ax9.set_title("I. Effect Size: VSA vs Each Condition", fontweight='bold')
    ax9.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Small (0.2)')
    ax9.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax9.axvline(x=0.8, color='gray', linestyle='-', alpha=0.5, label='Large (0.8)')
    ax9.axvline(x=0, color='black', linewidth=0.8)
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3, axis='x')

    plt.suptitle(
        "VSA-Constrained Liquid Neural Networks: Full 7-Condition Safety Benchmark\n"
        f"{N_STEPS} steps × {N_TRIALS} trials | D={D_VSA} | {len(obstacles)} obstacles | "
        f"Bonferroni α=0.05/{n_pairs}",
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches='tight')
    logger.info(f"  Figure saved: {FIGURE_PATH}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# 10. Save Results
# ══════════════════════════════════════════════════════════════════════

def save_results(all_results, obstacles, waypoints, stats, pairwise, bootstrap):
    """Save all results as JSON."""
    save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_steps": N_STEPS,
            "n_trials": N_TRIALS,
            "d_vsa": D_VSA,
            "drift_threshold": DRIFT_THRESHOLD,
            "timer_interval": TIMER_INTERVAL,
            "ema_alpha": EMA_ALPHA,
            "norm_threshold_frac": NORM_THRESHOLD_FRAC,
            "cbf_gamma": CBF_GAMMA,
            "n_obstacles": len(obstacles),
            "obstacles": [list(o) for o in obstacles],
            "n_bootstrap": N_BOOTSTRAP,
            "bonferroni_pairs": len(list(combinations(CONDITIONS, 2))),
            "bonferroni_alpha": 0.05 / len(list(combinations(CONDITIONS, 2))),
        },
        "summary": {},
        "pairwise_tests": {},
        "bootstrap_cis": bootstrap,
    }

    for c in CONDITIONS:
        save["summary"][c] = stats[c]

    for (c1, c2), res in pairwise.items():
        save["pairwise_tests"][f"{c1}_vs_{c2}"] = res

    # Trial-level data (no positions — too large)
    for c in CONDITIONS:
        save[c] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in t.items() if k != "positions"}
                    for t in all_results[c]]

    with open(RESULTS_JSON, 'w') as f:
        json.dump(save, f, indent=2)
    logger.info(f"  Results saved: {RESULTS_JSON}")


# ══════════════════════════════════════════════════════════════════════
# 11. Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log_banner("VSA-Constrained LTC Safety: Full 7-Condition Benchmark")
    logger.info(f"  Python {sys.version.split()[0]}")
    logger.info(f"  NumPy {np.__version__}")
    logger.info(f"  Output dir: {SCRIPT_DIR}")
    logger.info(f"  Log file: {LOG_FILE}")

    # Config summary
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Steps per trial:     {N_STEPS}")
    logger.info(f"  Trials per condition: {N_TRIALS}")
    logger.info(f"  Conditions:          {len(CONDITIONS)}")
    logger.info(f"  Total trials:        {len(CONDITIONS) * N_TRIALS}")
    logger.info(f"  VSA dimensionality:  {D_VSA}")
    logger.info(f"  Timer interval:      {TIMER_INTERVAL}")
    logger.info(f"  VSA drift threshold: {DRIFT_THRESHOLD}")
    logger.info(f"  EMA alpha:           {EMA_ALPHA}")
    logger.info(f"  Norm threshold:      {NORM_THRESHOLD_FRAC} × ||x_ref||")
    logger.info(f"  CBF gamma:           {CBF_GAMMA}")
    logger.info(f"  Bootstrap resamples: {N_BOOTSTRAP}")

    logger.info("")
    logger.info("Power analysis (pre-experiment):")
    logger.info(f"  Target: detect d≥0.67 at Bonferroni α=0.05/21≈0.0024, 80% power")
    logger.info(f"  Required n ≈ 48 per condition. Using n={N_TRIALS}.")

    # Run
    all_results, obs, wps = run_experiment()

    # Analyze
    stats, pairwise, bootstrap = analyze(all_results, obs, wps)

    # Figures
    n_pairs = len(list(combinations(CONDITIONS, 2)))
    make_figures(all_results, obs, wps, stats)

    # Save
    save_results(all_results, obs, wps, stats, pairwise, bootstrap)

    # Final summary
    logger.info("")
    log_banner("EXPERIMENT COMPLETE")
    logger.info("")
    logger.info("  RANKING (by mean violations):")
    ranked = sorted(CONDITIONS, key=lambda c: stats[c]['vm'])
    for rank, c in enumerate(ranked, 1):
        s = stats[c]
        logger.info(f"    {rank}. {CONDITION_LABELS[c]:<18} {s['vm']:>7.1f} ± {s['vs']:.1f} violations  "
                     f"(retention: {s['rf']:.3f})")

    logger.info("")
    logger.info("  KEY COMPARISONS:")
    vsa_timer = pairwise.get(("timer", "vsa")) or pairwise.get(("vsa", "timer"))
    if vsa_timer:
        logger.info(f"    VSA vs Timer:          p_adj = {vsa_timer['p_adj']:.6f} ({vsa_timer['sig']})  d = {vsa_timer['cohens_d']:+.2f}")
    vsa_cbf = pairwise.get(("cbf", "vsa")) or pairwise.get(("vsa", "cbf"))
    if vsa_cbf:
        logger.info(f"    VSA vs CBF:            p_adj = {vsa_cbf['p_adj']:.6f} ({vsa_cbf['sig']})  d = {vsa_cbf['cohens_d']:+.2f}")
    vsa_ema = pairwise.get(("ema", "vsa")) or pairwise.get(("vsa", "ema"))
    if vsa_ema:
        logger.info(f"    VSA vs EMA:            p_adj = {vsa_ema['p_adj']:.6f} ({vsa_ema['sig']})  d = {vsa_ema['cohens_d']:+.2f}")
    vsa_norm = pairwise.get(("norm_monitor", "vsa")) or pairwise.get(("vsa", "norm_monitor"))
    if vsa_norm:
        logger.info(f"    VSA vs Norm Monitor:   p_adj = {vsa_norm['p_adj']:.6f} ({vsa_norm['sig']})  d = {vsa_norm['cohens_d']:+.2f}")

    logger.info("")
    logger.info(f"  Log file: {LOG_FILE}")
    logger.info(f"  Results:  {RESULTS_JSON}")
    logger.info(f"  Figure:   {FIGURE_PATH}")
    logger.info("")
