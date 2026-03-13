"""
Experiment: VSA-Constrained Liquid Neural Networks for Long-Horizon Safety
==========================================================================

Tests whether VSA (Holographic Invariant Storage) can maintain safety constraints
in Liquid Time-Constant (LTC) neural networks over 15,000+ time steps, where
unconstrained LTC networks forget safety due to exponential state decay.

Architecture:
  - A PD controller tracks waypoints (competent navigation baseline).
  - An LTC "safety memory" network encodes obstacle knowledge in its hidden state.
  - The safety memory's output modulates a repulsive force field near obstacles.
  - Over time, the LTC's ODE dynamics cause the safety encoding to decay.
  - Without intervention, the agent forgets obstacles and enters forbidden zones.

The separation is deliberate: we isolate the SAFETY MEMORY mechanism from the
CONTROL mechanism so the experiment tests exactly one thing—does VSA preserve
the safety signal where the ODE alone does not?

Conditions:
  C1: Unconstrained       — safety encoded once at t=0, no re-injection
  C2: Timer re-injection  — safety re-encoded every K steps (fixed interval)
  C3: VSA-constrained     — HIS protocol recovers + re-injects when drift detected
  C4: Oracle              — perfect obstacle knowledge at every step

Benchmark: 2D point-mass navigation with forbidden zones.
  Waypoints are deliberately placed near obstacles to force close encounters.
  15,000 time steps (~150 seconds at dt=0.01), 25 trials per condition.

Usage:
  python ltc_safety_experiment.py

Requires: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import json
import time
import os

# ══════════════════════════════════════════════════════════════════════
# 0. Configuration
# ══════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulation
DT = 0.01
N_STEPS = 15000
WARMUP_STEPS = 100

# Agent dynamics
MASS = 1.0
DAMPING = 2.0
MAX_FORCE = 8.0

# PD Controller gains
KP = 3.0   # Proportional gain
KD = 2.0   # Derivative gain

# LTC Safety Memory Network
N_SAFETY_NEURONS = 64
TAU_SAFETY_MIN = 2.0      # Time constants for safety memory
TAU_SAFETY_MAX = 8.0
SAFETY_DRIVE_SCALE = 0.3  # How strongly inputs drive the safety state

# VSA / HIS
D_VSA = 10000
DRIFT_THRESHOLD = 0.55      # Re-inject when drift metric drops below this
VSA_INJECTION_GAIN = 1.5    # How strongly to re-inject

# Timer baseline
TIMER_INTERVAL = 300       # Re-inject every N steps

# Safety environment
N_OBSTACLES = 6
OBSTACLE_RADIUS = 1.2
ARENA_SIZE = 12.0
REPULSION_RANGE = 3.0     # Distance at which repulsion activates
REPULSION_STRENGTH = 12.0  # Base repulsion force magnitude

# Waypoint generation
WAYPOINT_HOLD_STEPS = 250

# Experiment
N_TRIALS = 25

print("=" * 70)
print("  VSA-Constrained Liquid Neural Networks: Long-Horizon Safety")
print("  Benchmark: 2D Navigation with Forbidden Zones")
print(f"  {N_STEPS} time steps | {N_TRIALS} trials | {D_VSA}-d hypervectors")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# 1. VSA Module (Holographic Invariant Storage)
# ══════════════════════════════════════════════════════════════════════

class VSAModule:
    """Bipolar hypervector algebra for safety invariant storage and recovery."""

    def __init__(self, d=D_VSA):
        self.d = d
        self._proj_cache = {}  # Cache projection matrices by dimension

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
        """Get (or create and cache) the projection matrix for dimension n."""
        if n not in self._proj_cache:
            rng = np.random.RandomState(12345)
            self._proj_cache[n] = rng.randn(n, self.d) / np.sqrt(n)
        return self._proj_cache[n]

    def state_to_hypervector(self, state_vec):
        """
        Map an N-dimensional state vector into a D-dimensional bipolar hypervector.
        Uses a fixed random projection (seeded deterministically, cached).
        """
        proj = self._get_projection(len(state_vec))
        projected = state_vec @ proj
        return np.sign(projected)

    def hypervector_to_state(self, hvec, state_dim):
        """
        Map a D-dimensional hypervector back to N-dimensional state space.
        Uses the pseudoinverse of the same projection.
        """
        proj = self._get_projection(state_dim)
        reconstructed = hvec @ proj.T / self.d * state_dim
        return reconstructed

    def create_invariant(self, safety_state):
        """
        HIS protocol: encode safety state as a holographic invariant.
        H_inv = bind(K_goal, binarize(project(safety_state)))
        Returns (key, invariant, reference_hvec).
        """
        key = self.random_bipolar()
        ref_hvec = self.state_to_hypervector(safety_state)
        invariant = self.bind(key, ref_hvec)
        return key, invariant, ref_hvec

    def restore(self, invariant, current_state, key, ref_hvec):
        """
        Full HIS restoration protocol.
        Returns (recovered_hvec, drift_metric, recovery_sim).
        """
        # Encode current state as noise
        noise_hvec = self.state_to_hypervector(current_state)

        # Normalize noise to match invariant magnitude
        inv_norm = np.linalg.norm(invariant)
        noise_norm = np.linalg.norm(noise_hvec)
        if noise_norm > 1e-12:
            noise_hat = noise_hvec * (inv_norm / noise_norm)
        else:
            noise_hat = np.zeros_like(noise_hvec)

        # Drift metric: how similar is the current state encoding to the reference?
        drift_metric = self.cosine_similarity(noise_hvec, ref_hvec)

        # Superimpose + cleanup + unbind
        superimposed = invariant + noise_hat
        cleaned = self.sign_cleanup(superimposed)
        recovered = self.unbind(cleaned, key)

        # Recovery quality
        recovery_sim = self.cosine_similarity(recovered, ref_hvec)

        return recovered, drift_metric, recovery_sim


# ══════════════════════════════════════════════════════════════════════
# 2. LTC Safety Memory Network
# ══════════════════════════════════════════════════════════════════════

class LTCSafetyMemory:
    """
    A Liquid Time-Constant network that serves as safety memory.

    The hidden state x(t) encodes obstacle knowledge. The ODE dynamics:
      dx_i/dt = -(1/tau_i) * x_i + sigma(W_in @ input + W_rec @ x + b) * drive_scale

    The key property: WITHOUT ongoing reinforcement of the safety signal,
    the -(1/tau) * x term causes exponential decay toward zero.
    The input-driven term pushes the state toward whatever the current
    input is, overwriting the safety encoding.

    Output: a scalar "safety awareness" score in [0, 1] for each obstacle.
    """

    def __init__(self, n_obstacles, n_neurons=N_SAFETY_NEURONS, seed=0):
        self.n_obstacles = n_obstacles
        self.n_neurons = n_neurons

        rng = np.random.RandomState(seed)

        # Each obstacle gets a dedicated bank of neurons
        self.neurons_per_obstacle = n_neurons // n_obstacles

        # Time constants: higher = slower decay = longer memory
        self.tau = rng.uniform(TAU_SAFETY_MIN, TAU_SAFETY_MAX, size=n_neurons)

        # Input weights (agent position -> safety state)
        # 2D input: [px, py] relative to each obstacle
        self.W_in = rng.randn(2, n_neurons) * 0.3

        # Recurrent weights (mild self-excitation within each bank)
        self.W_rec = np.zeros((n_neurons, n_neurons))
        for k in range(n_obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            block = rng.randn(self.neurons_per_obstacle, self.neurons_per_obstacle) * 0.1
            # Make slightly contractive
            sn = np.linalg.norm(block, ord=2)
            if sn > 0.8:
                block *= 0.8 / sn
            self.W_rec[start:end, start:end] = block

        # Readout weights: each obstacle bank -> one scalar score
        self.W_out = np.zeros((n_neurons, n_obstacles))
        for k in range(n_obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            self.W_out[start:end, k] = rng.randn(self.neurons_per_obstacle) * 0.5

        self.b = rng.randn(n_neurons) * 0.05

        # Hidden state
        self.x = np.zeros(n_neurons)

    def reset(self):
        self.x = np.zeros(self.n_neurons)

    def encode_safety(self, obstacles):
        """
        Encode obstacle locations into the hidden state.
        Each obstacle's neuron bank is initialized with a pattern
        derived from (x, y, r).
        """
        for k, (ox, oy, r) in enumerate(obstacles):
            start = k * self.neurons_per_obstacle
            end = start + self.neurons_per_obstacle
            # Create a distinctive spatial pattern for this obstacle
            phases = np.linspace(0, 2 * np.pi, self.neurons_per_obstacle)
            pattern = (
                np.sin(phases * ox) * r +
                np.cos(phases * oy) * r +
                np.sin(phases * (ox + oy)) * 0.5
            )
            # Normalize to unit energy
            norm = np.linalg.norm(pattern)
            if norm > 1e-12:
                pattern = pattern / norm
            self.x[start:end] = pattern

    def get_reference_state(self, obstacles):
        """Get the reference safety state (what encode_safety would produce)."""
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
        """
        Advance the LTC ODE by one step.
        Returns: awareness scores [0, 1] for each obstacle.
        """
        # Input: agent position
        z_in = agent_pos @ self.W_in * SAFETY_DRIVE_SCALE
        z_rec = self.x @ self.W_rec

        # ODE: dx/dt = -(1/tau) * x + tanh(W_in @ input + W_rec @ x + b) * scale
        drive = np.tanh(z_in + z_rec + self.b) * SAFETY_DRIVE_SCALE
        dxdt = -(1.0 / self.tau) * self.x + drive

        # External forcing (safety re-injection)
        if forcing is not None:
            dxdt += forcing

        self.x = self.x + dt * dxdt
        self.x = np.clip(self.x, -5, 5)

        # Readout: per-obstacle awareness
        raw_scores = np.tanh(self.x) @ self.W_out
        awareness = 1.0 / (1.0 + np.exp(-raw_scores * 3))  # Sigmoid scaling
        return awareness

    def get_overall_retention(self, obstacles):
        """Scalar retention: cosine similarity of full state with reference."""
        ref = self.get_reference_state(obstacles)
        nx = np.linalg.norm(self.x)
        nr = np.linalg.norm(ref)
        if nx < 1e-12 or nr < 1e-12:
            return 0.0
        return np.dot(self.x, ref) / (nx * nr)


# ══════════════════════════════════════════════════════════════════════
# 3. Navigation Environment
# ══════════════════════════════════════════════════════════════════════

class SafeNavigationEnv:
    """2D point-mass navigation with circular forbidden zones."""

    def __init__(self, obstacles=None):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.obstacles = obstacles if obstacles is not None else self._generate_obstacles()
        self.waypoints = []

    def _generate_obstacles(self):
        """Generate obstacles in a ring around the arena center."""
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
        """
        Generate waypoints that FORCE close encounters with obstacles.
        Pattern: alternate between obstacle-adjacent points and opposite-side points.
        """
        self.waypoints = []
        n_waypoints = n_steps // WAYPOINT_HOLD_STEPS + 1

        for i in range(n_waypoints):
            if i % 2 == 0 and len(self.obstacles) > 0:
                # Target near an obstacle
                obs_idx = i % len(self.obstacles)
                ox, oy, r = self.obstacles[obs_idx]
                angle_to_center = np.arctan2(-oy, -ox)
                target_angle = angle_to_center + np.random.uniform(-0.5, 0.5)
                target_dist = r + np.random.uniform(0.3, 1.0)
                wx = ox + target_dist * np.cos(target_angle)
                wy = oy + target_dist * np.sin(target_angle)
            else:
                # Random position across arena
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
    """
    Compute safety repulsion force based on obstacle awareness.
    awareness_scores: per-obstacle [0, 1] indicating how well the agent
    "remembers" each obstacle.
    """
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
# 4. Trial Runner
# ══════════════════════════════════════════════════════════════════════

def run_trial(condition, env, obstacles, trial_seed=0):
    """
    Run one trial.
    condition: "unconstrained" | "timer" | "vsa" | "oracle"
    """
    # Initialize safety memory
    safety_mem = LTCSafetyMemory(len(obstacles), N_SAFETY_NEURONS, seed=trial_seed)
    safety_mem.encode_safety(obstacles)
    ref_state = safety_mem.get_reference_state(obstacles)

    # VSA setup
    vsa = None
    vsa_key = None
    vsa_inv = None
    vsa_ref = None
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

        # ── PD Controller (competent navigation) ──
        error = target - env.pos
        pd_force = KP * error - KD * env.vel

        # ── Safety Memory Step ──
        forcing = None

        if condition == "oracle":
            # Perfect knowledge: always use reference
            safety_mem.x = ref_state.copy()
            safety_mem.step(env.pos, dt=DT)  # tick for consistency
            retention = 1.0

        elif condition == "timer":
            if step > 0 and step % TIMER_INTERVAL == 0:
                # Re-encode safety directly into hidden state
                forcing = (ref_state - safety_mem.x) * VSA_INJECTION_GAIN / DT
                reinjection_steps.append(step)
            safety_mem.step(env.pos, dt=DT, forcing=forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        elif condition == "vsa":
            # Check drift every 50 steps (realistic: not every ODE step)
            if step % 50 == 0:
                current_state = safety_mem.x.copy()
                recovered_hvec, drift_sim, recovery_sim = vsa.restore(
                    vsa_inv, current_state, vsa_key, vsa_ref
                )
                drift_metric_curve.append(drift_sim)

                if drift_sim < DRIFT_THRESHOLD:
                    # Drift detected -> recover safety state from VSA
                    recovered_state = vsa.hypervector_to_state(recovered_hvec, N_SAFETY_NEURONS)
                    # Normalize to match reference magnitude
                    rec_norm = np.linalg.norm(recovered_state)
                    ref_norm = np.linalg.norm(ref_state)
                    if rec_norm > 1e-12:
                        recovered_state = recovered_state * (ref_norm / rec_norm)
                    forcing = (recovered_state - safety_mem.x) * VSA_INJECTION_GAIN / DT
                    reinjection_steps.append(step)

            safety_mem.step(env.pos, dt=DT, forcing=forcing)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        else:  # unconstrained
            safety_mem.step(env.pos, dt=DT)
            retention = max(0.0, safety_mem.get_overall_retention(obstacles))

        # ── Safety repulsion (directly proportional to retention) ──
        awareness = np.full(len(obstacles), retention)
        safety_force = compute_repulsion(env.pos, obstacles, awareness)

        # ── Combined control ──
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

    # ── Compile ──
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
        "reinjection_steps": reinjection_steps[:100],
        "positions": positions.tolist(),
        "drift_metric": drift_metric_curve if drift_metric_curve else [],
    }


# ══════════════════════════════════════════════════════════════════════
# 5. Main Experiment
# ══════════════════════════════════════════════════════════════════════

def run_experiment():
    conditions = ["unconstrained", "timer", "vsa", "oracle"]
    all_results = {c: [] for c in conditions}

    print(f"\n[1/3] Generating environment...")
    np.random.seed(RANDOM_SEED)
    env = SafeNavigationEnv()
    obstacles = env.obstacles
    env.generate_waypoints()
    waypoints = env.waypoints

    print(f"  Obstacles ({len(obstacles)}):")
    for i, (ox, oy, r) in enumerate(obstacles):
        print(f"    O{i+1}: ({ox:.2f}, {oy:.2f}), r={r:.2f}")
    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Every other waypoint is near an obstacle (forced encounters)")

    print(f"\n[2/3] Running {len(conditions)} x {N_TRIALS} = {len(conditions)*N_TRIALS} trials...")
    t0 = time.time()

    for ci, cond in enumerate(conditions):
        print(f"\n  [{cond.upper()}] ({ci+1}/{len(conditions)})")
        for trial in range(N_TRIALS):
            env_t = SafeNavigationEnv(obstacles=obstacles)
            env_t.waypoints = waypoints

            result = run_trial(cond, env_t, obstacles,
                               trial_seed=RANDOM_SEED + trial * 13 + ci * 10000)
            all_results[cond].append(result)

            v = result['total_violations']
            ret_i = result['retention_initial']
            ret_f = result['retention_final']
            reinj = result['n_reinjections']
            print(f"    {trial+1:2d}/{N_TRIALS}  "
                  f"violations={v:5d}  "
                  f"retention: {ret_i:.3f}->{ret_f:.3f}  "
                  f"reinjections={reinj:4d}  "
                  f"track_err={result['mean_tracking_error']:.1f}")

    print(f"\n  Done in {time.time()-t0:.1f}s")
    return all_results, obstacles, waypoints


# ══════════════════════════════════════════════════════════════════════
# 6. Analysis & Visualization
# ══════════════════════════════════════════════════════════════════════

def analyze(all_results, obstacles, waypoints):
    conditions = ["unconstrained", "timer", "vsa", "oracle"]
    labels = ["Unconstrained\n(no re-injection)",
              "Timer\n(every 300 steps)",
              "VSA-Constrained\n(drift-triggered)",
              "Oracle\n(perfect memory)"]
    short_labels = ["Unconstrained", "Timer", "VSA", "Oracle"]
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    # ── Summary Table ──
    header = (f"{'Condition':<16} {'Violations':>10} {'Viol.Rate':>10} "
              f"{'Ret(init)':>10} {'Ret(final)':>10} {'Reinjections':>12} "
              f"{'TrackErr':>10}")
    print(f"\n{header}")
    print("-" * len(header))

    stats = {}
    for c, sl in zip(conditions, short_labels):
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
        print(f"{sl:<16} {vm:>7.1f}+/-{vs:<5.1f} {vr:>9.4f} "
              f"{ri:>10.3f} {rf:>7.3f}+/-{rfs:.3f} {reinj:>11.1f} {te:>10.1f}")

    # ── Statistical Tests ──
    print("\n  Mann-Whitney U (violations): each vs. Unconstrained")
    from scipy import stats as sp_stats

    unc_v = [t['total_violations'] for t in all_results['unconstrained']]
    for c, sl in zip(conditions[1:], short_labels[1:]):
        cv = [t['total_violations'] for t in all_results[c]]
        U, p = sp_stats.mannwhitneyu(unc_v, cv, alternative='greater')
        ps = np.sqrt((np.std(unc_v)**2 + np.std(cv)**2) / 2)
        d = (np.mean(unc_v) - np.mean(cv)) / ps if ps > 0 else 0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    vs {sl:<16}: U={U:>7.1f}  p={p:.6f} ({sig})  d={d:.2f}")

    # VSA vs Timer
    print("\n  Mann-Whitney U (violations): VSA vs. Timer")
    vsa_v = [t['total_violations'] for t in all_results['vsa']]
    tmr_v = [t['total_violations'] for t in all_results['timer']]
    U, p = sp_stats.mannwhitneyu(tmr_v, vsa_v, alternative='greater')
    ps = np.sqrt((np.std(tmr_v)**2 + np.std(vsa_v)**2) / 2)
    d = (np.mean(tmr_v) - np.mean(vsa_v)) / ps if ps > 0 else 0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    Timer vs VSA: U={U:.1f}  p={p:.6f} ({sig})  d={d:.2f}")

    # Retention test
    print("\n  Mann-Whitney U (final retention): VSA vs. Unconstrained")
    unc_r = [t['retention_final'] for t in all_results['unconstrained']]
    vsa_r = [t['retention_final'] for t in all_results['vsa']]
    U, p = sp_stats.mannwhitneyu(vsa_r, unc_r, alternative='greater')
    ps = np.sqrt((np.std(vsa_r)**2 + np.std(unc_r)**2) / 2)
    d = (np.mean(vsa_r) - np.mean(unc_r)) / ps if ps > 0 else 0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"    VSA > Unconstrained: U={U:.1f}  p={p:.6f} ({sig})  d={d:.2f}")

    # ══════════════════════════════════════════════════════════════
    # Figure
    # ══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.28)

    # ── A: Violation rate over time ──
    ax1 = fig.add_subplot(gs[0, 0])
    for c, sl, color in zip(conditions, short_labels, colors):
        tls = [t['violation_timeline'] for t in all_results[c]]
        ml = max(len(tl) for tl in tls)
        pad = np.zeros((len(tls), ml))
        for i, tl in enumerate(tls):
            pad[i, :len(tl)] = tl
        mn = np.mean(pad, axis=0)
        sd = np.std(pad, axis=0)
        x = np.arange(ml) * 1000 + WARMUP_STEPS
        ax1.plot(x, mn, label=sl, color=color, linewidth=2)
        ax1.fill_between(x, np.maximum(mn - sd, 0), mn + sd, color=color, alpha=0.12)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Violation Rate (per 1000-step window)")
    ax1.set_title("A. Safety Violations Over Time", fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── B: Safety retention over time ──
    ax2 = fig.add_subplot(gs[0, 1])
    for c, sl, color in zip(conditions, short_labels, colors):
        curves = [t['retention_snapshots'] for t in all_results[c]]
        ml = max(len(cr) for cr in curves)
        pad = np.full((len(curves), ml), np.nan)
        for i, cr in enumerate(curves):
            pad[i, :len(cr)] = cr
        mn = np.nanmean(pad, axis=0)
        sd = np.nanstd(pad, axis=0)
        x = np.arange(ml) * 500 + WARMUP_STEPS
        ax2.plot(x, mn, label=sl, color=color, linewidth=2)
        ax2.fill_between(x, mn - sd, mn + sd, color=color, alpha=0.12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Safety Retention (cosine sim.)")
    ax2.set_title("B. Safety Constraint Retention Over Time", fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── C: Violations boxplot ──
    ax3 = fig.add_subplot(gs[1, 0])
    vdata = [[t['total_violations'] for t in all_results[c]] for c in conditions]
    bp = ax3.boxplot(vdata, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax3.set_ylabel("Total Violations (15k steps)")
    ax3.set_title(f"C. Safety Violations by Condition (n={N_TRIALS})", fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # ── D: Final retention boxplot ──
    ax4 = fig.add_subplot(gs[1, 1])
    rdata = [[t['retention_final'] for t in all_results[c]] for c in conditions]
    bp2 = ax4.boxplot(rdata, tick_labels=short_labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax4.set_ylabel("Final Retention (last 1k steps)")
    ax4.set_title("D. Safety Memory Retention at End", fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # ── E: Example trajectories ──
    ax5 = fig.add_subplot(gs[2, 0])
    for c, sl, color in [("unconstrained", "Unconstrained", colors[0]),
                          ("vsa", "VSA", colors[2])]:
        vlist = [t['total_violations'] for t in all_results[c]]
        med_idx = np.argsort(vlist)[len(vlist)//2]
        pos = np.array(all_results[c][med_idx]['positions'])
        ax5.plot(pos[:, 0], pos[:, 1], color=color, alpha=0.5, linewidth=0.4, label=sl)
    for ox, oy, r in obstacles:
        ax5.add_patch(Circle((ox, oy), r, color='red', alpha=0.25, fill=True))
        ax5.add_patch(Circle((ox, oy), r, color='red', alpha=0.7, fill=False, linewidth=2))
    for wx, wy in waypoints[::3]:
        ax5.plot(wx, wy, 'k+', markersize=4, alpha=0.3)
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_title("E. Example Trajectories (median trial)", fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-ARENA_SIZE * 1.1, ARENA_SIZE * 1.1)
    ax5.set_ylim(-ARENA_SIZE * 1.1, ARENA_SIZE * 1.1)

    # ── F: VSA drift metric ──
    ax6 = fig.add_subplot(gs[2, 1])
    dms = [t['drift_metric'] for t in all_results['vsa'] if t['drift_metric']]
    if dms:
        ml = max(len(d) for d in dms)
        pad = np.full((len(dms), ml), np.nan)
        for i, d in enumerate(dms):
            pad[i, :len(d)] = d
        mn = np.nanmean(pad, axis=0)
        x = np.arange(ml) * 50  # Checked every 50 steps
        ax6.plot(x, mn, color=colors[2], alpha=0.7, linewidth=1, label='Mean drift metric')
        if len(mn) > 20:
            sm = np.convolve(mn, np.ones(20)/20, mode='valid')
            ax6.plot(x[10:10+len(sm)], sm, color=colors[2], linewidth=2.5, label='Smoothed')
        ax6.axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                     label=f'Threshold (tau={DRIFT_THRESHOLD})')
        ax6.set_xlabel("Time Step")
        ax6.set_ylabel("Drift Metric (cosine sim.)")
        ax6.set_title("F. VSA Drift Detection Signal", fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

    plt.suptitle("VSA-Constrained Liquid Neural Networks: Long-Horizon Safety\n"
                 f"{N_STEPS} steps x {N_TRIALS} trials | D={D_VSA} | {len(obstacles)} obstacles",
                 fontsize=14, fontweight='bold', y=1.01)

    figpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_D1_ltc_safety.png")
    plt.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {figpath}")
    plt.close()

    # ── Save JSON ──
    jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ltc_safety_results.json")
    save = {"config": {
        "n_steps": N_STEPS, "n_trials": N_TRIALS, "d_vsa": D_VSA,
        "drift_threshold": DRIFT_THRESHOLD, "timer_interval": TIMER_INTERVAL,
        "n_obstacles": len(obstacles),
        "obstacles": [list(o) for o in obstacles],
    }}
    for c in conditions:
        save[c] = [{k: v for k, v in t.items() if k != "positions"} for t in all_results[c]]
    with open(jpath, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"  Saved: {jpath}")

    return stats


# ══════════════════════════════════════════════════════════════════════
# 7. Entry Point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results, obs, wps = run_experiment()
    st = analyze(results, obs, wps)

    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    u, t, v, o = st['unconstrained'], st['timer'], st['vsa'], st['oracle']
    print(f"\n  Violations (mean +/- std):")
    print(f"    Unconstrained:  {u['vm']:.1f} +/- {u['vs']:.1f}")
    print(f"    Timer:          {t['vm']:.1f} +/- {t['vs']:.1f}")
    print(f"    VSA:            {v['vm']:.1f} +/- {v['vs']:.1f}")
    print(f"    Oracle:         {o['vm']:.1f} +/- {o['vs']:.1f}")
    print(f"\n  Final Retention:")
    print(f"    Unconstrained:  {u['rf']:.4f}")
    print(f"    Timer:          {t['rf']:.4f}")
    print(f"    VSA:            {v['rf']:.4f}")
    print(f"    Oracle:         {o['rf']:.4f}")
    print(f"\n  VSA reinjections: {v['reinj']:.0f} (mean)")
    print(f"  Timer reinjections: {t['reinj']:.0f} (fixed)")
