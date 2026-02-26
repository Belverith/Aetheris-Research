"""
Figure 10: Black-Box vs. WTA AASV Hunter
==========================================
Demonstrates why the Winner-Take-All (WTA) gradient decomposition is
essential for multi-modal barrier landscapes.

Panel (a): Single failure mode — finite-difference (FD) gradient
  discovers the violation without oracle knowledge.  No centroid saddle
  exists, so ∇h naturally guides the Hunter to the lone spike.

Panel (b): Multiple failure modes — the full ∇h (sum over all spikes)
  converges to the centroid saddle, not to any spike.  The WTA gradient
  (targeting one spike per restart) resolves all 3.

Key finding: automatic differentiation (oracle or AD) plus WTA
decomposition is the minimal requirement for multi-modal discovery.
For single-mode barriers typical of operational CBFs, FD suffices.

Output: figure_10.png
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DIMENSIONS      = 128
SPIKE_WIDTH     = 0.05
FUNNEL_STRENGTH = 1.0
FD_EPSILON      = 1e-4
MOMENTUM        = 0.9
LR              = 0.05
NOISE_SCALE     = 0.001
SURROGATE_ERROR = 0.005
HUNTER_STEPS    = 200
HUNTER_RESTARTS = 20

def spike_depth(n):
    return (n - 1) * FUNNEL_STRENGTH + 2.0

# ============================================================================
# SPIKE DIRECTIONS
# ============================================================================
rng = np.random.RandomState(42)
M = rng.randn(DIMENSIONS, 3)
Q, _ = np.linalg.qr(M)
ALL3_DIRS = [Q[:, i] for i in range(3)]   # 3 orthogonal spikes
SINGLE_DIR = [ALL3_DIRS[0]]               # just the first spike

# ============================================================================
# BARRIER + GRADIENTS
# ============================================================================
def barrier_function(x, spike_dirs):
    depth = spike_depth(len(spike_dirs))
    dist = np.linalg.norm(x)
    h = 1.0 - dist
    if dist > 0:
        for d in spike_dirs:
            sim = np.dot(x, d) / dist
            h += FUNNEL_STRENGTH * (1.0 - sim)
            h -= depth * np.exp(-(1.0 - sim) ** 2 / (2.0 * SPIKE_WIDTH ** 2))
    return h

def sum_gradient(x, spike_dirs):
    """True analytical ∇h — sum over ALL spikes."""
    depth = spike_depth(len(spike_dirs))
    dist = np.linalg.norm(x)
    if dist < 1e-12:
        return np.zeros_like(x)
    x_hat = x / dist
    grad = -x_hat.copy()
    for d in spike_dirs:
        sim = np.dot(x, d) / dist
        diff = 1.0 - sim
        spike_exp = np.exp(-diff ** 2 / (2.0 * SPIKE_WIDTH ** 2))
        sgc = depth * spike_exp * diff / (SPIKE_WIDTH ** 2)
        coeff = (-FUNNEL_STRENGTH - sgc) / dist
        tangent = d - sim * x_hat
        grad += coeff * tangent
    return grad

def wta_gradient(x, spike_dirs, memory_protos, rng_local=None,
                 block_thresh=0.7):
    """Winner-Take-All gradient — targets the NEAREST unblocked spike."""
    depth = spike_depth(len(spike_dirs))
    dist = np.linalg.norm(x)
    if dist < 1e-12:
        return np.zeros_like(x)
    x_hat = x / dist
    grad = -x_hat.copy()

    sims = [np.dot(x, d) / dist for d in spike_dirs]
    adj = list(sims)
    for i, d in enumerate(spike_dirs):
        for proto in memory_protos:
            if np.dot(d, proto) > block_thresh:
                adj[i] = -999.0
                break

    if max(adj) < -900 and len(memory_protos) > 0:
        if rng_local is None:
            rng_local = np.random.RandomState()
        rand_dir = rng_local.randn(len(x))
        for p in memory_protos:
            rand_dir -= np.dot(rand_dir, p) * p
        rand_dir -= np.dot(rand_dir, x_hat) * x_hat
        rn = np.linalg.norm(rand_dir)
        if rn > 1e-12:
            rand_dir /= rn
        grad += (-FUNNEL_STRENGTH / dist) * rand_dir
        return grad

    best = int(np.argmax(adj))
    d = spike_dirs[best]
    sim = sims[best]
    diff = 1.0 - sim
    spike_exp = np.exp(-diff ** 2 / (2.0 * SPIKE_WIDTH ** 2))
    sgc = depth * spike_exp * diff / (SPIKE_WIDTH ** 2)
    coeff = (-FUNNEL_STRENGTH - sgc) / dist
    tangent = d - sim * x_hat
    grad += coeff * tangent
    return grad

def fd_gradient(x, spike_dirs):
    """Finite-difference ∇h — no knowledge of spike locations."""
    h0 = barrier_function(x, spike_dirs)
    grad = np.zeros_like(x)
    for i in range(DIMENSIONS):
        e = np.zeros(DIMENSIONS)
        e[i] = FD_EPSILON
        grad[i] = (barrier_function(x + e, spike_dirs) - h0) / FD_EPSILON
    return grad

# ============================================================================
# ORTHOGONAL PROTOTYPE MEMORY
# ============================================================================
class OrthoMemory:
    def __init__(self, merge_thresh=0.3):
        self.prototypes = []
        self.merge_thresh = merge_thresh

    def add(self, v):
        v_n = v / np.linalg.norm(v)
        for i, p in enumerate(self.prototypes):
            if np.dot(v_n, p) > self.merge_thresh:
                self.prototypes[i] = p + 0.1 * v_n
                self.prototypes[i] /= np.linalg.norm(self.prototypes[i])
                return False
        self.prototypes.append(v_n.copy())
        return True

# ============================================================================
# HUNTER ENGINES
# ============================================================================
def run_wta_hunter(spike_dirs, n_restarts, seed_base=1000, label=""):
    """Hunter using WTA gradient (oracle knowledge of spike_dirs)."""
    memory = OrthoMemory()
    all_viol = []

    def refine(x, vel, rl):
        for _ in range(100):
            g = wta_gradient(x, spike_dirs, [], rng_local=rl,
                             block_thresh=999.0)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            vel = 0.3 * vel - 0.01 * g
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)
        return x

    hits = 0
    for restart in range(n_restarts):
        trial_rng = np.random.RandomState(seed_base + restart)
        x = trial_rng.randn(DIMENSIONS)
        x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)

        for _ in range(HUNTER_STEPS):
            g = wta_gradient(x, spike_dirs, memory.prototypes,
                             rng_local=trial_rng, block_thresh=0.7)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            g += trial_rng.randn(DIMENSIONS) * SURROGATE_ERROR
            for p in memory.prototypes:
                s = np.dot(x, p) / (np.linalg.norm(x) + 1e-8)
                g += 0.15 * s * p
            vel = MOMENTUM * vel - LR * g + trial_rng.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)

            if barrier_function(x, spike_dirs) < -1e-10:
                x = refine(x, vel, trial_rng)
                xn = x / np.linalg.norm(x)
                all_viol.append(xn)
                memory.add(x)
                hits += 1
                break

    # Phase 2: unblocked restarts
    for r in range(max(10, len(memory.prototypes) * 4)):
        rl = np.random.RandomState(seed_base + 20000 + r)
        x = rl.randn(DIMENSIONS)
        x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)
        for _ in range(HUNTER_STEPS):
            g = wta_gradient(x, spike_dirs, [], rng_local=rl,
                             block_thresh=999.0)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            g += rl.randn(DIMENSIONS) * SURROGATE_ERROR
            vel = MOMENTUM * vel - LR * g + rl.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)
            if barrier_function(x, spike_dirs) < -1e-10:
                x = refine(x, vel, rl)
                xn = x / np.linalg.norm(x)
                all_viol.append(xn)
                hits += 1
                break

    matched = _cluster_and_match(all_viol, spike_dirs)
    print(f"  [{label}] hits={hits}, matched={len(matched)}/{len(spike_dirs)}")
    return len(matched), hits


def run_sum_hunter(spike_dirs, n_restarts, use_fd=False,
                   seed_base=1000, label=""):
    """Hunter using full ∇h (sum over all spikes), oracle or FD."""
    memory = OrthoMemory()
    all_viol = []

    def grad_fn(x):
        if use_fd:
            return fd_gradient(x, spike_dirs)
        return sum_gradient(x, spike_dirs)

    def refine(x, vel):
        for _ in range(100):
            g = grad_fn(x)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            vel = 0.3 * vel - 0.01 * g
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)
        return x

    hits = 0
    for restart in range(n_restarts):
        trial_rng = np.random.RandomState(seed_base + restart)
        x = trial_rng.randn(DIMENSIONS)
        x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)

        for _ in range(HUNTER_STEPS):
            g = grad_fn(x)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            for p in memory.prototypes:
                s = np.dot(x, p) / (np.linalg.norm(x) + 1e-8)
                g += 0.15 * s * p
            vel = MOMENTUM * vel - LR * g + trial_rng.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)

            if barrier_function(x, spike_dirs) < -1e-10:
                x = refine(x, vel)
                xn = x / np.linalg.norm(x)
                all_viol.append(xn)
                memory.add(x)
                hits += 1
                break

    matched = _cluster_and_match(all_viol, spike_dirs)
    print(f"  [{label}] hits={hits}, matched={len(matched)}/{len(spike_dirs)}")
    return len(matched), hits


def _cluster_and_match(viol_dirs, spike_dirs):
    clusters = []
    for d in viol_dirs:
        merged = False
        for c in clusters:
            if np.dot(d, c['center']) > 0.95:
                c['center'] += 0.1 * d
                c['center'] /= np.linalg.norm(c['center'])
                c['count'] += 1
                merged = True
                break
        if not merged:
            clusters.append({'center': d.copy(), 'count': 1})
    matched = set()
    for c in clusters:
        for i, sd in enumerate(spike_dirs):
            if np.dot(c['center'], sd) > 0.85:
                matched.add(i)
    return matched

# ============================================================================
# EXPERIMENTS
# ============================================================================
print("=" * 60)
print("EXPERIMENT VI: WTA vs. Sum-Gradient Hunter")
print("=" * 60)

restart_counts = [5, 10, 15, 20, 30, 40]

# ----- (a) Single spike: FD gradient works ---
print("\n--- Single Spike (no centroid saddle) ---")
fd_single = []
for k in restart_counts:
    m, _ = run_sum_hunter(SINGLE_DIR, k, use_fd=True, seed_base=3000,
                          label=f"FD 1-spike k={k}")
    fd_single.append(m)

oracle_single = []
for k in restart_counts:
    m, _ = run_sum_hunter(SINGLE_DIR, k, use_fd=False, seed_base=3500,
                          label=f"Oracle 1-spike k={k}")
    oracle_single.append(m)

# ----- (b) 3 spikes: WTA vs sum-gradient ---
print("\n--- 3 Spikes: WTA vs Sum gradient ---")
wta_multi = []
for k in restart_counts:
    m, _ = run_wta_hunter(ALL3_DIRS, k, seed_base=5000,
                          label=f"WTA 3-spike k={k}")
    wta_multi.append(m)

sum_oracle_multi = []
for k in restart_counts:
    m, _ = run_sum_hunter(ALL3_DIRS, k, use_fd=False, seed_base=6000,
                          label=f"Sum-∇ 3-spike k={k}")
    sum_oracle_multi.append(m)

sum_fd_multi = []
for k in restart_counts:
    m, _ = run_sum_hunter(ALL3_DIRS, k, use_fd=True, seed_base=7000,
                          label=f"FD 3-spike k={k}")
    sum_fd_multi.append(m)

# ============================================================================
# PLOTTING
# ============================================================================
print("\nGenerating figure_10.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
BG = '#f8f9fa'

# ---- Panel (a): Single failure mode  ----
ax = axes[0]
ax.set_facecolor(BG)
ax.plot(restart_counts, fd_single, 's-', color='#8e44ad', lw=2.5, ms=9,
        label='Finite-difference gradient')
ax.plot(restart_counts, oracle_single, 'o--', color='#2ecc71', lw=2, ms=8,
        label='Analytical $\\nabla h$')
ax.axhline(1, color='#e74c3c', ls=':', lw=2, label='Ground truth (1 spike)')
ax.set_xlabel('Hunter Restarts ($k$)', fontsize=12)
ax.set_ylabel('Spikes Detected', fontsize=12)
ax.set_title('(a) Single Failure Mode\n'
             'FD gradient suffices — no centroid saddle',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.set_ylim(-0.3, 2.5)
ax.set_yticks([0, 1, 2])
ax.grid(True, alpha=0.2)
ax.text(0.03, 0.03,
        'With a single failure mode,\n'
        '$\\nabla h$ directly guides the\n'
        'Hunter to the violation.\n'
        'Black-box FD gradients suffice.',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='#e8f8e8', alpha=0.85))

# ---- Panel (b): Multi-modal landscape ----
ax = axes[1]
ax.set_facecolor(BG)
ax.plot(restart_counts, wta_multi, 'D-', color='#e67e22', lw=2.5, ms=9,
        label='WTA gradient (oracle)', zorder=5)
ax.plot(restart_counts, sum_oracle_multi, 'o--', color='#2ecc71', lw=2, ms=8,
        label='Sum $\\nabla h$ (oracle)')
ax.plot(restart_counts, sum_fd_multi, 's--', color='#8e44ad', lw=2, ms=8,
        label='Sum $\\nabla h$ (FD)')
ax.axhline(3, color='#e74c3c', ls=':', lw=2, label='Ground truth (3 spikes)')
ax.set_xlabel('Hunter Restarts ($k$)', fontsize=12)
ax.set_ylabel('Distinct Spikes Detected', fontsize=12)
ax.set_title('(b) Multi-Modal Failure Landscape (3 Spikes)\n'
             'WTA decomposition needed to escape centroid saddle',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9, loc='center right')
ax.set_ylim(-0.3, 4.5)
ax.set_yticks([0, 1, 2, 3, 4])
ax.grid(True, alpha=0.2)
ax.text(0.03, 0.03,
        'Sum $\\nabla h$ converges to the\n'
        'centroid of spike directions\n'
        '(a saddle, not a violation).\n'
        'WTA targets one spike per restart.',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fff3e0', alpha=0.85))

fig.suptitle('Experiment VI: Gradient Structure & Multi-Modal Discovery '
             '($\\mathbb{R}^{128}$, $\\theta_w = 0.05$ rad)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('core_safety/figure_10.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] Saved figure_10.png")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'k':<6} {'FD(1sp)':>8} {'∇h(1sp)':>8} "
      f"{'WTA(3sp)':>9} {'∇h(3sp)':>9} {'FD(3sp)':>8}")
print("-" * 52)
for i, k in enumerate(restart_counts):
    print(f"{k:<6} {fd_single[i]:>6}/1 {oracle_single[i]:>6}/1 "
          f"{wta_multi[i]:>7}/3 {sum_oracle_multi[i]:>7}/3 "
          f"{sum_fd_multi[i]:>6}/3")
