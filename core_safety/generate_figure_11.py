"""
Figure 11: Seed Sensitivity Sweep — 20 Spikes x 10 Seeds
==========================================================
Runs the 20-spike AASV Hunter (Experiment IV panel h) across 10
independent random seeds. Reports mean ± std detection counts,
replacing the single-seed result with a statistical claim.

Output: figure_11.png
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================================
# CONFIGURATION (matches Experiment IV, panel h)
# ============================================================================
DIMENSIONS = 128
N_SPIKES = 20
SPIKE_WIDTH = 0.05
FUNNEL_STRENGTH = 1.0
HUNTER_RESTARTS = 60
HUNTER_STEPS = 200
MOMENTUM = 0.9
LR = 0.05
NOISE_SCALE = 0.001
SURROGATE_ERROR = 0.005
N_SEEDS = 10

def spike_depth(n_spikes):
    return (n_spikes - 1) * FUNNEL_STRENGTH + 2.0

# ============================================================================
# BARRIER + WTA GRADIENT (from generate_figure_4.py)
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

def barrier_gradient_wta(x, spike_dirs, memory_protos, rng=None,
                         block_thresh=0.7):
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
        if rng is None:
            rng = np.random.RandomState()
        rand_dir = rng.randn(len(x))
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

# ============================================================================
# ORTHOGONAL PROTOTYPE MEMORY
# ============================================================================
class OrthoMemory:
    def __init__(self):
        self.prototypes = []

    def add(self, v, merge_thresh=0.3):
        v_n = v / np.linalg.norm(v)
        for i, p in enumerate(self.prototypes):
            if np.dot(v_n, p) > merge_thresh:
                self.prototypes[i] = p + 0.1 * v_n
                self.prototypes[i] /= np.linalg.norm(self.prototypes[i])
                return False
        self.prototypes.append(v_n.copy())
        return True

# ============================================================================
# FULL AASV PIPELINE (one seed)
# ============================================================================
def run_one_seed(seed):
    """Run full 20-spike AASV experiment with given seed."""
    rng_spike = np.random.RandomState(seed * 1000)
    spike_dirs = []
    for _ in range(N_SPIKES):
        v = rng_spike.randn(DIMENSIONS)
        v /= np.linalg.norm(v)
        spike_dirs.append(v)

    memory = OrthoMemory()
    all_viol_dirs = []

    def refine(x, vel, rng_local):
        for _ in range(100):
            g = barrier_gradient_wta(x, spike_dirs, [],
                                     rng=rng_local, block_thresh=999.0)
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

    # Phase 1: Standard WTA with blocking
    for restart in range(HUNTER_RESTARTS):
        trial_rng = np.random.RandomState(seed * 100 + restart)
        x = trial_rng.randn(DIMENSIONS)
        x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)

        for _ in range(HUNTER_STEPS):
            g = barrier_gradient_wta(x, spike_dirs, memory.prototypes,
                                     rng=trial_rng, block_thresh=0.7)
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
                all_viol_dirs.append(xn)
                memory.add(x)
                break

    # Phase 2: Unblocked restarts
    refine_k = max(10, len(memory.prototypes) * 4)
    for r in range(refine_k):
        trial_rng = np.random.RandomState(seed * 100 + HUNTER_RESTARTS + r)
        x = trial_rng.randn(DIMENSIONS)
        x /= np.linalg.norm(x)
        vel = np.zeros(DIMENSIONS)

        for _ in range(HUNTER_STEPS):
            g = barrier_gradient_wta(x, spike_dirs, [],
                                     rng=trial_rng, block_thresh=999.0)
            xh = x / np.linalg.norm(x)
            g -= np.dot(g, xh) * xh
            gn = np.linalg.norm(g)
            if gn > 2.0:
                g *= 2.0 / gn
            g += trial_rng.randn(DIMENSIONS) * SURROGATE_ERROR
            vel = MOMENTUM * vel - LR * g + trial_rng.randn(DIMENSIONS) * NOISE_SCALE
            vel -= np.dot(vel, xh) * xh
            x = x + vel
            x /= np.linalg.norm(x)

            if barrier_function(x, spike_dirs) < -1e-10:
                x = refine(x, vel, trial_rng)
                xn = x / np.linalg.norm(x)
                all_viol_dirs.append(xn)
                break

    # Post-hoc clustering
    CLUSTER_THRESH = 0.98
    clusters = []
    for d in all_viol_dirs:
        merged = False
        for c in clusters:
            if np.dot(d, c['center']) > CLUSTER_THRESH:
                c['center'] += 0.1 * d
                c['center'] /= np.linalg.norm(c['center'])
                c['count'] += 1
                merged = True
                break
        if not merged:
            clusters.append({'center': d.copy(), 'count': 1})

    # Match clusters to true spikes (cos > 0.9 = match)
    matched = set()
    for c in clusters:
        for i, sd in enumerate(spike_dirs):
            if np.dot(c['center'], sd) > 0.9:
                matched.add(i)

    return {
        'n_violations': len(all_viol_dirs),
        'n_clusters': len(clusters),
        'n_matched': len(matched),
        'spike_dirs': spike_dirs,
    }

# ============================================================================
# RUN SWEEP
# ============================================================================
print("=" * 60)
print("SEED SENSITIVITY SWEEP: 20 Spikes × 10 Seeds")
print("=" * 60)

results = []
for seed in range(N_SEEDS):
    t0 = time.perf_counter()
    res = run_one_seed(seed)
    elapsed = time.perf_counter() - t0
    results.append(res)
    print(f"  Seed {seed}: {res['n_matched']}/{N_SPIKES} spikes matched, "
          f"{res['n_clusters']} clusters, "
          f"{res['n_violations']} violations, "
          f"{elapsed:.1f}s")

matched_counts = [r['n_matched'] for r in results]
cluster_counts = [r['n_clusters'] for r in results]
violation_counts = [r['n_violations'] for r in results]

mean_matched = np.mean(matched_counts)
std_matched = np.std(matched_counts)
min_matched = np.min(matched_counts)
max_matched = np.max(matched_counts)

print(f"\n--- Summary across {N_SEEDS} seeds ---")
print(f"  Spikes matched: {mean_matched:.1f} ± {std_matched:.1f} "
      f"(min={min_matched}, max={max_matched})")
print(f"  Detection rate: {mean_matched/N_SPIKES*100:.1f}% ± "
      f"{std_matched/N_SPIKES*100:.1f}%")
print(f"  Mean clusters:  {np.mean(cluster_counts):.1f}")
print(f"  Mean violations: {np.mean(violation_counts):.1f}")

# ============================================================================
# PLOTTING
# ============================================================================
print("\nGenerating figure_11.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
BG = '#f8f9fa'

# --- Panel (a): Box plot of detection counts ---
ax = axes[0]
ax.set_facecolor(BG)

bp = ax.boxplot([matched_counts, cluster_counts],
                labels=['Spikes Matched\n(true positive)', 'Total Clusters\nDetected'],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor='#3498db', alpha=0.6),
                medianprops=dict(color='#2c3e50', lw=2),
                whiskerprops=dict(color='#2c3e50'),
                capprops=dict(color='#2c3e50'))

# Individual data points
for i, data in enumerate([matched_counts, cluster_counts]):
    x_jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(data)) + i + 1
    ax.scatter(x_jitter, data, s=40, c='#e74c3c', zorder=5, alpha=0.7,
               edgecolors='darkred', lw=0.5)

ax.axhline(N_SPIKES, color='#2ecc71', ls='--', lw=2,
           label=f'Ground truth: {N_SPIKES} spikes')
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'(a) Detection Across {N_SEEDS} Random Seeds\n'
             f'{N_SPIKES} spikes, $k={HUNTER_RESTARTS}$ restarts per seed',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')

# Stats annotation
ax.text(0.97, 0.97,
        f'Spikes matched:\n'
        f'  Mean: {mean_matched:.1f} / {N_SPIKES}\n'
        f'  Std:  {std_matched:.1f}\n'
        f'  Min:  {min_matched}\n'
        f'  Max:  {max_matched}\n\n'
        f'Detection rate:\n'
        f'  {mean_matched/N_SPIKES*100:.0f}% ± {std_matched/N_SPIKES*100:.0f}%',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9))

# --- Panel (b): Per-seed bar chart ---
ax = axes[1]
ax.set_facecolor(BG)

seeds = np.arange(N_SEEDS)
bars = ax.bar(seeds, matched_counts, color='#8e44ad', alpha=0.7,
              edgecolor='white', lw=1.5)
ax.axhline(N_SPIKES, color='#2ecc71', ls='--', lw=2,
           label=f'All {N_SPIKES} spikes')
ax.axhline(mean_matched, color='#e67e22', ls='-', lw=2,
           label=f'Mean = {mean_matched:.1f}')
ax.fill_between([-0.5, N_SEEDS - 0.5],
                mean_matched - std_matched, mean_matched + std_matched,
                alpha=0.15, color='#e67e22', label=f'±1σ = {std_matched:.1f}')

ax.set_xlabel('Random Seed', fontsize=12)
ax.set_ylabel('Spikes Matched', fontsize=12)
ax.set_title(f'(b) Per-Seed Detection ({N_SPIKES} spikes, '
             f'$k={HUNTER_RESTARTS}$)',
             fontsize=13, fontweight='bold')
ax.set_xticks(seeds)
ax.set_xticklabels([str(s) for s in seeds])
ax.set_ylim(0, N_SPIKES + 2)
ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.2, axis='y')

# Annotate each bar
for bar, val in zip(bars, matched_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', fontsize=10, fontweight='bold')

fig.suptitle('Experiment VII: Seed Sensitivity — 20 Spikes on $S^{127}$ '
             f'({N_SEEDS} Seeds, $k={HUNTER_RESTARTS}$ Restarts)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('core_safety/figure_11.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] Saved figure_11.png")
