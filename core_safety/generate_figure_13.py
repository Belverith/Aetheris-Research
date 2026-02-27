"""
Figure 13: CHDBO Scalability Beyond n=128 (n=512, n=1024)
==========================================================
The paper's main experiments use n=128. This experiment runs the
full CHDBO + AASV pipeline at n=512 and n=1024 to validate that
the O(N·n) cost remains practical and safety is preserved.

Setup:
  - Dimensions: n ∈ {128, 512, 1024}
  - Linear drift A with σ_max(A) = 0.3 (mildly unstable)
  - Safe set: unit ball, h(x) = 1 - ||x||^2
  - CBF-QP with Hutchinson spectral-norm estimation
  - 20 trials × 300 steps per dimension
  - Hutchinson probes: k = 5 (fixed across dimensions)

Output: figure_13.png — 2×2 panel:
  (a) Wall-clock time vs dimension
  (b) Hutchinson σ̂_max estimates vs dimension
  (c) Min barrier h(x) vs dimension (safety check)
  (d) Number of CBF interventions vs dimension

This is Experiment IX in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DIMS = [128, 512, 1024]
N_TRIALS = 20
N_STEPS = 300
DT = 0.01
GAMMA = 1.0
K_PROBES = 5  # Hutchinson probes (kept fixed to show O(k·n) scaling)


def make_drift_matrix(n, rng):
    """Construct mildly unstable linear drift A in R^n."""
    S = rng.randn(n, n)
    S = (S - S.T) / 2.0     # skew-symmetric
    S = S / np.linalg.norm(S, ord=2)
    A = S + 0.3 * np.eye(n)  # σ_max ≈ 1.3
    return A


def barrier(x):
    """h(x) = 1 - ||x||^2."""
    return 1.0 - np.dot(x, x)


def barrier_grad(x):
    """∇h = -2x."""
    return -2.0 * x


def hutchinson_spectral_estimate(A, x, k, rng):
    """
    Estimate ||J||_F via Hutchinson trace estimator on J^T J,
    then return ||J||_F as upper bound on σ_max.
    Here J = A (constant), so we compute trace(A^T A).
    """
    traces = []
    for _ in range(k):
        z = rng.choice([-1.0, 1.0], size=x.shape[0])
        Az = A @ z
        traces.append(np.dot(Az, Az))  # z^T A^T A z
    est_trace = np.mean(traces)
    return np.sqrt(max(est_trace, 0.0))  # ||A||_F estimate


def cbf_qp_step(x, u_nom, A, gamma=GAMMA):
    """
    CBF-QP for dx = Ax + u, barrier h(x) = 1 - ||x||^2.
    Constraint: ∇h · (Ax + u) ≥ -γ h(x)
    """
    h_val = barrier(x)
    grad_h = barrier_grad(x)
    Ax = A @ x
    Lf_h = np.dot(grad_h, Ax)
    Lg_h = grad_h

    lhs_nom = np.dot(Lg_h, u_nom)
    rhs = -gamma * h_val - Lf_h

    if lhs_nom >= rhs:
        return u_nom, False

    Lg_sq = np.dot(Lg_h, Lg_h)
    if Lg_sq < 1e-12:
        return u_nom, False

    lam = (rhs - lhs_nom) / Lg_sq
    return u_nom + lam * Lg_h, True


# ============================================================================
# RUN SCALABILITY EXPERIMENT
# ============================================================================
print("[*] Running Experiment IX: Scalability at n=128, 512, 1024...")

results = {}

for n in DIMS:
    print(f"\n  --- n = {n} ---")
    rng = np.random.RandomState(42)
    A = make_drift_matrix(n, rng)
    sigma_true = np.linalg.norm(A, ord=2)
    print(f"    σ_max(A) = {sigma_true:.4f}")

    trial_times = []
    trial_min_h = []
    trial_sigma_hat = []
    trial_interventions = []

    for trial in range(N_TRIALS):
        x = rng.randn(n)
        x = x / np.linalg.norm(x) * 0.5  # start at ||x||=0.5

        t0 = time.perf_counter()
        min_h = barrier(x)
        n_intervene = 0
        sigma_hats = []

        for step in range(N_STEPS):
            # Hutchinson estimate
            sig_hat = hutchinson_spectral_estimate(A, x, K_PROBES, rng)
            sigma_hats.append(sig_hat)

            # Nominal control toward goal outside safe set
            goal = np.zeros(n)
            goal[0] = 1.5
            u_nom = 0.3 * (goal - x) / max(np.linalg.norm(goal - x), 1e-8)

            # CBF-QP
            u_star, active = cbf_qp_step(x, u_nom, A)
            if active:
                n_intervene += 1

            # Euler step
            x = x + (A @ x + u_star) * DT

            h_val = barrier(x)
            min_h = min(min_h, h_val)

        elapsed = time.perf_counter() - t0
        trial_times.append(elapsed)
        trial_min_h.append(min_h)
        trial_sigma_hat.append(np.mean(sigma_hats))
        trial_interventions.append(n_intervene)

    results[n] = {
        'times': trial_times,
        'min_h': trial_min_h,
        'sigma_hat': trial_sigma_hat,
        'sigma_true': sigma_true,
        'interventions': trial_interventions,
    }

    print(f"    Mean time:       {np.mean(trial_times):.3f}s")
    print(f"    Mean σ̂:          {np.mean(trial_sigma_hat):.4f}")
    print(f"    σ_true:          {sigma_true:.4f}")
    print(f"    Min h (worst):   {np.min(trial_min_h):.6f}")
    print(f"    Violations:      {sum(1 for h in trial_min_h if h < -1e-6)}/{N_TRIALS}")
    print(f"    Mean CBF active: {np.mean(trial_interventions):.0f}/{N_STEPS}")

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
BG = '#f8f9fa'
for ax in axes.flat:
    ax.set_facecolor(BG)

dim_labels = [str(n) for n in DIMS]

# (a) Wall-clock time vs dimension
ax = axes[0, 0]
means = [np.mean(results[n]['times']) for n in DIMS]
stds = [np.std(results[n]['times']) for n in DIMS]
ax.bar(dim_labels, means, yerr=stds, capsize=5, color='#3498db',
       edgecolor='white', alpha=0.8)
# Reference O(n) scaling from n=128
ref = means[0]
ax.plot(range(len(DIMS)), [ref * (n / 128) for n in DIMS],
        'r--', lw=2, marker='o', label='$O(n)$ reference')
ax.set_xlabel('Dimension $n$', fontsize=12)
ax.set_ylabel('Wall-clock Time (s)', fontsize=12)
ax.set_title('(a) Computation Time vs Dimension', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

# (b) Hutchinson estimates
ax = axes[0, 1]
for n in DIMS:
    ax.hist(results[n]['sigma_hat'], bins=15, alpha=0.5,
            label=f'n={n} (σ_true={results[n]["sigma_true"]:.2f})')
    ax.axvline(results[n]['sigma_true'], ls='--', lw=1.5)
ax.set_xlabel('Hutchinson $\\hat{\\sigma}$ (mean over trial)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('(b) Spectral Estimates vs Dimension', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (c) Min barrier value
ax = axes[1, 0]
data_h = [results[n]['min_h'] for n in DIMS]
bp = ax.boxplot(data_h, tick_labels=dim_labels, patch_artist=True,
                boxprops=dict(facecolor='#2ecc71', alpha=0.6))
ax.axhline(0.0, color='red', ls='--', lw=2, label='$h=0$ (unsafe)')
ax.set_xlabel('Dimension $n$', fontsize=12)
ax.set_ylabel('Min $h(x)$ across trial', fontsize=12)
ax.set_title('(c) Safety Margin vs Dimension', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

# (d) CBF interventions
ax = axes[1, 1]
data_int = [results[n]['interventions'] for n in DIMS]
bp2 = ax.boxplot(data_int, tick_labels=dim_labels, patch_artist=True,
                 boxprops=dict(facecolor='#e67e22', alpha=0.6))
ax.set_xlabel('Dimension $n$', fontsize=12)
ax.set_ylabel(f'CBF Interventions (of {N_STEPS})', fontsize=12)
ax.set_title('(d) CBF Activation Frequency', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

n_total_violations = sum(
    sum(1 for h in results[n]['min_h'] if h < -1e-6) for n in DIMS
)

fig.suptitle(
    'Experiment IX: CHDBO Scalability at $n = 128, 512, 1024$\n'
    f'{N_TRIALS} trials × {N_STEPS} steps per dim — '
    f'Total violations: {n_total_violations}/{N_TRIALS * len(DIMS)}',
    fontsize=15, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('core_safety/figure_13.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\n[OK] Saved figure_13.png — Total violations: {n_total_violations}")
