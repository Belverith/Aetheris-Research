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

Output: figure_13a–d.png (individual panels) + figure_13.png (combined legacy)
  (a) Wall-clock time vs dimension
  (b) Hutchinson σ̂_max estimates vs dimension
  (c) Min barrier h(x) vs dimension (safety check)
  (d) Number of CBF interventions vs dimension

This is Experiment VI in the paper.

Usage:
  python generate_figure_13.py              # all panels + combined
  python generate_figure_13.py -a           # panel (a) only
  python generate_figure_13.py -b -d        # panels (b) and (d) only
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description='Generate Experiment VI scalability figures.')
parser.add_argument('-a', action='store_true', help='Generate panel (a): Wall-clock time')
parser.add_argument('-b', action='store_true', help='Generate panel (b): Spectral estimates')
parser.add_argument('-c', action='store_true', help='Generate panel (c): Safety margin')
parser.add_argument('-d', action='store_true', help='Generate panel (d): CBF activation')
args = parser.parse_args()
_any = args.a or args.b or args.c or args.d
_gen_a = args.a or not _any
_gen_b = args.b or not _any
_gen_c = args.c or not _any
_gen_d = args.d or not _any

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
    Estimate ||A||_F via Hutchinson trace estimator on A^T A,
    then return sqrt(trace(A^T A)) = ||A||_F.
    This is an upper bound on σ_max(A) since ||A||_F >= σ_max(A).
    """
    traces = []
    for _ in range(k):
        z = rng.choice([-1.0, 1.0], size=x.shape[0])
        Az = A @ z
        traces.append(np.dot(Az, Az))  # z^T A^T A z
    est_trace = np.mean(traces)
    return np.sqrt(max(est_trace, 0.0))  # ||A||_F estimate


def cbf_qp_step(x, u_nom, A, gamma=GAMMA, spectral_margin=0.0):
    """
    CBF-QP for dx = Ax + u, barrier h(x) = 1 - ||x||^2.
    Robust constraint: ∇h · (Ax + u) ≥ -γ (h(x) - spectral_margin)
    The spectral_margin term tightens the barrier to account for
    Hutchinson-estimated drift volatility (Equation 8 in paper).
    """
    h_val = barrier(x)
    grad_h = barrier_grad(x)
    Ax = A @ x
    Lf_h = np.dot(grad_h, Ax)
    Lg_h = grad_h

    lhs_nom = np.dot(Lg_h, u_nom)
    # Tighten RHS by the spectral safety tube ρ(x) = σ̂_max · d_step
    rhs = -gamma * (h_val - spectral_margin) - Lf_h

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
print("[*] Running Experiment VI: Scalability at n=128, 512, 1024...")

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

            # Compute adaptive spectral margin: ρ(x) = σ̂_max · d_step
            spectral_margin = sig_hat * DT

            # Nominal control toward goal outside safe set
            goal = np.zeros(n)
            goal[0] = 1.5
            u_nom = 0.3 * (goal - x) / max(np.linalg.norm(goal - x), 1e-8)

            # CBF-QP with Hutchinson-inflated tube margin
            u_star, active = cbf_qp_step(x, u_nom, A, spectral_margin=spectral_margin)
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
# PLOTTING — HELPERS
# ============================================================================
BG = '#f8f9fa'
dim_labels = [str(n) for n in DIMS]

n_total_violations = sum(
    sum(1 for h in results[n]['min_h'] if h < -1e-6) for n in DIMS
)


def _setup_ax(ax):
    ax.set_facecolor(BG)
    return ax


def _plot_a(ax):
    """(a) Wall-clock time vs dimension."""
    means = [np.mean(results[n]['times']) for n in DIMS]
    stds = [np.std(results[n]['times']) for n in DIMS]
    ax.bar(dim_labels, means, yerr=stds, capsize=5, color='#3498db',
           edgecolor='white', alpha=0.8)
    ref = means[0]
    ax.plot(range(len(DIMS)), [ref * (n / 128) for n in DIMS],
            'r--', lw=2, marker='o', label='$O(n)$ reference')
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel('Wall-clock Time (s)', fontsize=12)
    ax.set_title('(a) Computation Time vs Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')


def _plot_b(ax):
    """(b) Hutchinson spectral estimates."""
    data_sigma = [results[n]['sigma_hat'] for n in DIMS]
    colors_sigma = ['#3498db', '#e67e22', '#2ecc71']
    bp_sigma = ax.boxplot(data_sigma, tick_labels=dim_labels, patch_artist=True,
                          boxprops=dict(alpha=0.6))
    for patch, color in zip(bp_sigma['boxes'], colors_sigma):
        patch.set_facecolor(color)
    for i, n in enumerate(DIMS):
        st = results[n]['sigma_true']
        ax.hlines(st, i + 0.6, i + 1.4, colors='red', linestyles='--', lw=2,
                  label='$\\sigma_{\\max}(A)$' if i == 0 else '_nolegend_')
        ax.text(i + 1.0, st + 0.3, f'$\\sigma_{{\\max}}$={st:.2f}',
                fontsize=10, color='red', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='red', alpha=0.8))
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel(r'Hutchinson $\|A\|_F$ ($\geq \sigma_{\max}$)', fontsize=12)
    ax.set_title('(b) Spectral Estimates vs Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)


def _plot_c(ax):
    """(c) Min barrier value (safety margin)."""
    data_h = [results[n]['min_h'] for n in DIMS]
    ax.boxplot(data_h, tick_labels=dim_labels, patch_artist=True,
               boxprops=dict(facecolor='#2ecc71', alpha=0.6))
    ax.axhline(0.0, color='red', ls='--', lw=2, label='$h=0$ (unsafe)')
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel('Min $h(x)$ across trial', fontsize=12)
    ax.set_title('(c) Safety Margin vs Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')


def _plot_d(ax):
    """(d) CBF activation frequency."""
    data_int = [results[n]['interventions'] for n in DIMS]
    ax.boxplot(data_int, tick_labels=dim_labels, patch_artist=True,
               boxprops=dict(facecolor='#e67e22', alpha=0.6))
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel(f'CBF Interventions (of {N_STEPS})', fontsize=12)
    ax.set_title('(d) CBF Activation Frequency', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')


# ============================================================================
# INDIVIDUAL PANEL FIGURES (figure_13a–d.png)
# ============================================================================
panel_map = {'a': _plot_a, 'b': _plot_b, 'c': _plot_c, 'd': _plot_d}
panel_flags = {'a': _gen_a, 'b': _gen_b, 'c': _gen_c, 'd': _gen_d}

for key in 'abcd':
    if not panel_flags[key]:
        continue
    fig_i, ax_i = plt.subplots(figsize=(7, 5))
    _setup_ax(ax_i)
    panel_map[key](ax_i)
    fig_i.tight_layout()
    fname = f'figure_13{key}.png'
    fig_i.savefig(fname, dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_i)
    print(f'[OK] Saved {fname}')

# ============================================================================
# COMBINED LEGACY FIGURE (only when no flags specified)
# ============================================================================
if not _any:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax in axes.flat:
        _setup_ax(ax)
    _plot_a(axes[0, 0])
    _plot_b(axes[0, 1])
    _plot_c(axes[1, 0])
    _plot_d(axes[1, 1])
    fig.suptitle(
        'Experiment VI: CHDBO Scalability at $n = 128, 512, 1024$\n'
        f'{N_TRIALS} trials × {N_STEPS} steps per dim — '
        f'Total violations: {n_total_violations}/{N_TRIALS * len(DIMS)}',
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    fig.savefig('figure_13.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'[OK] Saved figure_13.png (combined)')

print(f'\n[OK] Total violations: {n_total_violations}/{N_TRIALS * len(DIMS)}')
