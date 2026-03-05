"""
Experiment X: Failure Modes and Limitations of CHDBO

A CREDIBILITY experiment showing WHERE the CHDBO framework breaks down.
Published papers gain trust by honestly reporting negative results.

This experiment demonstrates three failure regimes:

  (a) ACTUATION STARVATION: When the control budget u_max is too small relative
      to the Lipschitz-weighted drift, the CBF-QP becomes infeasible.
      → Shows the critical threshold: u_max < L_h * ||f||_max for guaranteed failure.

  (b) BARRIER CONDITIONING: When the barrier Lipschitz constant L_h grows large
      (e.g., narrow safe sets or high-curvature barriers), the CBF-QP requires
      proportionally larger interventions. In high dimensions, this can exceed
      any reasonable control budget.
      → Shows safety rate vs. L_h for fixed u_max.

  (c) MCBC SAMPLE COMPLEXITY vs. CONFIDENCE: While MCBC sample complexity is
      dimension-independent for fixed (ε, δ), the MINIMUM safe probability
      achievable depends on the barrier/dynamics interaction. For pathological
      dynamics (e.g., rapidly rotating flows), MCBC correctly reports low P_safe.
      → Shows MCBC honestly captures unsafe configurations.

Output: figure_21.png  (or figure_21a-c.png with -a/-b/-c flags)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

print("=" * 70)
print("  EXPERIMENT X: CHDBO Failure Modes and Limitations")
print("  Demonstrating WHERE the framework breaks down")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# Common setup
# ──────────────────────────────────────────────────────────────────────
DIM = 50
rng = np.random.RandomState(42)

# Spherical barrier: h(x) = R^2 - ||x||^2, safe set = {||x|| <= R}
R_SAFE = 1.0

def barrier_sphere(x, R=R_SAFE):
    return R**2 - np.dot(x, x)

def barrier_grad_sphere(x):
    return -2.0 * x

# CBF-QP for spherical barrier with bounded actuation
def cbf_qp_bounded(x, f_drift, gamma, u_max):
    """
    CBF-QP: min ||u||  s.t. ∇h·(f+u) >= -γh(x),  ||u|| <= u_max
    Closed-form for spherical barrier ∇h = -2x:
      Constraint: -2x·(f+u) >= -γ(R²-||x||²)
      Need: -2x·f - 2x·u >= -γh
      So: -2x·u >= -γh + 2x·f  →  u along -x direction
    Returns (u, feasible, h_next)
    """
    h_val = barrier_sphere(x)
    grad_h = barrier_grad_sphere(x)
    # grad_h · f
    gh_f = np.dot(grad_h, f_drift)
    # Required: grad_h · u >= -gamma * h - grad_h · f
    rhs = -gamma * h_val - gh_f

    if rhs <= 0:
        # No intervention needed
        x_next = x + f_drift
        return np.zeros(DIM), True, barrier_sphere(x_next)

    # Minimum-norm: u* = (rhs / ||∇h||²) * ∇h
    grad_h_sq = np.dot(grad_h, grad_h)
    if grad_h_sq < 1e-15:
        x_next = x + f_drift
        return np.zeros(DIM), False, barrier_sphere(x_next)

    lam = rhs / grad_h_sq
    u_star = lam * grad_h
    u_norm = np.linalg.norm(u_star)

    if u_norm <= u_max:
        x_next = x + f_drift + u_star
        return u_star, True, barrier_sphere(x_next)
    else:
        # Saturate: project onto the budget sphere
        u_sat = u_star * (u_max / u_norm)
        x_next = x + f_drift + u_sat
        h_next = barrier_sphere(x_next)
        return u_sat, (h_next >= 0), h_next


# ──────────────────────────────────────────────────────────────────────
# (a) ACTUATION STARVATION: safety vs. control budget for strong drift
#     Uses LINEAR barrier h(x) = w·x + b where CBF is EXACT (no
#     linearization error). This isolates the actuation-budget effect.
# ──────────────────────────────────────────────────────────────────────
print("\n[1/3] Actuation Starvation experiment (linear barrier)...")

# Random barrier normal (fixed across trials)
w_lin = rng.randn(DIM)
w_lin = w_lin / np.linalg.norm(w_lin)
b_lin = 0.0  # boundary passes through origin
DRIFT_STRENGTH = 0.15  # drift magnitude that pushes AGAINST barrier
NOISE_SCALE = 0.05
N_STEPS = 200
N_TRAJS = 200
GAMMA = 0.3

u_max_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 5.0]
safety_rates_a = []
mean_interventions_a = []
feasibility_rates_a = []

for u_max in u_max_values:
    n_safe = 0
    interv_norms = []
    n_feasible = 0
    n_total_steps = 0

    for _ in range(N_TRAJS):
        # Start in safe region (h(x) > 0)
        x = rng.randn(DIM)
        # Ensure h(x) = w·x + b > 0
        x = x - (np.dot(w_lin, x) + b_lin - 0.5) * w_lin  # project to h=0.5

        safe_so_far = True
        for t in range(N_STEPS):
            # Drift pushes against the barrier + random noise
            drift = -DRIFT_STRENGTH * w_lin + NOISE_SCALE * rng.randn(DIM)

            # CBF-QP for linear barrier (exact, no linearization error)
            h_val = np.dot(w_lin, x) + b_lin
            wf = np.dot(w_lin, drift)
            rhs = -GAMMA * h_val - wf  # need w·u >= rhs

            if rhs <= 0:
                u = np.zeros(DIM)
                n_feasible += 1
            else:
                # u* = rhs/||w||² * w (minimum norm)
                u_star = rhs * w_lin  # ||w||=1
                u_norm = np.linalg.norm(u_star)
                if u_norm <= u_max:
                    u = u_star
                    n_feasible += 1
                else:
                    u = u_star * (u_max / u_norm)  # saturate

            interv_norms.append(np.linalg.norm(u))
            n_total_steps += 1
            x = x + drift + u

            if np.dot(w_lin, x) + b_lin < -1e-6:
                safe_so_far = False
                break

        if safe_so_far:
            n_safe += 1

    safety_rate = n_safe / N_TRAJS
    feas_rate = n_feasible / n_total_steps if n_total_steps > 0 else 0
    safety_rates_a.append(safety_rate)
    mean_interventions_a.append(np.mean(interv_norms))
    feasibility_rates_a.append(feas_rate)
    print(f"  u_max={u_max:.3f}: safety={safety_rate:.2f}, "
          f"feasibility={feas_rate:.3f}, mean ||u||={np.mean(interv_norms):.4f}")

# Threshold: need u_max >= drift_strength (since ||w||=1)
u_threshold = DRIFT_STRENGTH
print(f"  Theoretical threshold: u_max ≈ {u_threshold:.2f} (drift opposing barrier)")


# ──────────────────────────────────────────────────────────────────────
# (b) BARRIER CONDITIONING: safety vs. drift/margin ratio
#     For a linear barrier with margin m, the CBF is exact. But as the
#     drift grows relative to the safety margin, even unconstrained CBF
#     struggles because barrier value erodes over multiple steps.
#     Uses BOUNDED actuation to show how tighter margins need more control.
# ──────────────────────────────────────────────────────────────────────
print("\n[2/3] Barrier Conditioning experiment (drift vs margin)...")

U_MAX_FIXED = 0.3
GAMMA_B = 0.3
N_STEPS_B = 200
N_TRAJS_B = 500

# Vary the drift magnitude — focus on the transition zone
drift_magnitudes = [0.10, 0.15, 0.18, 0.20,
                    0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245,
                    0.25, 0.255, 0.26, 0.265, 0.27, 0.28, 0.29, 0.30, 0.35]
safety_rates_b = []
lipschitz_values = []  # drift/u_max ratio (effective difficulty)

for drift_mag in drift_magnitudes:
    n_safe = 0
    for _ in range(N_TRAJS_B):
        # Linear barrier: h(x) = w·x + b, start with h = 0.5
        x = rng.randn(DIM)
        x = x - (np.dot(w_lin, x) + b_lin - 0.5) * w_lin

        safe_so_far = True
        for t in range(N_STEPS_B):
            # Drift opposing barrier + tangential noise
            drift = -drift_mag * w_lin + 0.02 * rng.randn(DIM)

            h_val = np.dot(w_lin, x) + b_lin
            wf = np.dot(w_lin, drift)
            rhs_val = -GAMMA_B * h_val - wf

            if rhs_val <= 0:
                u = np.zeros(DIM)
            else:
                u_star = rhs_val * w_lin
                if np.linalg.norm(u_star) <= U_MAX_FIXED:
                    u = u_star
                else:
                    u = u_star * (U_MAX_FIXED / np.linalg.norm(u_star))

            x = x + drift + u
            if np.dot(w_lin, x) + b_lin < -1e-6:
                safe_so_far = False
                break

        if safe_so_far:
            n_safe += 1

    safety_rate = n_safe / N_TRAJS_B
    ratio = drift_mag / U_MAX_FIXED
    safety_rates_b.append(safety_rate)
    lipschitz_values.append(ratio)
    print(f"  drift={drift_mag:.2f}: safety={safety_rate:.2f}, "
          f"drift/u_max={ratio:.2f}")


# ──────────────────────────────────────────────────────────────────────
# (c) MCBC HONESTY: correctly reports low P_safe for bad configurations
# ──────────────────────────────────────────────────────────────────────
print("\n[3/3] MCBC honesty experiment (pathological dynamics)...")

# Test MCBC with rotating flow that pushes states across the boundary
# The dynamics are designed so that ~X% of boundary points get pushed out
# MCBC should correctly report P_safe ≈ 1 - X%

N_MCBC = 5000
target_failure_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
measured_psafe = []
ci_lower = []
ci_upper = []

for target_fail in target_failure_rates:
    n_fail = 0
    for i in range(N_MCBC):
        # Sample on boundary: ||x|| = R
        x_bnd = rng.randn(DIM)
        x_bnd = x_bnd / np.linalg.norm(x_bnd) * R_SAFE

        # Dynamics that push outward with probability ~ target_fail
        # For target_fail fraction of directions, drift is strongly outward
        # For the rest, drift is weakly inward
        radial_component = np.dot(x_bnd, rng.randn(DIM))

        if rng.rand() < target_fail:
            # Strong outward drift
            f_drift = 0.5 * x_bnd / np.linalg.norm(x_bnd)
        else:
            # Weak inward drift
            f_drift = -0.1 * x_bnd / np.linalg.norm(x_bnd) + 0.02 * rng.randn(DIM)

        # CBF-QP with NO actuation constraint (u_max = infinity)
        # Even with infinite control, check if the dynamics are so bad that
        # the barrier still fails (this shouldn't happen for unconstrained CBF)
        # With BOUNDED control:
        U_MAX_MCBC = 0.3  # limited control authority
        h_val = barrier_sphere(x_bnd)  # ≈ 0 on boundary
        grad_h = barrier_grad_sphere(x_bnd)
        gh_f = np.dot(grad_h, f_drift)
        rhs = -GAMMA * h_val - gh_f  # ≈ -gh_f since h≈0

        if rhs <= 0:
            u = np.zeros(DIM)
        else:
            grad_h_sq = np.dot(grad_h, grad_h)
            lam = rhs / grad_h_sq
            u_star = lam * grad_h
            if np.linalg.norm(u_star) <= U_MAX_MCBC:
                u = u_star
            else:
                u = u_star * (U_MAX_MCBC / np.linalg.norm(u_star))

        x_next = x_bnd + f_drift + u
        if barrier_sphere(x_next) < 0:
            n_fail += 1

    p_safe = 1.0 - n_fail / N_MCBC
    # Clopper-Pearson 95% CI
    from scipy.stats import beta as beta_dist
    alpha_ci = 0.05
    n_success = N_MCBC - n_fail
    if n_success == 0:
        ci_lo = 0.0
    else:
        ci_lo = beta_dist.ppf(alpha_ci / 2, n_success, n_fail + 1)
    if n_success == N_MCBC:
        ci_hi = 1.0
    else:
        ci_hi = beta_dist.ppf(1 - alpha_ci / 2, n_success + 1, n_fail)

    measured_psafe.append(p_safe)
    ci_lower.append(ci_lo)
    ci_upper.append(ci_hi)
    print(f"  target_fail={target_fail:.2f}: P_safe={p_safe:.4f} "
          f"[{ci_lo:.4f}, {ci_hi:.4f}] (expected ~{1-target_fail:.2f})")


# ──────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────
BG = '#f8f9fa'
_DIR = os.path.dirname(os.path.abspath(__file__))

SUPTITLE = (
    f'Experiment X: CHDBO Failure Modes and Limitations\n'
    f'$n = {DIM}$  |  Honest reporting of WHERE the framework breaks down'
)


def _plot_a(ax):
    """(a) Actuation Starvation: safety vs control budget."""
    ax.semilogx(u_max_values, safety_rates_a, 'o-', color='crimson',
                linewidth=2, markersize=8, label='Safety rate')
    ax.axvline(x=u_threshold, color='navy', linestyle='--', linewidth=2,
               label=f'Threshold $u_{{max}} \\approx {u_threshold:.2f}$')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax.fill_between([u_max_values[0], u_threshold], 0, 1,
                    color='red', alpha=0.1, label='Infeasible regime')
    ax.fill_between([u_threshold, u_max_values[-1]], 0, 1,
                    color='green', alpha=0.1, label='Feasible regime')
    ax.set_xlabel('Control budget $u_{max}$', fontsize=12)
    ax.set_ylabel('Safety rate', fontsize=12)
    ax.set_title('(a) Actuation Starvation (linear barrier, exact CBF)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=8, loc='center left')
    ax.set_facecolor(BG)
    ax.text(0.98, 0.02,
            f'Drift: {DRIFT_STRENGTH}\n'
            f'Noise: {NOISE_SCALE}\n'
            f'N={N_TRAJS} × {N_STEPS} steps\n'
            f'γ = {GAMMA}',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def _plot_b(ax):
    """(b) Drift Overwhelms Control Budget."""
    ax.plot(lipschitz_values, safety_rates_b, 's-', color='darkorange',
            linewidth=2, markersize=5, label='Safety rate')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
               label='drift = $u_{max}$ (critical)')
    ax.set_xlabel('Drift magnitude / Control budget $u_{max}$', fontsize=11)
    ax.set_ylabel('Safety rate', fontsize=12)
    ax.set_title('(b) Drift Overwhelms Control Budget',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0.25, 1.25)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=9)
    ax.set_facecolor(BG)
    ax.text(0.985, 0.75,
            f'u_max = {U_MAX_FIXED}\n'
            f'γ = {GAMMA_B}\n'
            f'{N_TRAJS_B} trajs × {N_STEPS_B} steps',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def _plot_c(ax):
    """(c) MCBC Honesty: correctly reports low safety."""
    expected = [1 - f for f in target_failure_rates]
    measured = measured_psafe
    ci_err = [[m - lo for m, lo in zip(measured, ci_lower)],
              [hi - m for hi, m in zip(ci_upper, measured)]]
    ax.errorbar(expected, measured, yerr=ci_err, fmt='o', color='steelblue',
                markersize=10, capsize=6, capthick=2, linewidth=2,
                label=f'MCBC (N={N_MCBC})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal: measured = expected')
    ax.fill_between([0, 1], [0, 1], color='gray', alpha=0.1)
    ax.set_xlabel('Expected $P_{safe}$ (1 − target failure rate)', fontsize=12)
    ax.set_ylabel('MCBC measured $P_{safe}$', fontsize=12)
    ax.set_title('(c) MCBC Honesty: Correctly Reports Low Safety',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_facecolor(BG)
    ax.text(0.02, 0.98,
            f'u_max = {U_MAX_MCBC}\n'
            f'N = {N_MCBC} boundary samples\n'
            f'95% Clopper-Pearson CI',
            transform=ax.transAxes, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def _save_single(plot_fn, tag):
    """Save a single panel as figure_21{tag}.png."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_fn(ax)
    fig.suptitle(SUPTITLE, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(_DIR, f'figure_21{tag}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────────────
# CLI & GENERATION
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Generate Figure 21 panels')
parser.add_argument('-a', action='store_true', help='Panel (a) Actuation Starvation')
parser.add_argument('-b', action='store_true', help='Panel (b) Drift vs Budget')
parser.add_argument('-c', action='store_true', help='Panel (c) MCBC Honesty')
args = parser.parse_args()

_any = args.a or args.b or args.c
_gen_a = args.a or not _any
_gen_b = args.b or not _any
_gen_c = args.c or not _any

print("\nGenerating figure_21 panels...")

if _gen_a:
    _save_single(_plot_a, 'a')
if _gen_b:
    _save_single(_plot_b, 'b')
if _gen_c:
    _save_single(_plot_c, 'c')

# Combined legacy figure when no flags specified
if not _any:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _plot_a(axes[0])
    _plot_b(axes[1])
    _plot_c(axes[2])
    fig.suptitle(SUPTITLE, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(_DIR, 'figure_21.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  Saved figure_21.png (combined)")

print("\n[OK] Experiment X complete.")
