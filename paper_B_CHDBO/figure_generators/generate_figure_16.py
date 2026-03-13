"""
Figure 16: Safe Asymptotic Utility Convergence (Theorem 2 Validation)
=====================================================================
Empirically validates Theorem 2 by plotting utility U(x(t)) over time
for the CBF-QP controlled system. Shows convergence to the constrained
optimum x*_S while maintaining h(x(t)) >= 0.

Setup:
  - Dimension: n = 128
  - Safe set: unit ball, h(x) = 1 - ||x||^2
  - Utility: U(x) = -||x - x_goal||  (concave, target reaching)
  - Dynamics: dx = Ax + u (linear drift, sigma_max ~ 1.04)
  - 20 trials × 600 steps, dt = 0.01
  - Goal placed outside safe set (||x_goal|| = 1.5)

Output: figure_16.png — 2×2 panel:
  (a) Utility U(x(t)) vs time (convergence to constrained max)
  (b) Barrier h(x(t)) vs time (safety maintained)
  (c) ||x(t)|| vs time (trajectories approach boundary)
  (d) Distance to constrained optimum vs time

This validates Theorem 2 (Safe Asymptotic Convergence) from Section 4.3.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_DIM = 128
N_TRIALS = 20
N_STEPS = 1200
DT = 0.01
GAMMA = 2.0

GOAL_NORM = 1.5  # goal is outside safe set


def make_drift_matrix(n, rng):
    """Construct mildly unstable linear drift A in R^n."""
    S = rng.randn(n, n)
    S = (S - S.T) / 2.0
    S = S / np.linalg.norm(S, ord=2)
    A = S + 0.3 * np.eye(n)
    return A


def barrier(x):
    """h(x) = 1 - ||x||^2  (quadratic barrier)."""
    return 1.0 - np.dot(x, x)


def barrier_grad(x):
    """nabla h = -2x."""
    return -2.0 * x


def utility(x, goal):
    """U(x) = -||x - goal||  (target reaching, concave)."""
    return -np.linalg.norm(x - goal)


def cbf_qp_step(x, u_nom, A, gamma=GAMMA):
    """
    CBF-QP for dx = Ax + u, h(x) = 1 - ||x||^2.
    Constraint: nabla_h . (Ax + u) >= -gamma * h(x)
    """
    h_val = barrier(x)
    gh = barrier_grad(x)
    Ax = A @ x
    Lf_h = np.dot(gh, Ax)
    Lg_h = gh  # g(x) = I

    lhs = np.dot(Lg_h, u_nom)
    rhs = -gamma * h_val - Lf_h

    if lhs >= rhs:
        return u_nom, False  # constraint inactive

    Lg_sq = np.dot(Lg_h, Lg_h)
    if Lg_sq < 1e-12:
        return u_nom, False

    lam = (rhs - lhs) / Lg_sq
    return u_nom + lam * Lg_h, True


# The constrained optimum is the closest point on the boundary to the goal
# For ||goal|| > 1, x*_S = goal / ||goal||
def constrained_optimum(goal):
    return goal / np.linalg.norm(goal)


# ============================================================================
# RUN EXPERIMENT
# ============================================================================
print("[*] Running Experiment: Safe Asymptotic Utility Convergence (n=128)")

rng = np.random.RandomState(42)
A = make_drift_matrix(N_DIM, rng)
sigma_max = np.linalg.norm(A, ord=2)
print(f"    sigma_max(A) = {sigma_max:.4f}")

# Generate a random goal direction outside the safe set
goal_dir = rng.randn(N_DIM)
goal_dir = goal_dir / np.linalg.norm(goal_dir)
goal = goal_dir * GOAL_NORM
x_star = constrained_optimum(goal)
u_star = utility(x_star, goal)

print(f"    ||goal|| = {np.linalg.norm(goal):.2f}")
print(f"    U(x*_S) = {u_star:.4f}")

all_utilities = []
all_barriers = []
all_norms = []
all_distances = []

for trial in range(N_TRIALS):
    # Start from random interior point
    x = rng.randn(N_DIM)
    x = x / np.linalg.norm(x) * 0.3  # start at ||x|| = 0.3

    utilities = []
    barriers = []
    norms = []
    distances = []

    for step in range(N_STEPS):
        h_val = barrier(x)
        u_val = utility(x, goal)
        dist_to_opt = np.linalg.norm(x - x_star)

        utilities.append(u_val)
        barriers.append(h_val)
        norms.append(np.linalg.norm(x))
        distances.append(dist_to_opt)

        # Nominal control: gradient ascent on U(x)
        diff = goal - x
        diff_norm = np.linalg.norm(diff)
        if diff_norm > 1e-8:
            u_nom = 1.0 * diff / diff_norm
        else:
            u_nom = np.zeros(N_DIM)

        # CBF-QP safety filter
        u_star_ctrl, active = cbf_qp_step(x, u_nom, A)

        # Euler step
        x = x + (A @ x + u_star_ctrl) * DT

    all_utilities.append(utilities)
    all_barriers.append(barriers)
    all_norms.append(norms)
    all_distances.append(distances)

all_utilities = np.array(all_utilities)
all_barriers = np.array(all_barriers)
all_norms = np.array(all_norms)
all_distances = np.array(all_distances)

# Safety check
n_violations = np.sum(all_barriers < -1e-6)
print(f"    Total violations: {n_violations}/{N_TRIALS * N_STEPS}")
print(f"    Final mean U:     {np.mean(all_utilities[:, -1]):.4f}")
print(f"    Optimal U(x*_S):  {u_star:.4f}")
print(f"    Final mean dist to x*_S: {np.mean(all_distances[:, -1]):.4f}")

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
BG = '#f8f9fa'
for ax in axes.flat:
    ax.set_facecolor(BG)

time_axis = np.arange(N_STEPS) * DT

# (a) Utility convergence
ax = axes[0, 0]
for i in range(N_TRIALS):
    ax.plot(time_axis, all_utilities[i], alpha=0.25, color='#3498db', linewidth=0.8)
ax.plot(time_axis, np.mean(all_utilities, axis=0), 'b-', linewidth=2.5, label='Mean $U(x(t))$')
ax.axhline(u_star, color='red', ls='--', lw=2, label=f'$U(x^*_{{\\mathcal{{S}}}}) = {u_star:.2f}$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Utility $U(x)$', fontsize=12)
ax.set_title('(a) Utility Convergence to Constrained Optimum', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (b) Barrier values
ax = axes[0, 1]
for i in range(N_TRIALS):
    ax.plot(time_axis, all_barriers[i], alpha=0.25, color='#2ecc71', linewidth=0.8)
ax.plot(time_axis, np.mean(all_barriers, axis=0), 'g-', linewidth=2.5, label='Mean $h(x(t))$')
ax.axhline(0.0, color='red', ls='--', lw=2, label='$h = 0$ (unsafe)')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Barrier $h(x)$', fontsize=12)
ax.set_title('(b) Safety Maintained Throughout ($h \\geq 0$)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (c) Norm trajectories
ax = axes[1, 0]
for i in range(N_TRIALS):
    ax.plot(time_axis, all_norms[i], alpha=0.25, color='#e67e22', linewidth=0.8)
ax.plot(time_axis, np.mean(all_norms, axis=0), color='#e67e22', linewidth=2.5, label='Mean $\\|x(t)\\|$')
ax.axhline(1.0, color='red', ls='--', lw=2, label='Safe boundary $\\|x\\| = 1$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('$\\|x(t)\\|$', fontsize=12)
ax.set_title('(c) Trajectories Approach Boundary', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (d) Distance to constrained optimum
ax = axes[1, 1]
for i in range(N_TRIALS):
    ax.plot(time_axis, all_distances[i], alpha=0.25, color='#9b59b6', linewidth=0.8)
ax.plot(time_axis, np.mean(all_distances, axis=0), color='#9b59b6', linewidth=2.5,
        label='Mean $\\|x(t) - x^*_{\\mathcal{S}}\\|$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('$\\|x(t) - x^*_{\\mathcal{S}}\\|$', fontsize=12)
ax.set_title('(d) Convergence to Constrained Optimum', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_ylim(bottom=0)

fig.suptitle(
    f'Theorem 2 Validation: Safe Asymptotic Utility Convergence ($n={N_DIM}$)\n'
    f'{N_TRIALS} trials × {N_STEPS} steps — '
    f'Violations: {n_violations}/{N_TRIALS * N_STEPS} — '
    f'$\\sigma_{{\\max}}(A) = {sigma_max:.3f}$',
    fontsize=14, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_16.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\n[OK] Saved figure_16.png")
