"""
Figure 12: Nonlinear Drift Dynamics — Lorenz-type Attractor in R^128
=====================================================================
CHDBO with strongly nonlinear dynamics: a high-dimensional Lorenz-like
chaotic attractor scaled to R^128. The drift f(x) is genuinely nonlinear
(cubic coupling terms), not just linear or bilinear.

Setup:
  - State x ∈ R^128, grouped into 42 "Lorenz triplets" (126 dims) + 2 passive dims
  - Each triplet (x_{3i}, x_{3i+1}, x_{3i+2}) evolves with Lorenz-like coupling:
      dx_{3i}   = σ(x_{3i+1} - x_{3i})
      dx_{3i+1} = x_{3i}(ρ - x_{3i+2}) - x_{3i+1}
      dx_{3i+2} = x_{3i} * x_{3i+1} - β * x_{3i+2}
    (scaled so the attractor fits inside the unit ball)
  - Safe set: unit ball, h(x) = 1 - ||x||^2
  - CBF-QP enforces h(x) ≥ 0 at every step
  - 50 trials, 500 steps each, dt = 0.005

Output: figure_12.png — 2×2 panel:
  (a) Norm trajectories over time
  (b) h(x) barrier values over time
  (c) Distribution of ||f(x)|| (drift magnitude)
  (d) L_f h values showing nonlinear adverse drift

This is Experiment VIII in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
N = 128
N_TRIALS = 50
N_STEPS = 500
DT = 0.005
GAMMA = 1.0
GOAL_RADIUS = 1.5  # goal outside unit ball

# Lorenz parameters (standard values, rescaled)
SIGMA_L = 10.0
RHO_L = 28.0
BETA_L = 8.0 / 3.0

# Scaling factor to keep Lorenz dynamics inside unit ball
# Standard Lorenz attractor has amplitude ~20-30; we scale by 1/40
LORENZ_SCALE = 1.0 / 40.0

N_TRIPLETS = N // 3  # 42 triplets = 126 dims, 2 dims are passive

# ============================================================================
# LORENZ-TYPE DRIFT IN R^128
# ============================================================================
def lorenz_drift(x):
    """
    Nonlinear drift f(x) in R^128.
    Groups state into Lorenz triplets with cubic coupling.
    Extra dimensions have mild linear drift.
    """
    f = np.zeros_like(x)
    
    # Scale state to Lorenz coordinates
    xs = x / LORENZ_SCALE
    
    for i in range(N_TRIPLETS):
        i0, i1, i2 = 3*i, 3*i+1, 3*i+2
        xi, yi, zi = xs[i0], xs[i1], xs[i2]
        
        # Lorenz equations (nonlinear: xi*zi and xi*yi terms)
        f[i0] = SIGMA_L * (yi - xi)
        f[i1] = xi * (RHO_L - zi) - yi  # Nonlinear: xi * zi
        f[i2] = xi * yi - BETA_L * zi    # Nonlinear: xi * yi
    
    # Scale derivatives back
    f[:3*N_TRIPLETS] *= LORENZ_SCALE
    
    # Passive dims: mild linear drift
    for j in range(3*N_TRIPLETS, N):
        f[j] = -0.1 * x[j]
    
    return f


def barrier(x):
    """h(x) = 1 - ||x||^2 (quadratic barrier for the unit ball)."""
    return 1.0 - np.dot(x, x)


def barrier_gradient(x):
    """∇h(x) = -2x."""
    return -2.0 * x


def lie_derivative_f(x):
    """L_f h = ∇h · f(x) = -2 x · f(x)."""
    fx = lorenz_drift(x)
    return np.dot(barrier_gradient(x), fx)


def cbf_qp_step(x, u_nom, gamma=GAMMA):
    """
    Solve CBF-QP: min ||u - u_nom||^2
    s.t. L_f h + L_g h u >= -gamma * h(x)
    
    For single-integrator with drift: dx = f(x) + u
    L_f h = ∇h · f(x), L_g h = ∇h (since g = I)
    """
    h_val = barrier(x)
    grad_h = barrier_gradient(x)
    Lf_h = np.dot(grad_h, lorenz_drift(x))
    Lg_h = grad_h  # g(x) = I for single-integrator with drift
    
    # Constraint: Lg_h · u >= -gamma * h - Lf_h
    # i.e., grad_h · u >= -gamma * h - Lf_h
    lhs_nom = np.dot(Lg_h, u_nom)
    rhs = -gamma * h_val - Lf_h
    
    if lhs_nom >= rhs:
        # Constraint satisfied, use nominal
        return u_nom, False
    
    # Project: u* = u_nom + lambda* * Lg_h
    # lambda* = (rhs - Lg_h · u_nom) / ||Lg_h||^2
    Lg_norm_sq = np.dot(Lg_h, Lg_h)
    if Lg_norm_sq < 1e-12:
        return u_nom, False
    
    lam = (rhs - lhs_nom) / Lg_norm_sq
    u_star = u_nom + lam * Lg_h
    return u_star, True


# ============================================================================
# RUN SIMULATION
# ============================================================================
print("[*] Running Experiment VIII: Nonlinear Lorenz-type drift in R^128...")

all_norms = []
all_h_vals = []
all_Lfh = []
all_drift_mags = []
n_violations = 0

for trial in range(N_TRIALS):
    # Start inside safe set
    x = np.random.randn(N) * 0.3
    x = x / np.linalg.norm(x) * 0.5  # ||x|| = 0.5
    
    norms = [np.linalg.norm(x)]
    h_vals = [barrier(x)]
    lfh_vals = []
    drift_mags = []
    
    for step in range(N_STEPS):
        # Compute drift
        fx = lorenz_drift(x)
        drift_mags.append(np.linalg.norm(fx))
        
        # Compute L_f h
        lfh = np.dot(barrier_gradient(x), fx)
        lfh_vals.append(lfh)
        
        # Nominal control: drive toward goal outside the safe set
        goal = np.zeros(N)
        goal[0] = GOAL_RADIUS
        u_nom = 0.5 * (goal - x) / np.linalg.norm(goal - x)
        
        # CBF-QP
        u_star, active = cbf_qp_step(x, u_nom)
        
        # Euler step: dx = f(x) + u
        x = x + (fx + u_star) * DT
        
        norms.append(np.linalg.norm(x))
        h_val = barrier(x)
        h_vals.append(h_val)
        
        if h_val < -1e-6:
            n_violations += 1
    
    all_norms.append(norms)
    all_h_vals.append(h_vals)
    all_Lfh.extend(lfh_vals)
    all_drift_mags.extend(drift_mags)

print(f"    Violations: {n_violations}/{N_TRIALS}")
print(f"    Mean ||f(x)||: {np.mean(all_drift_mags):.4f}")
print(f"    Mean L_f h:    {np.mean(all_Lfh):.4f}")
print(f"    Min  L_f h:    {np.min(all_Lfh):.4f}")
print(f"    Max ||x||:     {max(max(n) for n in all_norms):.4f}")

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
BG = '#f8f9fa'
for ax in axes.flat:
    ax.set_facecolor(BG)

# (a) Norm trajectories
ax = axes[0, 0]
t_axis = np.arange(N_STEPS + 1) * DT
for i, norms in enumerate(all_norms):
    ax.plot(t_axis, norms, alpha=0.3, lw=0.8, color='#2ecc71')
ax.axhline(1.0, color='red', ls='--', lw=2, label='$\\|x\\| = 1$ (boundary)')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('$\\|x(t)\\|$', fontsize=12)
ax.set_title('(a) Norm Trajectories (50 trials)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.2)

# (b) Barrier values
ax = axes[0, 1]
for i, h_vals in enumerate(all_h_vals):
    ax.plot(t_axis, h_vals, alpha=0.3, lw=0.8, color='#3498db')
ax.axhline(0.0, color='red', ls='--', lw=2, label='$h(x) = 0$ (unsafe)')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('$h(x) = 1 - \\|x\\|^2$', fontsize=12)
ax.set_title('(b) Barrier Function Values', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (c) Drift magnitude distribution
ax = axes[1, 0]
ax.hist(all_drift_mags, bins=60, color='#e67e22', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(all_drift_mags), color='red', ls='--', lw=2,
           label=f'Mean $\\|f(x)\\| = {np.mean(all_drift_mags):.3f}$')
ax.set_xlabel('$\\|f(x)\\|$ (drift magnitude)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('(c) Nonlinear Drift Magnitude Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# (d) L_f h distribution
ax = axes[1, 1]
ax.hist(all_Lfh, bins=60, color='#9b59b6', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(all_Lfh), color='red', ls='--', lw=2,
           label=f'Mean $L_f h = {np.mean(all_Lfh):.3f}$')
ax.axvline(0, color='black', ls='-', lw=1)
ax.set_xlabel('$L_f h(x) = \\nabla h \\cdot f(x)$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('(d) Lie Derivative $L_f h$ Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

fig.suptitle(
    'Experiment VIII: CHDBO with Nonlinear Lorenz-type Drift ($\\mathbb{R}^{128}$)\n'
    f'{N_TRIALS} trials, {N_STEPS} steps, dt={DT} — '
    f'Violations: {n_violations}/{N_TRIALS}',
    fontsize=15, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('core_safety/figure_12.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\n[OK] Saved figure_12.png — {n_violations}/{N_TRIALS} safety violations")
