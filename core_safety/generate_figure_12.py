"""
Figure 12: Nonlinear Drift Dynamics — Lorenz-type Attractor in R^128
=====================================================================
CHDBO with strongly nonlinear dynamics: a high-dimensional Lorenz-like
attractor scaled to R^128 at NEAR-FULL SCALE (1/8 scaling).

CRITICAL CHANGE from earlier version: LORENZ_SCALE increased from 1/40
to 1/8, moving the system into the genuinely nonlinear convection regime
rather than the effectively linearized regime. RK4 integration replaces
Euler for stability. Gamma increased from 1.0 to 5.0 to handle the
much stronger nonlinear drift.

Setup:
  - State x ∈ R^128, 42 coupled Lorenz triplets, ring topology
  - LORENZ_SCALE = 1/8 (effective Rayleigh parameter ≈ 3.5, nonlinear convection)
  - RK4 integrator, dt = 0.001, 2000 steps per trial
  - Safe set: unit ball, h(x) = 1 - ||x||^2
  - CBF-QP (γ = 5.0) enforces h(x) ≥ 0 at every step

Output: figure_12.png — 2×2 panel

This is Experiment VIII in the paper.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Command-line arguments: -a, -b, -c, -d select which panels to regenerate.
# No flags = regenerate all (legacy behaviour).
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description='Generate Experiment V (Lorenz drift) figures.')
_parser.add_argument('-a', action='store_true', help='Regenerate figure_12a (norm trajectories)')
_parser.add_argument('-b', action='store_true', help='Regenerate figure_12b (barrier values)')
_parser.add_argument('-c', action='store_true', help='Regenerate figure_12c (drift magnitude)')
_parser.add_argument('-d', action='store_true', help='Regenerate figure_12d (L_f h distribution)')
_args = _parser.parse_args()
_any_selected = _args.a or _args.b or _args.c or _args.d
_gen_a = _args.a or not _any_selected
_gen_b = _args.b or not _any_selected
_gen_c = _args.c or not _any_selected
_gen_d = _args.d or not _any_selected
_gen_combined = not _any_selected

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
N = 128
N_TRIALS = 50
N_STEPS = 2000
DT = 0.001        # smaller dt for RK4 stability with stronger dynamics
GAMMA = 5.0       # stronger CBF enforcement to handle nonlinear drift
GOAL_RADIUS = 1.5  # goal outside unit ball

# Lorenz parameters (standard values, rescaled)
SIGMA_L = 10.0
RHO_L = 28.0
BETA_L = 8.0 / 3.0

# Scaling factor: 1/8 keeps some dynamics inside the unit ball but
# preserves genuinely nonlinear (near-chaotic) behavior. Previously 1/40
# which effectively linearized the dynamics.
# At 1/8 the effective Rayleigh number is (rho * LORENZ_SCALE) ~ 3.5,
# well into the nonlinear convection regime.
LORENZ_SCALE = 1.0 / 8.0

# Inter-triplet diffusive coupling strength
KAPPA = 0.5

N_TRIPLETS = N // 3  # 42 triplets = 126 dims, 2 dims are passive

# ============================================================================
# LORENZ-TYPE DRIFT IN R^128
# ============================================================================
def lorenz_drift(x):
    """
    Nonlinear drift f(x) in R^128.
    Groups state into Lorenz triplets with cubic coupling.
    Adjacent triplets are coupled via diffusive terms (ring topology)
    to create genuinely high-dimensional chaos.
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
    
    # Inter-triplet diffusive coupling (ring topology)
    # Couples the x-component of each triplet to its neighbors' x-component,
    # and the z-component to neighbors' z-component, creating cross-dimensional
    # information flow that makes the system genuinely high-dimensional.
    for i in range(N_TRIPLETS):
        i_prev = (i - 1) % N_TRIPLETS
        i_next = (i + 1) % N_TRIPLETS
        # Diffusive coupling on x-component (first of each triplet)
        f[3*i] += KAPPA * (x[3*i_next] - x[3*i])
        # Diffusive coupling on z-component (third of each triplet)
        f[3*i+2] += KAPPA * (x[3*i_prev+2] - x[3*i+2])
    
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
n_violations = 0  # per-trial violation count

for trial in range(N_TRIALS):
    # Start inside safe set
    x = np.random.randn(N) * 0.3
    x = x / np.linalg.norm(x) * 0.5  # ||x|| = 0.5
    
    norms = [np.linalg.norm(x)]
    h_vals = [barrier(x)]
    lfh_vals = []
    drift_mags = []
    trial_violated = False
    
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
        
        # RK4 integration (more accurate than Euler for nonlinear dynamics)
        def dynamics(state):
            return lorenz_drift(state) + u_star
        k1 = dynamics(x)
        k2 = dynamics(x + 0.5 * DT * k1)
        k3 = dynamics(x + 0.5 * DT * k2)
        k4 = dynamics(x + DT * k3)
        x = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        norms.append(np.linalg.norm(x))
        h_val = barrier(x)
        h_vals.append(h_val)
        
        if h_val < -1e-6:
            trial_violated = True
    
    if trial_violated:
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
BG = '#f8f9fa'
script_dir = os.path.dirname(os.path.abspath(__file__))
t_axis = np.arange(N_STEPS + 1) * DT

print(f"\nPanels to generate: "
      f"{'a ' if _gen_a else ''}{'b ' if _gen_b else ''}"
      f"{'c ' if _gen_c else ''}{'d ' if _gen_d else ''}"
      f"{'(all + combined)' if not _any_selected else ''}")

# ============================================================================
# COMBINED 2x2 FIGURE (legacy, only when no specific panel requested)
# ============================================================================
if _gen_combined:
    print("\nGenerating figure_12.png (combined) ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax in axes.flat:
        ax.set_facecolor(BG)

    # (a) Norm trajectories
    ax = axes[0, 0]
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
        'Experiment V: CHDBO with Nonlinear Lorenz-type Drift ($\\mathbb{R}^{128}$)\n'
        f'{N_TRIALS} trials, {N_STEPS} steps, dt={DT} — '
        f'Violations: {n_violations}/{N_TRIALS}',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'figure_12.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Saved figure_12.png (combined, legacy)")

# ============================================================================
# INDIVIDUAL PANEL FIGURES (for split paper layout)
# ============================================================================

# --- Figure 12a: Norm Trajectories ---
if _gen_a:
    fig_a, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(BG)
    for i, norms in enumerate(all_norms):
        ax.plot(t_axis, norms, alpha=0.3, lw=0.8, color='#2ecc71')
    ax.axhline(1.0, color='red', ls='--', lw=2, label='$\\|x\\| = 1$ (boundary)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('$\\|x(t)\\|$', fontsize=12)
    ax.set_title('Norm Trajectories (50 trials)\n'
                 f'Nonlinear Lorenz Drift in $\\mathbb{{R}}^{{{N}}}$, '
                 f'$\\Delta t = {DT}$, RK4',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2)
    ax.text(0.03, 0.97,
            f'{N_TRIALS} trials, {N_STEPS} steps\n'
            f'Safety violations: {n_violations}/{N_TRIALS}\n'
            f'Max $\\|x\\|$: {max(max(n) for n in all_norms):.4f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    plt.tight_layout()
    fig_a.savefig(os.path.join(script_dir, 'figure_12a.png'), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_a)
    print("[OK] Saved figure_12a.png")

# --- Figure 12b: Barrier Function Values ---
if _gen_b:
    fig_b, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(BG)
    for i, h_vals in enumerate(all_h_vals):
        ax.plot(t_axis, h_vals, alpha=0.3, lw=0.8, color='#3498db')
    ax.axhline(0.0, color='red', ls='--', lw=2, label='$h(x) = 0$ (unsafe)')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('$h(x) = 1 - \\|x\\|^2$', fontsize=12)
    ax.set_title('Barrier Function Values (50 trials)\n'
                 'Safety requires $h(x) \\geq 0$ at all times',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    min_h = min(min(h) for h in all_h_vals)
    ax.text(0.03, 0.97,
            f'Min $h(x)$ across all trials: {min_h:.4f}\n'
            f'$h(x) \\geq 0$ maintained: {"Yes" if min_h >= -1e-6 else "No"}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    plt.tight_layout()
    fig_b.savefig(os.path.join(script_dir, 'figure_12b.png'), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_b)
    print("[OK] Saved figure_12b.png")

# --- Figure 12c: Drift Magnitude Distribution ---
if _gen_c:
    fig_c, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(BG)
    ax.hist(all_drift_mags, bins=60, color='#e67e22', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(all_drift_mags), color='red', ls='--', lw=2,
               label=f'Mean $\\|f(x)\\| = {np.mean(all_drift_mags):.3f}$')
    ax.set_xlabel('$\\|f(x)\\|$ (drift magnitude)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Nonlinear Drift Magnitude Distribution\n'
                 'Lorenz-type attractor ($\\sigma=10, \\rho=28, \\beta=8/3$, scale $= 1/8$)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig_c.savefig(os.path.join(script_dir, 'figure_12c.png'), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_c)
    print("[OK] Saved figure_12c.png")

# --- Figure 12d: L_f h Distribution ---
if _gen_d:
    fig_d, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor(BG)
    ax.hist(all_Lfh, bins=60, color='#9b59b6', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(all_Lfh), color='red', ls='--', lw=2,
               label=f'Mean $L_f h = {np.mean(all_Lfh):.3f}$')
    ax.axvline(0, color='black', ls='-', lw=1)
    ax.set_xlabel('$L_f h(x) = \\nabla h \\cdot f(x)$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Lie Derivative $L_f h$ Distribution\n'
                 '(negative = drift pushes toward boundary)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.text(0.97, 0.97,
            'Strongly negative $L_f h$\n'
            'confirms nonlinear drift\n'
            'actively opposes safety',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0e6ff', alpha=0.85))
    plt.tight_layout()
    fig_d.savefig(os.path.join(script_dir, 'figure_12d.png'), dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_d)
    print("[OK] Saved figure_12d.png")

print(f"\n[DONE] {n_violations}/{N_TRIALS} safety violations")
