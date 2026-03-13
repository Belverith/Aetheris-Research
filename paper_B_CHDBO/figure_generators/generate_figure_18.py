"""
Figure 18: Union Bound Degradation and Re-Certification Strategy (Experiment XIII)
===================================================================================
Visualizes how trajectory-level safety probability degrades with horizon length T
under the union bound (Lemma 2), and demonstrates the re-certification mitigation.

Shows:
  (a) P_safe(T) = 1 - Tε for the naive union bound, becoming vacuous at T = 1/ε
  (b) Re-certification windows: resetting the union bound every T_w steps
  (c) Empirical safety rate from Monte Carlo simulation confirming the bound

Output: figure_18.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
EPSILON_VALUES = [1e-3, 1e-4, 1e-5, 1e-6]  # per-step failure bounds
T_MAX = 100_000                              # maximum horizon
N_EMPIRICAL_TRIALS = 500                     # Monte Carlo trials
T_RECERT_WINDOWS = [100, 1000, 10_000]       # re-certification window sizes

N_DIM = 128                                  # for empirical simulation

# ============================================================================
# THEORETICAL UNION BOUND CURVES
# ============================================================================
print("[*] Computing union bound degradation curves...")

T_range = np.logspace(0, np.log10(T_MAX), 500).astype(int)
T_range = np.unique(T_range)  # remove duplicates from int conversion

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
BG_COLOR = '#f8f9fa'

# --- Panel 1: Union bound P_safe(T) = max(0, 1 - Tε) ---
ax1 = axes[0]
ax1.set_facecolor(BG_COLOR)

colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
for eps, color in zip(EPSILON_VALUES, colors):
    P_safe = np.maximum(0, 1 - T_range * eps)
    vacuous_T = int(1.0 / eps)
    ax1.semilogx(T_range, P_safe, '-', color=color, lw=2.5,
                 label=f'$\\epsilon = 10^{{{int(np.log10(eps))}}}$'
                        f' (vacuous at $T = {vacuous_T:,}$)')
    # Mark where bound becomes vacuous
    ax1.axvline(x=vacuous_T, color=color, linestyle=':', alpha=0.3, lw=1)

ax1.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, lw=1,
            label='$P_{\\mathrm{safe}} = 0.99$')
ax1.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, lw=1,
            label='$P_{\\mathrm{safe}} = 0.95$')

ax1.set_xlabel('Trajectory Length $T$ (steps)', fontsize=12)
ax1.set_ylabel('$P_{\\mathrm{safe}}(T) \\geq 1 - T\\epsilon$', fontsize=12)
ax1.set_title('Union Bound Degradation\n(Lemma 2: Naive Bound)',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='lower left', framealpha=0.9)
ax1.grid(True, which='both', alpha=0.15)
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(1, T_MAX)

# --- Panel 2: Re-certification strategy ---
ax2 = axes[1]
ax2.set_facecolor(BG_COLOR)

eps_demo = 1e-4  # demonstration ε
recert_colors = ['#e74c3c', '#9b59b6', '#2ecc71']

# Naive union bound
P_naive = np.maximum(0, 1 - T_range * eps_demo)
ax2.semilogx(T_range, P_naive, '-', color='#bdc3c7', lw=2,
             label=f'Naive ($\\epsilon = 10^{{-4}}$)', alpha=0.7)

# Re-certification: P_safe(T, T_w) = (1 - T_w × ε)^(T/T_w) per window
for T_w, color in zip(T_RECERT_WINDOWS, recert_colors):
    n_windows = T_range / T_w
    P_per_window = 1 - T_w * eps_demo
    if P_per_window <= 0:
        P_recert = np.zeros_like(T_range, dtype=float)
    else:
        P_recert = np.power(np.maximum(P_per_window, 0), n_windows)
    ax2.semilogx(T_range, P_recert, '-', color=color, lw=2.5,
                 label=f'Re-certify every $T_w = {T_w:,}$\n'
                        f'($P_w = {P_per_window:.6f}$)')

ax2.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, lw=1)
ax2.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, lw=1)

ax2.set_xlabel('Trajectory Length $T$ (steps)', fontsize=12)
ax2.set_ylabel('$P_{\\mathrm{safe}}(T)$', fontsize=12)
ax2.set_title('Re-Certification Mitigation\n(reset union bound every $T_w$ steps)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, loc='lower left', framealpha=0.9)
ax2.grid(True, which='both', alpha=0.15)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlim(1, T_MAX)

# --- Panel 3: Empirical Monte Carlo validation ---
ax3 = axes[2]
ax3.set_facecolor(BG_COLOR)

print("\n[*] Running empirical Monte Carlo validation...")

# Simulate CBF-QP safety: at each step, the system has a tiny independent
# probability of violation ε_empirical. We measure the empirical trajectory
# safety rate as a function of T.
eps_empirical = 1e-4
T_empirical_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
empirical_safety_rates = []

for T_emp in T_empirical_values:
    n_safe = 0
    for trial in range(N_EMPIRICAL_TRIALS):
        # Each step has independent failure probability eps_empirical
        # This is a Bernoulli trial per step
        failures = np.random.random(T_emp) < eps_empirical
        if not np.any(failures):
            n_safe += 1
    rate = n_safe / N_EMPIRICAL_TRIALS
    empirical_safety_rates.append(rate)
    print(f"    T={T_emp:>6}: empirical P_safe = {rate:.4f}, "
          f"union bound = {max(0, 1 - T_emp * eps_empirical):.4f}, "
          f"exact = {(1 - eps_empirical)**T_emp:.4f}")

T_emp_arr = np.array(T_empirical_values)
emp_arr = np.array(empirical_safety_rates)

# Theoretical exact: (1-ε)^T
T_theory = np.logspace(0, np.log10(T_MAX), 500)
P_exact = (1 - eps_empirical) ** T_theory
P_union = np.maximum(0, 1 - T_theory * eps_empirical)

ax3.semilogx(T_theory, P_exact, '-', color='#3498db', lw=2.5,
             label='Exact: $(1-\\epsilon)^T$', zorder=3)
ax3.semilogx(T_theory, P_union, '--', color='#e74c3c', lw=2,
             label='Union bound: $1 - T\\epsilon$', zorder=2)
ax3.scatter(T_emp_arr, emp_arr, color='#2c3e50', s=60, zorder=5,
            label=f'Empirical ({N_EMPIRICAL_TRIALS} trials)', edgecolors='white', lw=0.5)

# Error bars (binomial CI)
for i, (t, p) in enumerate(zip(T_emp_arr, emp_arr)):
    se = np.sqrt(p * (1 - p) / N_EMPIRICAL_TRIALS) * 1.96
    ax3.plot([t, t], [p - se, p + se], '-', color='#2c3e50', lw=1, alpha=0.5)

ax3.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, lw=1)
ax3.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, lw=1)

ax3.set_xlabel('Trajectory Length $T$ (steps)', fontsize=12)
ax3.set_ylabel('$P_{\\mathrm{safe}}(T)$', fontsize=12)
ax3.set_title('Empirical Validation\n($\\epsilon = 10^{-4}$, i.i.d. per-step failures)',
              fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, loc='lower left', framealpha=0.9)
ax3.grid(True, which='both', alpha=0.15)
ax3.set_ylim(-0.05, 1.05)
ax3.set_xlim(1, T_MAX)

# Key insight box
textstr = ('Key Insight:\n'
           '• Union bound is conservative\n'
           '  but dimension-free\n'
           '• Exact $(1-\\epsilon)^T$ is tighter\n'
           '  (i.i.d. assumption)\n'
           '• Re-certification resets the\n'
           '  error accumulation')
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95,
             edgecolor='#e74c3c', lw=1.5)
ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes, fontsize=8,
         verticalalignment='top', ha='right', bbox=props, zorder=10)

fig.suptitle('Experiment XIII: Union Bound Horizon Analysis '
             '(Lemma 2 — Trajectory-Level Safety)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_18.png'), dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("\n[OK] Saved figure_18.png")
