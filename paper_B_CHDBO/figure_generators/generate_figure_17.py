"""
Figure 17: Scenario Approach vs. MCBC — Sample Complexity Comparison (Experiment IX)
=====================================================================================
Compares the sample count required by:
  (a) Scenario approach (Campi & Garatti 2008): N = O(n/ε) for convex DESIGN
  (b) MCBC (this paper): N = O(1/(ε² ln(1/δ))) for fixed-barrier VERIFICATION

IMPORTANT DISTINCTION (explicitly stated):
  - Scenario approach DESIGNS + VERIFIES a safe controller simultaneously.
    Its sample complexity depends on the number of decision variables (n).
  - MCBC only VERIFIES a GIVEN barrier h(x). It does not design h(x).
    Its sample complexity does not include the cost of barrier design.

This comparison is valid when the barrier is given (e.g., hand-designed or
pre-trained). For fair comparison, both are shown at the VERIFICATION task.

Output: figure_17a–b.png (individual panels) + figure_17.png (combined legacy)
  (a) Sample complexity vs dimension
  (b) Total computational cost scaling

This is Experiment IX in the paper.

Usage:
  python generate_figure_17.py              # all panels + combined
  python generate_figure_17.py -a           # panel (a) only
  python generate_figure_17.py -b           # panel (b) only
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate Experiment IX MCBC vs Scenario figures.')
parser.add_argument('-a', action='store_true', help='Generate panel (a): Sample complexity')
parser.add_argument('-b', action='store_true', help='Generate panel (b): Total cost scaling')
args = parser.parse_args()
_any = args.a or args.b
_gen_a = args.a or not _any
_gen_b = args.b or not _any

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DIMENSIONS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Scenario approach parameters (Campi & Garatti 2008, Theorem 1)
# Required samples: N_scenario >= (2/ε) * (ln(1/β) + n)
# where n = number of decision variables, ε = violation probability, β = confidence
EPSILON_SCENARIO = 0.01        # violation probability
BETA_SCENARIO = 1e-6           # confidence parameter

# MCBC parameters (this paper, Theorem 1)
# Required samples: N_mcbc = 1/(2ε²) * ln(2/δ)
EPSILON_MCBC = 0.01            # failure fraction tolerance
DELTA_MCBC = 1e-6              # confidence parameter

# Additional: effective MCBC cost is N × n (per-sample cost is O(n))
# while scenario approach per-sample cost depends on the LP/QP solver

# ============================================================================
# COMPUTE THEORETICAL SAMPLE COUNTS
# ============================================================================
print("[*] Computing theoretical sample counts...")

n_arr = np.array(DIMENSIONS)

# Scenario approach: N_scenario >= (2/ε)(ln(1/β) + n)  [Campi & Garatti 2008]
N_scenario = (2.0 / EPSILON_SCENARIO) * (np.log(1.0 / BETA_SCENARIO) + n_arr)

# MCBC: N_mcbc = 1/(2ε²) × ln(2/δ) — independent of n
N_mcbc_base = (1.0 / (2 * EPSILON_MCBC**2)) * np.log(2.0 / DELTA_MCBC)
N_mcbc = np.full_like(n_arr, N_mcbc_base, dtype=float)

# Total computational cost
# Scenario: N_scenario × O(n³) for solving LP/QP per sample (conservative)
# MCBC: N_mcbc × O(n) for evaluating h(x) per sample
cost_scenario = N_scenario * n_arr**2   # Optimistic: O(n²) per sample
cost_mcbc = N_mcbc * n_arr              # O(n) per sample (barrier evaluation)

print(f"  MCBC base sample count: {N_mcbc_base:.0f} (fixed for all n)")
print(f"  Scenario approach samples:")
for n_val, ns in zip(DIMENSIONS, N_scenario):
    print(f"    n={n_val:>5}: N_scenario = {ns:,.0f}")

# ============================================================================
# EMPIRICAL VERIFICATION: Run MCBC at each dimension with a non-trivial barrier
# ============================================================================
print("\n[*] Running empirical MCBC verification at each dimension...")
print("    (Using barrier with planted narrow failure wedge)")

empirical_results = {}  # n -> (safety_rate, actual_N)

for n in DIMENSIONS:
    # Unit ball barrier with a planted failure wedge:
    # h(x) = (1 - ||x||^2) - D * exp(-(1 - cos(angle(x, spike)))^2 / (2*w^2))
    # The spike has negligible volume fraction on S^{n-1}, so MCBC should
    # correctly report near-100% safety rate (the failure IS sub-epsilon).
    # This validates that MCBC's Hoeffding estimate is accurate, not that the
    # barrier is trivially safe.
    
    spike_dir = np.zeros(n)
    spike_dir[0] = 1.0  # fixed spike direction
    spike_width = 0.05  # narrow angular width (radians)
    spike_depth = 2.0   # enough to make h < 0 at spike center
    
    N_samples = min(int(N_mcbc_base), 50000)  # cap for speed at low dims
    violations = 0
    
    for _ in range(N_samples):
        # Sample uniformly on S^{n-1} 
        z = np.random.randn(n)
        z = z / np.linalg.norm(z)
        
        # Evaluate non-trivial barrier
        cos_angle = np.dot(z, spike_dir)
        diff = 1.0 - cos_angle
        h_base = 1.0 - np.dot(z, z)  # = 0 on boundary (by construction)
        h_spike = -spike_depth * np.exp(-diff**2 / (2 * spike_width**2))
        h_val = h_base + h_spike  # h < 0 only near spike direction
        
        # Check barrier condition feasibility:
        # For single integrator, CBF-QP is always feasible IF h ≥ 0.
        # A violation occurs when h(x) < 0 at this boundary point.
        grad_h = -2 * z - spike_depth * np.exp(-diff**2 / (2 * spike_width**2)) * \
                 (diff / (spike_width**2)) * (spike_dir - cos_angle * z)
        gh_norm = np.linalg.norm(grad_h)
        
        if h_val < 0 and gh_norm > 1e-10:
            violations += 1
    
    safety_rate = 1.0 - violations / N_samples
    empirical_results[n] = (safety_rate, N_samples, violations)
    print(f"    n={n:>5}: safety_rate={safety_rate:.6f}, violations={violations}/{N_samples}")

# ============================================================================
# PLOTTING — HELPERS
# ============================================================================
BG_COLOR = '#f8f9fa'
script_dir = os.path.dirname(os.path.abspath(__file__))
n_theory = np.logspace(np.log10(2), np.log10(1024), 200)


def _setup_ax(ax):
    ax.set_facecolor(BG_COLOR)
    return ax


def _plot_a(ax):
    """(a) Sample count vs. dimension."""
    ax.loglog(n_arr, N_scenario, 'o-', color='#e74c3c', lw=2.5, markersize=8,
              label='Scenario Approach\n$N = \\frac{2}{\\epsilon}(\\ln\\frac{1}{\\beta} + n)$',
              zorder=5)
    ax.loglog(n_arr, N_mcbc, 's-', color='#2ecc71', lw=2.5, markersize=8,
              label='MCBC (this paper)\n$N = \\frac{1}{2\\epsilon^2}\\ln\\frac{2}{\\delta}$',
              zorder=5)
    ax.loglog(n_theory,
              (2.0 / EPSILON_SCENARIO) * (np.log(1.0 / BETA_SCENARIO) + n_theory),
              '--', color='#e74c3c', alpha=0.3, lw=1.5, label='_nolegend_')
    ax.axhline(y=N_mcbc_base, color='#2ecc71', linestyle='--', alpha=0.4, lw=1.5)
    idx_1024 = DIMENSIONS.index(1024)
    ratio_1024 = N_scenario[idx_1024] / N_mcbc_base
    ax.annotate(f'{ratio_1024:.1f}× fewer\nsamples (MCBC)',
                xy=(1024, N_mcbc_base), xytext=(100, N_mcbc_base * 0.12),
                fontsize=10, fontweight='bold', color='#27ae60',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
    ax.axvline(x=349, color='gray', linestyle=':', alpha=0.5, lw=1.5)
    ax.text(349, ax.get_ylim()[0] * 2.0, 'Crossover\n$n \\approx 350$',
            fontsize=9, ha='center', va='bottom', color='gray')
    ax.set_xlabel('State Space Dimension ($n$)', fontsize=12)
    ax.set_ylabel('Required Samples ($N$)', fontsize=12)
    ax.set_title('Sample Complexity: Verification Comparison\n'
                 f'($\\epsilon = {EPSILON_SCENARIO}$, '
                 f'$\\delta = \\beta = 10^{{-6}}$)\n'
                 'Scenario → design; MCBC → verify given barrier',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.15)


def _plot_b(ax):
    """(b) Total computational cost."""
    ax.loglog(n_arr, cost_scenario, 'o-', color='#e74c3c', lw=2.5, markersize=8,
              label='Scenario: $N \\times O(n^2)$', zorder=5)
    ax.loglog(n_arr, cost_mcbc, 's-', color='#2ecc71', lw=2.5, markersize=8,
              label='MCBC: $N \\times O(n)$', zorder=5)
    ax.loglog(n_theory, cost_scenario[0] * (n_theory / n_arr[0])**3,
              '--', color='#e74c3c', alpha=0.2, lw=1.5, label='$O(n^3)$ ref')
    ax.loglog(n_theory, cost_mcbc[0] * (n_theory / n_arr[0])**1,
              '--', color='#2ecc71', alpha=0.2, lw=1.5, label='$O(n)$ ref')
    ax.axhline(y=1e8, color='#3498db', linestyle=':', lw=2, alpha=0.6,
               label='Approx. 1-second budget')
    ax.set_xlabel('State Space Dimension ($n$)', fontsize=12)
    ax.set_ylabel('Total Computational Cost (FLOPs proxy)', fontsize=12)
    ax.set_title('Total Verification Cost Scaling\n(samples × per-sample cost)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.15)
    textstr = ('VERIFICATION-ONLY comparison:\n'
               '• MCBC sample count is $O(1)$ in $n$\n'
               '  (for fixed $L_h$ and $\\varepsilon_s$)\n'
               '• Scenario approach is $O(n)$ in $n$\n'
               '• Total cost: MCBC $O(n)$ vs. Scenario $O(n^3)$\n'
               '\n'
               'Important caveat:\n'
               '• MCBC verifies a GIVEN $h(x)$\n'
               '• Scenario designs $h(x)$ simultaneously\n'
               '• Barrier design cost not included in MCBC')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95,
                 edgecolor='#2ecc71', lw=1.5)
    ax.text(0.02, 0.75, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', ha='left', bbox=props, zorder=10)


# ============================================================================
# INDIVIDUAL PANEL FIGURES (figure_17a–b.png)
# ============================================================================
panel_map = {'a': _plot_a, 'b': _plot_b}
panel_flags = {'a': _gen_a, 'b': _gen_b}

for key in 'ab':
    if not panel_flags[key]:
        continue
    fig_i, ax_i = plt.subplots(figsize=(8, 6))
    _setup_ax(ax_i)
    panel_map[key](ax_i)
    fig_i.tight_layout()
    fname = os.path.join(script_dir, f'figure_17{key}.png')
    fig_i.savefig(fname, dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_i)
    print(f'[OK] Saved figure_17{key}.png')

# ============================================================================
# COMBINED LEGACY FIGURE (only when no flags specified)
# ============================================================================
if not _any:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _setup_ax(ax1)
    _setup_ax(ax2)
    _plot_a(ax1)
    _plot_b(ax2)
    fig.suptitle('Experiment IX: MCBC (Verification) vs. Scenario Approach (Design+Verify)\n'
                 'Note: Different tasks — MCBC verifies a given barrier; Scenario designs one',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(script_dir, 'figure_17.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] Saved figure_17.png (combined)')

print(f"\nConclusion: MCBC requires {N_mcbc_base:.0f} samples regardless of dimension,")
print(f"while scenario approach requires {N_scenario[-1]:,.0f} samples at n={DIMENSIONS[-1]}.")
print(f"MCBC advantage: {N_scenario[-1] / N_mcbc_base:.0f}× fewer samples at n={DIMENSIONS[-1]}.")
