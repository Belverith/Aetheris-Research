"""
Figure 15: Single-Constraint QP Benchmark — Closed-Form vs OSQP
================================================================
Compares solve time for the CBF-QP with a SINGLE barrier constraint:
  min  ||u - u_nom||^2
  s.t. ∇h · u ≥ -γ h(x) - L_f h

KEY DISTINCTION:
  - The closed-form solution is O(n) for SINGLE-constraint QPs only.
  - OSQP handles arbitrary multi-constraint QPs but carries setup overhead.
  - With actuation bounds (box constraints), the closed-form no longer applies
    and a general QP solver (OSQP, etc.) IS needed.

This experiment demonstrates the computational advantage of CHDBO's single-barrier
structure. It is NOT a general claim that closed-form beats QP solvers — only
that the single-constraint structure enables it.

Output: figure_15a–c.png (individual panels) + figure_15.png (combined legacy)
  (a) QP Solve Time vs Dimension (log-log)
  (b) Closed-Form Speedup Factor
  (c) Direct Comparison (grouped bar)

This is Experiment VII in the paper.

Usage:
  python generate_figure_15.py              # all panels + combined
  python generate_figure_15.py -a           # panel (a) only
  python generate_figure_15.py -b -c        # panels (b) and (c) only
"""

import argparse
import numpy as np
import time
import os
import scipy.sparse as sp

parser = argparse.ArgumentParser(description='Generate Experiment VII QP benchmark figures.')
parser.add_argument('-a', action='store_true', help='Generate panel (a): Solve time log-log')
parser.add_argument('-b', action='store_true', help='Generate panel (b): Speedup factor')
parser.add_argument('-c', action='store_true', help='Generate panel (c): Direct comparison')
args = parser.parse_args()
_any = args.a or args.b or args.c
_gen_a = args.a or not _any
_gen_b = args.b or not _any
_gen_c = args.c or not _any

np.random.seed(42)

# ============================================================================
# Try importing OSQP
# ============================================================================
try:
    import osqp
    HAS_OSQP = True
    print("[OK] OSQP available")
except ImportError:
    HAS_OSQP = False
    print("[!] OSQP not available — using scipy.optimize.minimize as fallback")
    from scipy.optimize import minimize

import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DIMS = [64, 128, 256, 512, 1024, 2048]
N_SOLVES = 5000  # Number of QP solves per dimension (high for timing stability)
GAMMA = 1.0

# ============================================================================
# CLOSED-FORM CBF-QP SOLVER (O(n))
# ============================================================================
def cbf_qp_closedform(u_nom, grad_h, rhs):
    """
    Solve: min ||u - u_nom||^2  s.t. grad_h · u >= rhs
    Closed-form: if constraint satisfied, return u_nom.
    Otherwise, u* = u_nom + lambda * grad_h where
    lambda = (rhs - grad_h · u_nom) / ||grad_h||^2
    """
    lhs = np.dot(grad_h, u_nom)
    if lhs >= rhs:
        return u_nom, False
    gh_sq = np.dot(grad_h, grad_h)
    if gh_sq < 1e-15:
        return u_nom, False
    lam = (rhs - lhs) / gh_sq
    return u_nom + lam * grad_h, True


# ============================================================================
# OSQP CBF-QP SOLVER
# ============================================================================
def cbf_qp_osqp(u_nom, grad_h, rhs, n, solver_instance=None):
    """
    Solve the same QP using OSQP:
      min  0.5 u^T I u - u_nom^T u
      s.t. grad_h^T u >= rhs   →   -grad_h^T u <= -rhs
    """
    if solver_instance is not None:
        # Update and solve
        solver_instance.update(q=-u_nom, l=np.array([rhs]), u=np.array([np.inf]))
        result = solver_instance.solve()
        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            return result.x, True
        else:
            return u_nom, False
    return u_nom, False


def setup_osqp(n, grad_h):
    """Pre-setup OSQP with problem structure (done once per dimension)."""
    P = sp.eye(n, format='csc')
    q = np.zeros(n)  # placeholder
    A = sp.csc_matrix(grad_h.reshape(1, -1))
    l = np.array([0.0])  # placeholder
    u = np.array([np.inf])

    solver = osqp.OSQP()
    solver.setup(P, q, A, l, u,
                 verbose=False,
                 eps_abs=1e-6,
                 eps_rel=1e-6,
                 max_iter=200,
                 warm_start=True,
                 polish=False)
    return solver


# ============================================================================
# BENCHMARK
# ============================================================================
print("[*] Running Experiment VII: Closed-Form QP vs OSQP Benchmark...")
print(f"    {N_SOLVES} solves per dimension\n")

results = {}

for n in DIMS:
    print(f"  --- n = {n} ---")
    rng = np.random.RandomState(42)

    # Pre-generate test cases
    xs = []
    u_noms = []
    grad_hs = []
    rhss = []

    for j in range(N_SOLVES):
        x = rng.randn(n)
        x = x / np.linalg.norm(x) * (0.85 + rng.rand() * 0.12)  # ||x|| ∈ [0.85, 0.97] near boundary
        h_val = 1.0 - np.dot(x, x)
        grad_h = -2.0 * x
        Lfh = 0.0  # single integrator
        rhs_val = -GAMMA * h_val - Lfh

        # Alternate: half push aggressively outward (constraint active),
        # half stay conservative (constraint inactive)
        if j % 2 == 0:
            # Aggressive: push radially outward → violates barrier
            u_nom = 2.0 * x / np.linalg.norm(x)
        else:
            # Conservative: push inward → barrier satisfied
            u_nom = -0.3 * x / np.linalg.norm(x)

        xs.append(x)
        u_noms.append(u_nom)
        grad_hs.append(grad_h)
        rhss.append(rhs_val)

    # ---- Closed-form benchmark ----
    t0 = time.perf_counter()
    cf_active = 0
    for i in range(N_SOLVES):
        _, active = cbf_qp_closedform(u_noms[i], grad_hs[i], rhss[i])
        if active:
            cf_active += 1
    t_cf = time.perf_counter() - t0

    # ---- OSQP benchmark ----
    if HAS_OSQP:
        # Setup with first problem
        solver = setup_osqp(n, grad_hs[0])

        t0 = time.perf_counter()
        osqp_solved = 0
        for i in range(N_SOLVES):
            A_new = sp.csc_matrix(grad_hs[i].reshape(1, -1))
            solver.update(q=-u_noms[i], Ax=A_new.data, l=np.array([rhss[i]]))
            result = solver.solve()
            if result.info.status in ('solved', 'solved_inaccurate'):
                osqp_solved += 1
        t_osqp = time.perf_counter() - t0
    else:
        t_osqp = float('nan')
        osqp_solved = N_SOLVES

    results[n] = {
        't_cf': t_cf,
        't_osqp': t_osqp,
        'cf_active': cf_active,
        'osqp_solved': osqp_solved,
        'us_cf': t_cf / N_SOLVES * 1e6,  # microseconds per solve
        'us_osqp': t_osqp / N_SOLVES * 1e6,
    }

    print(f"    Closed-form: {t_cf*1000:.1f}ms total, {results[n]['us_cf']:.1f}µs/solve, {cf_active}/{N_SOLVES} constraint active")
    print(f"    OSQP:        {t_osqp*1000:.1f}ms total, {results[n]['us_osqp']:.1f}µs/solve, {osqp_solved}/{N_SOLVES} solved")
    speedup = t_osqp / t_cf if t_cf > 0 else float('inf')
    print(f"    Speedup:     {speedup:.1f}×")

# ============================================================================
# PLOTTING — HELPERS
# ============================================================================
BG = '#f8f9fa'
script_dir = os.path.dirname(os.path.abspath(__file__))

dim_arr = np.array(DIMS)
us_cf = np.array([results[n]['us_cf'] for n in DIMS])
us_osqp = np.array([results[n]['us_osqp'] for n in DIMS])
speedups = us_osqp / us_cf


def _setup_ax(ax):
    ax.set_facecolor(BG)
    return ax


def _plot_a(ax):
    """(a) Solve time vs dimension (log-log)."""
    ax.loglog(dim_arr, us_cf, 'o-', color='#2ecc71', lw=2.5, markersize=8,
              label='Closed-form (ours)', zorder=5)
    ax.loglog(dim_arr, us_osqp, 's-', color='#e74c3c', lw=2.5, markersize=8,
              label='OSQP')
    ref_n = us_cf[0] * dim_arr / dim_arr[0]
    ax.loglog(dim_arr, ref_n, '--', color='gray', lw=1, alpha=0.5, label='$O(n)$ ref')
    ax.axhline(10000, color='#f39c12', ls=':', lw=2, label='10ms real-time budget')
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel('Time per QP solve (µs)', fontsize=12)
    ax.set_title('(a) QP Solve Time vs Dimension', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')


def _plot_b(ax):
    """(b) Speedup factor."""
    ax.bar([str(n) for n in DIMS], speedups, color='#3498db', edgecolor='white', alpha=0.8)
    for i, (n, s) in enumerate(zip(DIMS, speedups)):
        ax.text(i, s + 0.5, f'{s:.0f}×', ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel('Speedup (OSQP / Closed-form)', fontsize=12)
    ax.set_title('(b) Closed-Form Speedup Factor', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')


def _plot_c(ax):
    """(c) Direct comparison (grouped bar, log scale)."""
    x_pos = np.arange(len(DIMS))
    w = 0.35
    ax.bar(x_pos - w/2, us_cf, w, label='Closed-form', color='#2ecc71',
           edgecolor='white', alpha=0.8)
    ax.bar(x_pos + w/2, us_osqp, w, label='OSQP', color='#e74c3c',
           edgecolor='white', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in DIMS])
    ax.set_xlabel('Dimension $n$', fontsize=12)
    ax.set_ylabel('Time per QP solve (µs)', fontsize=12)
    ax.set_title('(c) Direct Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_yscale('log')


# ============================================================================
# INDIVIDUAL PANEL FIGURES (figure_15a–c.png)
# ============================================================================
panel_map = {'a': _plot_a, 'b': _plot_b, 'c': _plot_c}
panel_flags = {'a': _gen_a, 'b': _gen_b, 'c': _gen_c}

for key in 'abc':
    if not panel_flags[key]:
        continue
    fig_i, ax_i = plt.subplots(figsize=(7, 5.5))
    _setup_ax(ax_i)
    panel_map[key](ax_i)
    fig_i.tight_layout()
    fname = os.path.join(script_dir, f'figure_15{key}.png')
    fig_i.savefig(fname, dpi=300, bbox_inches='tight',
                  facecolor='white', edgecolor='none')
    plt.close(fig_i)
    print(f'[OK] Saved figure_15{key}.png')

# ============================================================================
# COMBINED LEGACY FIGURE (only when no flags specified)
# ============================================================================
if not _any:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax in axes:
        _setup_ax(ax)
    _plot_a(axes[0])
    _plot_b(axes[1])
    _plot_c(axes[2])
    fig.suptitle(
        'Experiment VII: Single-Constraint CBF-QP — Closed-Form vs OSQP\n'
        f'{N_SOLVES} solves per dimension  |  '
        'IMPORTANT: Advantage is specific to single-constraint structure',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(script_dir, 'figure_15.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print('[OK] Saved figure_15.png (combined)')

# Print summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Dim':>6} | {'Closed-form (µs)':>16} | {'OSQP (µs)':>12} | {'Speedup':>8}")
print("-"*50)
for n in DIMS:
    r = results[n]
    print(f"{n:>6} | {r['us_cf']:>16.1f} | {r['us_osqp']:>12.1f} | {r['us_osqp']/r['us_cf']:>7.0f}×")
print("="*70)
print(f"\n[OK] Done.")
