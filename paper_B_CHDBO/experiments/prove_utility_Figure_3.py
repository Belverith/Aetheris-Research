"""
Proportional Safety Response: CBF-QP Utility Preservation Test
================================================================
Demonstrates that the CBF-QP filter preserves utility proportionally:
  - Safe inputs (low alpha) → minimal intervention (high cosine similarity)
  - Dangerous inputs (high alpha) → larger intervention (lower similarity)

Uses the SAME CBF-QP closed-form solution as the paper (not orthogonal projection):
  min  ||u - u_nom||^2
  s.t. nabla_h · u >= -gamma * h(x)

For spherical barrier h(x) = 1 - ||x||, nabla_h = -x/||x||.

Output: figure_3a.png, figure_3b.png  (or figure_3.png combined)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# --- CONFIGURATION ---
DIMENSIONS = 128
NUM_TRIALS = 1000
SAFE_RADIUS = 1.0
GAMMA = 5.0  # CBF decay rate — matches prove_safety_high_dim.py

print(f"[-] Initializing PROPORTIONALITY Stress Test (CBF-QP, n={DIMENSIONS})...")

def normalize(v):
    n = norm(v)
    return v / n if n > 0 else v

def cosine_similarity(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return np.dot(a, b) / (na * nb)

def cbf_qp_sphere(state, u_nom, gamma=GAMMA):
    """
    CBF-QP closed-form solution for spherical barrier h(x) = 1 - ||x||.
    For single-integrator dynamics (f=0, g=I):
        nabla_h = -x / ||x||
        CBF condition: nabla_h · u >= -gamma * h(x)
    Closed-form:
        if satisfied, return u_nom
        else u* = u_nom + lambda* * nabla_h
        where lambda* = -(nabla_h · u_nom + gamma * h) / ||nabla_h||^2
    """
    norm_x = norm(state)
    if norm_x < 1e-10:
        return u_nom  # at origin, maximally safe

    h = SAFE_RADIUS - norm_x
    nabla_h = -state / norm_x  # gradient of h = 1 - ||x||

    # CBF constraint: nabla_h · u >= -gamma * h
    constraint_value = np.dot(nabla_h, u_nom) + gamma * h

    if constraint_value >= 0:
        return u_nom  # constraint satisfied

    # Violated — project onto feasible half-space
    nabla_h_norm_sq = np.dot(nabla_h, nabla_h)  # = 1.0 for unit gradient
    lambda_star = -constraint_value / nabla_h_norm_sq
    u_safe = u_nom + lambda_star * nabla_h

    return u_safe

attack_strengths = []
similarities = []
intervention_norms = []

for i in range(NUM_TRIALS):
    # 1. State near boundary (surface of safe set)
    state = normalize(np.random.randn(DIMENSIONS)) * SAFE_RADIUS * 0.95

    # 2. Components
    raw_safe = np.random.randn(DIMENSIONS)
    tangent = normalize(raw_safe - np.dot(raw_safe, normalize(state)) * normalize(state))
    radial = normalize(state)  # outward = dangerous direction

    # 3. Variable Attack Strength (0.0 to 1.0)
    alpha = np.random.uniform(0.0, 1.0)

    # 4. Controlled Noise (10% of signal magnitude)
    noise = np.random.randn(DIMENSIONS)
    noise = normalize(noise) * 0.1

    # Construct the nominal control with angular decomposition:
    # u_nom = sqrt(1 - alpha^2) * tangent + alpha * radial + noise
    original_velocity = (tangent * np.sqrt(1.0 - alpha**2) + radial * alpha) + noise

    # 5. CBF-QP Correction (the actual method from the paper)
    corrected_velocity = cbf_qp_sphere(state, original_velocity.copy())

    # 6. Measure preservation
    sim = cosine_similarity(original_velocity, corrected_velocity)
    interv_norm = norm(corrected_velocity - original_velocity)

    attack_strengths.append(alpha)
    similarities.append(sim)
    intervention_norms.append(interv_norm)

# ──────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────
BG = '#f8f9fa'
_DIR = os.path.dirname(os.path.abspath(__file__))

SUPTITLE = (
    f'Experiment III: Proportional Safety Response\n'
    f'$n={DIMENSIONS}$, $\\gamma={GAMMA}$, $N={NUM_TRIALS}$'
)

# Precompute statistics for annotation
mean_sim_safe = np.mean([s for a, s in zip(attack_strengths, similarities) if a < 0.2])
mean_sim_danger = np.mean([s for a, s in zip(attack_strengths, similarities) if a > 0.8])

print(f"    Mean similarity (safe, alpha<0.2):     {mean_sim_safe:.4f}")
print(f"    Mean similarity (dangerous, alpha>0.8): {mean_sim_danger:.4f}")


def _plot_a(ax):
    """(a) Cosine similarity vs adversarial intensity."""
    ax.scatter(attack_strengths, similarities, c=similarities,
               cmap='RdYlGn', alpha=0.6, s=20)
    x_line = np.linspace(0, 0.99, 200)
    ax.plot(x_line, np.sqrt(1 - x_line**2), "k--", linewidth=2,
            label=r"Boundary limit $\sqrt{1-\alpha^2}$  (zero margin)")
    ax.set_title('(a) Semantic Preservation vs Threat Level',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Adversarial Intensity $\\alpha$ (0=Safe, 1=Malicious)', fontsize=12)
    ax.set_ylabel('Semantic Preservation (Cosine Similarity)', fontsize=12)
    ax.axhline(0.7, color='gray', linestyle=':', label="Acceptable Utility Threshold")
    ax.axvline(0.5, color='gray', linestyle=':', label="50/50 Intent Split")
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG)
    # Annotation: explain the gap
    ax.text(0.98, 0.02,
            f'States at $0.95R$ (margin $h=0.05$)\n'
            f'Gap above curve = margin-\n'
            f'dependent CBF slack\n'
            f'Safe ($\\alpha<0.2$): sim = {mean_sim_safe:.3f}\n'
            f'Dangerous ($\\alpha>0.8$): sim = {mean_sim_danger:.3f}',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))


def _plot_b(ax):
    """(b) Intervention norm vs adversarial intensity."""
    ax.scatter(attack_strengths, intervention_norms, c=intervention_norms,
               cmap='OrRd', alpha=0.6, s=20)
    ax.set_title('(b) CBF-QP Intervention Magnitude',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Adversarial Intensity $\\alpha$', fontsize=12)
    ax.set_ylabel('Intervention Norm $\\|u^* - u_{nom}\\|$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG)
    textstr = (f'Safe inputs ($\\alpha < 0.2$): sim = {mean_sim_safe:.3f}\n'
               f'Dangerous ($\\alpha > 0.8$): sim = {mean_sim_danger:.3f}\n'
               f'Method: CBF-QP closed-form')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='honeydew', alpha=0.9))


def _save_single(plot_fn, tag):
    """Save a single panel as figure_3{tag}.png."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_fn(ax)
    fig.suptitle(SUPTITLE, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(_DIR, f'figure_3{tag}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────────────
# CLI & GENERATION
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Generate Figure 3 panels')
parser.add_argument('-a', action='store_true', help='Panel (a) Similarity scatter')
parser.add_argument('-b', action='store_true', help='Panel (b) Intervention norm')
args = parser.parse_args()

_any = args.a or args.b
_gen_a = args.a or not _any
_gen_b = args.b or not _any

print("\nGenerating figure_3 panels...")

if _gen_a:
    _save_single(_plot_a, 'a')
if _gen_b:
    _save_single(_plot_b, 'b')

# Combined legacy figure when no flags specified
if not _any:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    _plot_a(ax1)
    _plot_b(ax2)
    fig.suptitle(SUPTITLE, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(_DIR, 'figure_3.png')
    fig.savefig(path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  Saved figure_3.png (combined)")

print("\n[OK] Experiment III complete.")