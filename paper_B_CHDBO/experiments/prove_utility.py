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

Output: proportionality_proof.png
"""

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

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Cosine similarity vs attack strength
sc = ax1.scatter(attack_strengths, similarities, c=similarities, cmap='RdYlGn', alpha=0.6, s=20)
x_line = np.linspace(0, 0.99, 200)
# For CBF-QP with gamma, the theoretical curve is not exactly sqrt(1-alpha^2)
# but we show the empirical trend + the geometric baseline
ax1.plot(x_line, np.sqrt(1 - x_line**2), "k--", linewidth=2,
         label=r"Geometric baseline $\sqrt{1-\alpha^2}$")

ax1.set_title(f'CBF-QP Proportional Safety Response (N={NUM_TRIALS})\n'
              f'$n={DIMENSIONS}$, $\\gamma={GAMMA}$', fontsize=14)
ax1.set_xlabel('Adversarial Intensity $\\alpha$ (0=Safe, 1=Malicious)', fontsize=12)
ax1.set_ylabel('Semantic Preservation (Cosine Similarity)', fontsize=12)
ax1.axhline(0.7, color='gray', linestyle=':', label="Acceptable Utility Threshold")
ax1.axvline(0.5, color='gray', linestyle=':', label="50/50 Intent Split")
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

# Panel 2: Intervention norm vs attack strength
ax2.scatter(attack_strengths, intervention_norms, c=intervention_norms,
            cmap='OrRd', alpha=0.6, s=20)
ax2.set_title(f'CBF-QP Intervention Magnitude\n'
              f'$\\|u^* - u_{{nom}}\\|$ vs Threat Level', fontsize=14)
ax2.set_xlabel('Adversarial Intensity $\\alpha$', fontsize=12)
ax2.set_ylabel('Intervention Norm $\\|u^* - u_{nom}\\|$', fontsize=12)
ax2.grid(True, alpha=0.3)

# Key result
mean_sim_safe = np.mean([s for a, s in zip(attack_strengths, similarities) if a < 0.2])
mean_sim_danger = np.mean([s for a, s in zip(attack_strengths, similarities) if a > 0.8])
textstr = (f'Safe inputs ($\\alpha < 0.2$): similarity = {mean_sim_safe:.3f}\n'
           f'Dangerous ($\\alpha > 0.8$): similarity = {mean_sim_danger:.3f}\n'
           f'Method: CBF-QP closed-form (Theorem 1)')
props = dict(boxstyle='round,pad=0.5', facecolor='honeydew', alpha=0.9)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure_3.png'), dpi=300, bbox_inches='tight')
print(f"[+] CBF-QP Proportionality proof saved as figure_3.png.")
print(f"    Mean similarity (safe, alpha<0.2):     {mean_sim_safe:.4f}")
print(f"    Mean similarity (dangerous, alpha>0.8): {mean_sim_danger:.4f}")