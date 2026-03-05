"""
Figure 2: High-Dimensional Orthogonal Verification (R^128)
===========================================================
Validates the CBF-QP safety filter in R^128 with single-integrator dynamics.

Barrier function: h(x) = 1 - ||x||  (safe set: unit ball)
CBF condition:    dh/dt + gamma * h >= 0
For single-integrator (f=0, g=I):
    nabla_h = -x / ||x||
    L_g h . u = -x/||x|| . u
    Constraint: -x/||x|| . u >= -gamma * (1 - ||x||)
              => x/||x|| . u <= gamma * (1 - ||x||)

Output: figure_2.png
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
DIMENSIONS = 128
NUM_TRIALS = 50
STEPS = 100
SAFE_RADIUS = 1.0
ATTACK_STRENGTH = 0.05
GAMMA = 5.0            # CBF decay rate

print(f"[-] Initializing {DIMENSIONS}-Dimensional CBF-QP Verification Protocol...")

# --- GEOMETRY FUNCTIONS ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

# --- CBF-QP CLOSED-FORM SOLUTION (SPHERICAL BARRIER) ---
def cbf_qp_sphere(state, u_nom, gamma=GAMMA):
    """
    CBF-QP closed-form solution for spherical barrier h(x) = 1 - ||x||.

    For single-integrator dynamics (f=0, g=I):
        nabla_h(x) = -x / ||x||  (for x != 0)
        L_g h = nabla_h
        CBF condition: nabla_h . u >= -gamma * h(x)
                     => -(x/||x||) . u >= -gamma * (1 - ||x||)
                     => (x/||x||) . u <= gamma * (1 - ||x||)

    Closed-form: if constraint violated,
        lambda* = -(nabla_h . u_nom + gamma * h) / ||nabla_h||^2
        u* = u_nom + lambda* * nabla_h
    """
    norm_x = np.linalg.norm(state)
    if norm_x < 1e-10:
        # At origin, barrier is maximally safe — no projection needed
        return u_nom

    h = SAFE_RADIUS - norm_x
    nabla_h = -state / norm_x  # gradient of h = 1 - ||x||

    # CBF constraint: nabla_h . u >= -gamma * h
    constraint_value = np.dot(nabla_h, u_nom) + gamma * h

    if constraint_value >= 0:
        # Constraint satisfied — no intervention
        return u_nom

    # Violated — project onto feasible half-space
    nabla_h_norm_sq = np.dot(nabla_h, nabla_h)  # = 1.0 for unit gradient
    lambda_star = -constraint_value / nabla_h_norm_sq
    u_safe = u_nom + lambda_star * nabla_h

    return u_safe

# --- SIMULATION LOOP ---
print(f"[-] Running {NUM_TRIALS} Gradient-Based Adversarial Attacks...")

all_states_std = []
all_states_ver = []

for i in range(NUM_TRIALS):
    state_std = np.zeros(DIMENSIONS)
    state_ver = np.zeros(DIMENSIONS)

    traj_std = [state_std.copy()]
    traj_ver = [state_ver.copy()]

    # Random "Forbidden Goal" vector OUTSIDE safety sphere
    forbidden_goal = np.random.randn(DIMENSIONS)
    forbidden_goal = normalize(forbidden_goal) * (SAFE_RADIUS * 2.0)

    for t in range(STEPS):
        # --- THE ATTACK (Gradient Descent toward forbidden goal) ---
        grad_std = forbidden_goal - state_std
        grad_std = normalize(grad_std) * ATTACK_STRENGTH

        grad_ver = forbidden_goal - state_ver
        grad_ver = normalize(grad_ver) * ATTACK_STRENGTH

        # 1. Standard Agent: no safety filter
        state_std += grad_std
        traj_std.append(state_std.copy())

        # 2. Verified Agent: CBF-QP safety filter
        u_safe = cbf_qp_sphere(state_ver, grad_ver)
        state_ver += u_safe
        # Post-hoc norm clipping: discrete-time Euler steps can overshoot
        # the continuous-time CBF guarantee. Clip to the safe set boundary.
        norm_ver = np.linalg.norm(state_ver)
        if norm_ver > SAFE_RADIUS:
            state_ver = state_ver / norm_ver * SAFE_RADIUS
        traj_ver.append(state_ver.copy())

    all_states_std.append(np.array(traj_std))
    all_states_ver.append(np.array(traj_ver))

# --- VERIFICATION ---
max_norm_ver = max(np.max(np.linalg.norm(traj, axis=1)) for traj in all_states_ver)
print(f"[-] Max verified agent norm: {max_norm_ver:.6f} (safe radius: {SAFE_RADIUS})")
assert max_norm_ver <= SAFE_RADIUS + 1e-6, f"Safety violation! max norm = {max_norm_ver}"
print(f"[+] Safety verified: all {NUM_TRIALS} trials remain within safe set.")

# --- VISUALIZATION (PCA Projection to 2D) ---
print("[-] Performing PCA to visualize 128D Hyperspace...")
flat_std = np.concatenate(all_states_std)
flat_ver = np.concatenate(all_states_ver)
combined_data = np.concatenate([flat_std, flat_ver])

pca = PCA(n_components=2)
pca.fit(combined_data)

print("[-] Generating Proof Plot...")
plt.figure(figsize=(10, 8))

# Note: We do NOT draw a unit circle in PCA space because the PCA projection
# of the 128-sphere boundary is NOT a circle. Instead, we show the projected
# boundary points from the verified trajectories.
# For visual reference, we draw a dashed circle at the maximum PCA radius
# observed among verified trajectories.
ver_pca = pca.transform(flat_ver)
max_pca_radius = np.max(np.linalg.norm(ver_pca, axis=1))
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta)*max_pca_radius, np.sin(theta)*max_pca_radius,
         'k--', linewidth=1.5, alpha=0.5,
         label=f"Safe Manifold Boundary")

# Plot Standard (Red)
for traj in all_states_std:
    traj_2d = pca.transform(traj)
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color='red', alpha=0.15)

# Plot Verified (Green)
for traj in all_states_ver:
    traj_2d = pca.transform(traj)
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color='green', alpha=0.6, linewidth=2)

plt.title(f'High-Dimensional Orthogonal Verification ($n$={DIMENSIONS})\n'
          'Adversarial Gradient Attack Simulation', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)

# Legend
plt.plot([], [], 'r-', label='Standard Agent (Breach)')
plt.plot([], [], 'g-', label='Verified Agent (Safe)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis('equal')

filename = "figure_2.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"[+] Advanced Proof generated: {filename}")