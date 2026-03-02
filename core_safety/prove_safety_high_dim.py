import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
DIMENSIONS = 128          # We are now in Hyperspace (Emulating Embeddings)
NUM_TRIALS = 50
STEPS = 100
SAFE_RADIUS = 1.0         # The "Golden Manifold" is a unit hypersphere
ATTACK_STRENGTH = 0.05    # How hard the adversary pushes

print(f"[-] Initializing {DIMENSIONS}-Dimensional Orthogonal Verification Protocol...")

# --- GEOMETRY FUNCTIONS ---
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: return v
    return v / norm

def is_safe_hypersphere(state):
    """Checks if state is inside the Safe Unit Hypersphere."""
    # Distance from origin must be <= SAFE_RADIUS
    return np.linalg.norm(state) <= SAFE_RADIUS

def orthogonal_projection_hypersphere(state, velocity):
    """
    If velocity pushes state OUT of the sphere, project velocity
    onto the TANGENT PLANE of the sphere surface.
    """
    current_norm = np.linalg.norm(state)
    
    # Predict next position
    next_state = state + velocity
    next_norm = np.linalg.norm(next_state)
    
    if next_norm > SAFE_RADIUS:
        # We are breaching the manifold.
        # 1. Calculate Normal Vector (direction from origin to surface)
        normal = normalize(state)
        
        # 2. Project velocity onto the normal to see how much is "pushing out"
        push_out = np.dot(velocity, normal)
        
        # 3. If pushing out, remove that component (Orthonormalization)
        if push_out > 0:
            velocity = velocity - (push_out * normal)
            
    return velocity

# --- SIMULATION LOOP ---
print(f"[-] Running {NUM_TRIALS} Gradient-Based Adversarial Attacks...")

# Storage for PCA visualization
all_states_std = []
all_states_ver = []

for i in range(NUM_TRIALS):
    # Start at origin
    state_std = np.zeros(DIMENSIONS)
    state_ver = np.zeros(DIMENSIONS)
    
    traj_std = [state_std.copy()]
    traj_ver = [state_ver.copy()]
    
    # Generate a random "Forbidden Goal" vector for this trial
    # The agent REALLY wants to go here.
    forbidden_goal = np.random.randn(DIMENSIONS)
    forbidden_goal = normalize(forbidden_goal) * (SAFE_RADIUS * 2.0) # Goal is OUTSIDE safety
    
    for t in range(STEPS):
        # --- THE ATTACK (Gradient Descent) ---
        # Calculate direction toward the forbidden goal
        grad_std = forbidden_goal - state_std
        grad_std = normalize(grad_std) * ATTACK_STRENGTH
        
        grad_ver = forbidden_goal - state_ver
        grad_ver = normalize(grad_ver) * ATTACK_STRENGTH
        
        # 1. Standard Agent: Just goes for it.
        state_std += grad_std
        traj_std.append(state_std.copy())
        
        # 2. Verified Agent: Tries to go, but gets projected.
        # Check safety of the INTENDED move
        safe_velocity = orthogonal_projection_hypersphere(state_ver, grad_ver)
        state_ver += safe_velocity
        # Re-project to sphere surface to prevent numerical drift
        norm_ver = np.linalg.norm(state_ver)
        if norm_ver > SAFE_RADIUS:
            state_ver = (state_ver / norm_ver) * SAFE_RADIUS
        traj_ver.append(state_ver.copy())

    all_states_std.append(np.array(traj_std))
    all_states_ver.append(np.array(traj_ver))

# --- VISUALIZATION (PCA Projection to 2D) ---
print("[-] Performing PCA to visualize 128D Hyperspace...")
# Flatten data to fit PCA
flat_std = np.concatenate(all_states_std)
flat_ver = np.concatenate(all_states_ver)
combined_data = np.concatenate([flat_std, flat_ver])

pca = PCA(n_components=2)
pca.fit(combined_data)

print("[-] Generating Proof Plot...")
plt.figure(figsize=(10, 8))

# Draw the "Safe Boundary" in PCA space (Approximate)
theta = np.linspace(0, 2*np.pi, 100)
# We approximate the unit circle in PCA space for visual reference
plt.plot(np.cos(theta)*SAFE_RADIUS, np.sin(theta)*SAFE_RADIUS, 'k--', linewidth=2, label="Safe Manifold Boundary")

# Plot Standard (Red)
for traj in all_states_std:
    traj_2d = pca.transform(traj)
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color='red', alpha=0.15)

# Plot Verified (Green)
for traj in all_states_ver:
    traj_2d = pca.transform(traj)
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color='green', alpha=0.6, linewidth=2)

plt.title(f'High-Dimensional Orthogonal Verification (D={DIMENSIONS})\nAdversarial Gradient Attack Simulation', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)


# Add dummy lines for legend
plt.plot([], [], 'r-', label='Standard Agent (Breach)')
plt.plot([], [], 'g-', label='Verified Agent (Safe)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis('equal') # Important for geometry!

filename = "core_safety/high_dim_safety_proof.png"
plt.savefig(filename, dpi=300)
print(f"[+] Advanced Proof generated: {filename}")