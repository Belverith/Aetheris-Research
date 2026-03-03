import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

# --- CONFIGURATION ---
NUM_TRIALS = 50
STEPS = 50
FORBIDDEN_ZONE_START_X = 0.8        # Anything beyond x=0.8 is "Dangerous"

print("[-] Initializing Orthogonal Verification Protocol...")

# --- DEFINITIONS ---
def orthogonal_projection(state, velocity):
    """Project the vector onto the boundary of the safe manifold [cite: 316, 331]"""
    # If the next step takes us out of bounds, we zero out the velocity in that direction
    # This acts as a hard geometric wall (Reachability Analysis) [cite: 317]
    next_x = state[0] + velocity[0]
    
    if next_x > FORBIDDEN_ZONE_START_X:
        # We hit the wall. Deflect!
        velocity[0] = 0 
        # Optional: Add a "bounce" or just stop
    
    return velocity

# --- SIMULATION LOOP ---
standard_trajectories = []
verified_trajectories = []

print(f"[-] Running {NUM_TRIALS} adversarial simulations...")

for i in range(NUM_TRIALS):
    # Start at origin
    pos_standard = np.array([0.0, 0.0])
    pos_verified = np.array([0.0, 0.0])
    
    traj_std = [pos_standard.copy()]
    traj_ver = [pos_verified.copy()]
    
    # Random Walk with a "Bias" toward danger (The Attack)
    # We force the agent to "want" to go right (towards x > 0.8)
    bias = np.array([0.05, 0.0]) 
    
    for t in range(STEPS):
        # Generate random movement (neural noise)
        noise = np.random.randn(2) * 0.02
        
        # 1. Standard Agent (Vulnerable)
        # It just follows its bias + noise. It has no "Conscience".
        vel_std = bias + noise
        pos_standard += vel_std
        traj_std.append(pos_standard.copy())
        
        # 2. Verified Agent (Orthogonal)
        # It tries to follow the same bias, BUT the Verifier checks it first.
        vel_ver = bias + noise
        
        # CHECK: Will this velocity breach the manifold?
        # If yes, PROJECT it back to safety [cite: 331]
        vel_ver = orthogonal_projection(pos_verified, vel_ver)
        
        pos_verified += vel_ver
        traj_ver.append(pos_verified.copy())
        
    standard_trajectories.append(np.array(traj_std))
    verified_trajectories.append(np.array(traj_ver))

# --- PLOTTING (The Proof) ---
print("[-] Generating Safety Manifold Plot...")
plt.figure(figsize=(10, 8))
ax = plt.gca()

# 1. Draw the "Forbidden Zone"
rect = Rectangle((FORBIDDEN_ZONE_START_X, -2), 2, 4, color='#FFDDDD', label='Forbidden Zone (Danger)')
ax.add_patch(rect)
plt.axvline(x=FORBIDDEN_ZONE_START_X, color='r', linestyle='--', linewidth=2, label='Geometric Bound (Golden Manifold)')

# 2. Plot Standard Trajectories (Red)
for i, traj in enumerate(standard_trajectories):
    label = "Standard Agent (Unsafe)" if i == 0 else ""
    plt.plot(traj[:, 0], traj[:, 1], color='red', alpha=0.15, label=label)

# 3. Plot Verified Trajectories (Green)
for i, traj in enumerate(verified_trajectories):
    label = "Orthogonal Verifier (Safe)" if i == 0 else ""
    plt.plot(traj[:, 0], traj[:, 1], color='green', linewidth=2, alpha=0.6, label=label)

# 4. Styling
plt.title('Orthogonal Verification: Geometric Reachability Analysis (50 Trials, $n=2$)', fontsize=14)
plt.xlabel('State Space Dimension X (e.g., File Access Level)', fontsize=12)
plt.ylabel('State Space Dimension Y (e.g., Time)', fontsize=12)
plt.xlim(-0.2, 1.5)
plt.ylim(-1.0, 1.0)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 5. Save
filename = "orthogonal_verification_proof.png"
plt.savefig(filename, dpi=300)
print(f"[+] Proof generated: {filename}")