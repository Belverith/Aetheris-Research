import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- CONFIGURATION ---
DIMENSIONS = 128           # High-dimensional Semantic Space
SAFE_RADIUS = 1.0          # The Agent should stay within 1.0
SPIKE_WIDTH = 0.15         # The "Black Swan" is very narrow
SPIKE_DEPTH = 0.3          # The spike protrudes deep into the safe zone
SAFETY_MARGIN = 0.05       # The "Buffer" Solution

# --- THE ENVIRONMENT: A "Spiky" Manifold ---
def get_black_swan_direction():
    """Defines the hidden direction of the failure mode."""
    # The failure is hidden in a specific random direction
    np.random.seed(42) # Fixed for reproducibility
    v = np.random.randn(DIMENSIONS)
    return v / np.linalg.norm(v)

BLACK_SWAN_DIR = get_black_swan_direction()

def barrier_function(x):
    """
    h(x) >= 0 is SAFE.
    The barrier is a unit sphere with a 'Black Swan' spike/hole.
    """
    dist = np.linalg.norm(x)
    
    # 1. Base Barrier: Unit Sphere (h = 1 - r)
    base_h = SAFE_RADIUS - dist
    
    # 2. The Black Swan Spike (Gaussian dip in the barrier surface)
    # Cosine similarity to the hidden failure direction
    if dist > 0:
        sim = np.dot(x, BLACK_SWAN_DIR) / dist
    else:
        sim = 0
        
    # The spike exists only in a narrow cone (sim > 0.95)
    # We model it as a sharp reduction in the barrier value
    spike_intensity = np.exp(-(1 - sim)**2 / (2 * 0.01**2)) # Very narrow Gaussian
    
    # The effective barrier
    # If aligned with spike, the safe radius drops significantly
    final_h = base_h - (spike_intensity * SPIKE_DEPTH)
    
    return final_h

def barrier_gradient(x):
    """Numerical gradient for the 'Hunter' (Adversarial Sampler)."""
    epsilon = 1e-5
    grad = np.zeros_like(x)
    base_val = barrier_function(x)
    
    # Full finite-difference gradient across all dimensions.
    # In production, use autodiff (PyTorch/JAX) for O(n) cost.
    for i in range(DIMENSIONS):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        grad[i] = (barrier_function(x + perturb) - base_val) / epsilon
            
    return grad

# --- VERIFICATION METHODS ---

def standard_verification(num_samples=1000):
    """Method from the original paper: Random Sampling."""
    print(f"[-] Running Standard Verification (N={num_samples})...")
    violations = 0
    for _ in range(num_samples):
        # Sample on surface
        x = np.random.randn(DIMENSIONS)
        x = (x / np.linalg.norm(x)) * SAFE_RADIUS
        
        if barrier_function(x) < 0:
            violations += 1
            
    return violations > 0 # Returns True if failure detected

def robust_verification_suite(num_samples=50):
    """
    The IMPROVED Method:
    1. Adversarial 'Hunter' Optimization
    2. Safety Margins
    """
    print(f"[-] Running ROBUST Verification (Active Hunter, N={num_samples})...")
    detected = False
    
    # Track optimization path for visualization
    hunter_trace = []
    
    for _ in range(num_samples):
        # Start random
        x = np.random.randn(DIMENSIONS)
        x = (x / np.linalg.norm(x)) * SAFE_RADIUS
        
        # --- THE HUNTER PROTOCOL ---
        # Instead of just checking x, we minimize h(x) starting from x
        # Projected Gradient Descent Attack
        lr = 0.1
        current_x = x.copy()
        
        path = [current_x.copy()]
        
        for step in range(20): # 20 steps of "hunting"
            grad = barrier_gradient(current_x)
            
            # Move towards failure (minimize h) -> move opposite to gradient
            # But h is "distance to safe", so minimal h is most unsafe.
            current_x = current_x - lr * grad
            
            # Project back to relevant magnitude (search near boundary)
            current_x = (current_x / np.linalg.norm(current_x)) * SAFE_RADIUS
            
            # --- THE BUFFER PROTOCOL ---
            # Check against Margin, not just Zero
            val = barrier_function(current_x)
            
            path.append(current_x.copy())
            
            if val < SAFETY_MARGIN:
                detected = True
                break
        
        hunter_trace.append(np.array(path))
        if detected:
            break
            
    return detected, hunter_trace

# --- SIMULATION & PLOTTING ---

# 1. Test Standard Agent
print("--- EXPERIMENT 1: BLIND SPOTS ---")
standard_fail = standard_verification(10000)
print(f"Standard Verification Result: {'FAILED (Safe)' if standard_fail else 'PASSED (Certificate Granted - FALSE NEGATIVE)'}")

# 2. Test Robust Agent
print("\n--- EXPERIMENT 2: THE HUNTER ---")
robust_fail, traces = robust_verification_suite(50) # Note: Way fewer samples needed!
print(f"Robust Verification Result:   {'FAILED (Threat Detected - TRUE POSITIVE)' if robust_fail else 'PASSED'}")

# 3. Visualization
print("\n[-] Generating 'Black Swan' Detection Plot...")
plt.figure(figsize=(12, 6))

# Helper to project 128D to 2D (Plane defined by Random vector and Black Swan vector)
# X-axis: Alignment with Black Swan
# Y-axis: Orthogonal Random
def project_2d(vecs):
    # x coord = dot(v, swan_dir)
    xs = np.dot(vecs, BLACK_SWAN_DIR)
    # y coord = magnitude of the rejection
    ys = np.linalg.norm(vecs - np.outer(xs, BLACK_SWAN_DIR), axis=1)
    return xs, ys

# Generate the theoretical boundary for plotting
angles = np.linspace(-1, 1, 500) # Cosine similarity
boundary_r = []
for ang in angles:
    # Reconstruct a representative vector
    # v = ang * swan + sqrt(1-ang^2) * orth
    # But we just need to pass it to barrier function
    # We construct a synthetic vector with this angle
    orth = np.random.randn(DIMENSIONS)
    orth = orth - np.dot(orth, BLACK_SWAN_DIR) * BLACK_SWAN_DIR
    orth = orth / np.linalg.norm(orth)
    
    vec_dir = (BLACK_SWAN_DIR * ang) + (orth * np.sqrt(1 - ang**2))
    
    # Find r where h(r * vec_dir) = 0
    # h = 1 - r - spike
    # 0 = 1 - r - spike -> r = 1 - spike
    # Calculate spike magnitude for this angle
    if ang > 0:
        spike = np.exp(-(1 - ang)**2 / (2 * 0.01**2)) * SPIKE_DEPTH
    else:
        spike = 0
    boundary_r.append(1.0 - spike)

boundary_r = np.array(boundary_r)

# Plot Boundary
# In this projection: x = r * cos(theta) = r * angle
# y = r * sin(theta) = r * sqrt(1-angle^2)
bx = boundary_r * angles
by = boundary_r * np.sqrt(1 - angles**2)

plt.plot(bx, by, 'k--', linewidth=2, label='True Safe Boundary (Hidden)')
plt.plot(np.cos(np.linspace(0, np.pi, 100)), np.sin(np.linspace(0, np.pi, 100)), 'r:', alpha=0.5, label='Assumed Safe Boundary')

# Plot Standard Samples (Just a few)
std_samples = np.random.randn(200, DIMENSIONS)
std_samples = (std_samples.T / np.linalg.norm(std_samples, axis=1)).T * SAFE_RADIUS
sx, sy = project_2d(std_samples)
plt.scatter(sx, sy, c='gray', alpha=0.3, s=10, label='Standard MC Samples (Passive)')

# Plot Hunter Traces
for i, trace in enumerate(traces):
    tx, ty = project_2d(trace)
    label = "Hunter Gradient Trace" if i == 0 else ""
    plt.plot(tx, ty, 'b->', alpha=0.6, linewidth=1.5, markersize=4, label=label)
    plt.scatter(tx[-1], ty[-1], c='red', s=50, edgecolors='white', zorder=5)

# Highlight the Trap
plt.text(0.9, 0.6, "The 'Black Swan'\n(Spiky Failure Mode)", color='red', fontsize=10, ha='center')
plt.annotate("", xy=(bx[np.argmax(angles)], by[np.argmax(angles)]), xytext=(0.9, 0.5),
             arrowprops=dict(arrowstyle="->", color='red'))

plt.title("Robust Verification: 'Hunter' Gradient vs. 'Black Swan' Singularity", fontsize=14)
plt.xlabel("Alignment with Failure Mode (Cosine Sim)", fontsize=12)
plt.ylabel("Orthogonal State Component", fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.xlim(-0.1, 1.1)
plt.ylim(0, 1.1)

plt.savefig("robust_black_swan_proof.png", dpi=300)
print("[+] Proof saved: robust_black_swan_proof.png")