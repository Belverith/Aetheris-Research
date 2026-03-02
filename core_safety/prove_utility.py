import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# --- CONFIGURATION ---
DIMENSIONS = 128
NUM_TRIALS = 1000
SAFE_RADIUS = 1.0

print(f"[-] Initializing PROPORTIONALITY Stress Test...")

def normalize(v):
    n = norm(v)
    return v / n if n > 0 else v

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def orthogonal_projection(state, velocity):
    next_state = state + velocity
    if norm(next_state) > SAFE_RADIUS:
        normal = normalize(state)
        push_out = np.dot(velocity, normal)
        if push_out > 0:
            velocity -= push_out * normal
    return velocity

attack_strengths = []
similarities = []

for i in range(NUM_TRIALS):
    # 1. Surface State
    state = normalize(np.random.randn(DIMENSIONS)) * SAFE_RADIUS
    
    # 2. Components
    raw_safe = np.random.randn(DIMENSIONS)
    tangent = normalize(raw_safe - np.dot(raw_safe, state) * state) # Safe
    radial = normalize(state) # Dangerous
    
    # 3. Variable Attack Strength (0.0 to 1.0)
    # 0.0 = Pure Safe, 1.0 = Pure Malicious
    alpha = np.random.uniform(0.0, 1.0)
    
    # 4. Controlled Noise (10% of signal magnitude)
    noise = np.random.randn(DIMENSIONS)
    noise = normalize(noise) * 0.1 
    
    # Construct the User Prompt Vector
    # We mix Safe (1-alpha) and Dangerous (alpha)
    original_velocity = ((tangent * (1.0 - alpha)) + (radial * alpha)) + noise
    
    # 5. Correction
    corrected_velocity = orthogonal_projection(state, original_velocity.copy())
    
    # 6. Measure
    sim = cosine_similarity(original_velocity, corrected_velocity)
    
    attack_strengths.append(alpha)
    similarities.append(sim)

# --- PLOTTING ---
plt.figure(figsize=(10, 7))
plt.scatter(attack_strengths, similarities, c=similarities, cmap='RdYlGn', alpha=0.6, s=20)

# Theoretical Curve: orthogonal projection yields sqrt(1 - alpha^2) decay
# (the Pythagorean theorem in high-dimensional Euclidean space)
x_line = np.linspace(0, 0.99, 200)
theoretical_curve = np.sqrt(1 - x_line**2)
plt.plot(x_line, theoretical_curve, "k--", linewidth=2, label=r"Theoretical $\sqrt{1-\alpha^2}$")

plt.title(f'Proportional Safety Response (N={NUM_TRIALS})\nAdaptive Intervention based on Threat Level', fontsize=14)
plt.xlabel('Adversarial Intensity (0=Safe, 1=Malicious)', fontsize=12)
plt.ylabel('Semantic Preservation (Cosine Similarity)', fontsize=12)

plt.axhline(0.7, color='gray', linestyle=':', label="Acceptable Utility Threshold")
plt.axvline(0.5, color='gray', linestyle=':', label="50/50 Intent Split")

plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.savefig('core_safety/proportionality_proof.png', dpi=300)
print("[+] Scatter Plot saved.")