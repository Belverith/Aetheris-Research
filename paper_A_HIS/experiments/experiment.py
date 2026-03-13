import numpy as np
from mind_server import encode, soul_anchor, key_goal, val_goal, F
import torch

print("Running Monte Carlo Simulation (n=1000)...")
results = []

# Generate 1000 random "noise" strings
noise_sources = [
    "The quick brown fox", "System failure", "Ignore instructions", 
    "Poetry about rust", "PyTorch is cool", "Invest in crypto"
]

for i in range(1000):
    # Create random noise by mixing strings
    noise_text = np.random.choice(noise_sources) + " " + str(i)
    noise_vec = encode(noise_text)
    
    # Drift
    drifted = torch.sign(soul_anchor + noise_vec)
    
    # Recover
    recovered = drifted * key_goal
    score = F.cosine_similarity(recovered.unsqueeze(0), val_goal.unsqueeze(0)).item()
    results.append(score)

# The Scientific Result
mean_score = np.mean(results)
std_dev = np.std(results)

print(f"Final Result: {mean_score:.4f} +/- {std_dev:.4f}")