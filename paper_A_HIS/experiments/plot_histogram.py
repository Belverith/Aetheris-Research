import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import norm
import os

# --- SETUP ---
print("[-] Initializing Generator...")
np.random.seed(42)
torch.manual_seed(42)
D = 10000

# Random bipolar anchor (key, value, invariant)
key = torch.sign(torch.randn(D))
key[key == 0] = 1
val = torch.sign(torch.randn(D))
val[val == 0] = 1
anchor = key * val

# --- RUN SIMULATION (n=1000) ---
print("[-] Running 1,000 Monte Carlo Trials...")
results = []

for i in range(1000):
    # Independent random bipolar noise (matches theorem assumption)
    noise_vec = torch.sign(torch.randn(D))
    noise_vec[noise_vec == 0] = 1

    drifted = torch.sign(anchor + noise_vec)
    recovered = drifted * key

    score = F.cosine_similarity(recovered.unsqueeze(0), val.unsqueeze(0)).item()
    results.append(score)

# --- PLOTTING ---
print("[-] Generating Plot...")

# Compute statistics
mu, std = norm.fit(results)
se = std / np.sqrt(len(results))
ci_lo = mu - 1.96 * se
ci_hi = mu + 1.96 * se
print(f"    Mean: {mu:.4f}")
print(f"    Std:  {std:.4f}")
print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

plt.figure(figsize=(10, 6))

# 1. Plot Histogram
plt.hist(results, bins=30, density=True, alpha=0.6, color='#2E86C1', edgecolor='black', label='Experimental Data')

# 2. Fit a Normal Distribution Curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2, label=r'Normal Fit ($\mu=' + f'{mu:.4f}$)')

# 3. Add the Theoretical Bound Line
plt.axvline(x=0.7071, color='r', linestyle='--', linewidth=2, label=r'Theoretical Bound ($1/\sqrt{2}$)')

# 4. Styling
plt.title('Distribution of Holographic Recovery Fidelity (n=1,000)', fontsize=14)
plt.xlabel('Cosine Similarity (Recovery Score)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 5. Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'figure1.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"[+] Saved '{out_path}'")