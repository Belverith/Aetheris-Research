"""
Figure 3: Signal-Level Baseline Comparisons
============================================
Compares four recovery strategies under increasing additive noise depth K.

Methods:
  1. No Intervention:    cosine(sign(S + sum(N_i)), V)
  2. Re-prompting:       cosine(sign(S + N_K), V)   (only last noise kept)
  3. Similarity (RAG):   max cosine(N_i, V)          (nearest-neighbour lookup)
  4. HIS (unbind):       cosine(sign(S + sum(N_i)) * K_key, V)

All vectors are bipolar {-1, +1}^D with D = 10,000.
S = K_key * V  (holographic binding).  N_1 ... N_K are iid random bipolar.
200 trials per K value; error bars show ±1 std dev.

Output: core_memory/assets/figure3_baselines.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Parameters ──────────────────────────────────────────────────────────────
D = 10_000          # Dimensionality
TRIALS = 200        # Trials per noise depth
K_MAX = 20          # Maximum number of additive noise vectors
K_RANGE = range(1, K_MAX + 1)

np.random.seed(42)

def rand_bipolar(shape):
    """Generate iid Rademacher vector(s) in {-1, +1}^D."""
    return 2 * np.random.randint(0, 2, size=shape).astype(np.float64) - 1

def cosine(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def sign_cleanup(v):
    """Bipolar sign cleanup: sign(0) = 0 (ternary abstention)."""
    return np.sign(v)

# ── Pre-generate fixed key and value ────────────────────────────────────────
K_key = rand_bipolar(D)
V = rand_bipolar(D)
S = K_key * V                         # Holographic binding

# ── Collect results ─────────────────────────────────────────────────────────
methods = {
    "No Intervention":  {"mean": [], "std": []},
    "Re-prompting":     {"mean": [], "std": []},
    "Similarity (RAG)": {"mean": [], "std": []},
    "HIS (unbind)":     {"mean": [], "std": []},
}

for K in K_RANGE:
    scores = {m: [] for m in methods}

    for _ in range(TRIALS):
        # Generate K iid noise vectors
        noise_vecs = rand_bipolar((K, D))
        noise_sum = noise_vecs.sum(axis=0)

        # Drifted composite
        drifted = sign_cleanup(S + noise_sum)

        # --- Method 1: No Intervention ---
        scores["No Intervention"].append(cosine(drifted, V))

        # --- Method 2: Re-prompting (keep only last noise) ---
        reprompt = sign_cleanup(S + noise_vecs[-1])
        scores["Re-prompting"].append(cosine(reprompt, V))

        # --- Method 3: Similarity / RAG (best cosine among noise vectors) ---
        best_sim = max(cosine(n, V) for n in noise_vecs)
        scores["Similarity (RAG)"].append(best_sim)

        # --- Method 4: HIS (unbind from drifted state) ---
        recovered = drifted * K_key
        scores["HIS (unbind)"].append(cosine(recovered, V))

    for m in methods:
        methods[m]["mean"].append(np.mean(scores[m]))
        methods[m]["std"].append(np.std(scores[m]))

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colours = {
    "No Intervention":  "#d62728",
    "Re-prompting":     "#ff7f0e",
    "Similarity (RAG)": "#2ca02c",
    "HIS (unbind)":     "#1f77b4",
}
markers = {
    "No Intervention":  "x",
    "Re-prompting":     "s",
    "Similarity (RAG)": "^",
    "HIS (unbind)":     "o",
}

K_arr = np.array(list(K_RANGE))

for m in methods:
    mu = np.array(methods[m]["mean"])
    sd = np.array(methods[m]["std"])
    ax.plot(K_arr, mu, marker=markers[m], color=colours[m], label=m, linewidth=1.5, markersize=5)
    ax.fill_between(K_arr, mu - sd, mu + sd, alpha=0.12, color=colours[m])

# Theoretical bound line
ax.axhline(y=1 / np.sqrt(2), color="#1f77b4", linestyle="--", alpha=0.5,
           label=r"Theoretical $1/\sqrt{2}$")

ax.set_xlabel("Noise Depth $K$ (number of additive noise vectors)", fontsize=12)
ax.set_ylabel("Cosine Similarity to Ground-Truth Value $V$", fontsize=12)
ax.set_title("Signal-Level Recovery: HIS vs. Baselines", fontsize=13)
ax.legend(loc="center right", fontsize=9, framealpha=0.9)
ax.set_xlim(1, K_MAX)
ax.set_ylim(-0.1, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()

# ── Save ────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "figure3_baselines.png")
fig.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
plt.close(fig)
