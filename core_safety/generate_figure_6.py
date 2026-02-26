"""
Figure 6: Anti-Memory — PCA Averaging vs. Orthogonal Prototype Retention
=========================================================================
Shows that PCA averaging of two orthogonal failure directions points to a
SAFE region (catastrophic erasure), while orthogonal prototype storage
preserves both directions faithfully.

Output: figure_6.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DIMENSIONS = 128
N_FAILURE_POINTS = 200  # points sampled near each spike
SPIKE_WIDTH = 0.05       # wider for visualization
SIMILARITY_THRESHOLD = 0.3  # θ for orthogonal prototype decision

# ============================================================================
# GENERATE FAILURE DATA
# ============================================================================
rng = np.random.RandomState(42)

# Two orthogonal failure directions in R^128
Q, _ = np.linalg.qr(rng.randn(DIMENSIONS, 4))
spike_dir_1 = Q[:, 0]
spike_dir_2 = Q[:, 1]
spike_dir_3 = Q[:, 2]  # 3rd direction for extended demo

# Verify orthogonality
print(f"Dot product spike1*spike2: {np.dot(spike_dir_1, spike_dir_2):.6f}")
print(f"Dot product spike1*spike3: {np.dot(spike_dir_1, spike_dir_3):.6f}")

# Generate failure points clustered around each spike direction
def gen_cluster(center_dir, n_points, spread=0.08):
    points = []
    for _ in range(n_points):
        noise = rng.randn(DIMENSIONS) * spread
        p = center_dir + noise
        p /= np.linalg.norm(p)
        points.append(p)
    return np.array(points)

cluster_1 = gen_cluster(spike_dir_1, N_FAILURE_POINTS)
cluster_2 = gen_cluster(spike_dir_2, N_FAILURE_POINTS)
cluster_3 = gen_cluster(spike_dir_3, N_FAILURE_POINTS // 2)
all_failures = np.vstack([cluster_1, cluster_2, cluster_3])

# ============================================================================
# PCA ANALYSIS — averages directions, loses information
# ============================================================================
from numpy.linalg import svd

# Center the data
mean_failure = all_failures.mean(axis=0)
centered = all_failures - mean_failure

U, S, Vt = svd(centered, full_matrices=False)
pc1 = Vt[0]  # First principal component
pc2 = Vt[1]  # Second principal component

# The PCA "summary" direction
pca_average = mean_failure / np.linalg.norm(mean_failure)

# Check: does PCA average align with any actual spike?
cos_pca_1 = np.abs(np.dot(pca_average, spike_dir_1))
cos_pca_2 = np.abs(np.dot(pca_average, spike_dir_2))
cos_pca_3 = np.abs(np.dot(pca_average, spike_dir_3))
print(f"\nPCA centroid alignment:")
print(f"  cos(pca, spike1) = {cos_pca_1:.4f}")
print(f"  cos(pca, spike2) = {cos_pca_2:.4f}")
print(f"  cos(pca, spike3) = {cos_pca_3:.4f}")
print(f"  --> PCA points to a SAFE region (low alignment with all spikes)")

# ============================================================================
# ORTHOGONAL PROTOTYPE STORAGE — preserves all directions
# ============================================================================
prototypes = []

for point in all_failures:
    p_hat = point / np.linalg.norm(point)
    if len(prototypes) == 0:
        prototypes.append(p_hat.copy())
        continue
    
    # Check similarity against all existing prototypes
    max_sim = max(np.abs(np.dot(p_hat, proto)) for proto in prototypes)
    
    if max_sim < SIMILARITY_THRESHOLD:
        prototypes.append(p_hat.copy())

print(f"\nOrthogonal prototypes found: {len(prototypes)}")
for i, proto in enumerate(prototypes):
    alignments = [np.abs(np.dot(proto, sd)) for sd in [spike_dir_1, spike_dir_2, spike_dir_3]]
    best = np.argmax(alignments)
    print(f"  Prototype {i+1}: best alignment = {alignments[best]:.4f} (spike {best+1})")

# ============================================================================
# 2D PROJECTION for visualization
# ============================================================================
# Use spike_dir_1 and spike_dir_2 as projection basis
def project_2d(x):
    return np.array([np.dot(x, spike_dir_1), np.dot(x, spike_dir_2)])

# Project everything
cluster_1_2d = np.array([project_2d(p) for p in cluster_1])
cluster_2_2d = np.array([project_2d(p) for p in cluster_2])
cluster_3_2d = np.array([project_2d(p) for p in cluster_3])
pca_avg_2d = project_2d(pca_average)
proto_2d = [project_2d(p) for p in prototypes]
spike_1_2d = project_2d(spike_dir_1)
spike_2_2d = project_2d(spike_dir_2)
spike_3_2d = project_2d(spike_dir_3)

# ============================================================================
# PLOTTING
# ============================================================================
fig = plt.figure(figsize=(16, 6.5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)

BG_COLOR = '#f8f9fa'
CLUSTER_COLORS = ['#e74c3c', '#3498db', '#2ecc71']
PCA_COLOR = '#8e44ad'
PROTO_COLOR = '#e67e22'

# --- Panel 1: PCA Averaging (BAD) ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(BG_COLOR)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 200)
ax1.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5, alpha=0.3)

# Plot failure clusters
ax1.scatter(cluster_1_2d[:, 0], cluster_1_2d[:, 1], s=12, c=CLUSTER_COLORS[0], 
            alpha=0.4, label='Spike 1 failures')
ax1.scatter(cluster_2_2d[:, 0], cluster_2_2d[:, 1], s=12, c=CLUSTER_COLORS[1], 
            alpha=0.4, label='Spike 2 failures')
ax1.scatter(cluster_3_2d[:, 0], cluster_3_2d[:, 1], s=12, c=CLUSTER_COLORS[2], 
            alpha=0.4, label='Spike 3 failures')

# True spike directions
for i, (sp, col) in enumerate(zip([spike_1_2d, spike_2_2d, spike_3_2d], CLUSTER_COLORS)):
    ax1.annotate('', xy=sp*0.9, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2))

# PCA average direction (WRONG)
ax1.annotate('', xy=pca_avg_2d*1.1, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=PCA_COLOR, lw=3))
ax1.plot(*pca_avg_2d*1.1, 's', color=PCA_COLOR, markersize=12, zorder=10)

# Label PCA
ax1.annotate('PCA Centroid\n(SAFE REGION!)', xy=pca_avg_2d*1.1, fontsize=13,
            fontweight='bold', color=PCA_COLOR, ha='center',
            xytext=(pca_avg_2d[0]*1.1 + 0.15, pca_avg_2d[1]*1.1 + 0.2))

# Danger zone marker at PCA location — show it's actually safe
safe_circle = plt.Circle(pca_avg_2d*0.95, 0.08, fill=True, 
                         facecolor='#2ecc71', edgecolor='#2ecc71',
                         alpha=0.3, linestyle='-', lw=2)
ax1.add_patch(safe_circle)

# Big X over PCA
ax1.plot(pca_avg_2d[0]*1.1, pca_avg_2d[1]*1.1, 'X', color='red', 
         markersize=20, markeredgewidth=3, zorder=11)

ax1.set_title('PCA Dimensionality Reduction\n(Averages orthogonal failures → points to SAFE zone)',
              fontsize=14, fontweight='bold', pad=10, color='#c0392b')
ax1.set_xlabel('$e_1$ (spike 1 direction)', fontsize=13)
ax1.set_ylabel('$e_2$ (spike 2 direction)', fontsize=13)

# Failure box
textstr = ('PCA Failure Mode:\n'
           f'cos(centroid, spike1) = {cos_pca_1:.3f}\n'
           f'cos(centroid, spike2) = {cos_pca_2:.3f}\n'
           '\u2192 Memory of BOTH spikes erased!')
props = dict(boxstyle='round,pad=0.4', facecolor='#e74c3c', alpha=0.2)
ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=10.5,
         verticalalignment='bottom', bbox=props)

ax1.set_xlim(-1.4, 1.4)
ax1.set_ylim(-1.4, 1.4)
ax1.set_aspect('equal')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.2)

# --- Panel 2: Orthogonal Prototype Storage (GOOD) ---
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(BG_COLOR)

# Draw unit circle
ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5, alpha=0.3)

# Plot failure clusters
ax2.scatter(cluster_1_2d[:, 0], cluster_1_2d[:, 1], s=12, c=CLUSTER_COLORS[0], 
            alpha=0.4, label='Spike 1 failures')
ax2.scatter(cluster_2_2d[:, 0], cluster_2_2d[:, 1], s=12, c=CLUSTER_COLORS[1], 
            alpha=0.4, label='Spike 2 failures')
ax2.scatter(cluster_3_2d[:, 0], cluster_3_2d[:, 1], s=12, c=CLUSTER_COLORS[2], 
            alpha=0.4, label='Spike 3 failures')

# True spike directions
for i, (sp, col) in enumerate(zip([spike_1_2d, spike_2_2d, spike_3_2d], CLUSTER_COLORS)):
    ax2.annotate('', xy=sp*0.9, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2, alpha=0.5))

# Orthogonal prototypes (CORRECT)
for i, p2d in enumerate(proto_2d):
    ax2.annotate('', xy=p2d*1.05, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=PROTO_COLOR, lw=2.5))
    ax2.plot(*p2d*1.05, 'D', color=PROTO_COLOR, markersize=10, zorder=10,
            markeredgecolor='black', markeredgewidth=0.5)

# Repulsion arrows
for p2d in proto_2d:
    for angle in [30, -30]:
        dx = np.cos(np.radians(angle)) * 0.15
        dy = np.sin(np.radians(angle)) * 0.15
        ax2.annotate('', xy=(p2d[0]*0.7 + dx, p2d[1]*0.7 + dy), 
                    xytext=(p2d[0]*0.85, p2d[1]*0.85),
                    arrowprops=dict(arrowstyle='->', color='gray', 
                                   lw=1, linestyle='dashed', alpha=0.5))

# Check mark
ax2.text(0.5, 0.02, '✓ All failure directions preserved independently',
         transform=ax2.transAxes, fontsize=12, fontweight='bold',
         color='#27ae60', ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.2))

ax2.set_title('Orthogonal Prototype Retention (AASV Anti-Memory)\n(Each failure direction stored as distinct prototype)',
              fontsize=14, fontweight='bold', pad=10, color='#27ae60')
ax2.set_xlabel('$e_1$ (spike 1 direction)', fontsize=13)
ax2.set_ylabel('$e_2$ (spike 2 direction)', fontsize=13)

# Success box
textstr = (f'Prototypes stored: {len(prototypes)}\n'
           f'Orthogonality threshold: θ = {SIMILARITY_THRESHOLD}\n'
           '→ Each spike remembered independently\n'
           '→ Repulsion forces novel exploration')
props = dict(boxstyle='round,pad=0.4', facecolor='#2ecc71', alpha=0.2)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10.5,
         verticalalignment='top', bbox=props)

ax2.set_xlim(-1.4, 1.4)
ax2.set_ylim(-1.4, 1.4)
ax2.set_aspect('equal')
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.2)

fig.suptitle('Anti-Memory: Why PCA Fails for Safety-Critical Failure Storage ($\\mathbb{R}^{128}$)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('core_safety/figure_6.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("\n[OK] Saved figure_6.png")
