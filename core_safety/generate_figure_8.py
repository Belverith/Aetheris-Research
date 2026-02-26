"""
Figure 8: Adaptive Tube-Based Safety Margins
==============================================
Shows the safety "tube" thickening near high-curvature regions (volatile dynamics)
and thinning in smooth regions, compared to a fixed global Lipschitz margin.
Demonstrates why adaptive margins avoid the "Frozen Robot" problem.

Output: figure_8.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_POINTS = 500
GLOBAL_LIPSCHITZ = 0.25   # conservative global margin
NOISE_BOUND = 0.02
SURROGATE_ERROR = 0.03

# ============================================================================
# DEFINE A 2D TRAJECTORY WITH VARYING CURVATURE
# ============================================================================
t = np.linspace(0, 2 * np.pi, N_POINTS)

# Non-convex safe boundary with a "pinch" (high curvature) region
# Parametric curve: a deformed circle
r_boundary = 1.0 + 0.3 * np.cos(3 * t) + 0.15 * np.sin(5 * t)
boundary_x = r_boundary * np.cos(t)
boundary_y = r_boundary * np.sin(t)

# Agent trajectory — follows inside the boundary
r_agent = 0.65 + 0.15 * np.cos(3 * t) + 0.08 * np.sin(5 * t)
agent_x = r_agent * np.cos(t + 0.1)
agent_y = r_agent * np.sin(t + 0.1)

# ============================================================================
# COMPUTE LOCAL CURVATURE (proxy for spectral norm of Jacobian)
# ============================================================================
# Curvature of the boundary = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
dx = np.gradient(boundary_x)
dy = np.gradient(boundary_y)
ddx = np.gradient(dx)
ddy = np.gradient(dy)
curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

# Normalize curvature to [0, 1]
curvature_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min())

# ============================================================================
# COMPUTE ADAPTIVE MARGINS
# ============================================================================
# δ(x) = σ_max(J_f) * d_step  (proportional to curvature)
# Total margin = δ(x) + ε_model + Δ_noise
adaptive_margin = 0.05 + 0.2 * curvature_norm + SURROGATE_ERROR + NOISE_BOUND
# Clamp
adaptive_margin = np.clip(adaptive_margin, 0.05, 0.35)

# Fixed global margin
fixed_margin = np.ones(N_POINTS) * GLOBAL_LIPSCHITZ

# ============================================================================
# COMPUTE TUBE BOUNDARIES
# ============================================================================
# Normal directions along boundary
normals_x = -dy / np.sqrt(dx**2 + dy**2)
normals_y = dx / np.sqrt(dx**2 + dy**2)

# Inner tubes (where agent must stay)
# Adaptive tube
adapt_inner_x = boundary_x + adaptive_margin * normals_x
adapt_inner_y = boundary_y + adaptive_margin * normals_y

# Fixed tube
fixed_inner_x = boundary_x + fixed_margin * normals_x
fixed_inner_y = boundary_y + fixed_margin * normals_y

# ============================================================================
# PLOTTING
# ============================================================================
fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)

BG_COLOR = '#f8f9fa'
SAFE_COLOR = '#2ecc71'
DANGER_COLOR = '#e74c3c'
FIXED_COLOR = '#e67e22'
ADAPTIVE_COLOR = '#3498db'

# --- Panel 1: Fixed Global Margin (causes Frozen Robot) ---
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(BG_COLOR)

# Fill unsafe region (outside boundary)
ax1.fill(boundary_x, boundary_y, alpha=0.05, color='green')

# Boundary
ax1.plot(boundary_x, boundary_y, 'k-', lw=2.5, label='$\\partial\\mathcal{S}$ (safety boundary)')

# Fixed margin tube  
ax1.plot(fixed_inner_x, fixed_inner_y, '--', color=FIXED_COLOR, lw=2,
         label=f'Fixed margin $\\delta = {GLOBAL_LIPSCHITZ}$ (global Lipschitz)')

# Fill the "forbidden zone" between boundary and fixed tube
ax1.fill(np.concatenate([boundary_x, fixed_inner_x[::-1]]),
         np.concatenate([boundary_y, fixed_inner_y[::-1]]),
         alpha=0.2, color=FIXED_COLOR, label='Dead zone (Frozen Robot)')

# Agent trajectory
ax1.plot(agent_x, agent_y, '-', color='#2c3e50', lw=1.5, alpha=0.7, label='Agent trajectory')

# Mark regions where agent violates fixed margin
for i in range(N_POINTS):
    dist_to_boundary = np.sqrt((agent_x[i] - boundary_x)**2 + (agent_y[i] - boundary_y)**2).min()
    dist_to_fixed = np.sqrt((agent_x[i] - fixed_inner_x)**2 + (agent_y[i] - fixed_inner_y)**2).min()
    # Check if agent is between boundary and fixed tube
    r_agent_pt = np.sqrt(agent_x[i]**2 + agent_y[i]**2)
    r_fixed_pt = np.sqrt(fixed_inner_x[i]**2 + fixed_inner_y[i]**2)
    r_boundary_pt = np.sqrt(boundary_x[i]**2 + boundary_y[i]**2)

# Highlight high-curvature zones with markers
high_curv_mask = curvature_norm > 0.6
ax1.scatter(boundary_x[high_curv_mask], boundary_y[high_curv_mask], 
            s=5, c=DANGER_COLOR, alpha=0.5, zorder=3)

# Frozen robot illustration
frozen_idx = np.argmax(curvature_norm)
ax1.annotate('FROZEN\nROBOT', 
            xy=(agent_x[frozen_idx], agent_y[frozen_idx]),
            xytext=(agent_x[frozen_idx] + 0.4, agent_y[frozen_idx] + 0.4),
            fontsize=14, fontweight='bold', color=DANGER_COLOR,
            arrowprops=dict(arrowstyle='->', color=DANGER_COLOR, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.2))

ax1.set_title('Fixed Global Lipschitz Margin\n(Overly conservative \u2192 "Frozen Robot")',
              fontsize=14, fontweight='bold', pad=10, color=DANGER_COLOR)
ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.legend(loc='lower left', fontsize=10.5, framealpha=0.9)
ax1.set_xlim(-1.8, 1.8)
ax1.set_ylim(-1.8, 1.8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.15)

# --- Panel 2: Adaptive Tube-Based Margin (AASV Buffer) ---
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(BG_COLOR)

# Fill safe region
ax2.fill(boundary_x, boundary_y, alpha=0.05, color='green')

# Boundary
ax2.plot(boundary_x, boundary_y, 'k-', lw=2.5, label='$\\partial\\mathcal{S}$ (safety boundary)')

# Color the adaptive margin by local curvature
# Create colored segments for the adaptive tube
points = np.array([adapt_inner_x, adapt_inner_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = Normalize(vmin=0, vmax=1)
lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm, linewidths=2.5)
lc.set_array(curvature_norm[:-1])
ax2.add_collection(lc)

# Fill adaptive tube with varying opacity
for i in range(0, N_POINTS-1, 3):
    triangle_x = [boundary_x[i], adapt_inner_x[i], adapt_inner_x[i+1], boundary_x[i+1]]
    triangle_y = [boundary_y[i], adapt_inner_y[i], adapt_inner_y[i+1], boundary_y[i+1]]
    alpha = 0.05 + 0.15 * curvature_norm[i]
    ax2.fill(triangle_x, triangle_y, alpha=alpha, color=ADAPTIVE_COLOR, linewidth=0)

# Agent trajectory
ax2.plot(agent_x, agent_y, '-', color='#2c3e50', lw=1.5, alpha=0.7, label='Agent trajectory')

# Highlight thick vs thin margin regions
thick_idx = np.argmax(adaptive_margin)
thin_idx = np.argmin(adaptive_margin)

# Thick margin annotation
ax2.annotate(f'High curvature\n$\\delta = {adaptive_margin[thick_idx]:.2f}$\n(wide tube)', 
            xy=(boundary_x[thick_idx], boundary_y[thick_idx]),
            xytext=(boundary_x[thick_idx] + 0.5, boundary_y[thick_idx] + 0.5),
            fontsize=9, fontweight='bold', color=DANGER_COLOR,
            arrowprops=dict(arrowstyle='->', color=DANGER_COLOR, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e74c3c', alpha=0.15))

# Thin margin annotation  
ax2.annotate(f'Low curvature\n$\\delta = {adaptive_margin[thin_idx]:.2f}$\n(narrow tube)', 
            xy=(boundary_x[thin_idx], boundary_y[thin_idx]),
            xytext=(boundary_x[thin_idx] - 0.6, boundary_y[thin_idx] - 0.5),
            fontsize=9, fontweight='bold', color=SAFE_COLOR,
            arrowprops=dict(arrowstyle='->', color=SAFE_COLOR, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2ecc71', alpha=0.15))

# Colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label('Local Curvature $\\tilde{\\sigma}_{\\max}(J_f)$', fontsize=10)

ax2.set_title('Adaptive Tube-Based Margin (AASV Buffer)\n($\\delta(x) = \\tilde{\\sigma}_{\\max}(J_f) \\cdot d_{step} + \\epsilon_{model} + \\Delta_{noise}$)',
              fontsize=12, fontweight='bold', pad=10, color='#27ae60')
ax2.set_xlabel('$x_1$', fontsize=12)
ax2.set_ylabel('$x_2$', fontsize=12)
ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.8, 1.8)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.15)

fig.suptitle('Tube-Based Adaptive Safety Margins: Avoiding the Frozen Robot Problem',
             fontsize=16, fontweight='bold', y=1.02)

# Formula as compact caption below the figure (keeps plot area clean)
fig.text(0.5, -0.02,
         r'Robust Barrier Condition: $h(x) \geq \delta(x) + \epsilon_{model} + \Delta_{noise}$'
         r'   \u2022   Thick where dynamics are volatile   \u2022   Thin where dynamics are smooth'
         r'   \u2022   Agent retains mobility everywhere',
         ha='center', fontsize=11.5, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#eaf6ff', alpha=0.8, edgecolor=ADAPTIVE_COLOR, lw=1))

plt.tight_layout()
plt.savefig('core_safety/figure_8.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("\n[OK] Saved figure_8.png")
