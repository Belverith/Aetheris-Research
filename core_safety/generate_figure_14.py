"""
Figure 14: CHDBO System Architecture — Data-Flow Diagram
=========================================================
Visual diagram showing the runtime interaction between:
  - MCBC (offline/periodic statistical certification)
  - AASV Hunter (online adversarial verification)
  - CBF-QP (real-time safety filter)
  - Safe Backup Controller

This corresponds to Algorithm 2 in the paper.

Output: figure_14.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-1.0, 10.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ============================================================================
# COLOR SCHEME
# ============================================================================
COL_OFFLINE = '#3498db'     # blue - offline
COL_ONLINE  = '#2ecc71'     # green - online
COL_SAFETY  = '#e74c3c'     # red - safety enforcement
COL_AGENT   = '#9b59b6'     # purple - agent/plant
COL_BACKUP  = '#e67e22'     # orange - backup
COL_MEMORY  = '#1abc9c'     # teal - memory
COL_BG_OFF  = '#ebf5fb'     # light blue bg
COL_BG_ON   = '#eafaf1'     # light green bg

# ============================================================================
# BACKGROUND REGIONS
# ============================================================================
# Offline region
offline_bg = FancyBboxPatch((0.2, 7.0), 5.6, 3.3,
    boxstyle="round,pad=0.15", facecolor=COL_BG_OFF, edgecolor=COL_OFFLINE,
    linewidth=2, linestyle='--', alpha=0.5)
ax.add_patch(offline_bg)
ax.text(3.0, 10.05, 'OFFLINE / PERIODIC', fontsize=14, fontweight='bold',
        color=COL_OFFLINE, ha='center', style='italic')

# Online region
online_bg = FancyBboxPatch((0.2, 0.0), 15.8, 6.5,
    boxstyle="round,pad=0.15", facecolor=COL_BG_ON, edgecolor=COL_ONLINE,
    linewidth=2, linestyle='--', alpha=0.3)
ax.add_patch(online_bg)
ax.text(8.0, 6.25, 'ONLINE — PER CONTROL STEP ($\\mathbf{O(n)}$ per step)',
        fontsize=14, fontweight='bold', color='#1a8a4a', ha='center', style='italic')

# ============================================================================
# BOXES
# ============================================================================
def draw_box(ax, x, y, w, h, label, sublabel, color, text_color='white'):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.12", facecolor=color, edgecolor='#2c3e50',
        linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.18, label, fontsize=12, fontweight='bold',
            color=text_color, ha='center', va='center')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.28, sublabel, fontsize=10,
                color=text_color, ha='center', va='center', alpha=0.95)

# Offline: MCBC
draw_box(ax, 0.8, 7.5, 4.5, 2.0,
         'MCBC Engine', '(Algorithm 1)\nBoundary sampling + Hoeffding bound',
         COL_OFFLINE)

# Online boxes
# Utility / Nominal Control
draw_box(ax, 0.5, 4.2, 3.0, 1.5,
         'Utility $\\nabla U(x)$', '$u_{\\mathrm{nom}} = \\nabla U(x)$',
         COL_AGENT)

# CBF-QP
draw_box(ax, 4.5, 4.2, 3.5, 1.5,
         'CBF-QP Filter', 'Eq. (4): $\\min \\|u - u_{\\mathrm{nom}}\\|^2$\ns.t. $L_fh + L_ghu \\geq -\\gamma h$',
         COL_SAFETY)

# AASV Hunter
draw_box(ax, 9.0, 4.2, 3.5, 1.5,
         'AASV Hunter', 'Momentum PGD × $k$ restarts\non $\\mathcal{B}(x_{\\mathrm{pred}}, \\epsilon)$',
         '#c0392b')

# Anti-Memory
draw_box(ax, 13.0, 4.2, 3.0, 1.5,
         'Anti-Memory', 'Orthogonal Prototype\nStorage $\\mathcal{M}_{\\mathrm{ban}}$',
         COL_MEMORY)

# Plant / Agent
draw_box(ax, 4.5, 1.0, 3.5, 1.5,
         'Plant / Agent', '$\\dot{x} = f(x) + g(x)u^*$',
         COL_AGENT)

# Safe Backup
draw_box(ax, 9.0, 1.0, 3.5, 1.5,
         'Safe Backup', 'Invariant orbit / anchor\nembedding reversion',
         COL_BACKUP)

# Spectral Margin
draw_box(ax, 8.5, 7.5, 4.0, 2.0,
         'Spectral Margin $\\delta(x)$', 'Hutchinson / Power Iteration\n$\\delta = \\tilde{\\sigma}_{\\max}(J_f) \\cdot d_{\\mathrm{step}}$',
         '#8e44ad')

# Output / Certificate
draw_box(ax, 13.0, 7.5, 3.0, 2.0,
         'Certificate', '$\\hat{P}_{\\mathrm{fail}} < \\epsilon$\n$P(\\mathrm{miss}) \\leq (1{-}p_{\\mathrm{hit}})^k$',
         '#2c3e50')

# ============================================================================
# ARROWS
# ============================================================================
arrow_kw = dict(arrowstyle='->', color='#2c3e50', lw=2,
                connectionstyle='arc3,rad=0.0', mutation_scale=18)
arrow_kw_curved = dict(arrowstyle='->', color='#2c3e50', lw=2,
                       mutation_scale=18)

# Utility → CBF-QP
ax.annotate('', xy=(4.5, 4.95), xytext=(3.5, 4.95), arrowprops=arrow_kw)
ax.text(4.0, 5.25, '$u_{\\mathrm{nom}}$', fontsize=14, ha='center', fontweight='bold')

# CBF-QP → Plant
ax.annotate('', xy=(6.25, 4.2), xytext=(6.25, 2.5), arrowprops=arrow_kw)
ax.text(5.8, 3.35, '$u^*$', fontsize=13, ha='center', fontweight='bold', color=COL_SAFETY)

# CBF-QP → AASV (predict)
ax.annotate('', xy=(9.0, 4.95), xytext=(8.0, 4.95), arrowprops=arrow_kw)
ax.text(8.5, 5.3, '$x_{\\mathrm{pred}}$', fontsize=14, ha='center', fontweight='bold')

# AASV ↔ Anti-Memory
ax.annotate('', xy=(13.0, 5.4), xytext=(12.5, 5.4), arrowprops=arrow_kw)
ax.annotate('', xy=(12.5, 4.5), xytext=(13.0, 4.5),
            arrowprops=dict(arrowstyle='->', color=COL_MEMORY, lw=2, mutation_scale=18))
ax.text(12.75, 5.9, 'store', fontsize=10, ha='center', color='#2c3e50', fontweight='bold')
ax.text(12.75, 3.85, 'repel', fontsize=10, ha='center', color=COL_MEMORY, fontweight='bold')

# AASV → Safe Backup (violation)
ax.annotate('', xy=(10.75, 2.5), xytext=(10.75, 4.2),
            arrowprops=dict(arrowstyle='->', color=COL_BACKUP, lw=2.5,
                            mutation_scale=18, linestyle='dashed'))
ax.text(11.65, 3.35, 'violation\ndetected', fontsize=10, ha='center',
        color=COL_BACKUP, fontweight='bold')

# AASV → CBF-QP (certified safe → execute)
ax.annotate('', xy=(7.5, 4.5), xytext=(9.0, 4.5),
            arrowprops=dict(arrowstyle='->', color=COL_ONLINE, lw=2,
                            mutation_scale=18, connectionstyle='arc3,rad=-0.3'))
ax.text(8.0, 3.8, 'safe ✓', fontsize=11, ha='center',
        color=COL_ONLINE, fontweight='bold')

# Plant → Utility (state feedback)
ax.annotate('', xy=(2.0, 4.2), xytext=(4.5, 1.75),
            arrowprops=dict(arrowstyle='->', color=COL_AGENT, lw=1.5,
                            mutation_scale=15, connectionstyle='arc3,rad=0.4'))
ax.text(2.3, 2.9, '$x(t)$', fontsize=11, ha='center', color=COL_AGENT, fontweight='bold')

# Spectral → AASV
ax.annotate('', xy=(10.5, 5.7), xytext=(10.5, 7.5),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5,
                            mutation_scale=15))
ax.text(11, 6.9, '$\\delta(x)$', fontsize=12, ha='center',
        color='#8e44ad', fontweight='bold')

# MCBC → Certificate
ax.annotate('', xy=(5.8, 8.5), xytext=(5.3, 8.5),
            arrowprops=dict(arrowstyle='->', color=COL_OFFLINE, lw=1.5,
                            mutation_scale=15, connectionstyle='arc3,rad=0.0'))
# extend to certificate
ax.annotate('', xy=(13.0, 8.5), xytext=(5.8, 8.5),
            arrowprops=dict(arrowstyle='->', color=COL_OFFLINE, lw=1.5,
                            mutation_scale=15))

# Spectral → Certificate
ax.annotate('', xy=(13.0, 8.2), xytext=(12.5, 8.2),
            arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5,
                            mutation_scale=15))

# ============================================================================
# TIMING ANNOTATIONS
# ============================================================================
ax.text(1.2, 0.3, 'Time $t$: Execute $u^*_t$   |   Hunt $x_{\\mathrm{pred}}(t{+}1)$   |   Certify or Reject',
        fontsize=11, color='#2c3e50', ha='left', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor='#f39c12', lw=1.5))

# Legend
legend_items = [
    mpatches.Patch(facecolor=COL_OFFLINE, edgecolor='#2c3e50', label='Offline: MCBC statistical certificate'),
    mpatches.Patch(facecolor=COL_SAFETY, edgecolor='#2c3e50', label='Online: CBF-QP safety filter ($O(n)$)'),
    mpatches.Patch(facecolor='#c0392b', edgecolor='#2c3e50', label='Online: AASV adversarial verification'),
    mpatches.Patch(facecolor=COL_BACKUP, edgecolor='#2c3e50', label='Fallback: Safe backup trajectory'),
    mpatches.Patch(facecolor='#8e44ad', edgecolor='#2c3e50', label='Spectral margin estimation'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=10, framealpha=0.9,
          edgecolor='#bdc3c7', ncol=2)

# Title
ax.set_title(
    'CHDBO System Architecture: Runtime Verification Pipeline\n'
    '(Algorithm 2 — MCBC + AASV + CBF-QP)',
    fontsize=16, fontweight='bold', pad=20
)

plt.tight_layout()
plt.savefig('core_safety/figure_14.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("[OK] Saved figure_14.png — System architecture diagram")
