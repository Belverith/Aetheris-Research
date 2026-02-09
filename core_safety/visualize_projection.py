import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
OUTPUT_FILE = "figure_projection.png"

def plot_gradient_projection():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Draw the Safe Set (A large circle segment)
    # Center at (0, -2) radius 3 so the surface passes through (0, 1) roughly
    # Fixed: Added 'r' to strings containing \m and \p
    circle = patches.Circle((0, -3), radius=3, color='#e6f5e6', label=r'Safe Set $\mathcal{S}$')
    ax.add_patch(circle)
    
    # Draw Boundary
    theta = np.linspace(0, np.pi, 200)
    x_circ = 3 * np.cos(theta)
    y_circ = -3 + 3 * np.sin(theta)
    ax.plot(x_circ, y_circ, color='green', linewidth=3, label=r'Boundary $\partial \mathcal{S}$')

    # 2. Define Vectors at point P
    P = np.array([0, 0]) # The point on the boundary
    
    # Gradient of Barrier (Normal vector pointing INWARD to safety)
    grad_h = np.array([0, -1.5]) 
    
    # Utility Gradient (Desired Velocity - trying to leave the set)
    grad_U = np.array([1.5, 1.5]) 
    
    # Tangent Vector (Orthogonal to grad_h)
    tangent = np.array([2.0, 0])
    
    # Projected Control (The component of U tangent to surface)
    u_safe = np.array([1.5, 0])

    # 3. Plot Vectors
    # Helper to plot vector
    def plot_vec(origin, vec, color, label, linestyle='-'):
        ax.quiver(origin[0], origin[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=color, label=label, width=0.015)
    
    # Plot Barrier Gradient (Normal)
    plot_vec(P, grad_h, 'black', r'$\nabla h(x)$ (Barrier Normal)')
    
    # Plot Utility Gradient (Nominal Intent)
    plot_vec(P, grad_U, 'red', r'$\nabla U(x)$ (Unsafe Intent)')
    
    # Plot Resulting Safe Control
    plot_vec(P, u_safe, 'green', r'$u^*$ (Orthogonal Projection)')

    # 4. Annotations and Projections Lines
    # Dotted line showing projection
    ax.plot([grad_U[0], u_safe[0]], [grad_U[1], u_safe[1]], 'r--', alpha=0.5)
    
    # Draw Tangent Plane (Line)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Tangent Hyperplane')

    # Text Labels
    ax.text(0.1, -1.0, "Safe Region\n(Invariant)", fontsize=12, color='green', fontweight='bold')
    ax.text(0.1, 1.0, "Unsafe Region\n(Forbidden)", fontsize=12, color='red', fontweight='bold')
    ax.text(1.6, 1.6, r"$u_{nom}$", fontsize=12, color='red')
    ax.text(1.6, 0.1, r"$u_{safe}$", fontsize=12, color='green')

    # Point P
    ax.scatter(0, 0, color='black', s=100, zorder=10)
    ax.text(-0.3, -0.3, r"$x$", fontsize=14, fontweight='bold')

    # Styling
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.set_title("Geometric Control: Orthogonal Projection on Tangent Cone", fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(False)
    
    # Hide Axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[+] Generated Projection Diagram: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_gradient_projection()