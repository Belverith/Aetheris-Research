import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
# Comparing standard dimensions vs the VSA/Semantic dimension (128)
DIMENSIONS = [3, 10, 50, 128]  
RESOLUTION = 1000
OUTPUT_FILE = "figure_concentration.png"

def plot_volume_concentration():
    r = np.linspace(0, 1, RESOLUTION)
    
    plt.figure(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Plot densities
    for i, n in enumerate(DIMENSIONS):
        # The volume element in n-dimensions scales proportional to r^(n-1)
        # We normalize it so the integral over [0,1] is 1 (PDF)
        # PDF(r) = n * r^(n-1)
        pdf = n * (r ** (n - 1))
        
        # Using raw strings r"..." to prevent SyntaxWarnings with LaTeX
        label = f"Dimension $n={n}$"
        if n == 128:
            label += " (Semantic Space)"
            color = 'crimson'
            linewidth = 3
        else:
            color = colors[i % len(colors)]
            linewidth = 2
            
        plt.plot(r, pdf, label=label, color=color, linewidth=linewidth)
        
        # Fill under the curve for the highest dimension to emphasize the "shell"
        if n == 128:
            plt.fill_between(r, pdf, alpha=0.1, color='crimson')

    # Add annotations
    plt.axvline(0.9, color='gray', linestyle=':', alpha=0.5)
    plt.text(0.5, 4, "Deep Interior\n(Empty in High Dim)", ha='center', color='gray', fontsize=10)
    plt.text(0.95, 20, "Active Safety Shell\n(Mass Concentrates Here)", ha='right', color='crimson', fontsize=10, fontweight='bold')

    # Styling
    plt.title("The 'Hollow Ball': Concentration of Measure in High Dimensions", fontsize=14)
    plt.xlabel(r"Distance from Center (Radius $r$)", fontsize=12)
    plt.ylabel("Probability Density of Volume", fontsize=12)
    plt.xlim(0, 1.05)
    plt.ylim(0, 50) # Cap y-axis to keep low-dim curves visible
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"[+] Generated Concentration Diagram: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_volume_concentration()