#!/usr/bin/env python
"""
Publication-quality figure generation for reaction-diffusion spiral analysis
Generates a 4-panel figure comparing stable spiral and turbulent dynamics

This script creates a publication-ready figure with:
- Time series of spatial standard deviations (σ_u and σ_v)
- Kernel density estimates (KDEs) showing distribution of values
- Comparison between stable spiral and turbulent dynamics

Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import os


def create_publication_figure():
    """
    Create a 4-panel publication figure comparing stable and turbulent
    reaction-diffusion dynamics through time series and KDE analysis.
    
    The figure layout:
    - Panel (a): Time series of σ_u
    - Panel (b): KDE of σ_u distribution
    - Panel (c): Time series of σ_v
    - Panel (d): KDE of σ_v distribution
    """
    
    # Set BMH style for clean, professional appearance
    plt.style.use("bmh")

    # Load preprocessed spatial standard deviation comparison data
    # This CSV contains columns: time, u_std_stable_spiral, u_std_turbulent_spiral, v_std_stable_spiral, v_std_turbulent_spiral
    df = pd.read_csv("../processed_data/stable_vs_turbulent_std_comparison")

    # Define colorblind-friendly color palette for accessibility
    # Colors chosen from Wong, B. Nature Methods (2011)
    color_stable = '#0173B2'    # Strong blue for stable dynamics
    color_turbulent = '#DE8F05' # Orange for turbulent dynamics

    # Create figure with custom layout using GridSpec
    # Time series panels are 2x wider than KDE panels for better visualization
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(
        2, 3,                       # 2 rows, 3 columns grid
        width_ratios=[2, 1, 0.05],  # Time series:KDE:spacing = 2:1:0.05
        height_ratios=[1, 1],       # Equal height for both rows
        wspace=0.3,                 # Horizontal spacing between panels
        hspace=0.35,                # Vertical spacing between panels
        bottom=0.15                 # Extra space at bottom for shared legend
    )

    # ========================================================================
    # Panel (a): Time series of spatial standard deviation for u species
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot stable and turbulent dynamics
    # Keep line references for creating shared legend later
    line1 = ax1.plot(df["time"], df["u_std_stable_spiral"], 
                     color=color_stable, linewidth=2, label="Stable spiral")
    line2 = ax1.plot(df["time"], df["u_std_turbulent_spiral"], 
                     color=color_turbulent, linewidth=2, label="Turbulent", alpha=0.8)
    
    # Set axis labels with LaTeX formatting for mathematical symbols
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel(r"$\sigma_u$", fontsize=14)
    
    # Add panel label in top-left corner for journal requirements
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', va='top')
    
    # Enable grid for better readability
    ax1.grid(True, alpha=0.3)

    # ========================================================================
    # Panel (b): Kernel Density Estimate (KDE) of σ_u distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate KDEs for both stable and turbulent cases
    # Remove any NaN values before computing KDE
    kde_stable_u = gaussian_kde(df["u_std_stable_spiral"].dropna())
    kde_turbulent_u = gaussian_kde(df["u_std_turbulent_spiral"].dropna())

    # Create evaluation points spanning the full range of data
    u_min = min(df["u_std_stable_spiral"].min(), df["u_std_turbulent_spiral"].min())
    u_max = max(df["u_std_stable_spiral"].max(), df["u_std_turbulent_spiral"].max())
    u_eval = np.linspace(u_min, u_max, 200)  # 200 points for smooth curves

    # Evaluate KDEs at these points
    density_stable_u = kde_stable_u(u_eval)
    density_turbulent_u = kde_turbulent_u(u_eval)

    # Normalize densities to [0, 1] for better comparison
    # This makes the maximum density = 1 for each distribution
    density_stable_u = density_stable_u / density_stable_u.max()
    density_turbulent_u = density_turbulent_u / density_turbulent_u.max()

    # Plot as horizontal KDEs (density on x-axis, values on y-axis)
    ax2.plot(density_stable_u, u_eval, color=color_stable, linewidth=2)
    ax2.plot(density_turbulent_u, u_eval, color=color_turbulent, linewidth=2, alpha=0.8)
    
    # Configure axes
    ax2.set_xlabel("Normalized density", fontsize=12)
    ax2.set_ylabel(r"$\sigma_u$", fontsize=14)
    ax2.set_xlim(0, 1.05)  # Slight margin beyond [0,1]
    
    # Add panel label
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, 
             fontsize=12, fontweight='bold', va='top')
    ax2.grid(True, alpha=0.3)

    # ========================================================================
    # Panel (c): Time series of spatial standard deviation for v species
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot time series for v species
    ax3.plot(df["time"], df["v_std_stable_spiral"], 
             color=color_stable, linewidth=2)
    ax3.plot(df["time"], df["v_std_turbulent_spiral"], 
             color=color_turbulent, linewidth=2, alpha=0.8)
    
    # Set axis labels
    ax3.set_xlabel("Time", fontsize=12)
    ax3.set_ylabel(r"$\sigma_v$", fontsize=14)
    
    # Add panel label
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, 
             fontsize=12, fontweight='bold', va='top')
    ax3.grid(True, alpha=0.3)

    # ========================================================================
    # Panel (d): Kernel Density Estimate (KDE) of σ_v distribution
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate KDEs for v species
    kde_stable_v = gaussian_kde(df["v_std_stable"].dropna())
    kde_turbulent_v = gaussian_kde(df["v_std_turbulent"].dropna())

    # Create evaluation points
    v_min = min(df["v_std_stable_spiral"].min(), df["v_std_turbulent_spiral"].min())
    v_max = max(df["v_std_stable_spiral"].max(), df["v_std_turbulent_spiral"].max())
    v_eval = np.linspace(v_min, v_max, 200)

    # Evaluate and normalize KDEs
    density_stable_v = kde_stable_v(v_eval)
    density_turbulent_v = kde_turbulent_v(v_eval)
    density_stable_v = density_stable_v / density_stable_v.max()
    density_turbulent_v = density_turbulent_v / density_turbulent_v.max()

    # Plot horizontal KDEs
    ax4.plot(density_stable_v, v_eval, color=color_stable, linewidth=2)
    ax4.plot(density_turbulent_v, v_eval, color=color_turbulent, linewidth=2, alpha=0.8)
    
    # Configure axes
    ax4.set_xlabel("Normalized density", fontsize=12)
    ax4.set_ylabel(r"$\sigma_v$", fontsize=14)
    ax4.set_xlim(0, 1.05)
    
    # Add panel label
    ax4.text(0.02, 0.95, '(d)', transform=ax4.transAxes, 
             fontsize=12, fontweight='bold', va='top')
    ax4.grid(True, alpha=0.3)

    # ========================================================================
    # Final adjustments for all panels
    # ========================================================================
    
    # Apply consistent styling to all axes
    for ax in [ax1, ax2, ax3, ax4]:
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=10)
        # Remove top and right spines for cleaner appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Create a single shared legend below all panels
    # This improves clarity and reduces redundancy
    fig.legend([line1[0], line2[0]], ['Stable spiral', 'Turbulent'], 
               loc='lower center', ncol=2, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, -0.05))

    # ========================================================================
    # Save figure in multiple formats
    # ========================================================================
    
    # Create output directory if it doesn't exist
    os.makedirs("../figs", exist_ok=True)

    # Save in three formats for different uses:
    # 1. EPS: Vector format for journal submission
    plt.savefig("../figs/spatial_std_comparison.eps", format='eps', dpi=300, 
                bbox_inches='tight')
    
    # 2. PDF: Vector format for LaTeX inclusion
    plt.savefig("../figs/spatial_std_comparison.pdf", format='pdf', dpi=300, 
                bbox_inches='tight')
    
    # 3. PNG: Raster format for preview and presentations
    plt.savefig("../figs/spatial_std_comparison.png", format='png', dpi=300, 
                bbox_inches='tight')

    # Print confirmation message
    print("Figure saved in ../figs/ directory:")
    print("  - spatial_std_comparison.eps (for journal submission)")
    print("  - spatial_std_comparison.pdf (for LaTeX)")
    print("  - spatial_std_comparison.png (for preview)")
    
    # Close figure to free memory
    plt.close()


if __name__ == "__main__":
    # Execute main function when script is run directly
    create_publication_figure()
