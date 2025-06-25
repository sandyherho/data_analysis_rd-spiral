#!/usr/bin/env python
"""
Spiral Wave Comparison Visualization
Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: June 2025

To make executable: chmod +x spiral_wave_comparison.py
"""

import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # Uncomment for headless systems (servers without display)
import matplotlib.pyplot as plt
import xarray as xr
import os

# Enable LaTeX rendering
plt.rcParams['text.usetex'] = False  # Use matplotlib's mathtext instead of full LaTeX
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['figure.dpi'] = 150  # Screen display DPI

# Load the datasets
# You'll need to adjust these paths to your actual file locations
stable_path = '../../rd_outputs/stable_spiral/solution.nc'
turbulent_path = '../../rd_outputs/turbulent_spiral/solution.nc'

# Load data
stable = xr.open_dataset(stable_path)
turbulent = xr.open_dataset(turbulent_path)

# Time points to visualize
time_points = [5.0, 100.0, 200.0]

# Create figure with 2x3 subplots
# Figure size optimized for two-column journal layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Find vmin and vmax across all data for consistent colorbar
all_data = []

for t in time_points:
    # Get closest time indices
    stable_idx = np.argmin(np.abs(stable.time.values - t))
    turbulent_idx = np.argmin(np.abs(turbulent.time.values - t))
    
    # Mix u and v concentrations (using u + v combination)
    stable_mixed = stable.u.isel(time=stable_idx).values + stable.v.isel(time=stable_idx).values
    turbulent_mixed = turbulent.u.isel(time=turbulent_idx).values + turbulent.v.isel(time=turbulent_idx).values
    
    all_data.extend([stable_mixed, turbulent_mixed])

# Calculate global min/max for consistent color scale
vmin = np.min([d.min() for d in all_data])
vmax = np.max([d.max() for d in all_data])

# Use symmetric color scale around zero for better visualization
vabs = max(abs(vmin), abs(vmax))
vmin, vmax = -vabs, vabs

# Plot data
for j, t in enumerate(time_points):
    # Get closest time indices
    stable_idx = np.argmin(np.abs(stable.time.values - t))
    turbulent_idx = np.argmin(np.abs(turbulent.time.values - t))
    
    # Mix u and v concentrations
    stable_mixed = stable.u.isel(time=stable_idx).values + stable.v.isel(time=stable_idx).values
    turbulent_mixed = turbulent.u.isel(time=turbulent_idx).values + turbulent.v.isel(time=turbulent_idx).values
    
    # Get spatial coordinates
    # Assuming spatial coordinates are implicit (0 to L)
    L_stable = stable.attrs['L']
    L_turbulent = turbulent.attrs['L']
    
    # Create coordinate arrays
    x_stable = np.linspace(-L_stable/2, L_stable/2, stable.attrs['n'])
    x_turbulent = np.linspace(-L_turbulent/2, L_turbulent/2, turbulent.attrs['n'])
    
    # Plot stable (top row)
    im = axes[0, j].imshow(stable_mixed, 
                          extent=[-L_stable/2, L_stable/2, -L_stable/2, L_stable/2],
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                          origin='lower', interpolation='bilinear')
    axes[0, j].set_title(f'$t = {t:.0f}$', fontsize=16)
    axes[0, j].set_aspect('equal')
    axes[0, j].tick_params(labelsize=12, width=1)
    axes[0, j].grid(False)
    
    # Plot turbulent (bottom row)
    im = axes[1, j].imshow(turbulent_mixed,
                          extent=[-L_turbulent/2, L_turbulent/2, -L_turbulent/2, L_turbulent/2],
                          cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                          origin='lower', interpolation='bilinear')
    axes[1, j].set_aspect('equal')
    axes[1, j].tick_params(labelsize=12, width=1)
    axes[1, j].grid(False)
    
    # Add axis labels for leftmost column
    if j == 0:
        axes[0, j].set_ylabel('$y$', fontsize=16)
        axes[1, j].set_ylabel('$y$', fontsize=16)
    
    # Add x labels for bottom row
    axes[1, j].set_xlabel('$x$', fontsize=16)

# Add row labels
fig.text(0.09, 0.71, 'Stable', fontsize=16, ha='center', va='center', weight='bold')
fig.text(0.09, 0.29, 'Turbulent', fontsize=16, ha='center', va='center', weight='bold')

# Create single colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$u + v$', fontsize=16, rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=12, width=1)
cbar.outline.set_linewidth(1)

# Adjust layout
plt.subplots_adjust(left=0.15, right=0.9, top=0.98, bottom=0.08, wspace=0.15, hspace=0.15)

# Create output directory if it doesn't exist
output_dir = '../figs'
os.makedirs(output_dir, exist_ok=True)

# Save figure in multiple formats with publication quality
base_filename = 'spiral_wave_comparison_2x3'

# Save as PNG with high DPI
plt.savefig(os.path.join(output_dir, f'{base_filename}.png'), 
            dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

# Save as PDF (vector format)
plt.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), 
            bbox_inches='tight', facecolor='white', edgecolor='none')

# Save as EPS (vector format)
plt.savefig(os.path.join(output_dir, f'{base_filename}.eps'), 
            format='eps', bbox_inches='tight', facecolor='white', edgecolor='none')

print(f"Figures saved to {output_dir}/")
print(f"  - {base_filename}.png (600 DPI)")
print(f"  - {base_filename}.pdf (vector)")
print(f"  - {base_filename}.eps (vector)")

# plt.show()  # Uncomment if you want to display the figure interactively

# Close figure to free memory
plt.close(fig)

# Close datasets
stable.close()
turbulent.close()