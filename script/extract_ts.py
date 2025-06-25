#!/usr/bin/env python
"""
extract_ts.py
=============================
Extract and combine spatial standard deviation statistics from rd-spiral experiments.

This script processes the output statistics from stable and turbulent spiral
reaction-diffusion simulations and creates a combined dataset for comparative
analysis. The data is filtered to a common time range for consistency.

Usage:
    python extract_ts.py

Output:
    - Combined CSV file with spatial standard deviations from both experiments
    - Summary statistics and verification output to console

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: June 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


def load_experiment_data(base_dir: str, experiment_names: list) -> dict:
    """
    Load statistics data from multiple experiments.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment outputs
    experiment_names : list
        List of experiment directory names
    
    Returns
    -------
    dict
        Dictionary mapping experiment names to their dataframes
    
    Raises
    ------
    FileNotFoundError
        If any experiment data file is missing
    """
    data = {}
    
    for exp_name in experiment_names:
        stats_path = os.path.join(base_dir, exp_name, "stats.csv")
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Statistics file not found: {stats_path}\n"
                f"Please ensure the '{exp_name}' experiment has been run."
            )
        
        print(f"Loading {exp_name} data from: {stats_path}")
        data[exp_name] = pd.read_csv(stats_path)
        
    return data


def filter_by_time(data: dict, time_limit: float) -> dict:
    """
    Filter all experiment data to include only times up to the specified limit.
    
    Parameters
    ----------
    data : dict
        Dictionary of experiment dataframes
    time_limit : float
        Maximum time to include in the analysis
    
    Returns
    -------
    dict
        Dictionary of filtered dataframes with reset indices
    """
    filtered_data = {}
    
    print(f"\nFiltering data to time <= {time_limit}...")
    
    for exp_name, df in data.items():
        # Filter by time limit
        filtered_df = df[df['time'] <= time_limit].copy()
        
        # Reset index for clean processing
        filtered_df = filtered_df.reset_index(drop=True)
        
        filtered_data[exp_name] = filtered_df
        
        # Report filtering results
        original_count = len(df)
        filtered_count = len(filtered_df)
        print(f"  {exp_name}: {original_count} → {filtered_count} time points "
              f"(removed {original_count - filtered_count} points)")
    
    return filtered_data


def verify_time_consistency(data: dict) -> bool:
    """
    Verify that all experiments have consistent time points.
    
    Parameters
    ----------
    data : dict
        Dictionary of experiment dataframes
    
    Returns
    -------
    bool
        True if all experiments have identical time arrays
    """
    print("\nVerifying time consistency across experiments...")
    
    # Get reference time array from first experiment
    ref_name = list(data.keys())[0]
    ref_time = data[ref_name]['time'].values
    
    all_consistent = True
    
    for exp_name, df in data.items():
        t_start = df['time'].iloc[0]
        t_end = df['time'].iloc[-1]
        n_points = len(df)
        
        print(f"  {exp_name:15s}: {n_points:4d} points, "
              f"t ∈ [{t_start:6.2f}, {t_end:6.2f}]")
        
        # Check if time arrays match
        if exp_name != ref_name:
            if not np.allclose(df['time'].values, ref_time, rtol=1e-9):
                print(f"    ⚠️  Warning: Time points differ from {ref_name}")
                all_consistent = False
    
    if all_consistent:
        print("  ✓ All experiments have identical time points")
    else:
        print("  ⚠️  Time points are not perfectly aligned - using stable_spiral as reference")
    
    return all_consistent


def create_combined_dataframe(data: dict) -> pd.DataFrame:
    """
    Create a combined dataframe with spatial standard deviations from all experiments.
    
    Parameters
    ----------
    data : dict
        Dictionary of experiment dataframes
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns for each experiment's u_std and v_std
    """
    # Use stable_spiral time as reference (assuming it exists)
    if 'stable_spiral' not in data:
        raise ValueError("stable_spiral experiment data is required as time reference")
    
    # Initialize with time column
    combined_df = pd.DataFrame({
        'time': data['stable_spiral']['time']
    })
    
    # Add std columns from each experiment
    for exp_name, df in data.items():
        # Create descriptive column names
        u_col = f'u_std_{exp_name}'
        v_col = f'v_std_{exp_name}'
        
        # Add the data
        combined_df[u_col] = df['u_std']
        combined_df[v_col] = df['v_std']
    
    return combined_df


def save_results(combined_df: pd.DataFrame, output_dir: str) -> str:
    """
    Save the combined dataframe to CSV file.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined experiment data
    output_dir : str
        Directory to save output file
    
    Returns
    -------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, "stable_vs_turbulent_std_comparison.csv")
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined stable vs turbulent data saved to: {output_file}")
    
    return output_file


def print_summary(combined_df: pd.DataFrame):
    """
    Print summary statistics and sample data for verification.
    
    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined experiment data
    """
    print(f"  Shape: {combined_df.shape}")
    print(f"  Columns: {', '.join(combined_df.columns)}")
    print(f"  Time range: {combined_df['time'].min():.2f} to {combined_df['time'].max():.2f}")
    
    # Display first few rows
    print("\nFirst 5 rows of combined data:")
    print(combined_df.head().to_string(float_format='%.6f'))
    
    # Display last few rows
    print("\nLast 5 rows of combined data:")
    print(combined_df.tail().to_string(float_format='%.6f'))
    
    # Summary statistics
    print("\nSummary of spatial standard deviations:")
    print(combined_df.describe().to_string(float_format='%.6f'))
    
    # Additional analysis: equilibrium behavior
    print("\nEquilibrium analysis (last 10% of data):")
    print("Comparing stable (dynamic equilibrium) vs turbulent (chaotic) behavior:")
    n_eq = max(int(0.1 * len(combined_df)), 5)
    
    for col in combined_df.columns:
        if col != 'time' and 'std' in col:
            mean_val = combined_df[col].iloc[-n_eq:].mean()
            std_val = combined_df[col].iloc[-n_eq:].std()
            # Stable spiral should have low std_val (steady rotation)
            # Turbulent spiral should have high std_val (chaotic dynamics)
            print(f"  {col:25s}: mean={mean_val:.6f}, std={std_val:.6f}")


def main():
    """
    Main function to extract and combine stable vs turbulent spiral results.
    """
    print("="*70)
    print("RD-SPIRAL EXPERIMENT DATA EXTRACTION")
    print("Extract spatial standard deviations from stable vs turbulent spirals")
    print("="*70)
    
    # Configuration
    base_dir = "../../rd_outputs"  # Adjust path as needed
    output_dir = "../processed_data"
    time_limit = 200.0  # Focus on t ≤ 200 for all experiments
    
    # Define experiments to process
    experiment_names = [
        "stable_spiral",
        "turbulent_spiral"
    ]
    
    try:
        # Step 1: Load all experiment data
        print("\nStep 1: Loading experiment data...")
        data = load_experiment_data(base_dir, experiment_names)
        
        # Step 2: Filter by time limit
        print("\nStep 2: Filtering by time limit...")
        filtered_data = filter_by_time(data, time_limit)
        
        # Step 3: Verify time consistency
        print("\nStep 3: Verifying time consistency...")
        verify_time_consistency(filtered_data)
        
        # Step 4: Create combined dataframe
        print("\nStep 4: Creating combined dataframe...")
        combined_df = create_combined_dataframe(filtered_data)
        
        # Step 5: Save results
        print("\nStep 5: Saving results...")
        output_file = save_results(combined_df, output_dir)
        
        # Step 6: Print summary
        print("\nStep 6: Summary and verification...")
        print_summary(combined_df)
        
        print("\n" + "="*70)
        print("✓ PROCESSING COMPLETE!")
        print(f"  Output saved to: {output_file}")
        print(f"  Data ready for stable vs turbulent dynamics comparison")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the experiments first using:")
        for exp in experiment_names:
            print(f"  rd-spiral configs/{exp}.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function only if script is executed directly
    main()