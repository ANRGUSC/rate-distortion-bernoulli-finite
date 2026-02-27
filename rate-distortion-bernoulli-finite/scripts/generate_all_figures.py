"""
Master script to generate all figures for the rate-distortion tutorial.

Runs each plotting module in sequence with consistent matplotlib styling.
All figures are saved as PDF in the figures/ directory.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for batch generation
import matplotlib.pyplot as plt

# Ensure the scripts directory is on the path so imports work.
sys.path.insert(0, os.path.dirname(__file__))

# Consistent matplotlib defaults across all figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.figsize": (6, 4.5),
    "savefig.dpi": 300,
})


def main():
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("Generating all figures for the rate-distortion tutorial")
    print("=" * 60)

    # --- rate_distortion.py ---
    print("\n[1/5] rate_distortion.py")
    from rate_distortion import plot_binary_entropy, plot_rate_distortion_curves
    plot_binary_entropy()
    plt.close("all")
    plot_rate_distortion_curves()
    plt.close("all")

    # --- blahut_arimoto.py ---
    print("\n[2/5] blahut_arimoto.py")
    from blahut_arimoto import plot_convergence, plot_ba_vs_closedform
    plot_convergence()
    plt.close("all")
    plot_ba_vs_closedform()
    plt.close("all")

    # --- dispersion.py ---
    print("\n[3/5] dispersion.py")
    from dispersion import plot_dtilted_info, plot_dispersion_curves, verify
    verify()
    plot_dtilted_info()
    plt.close("all")
    plot_dispersion_curves()
    plt.close("all")

    # --- clt_histogram.py ---
    print("\n[4/5] clt_histogram.py")
    from clt_histogram import plot_clt_histogram
    plot_clt_histogram()
    plt.close("all")

    # --- finite_blocklength.py ---
    print("\n[5/5] finite_blocklength.py")
    from finite_blocklength import (
        plot_rate_vs_blocklength,
        plot_finite_blocklength_bounds,
        plot_comprehensive_comparison,
        verify_convergence,
    )
    verify_convergence()
    plot_rate_vs_blocklength()
    plt.close("all")
    plot_finite_blocklength_bounds()
    plt.close("all")
    plot_comprehensive_comparison()
    plt.close("all")

    print("\n" + "=" * 60)
    print("All figures generated successfully.")
    print(f"Output directory: {os.path.abspath(figures_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
