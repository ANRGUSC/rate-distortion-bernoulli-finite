"""
Rate-distortion theory for a Bernoulli(p) source with Hamming distortion.

Computes and plots:
  1. The binary entropy function H(p).
  2. Rate-distortion curves R(D) = H(p) - H(D) for several values of p.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Matplotlib: publication-quality defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.figsize": (6, 4.5),
    "savefig.dpi": 300,
})

COLOR_CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def binary_entropy(p: float | np.ndarray) -> float | np.ndarray:
    """Compute H(p) = -p*log2(p) - (1-p)*log2(1-p), handling p=0 and p=1."""
    p = np.asarray(p, dtype=float)
    h = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    h[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    # Squeeze back to scalar when the input was scalar.
    return float(h) if h.ndim == 0 else h


def rate_distortion_bernoulli(
    p: float, D: float | np.ndarray
) -> float | np.ndarray:
    """
    Rate-distortion function for Bernoulli(p) source, Hamming distortion.

    R(D) = H(p) - H(D)   for 0 <= D <= min(p, 1-p)
    R(D) = 0              otherwise
    """
    D = np.asarray(D, dtype=float)
    Hp = binary_entropy(p)
    HD = binary_entropy(D)
    R = Hp - HD

    # R(D) is zero outside [0, min(p, 1-p)]
    D_max = min(p, 1 - p)
    R = np.where((D >= 0) & (D <= D_max), R, 0.0)

    # Clamp any tiny negative values from floating-point arithmetic.
    R = np.maximum(R, 0.0)

    return float(R) if R.ndim == 0 else R


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_binary_entropy(save: bool = True) -> plt.Figure:
    """Plot the binary entropy function H(p) for p in [0, 1]."""
    p = np.linspace(0, 1, 500)
    h = binary_entropy(p)

    fig, ax = plt.subplots()
    ax.plot(p, h, color=COLOR_CYCLE[0], linewidth=2.0)
    ax.set_xlabel("$p$", fontsize=12)
    ax.set_ylabel("$H(p)$ (bits)", fontsize=12)
    ax.set_title("Binary Entropy Function", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "binary_entropy.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


def plot_rate_distortion_curves(save: bool = True) -> plt.Figure:
    """Plot R(D) curves for several Bernoulli(p) sources."""
    p_values = [0.11, 0.2, 0.3, 0.5]
    D = np.linspace(0, 0.5, 500)

    fig, ax = plt.subplots()

    for i, p in enumerate(p_values):
        R = rate_distortion_bernoulli(p, D)
        ax.plot(
            D,
            R,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            linewidth=2.0,
            label=f"$p = {p}$",
        )

    ax.set_xlabel("Distortion $D$", fontsize=12)
    ax.set_ylabel("Rate $R(D)$ (bits)", fontsize=12)
    ax.set_title("Rate-Distortion: Bernoulli Source, Hamming Distortion", fontsize=14)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "rate_distortion_curves.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_binary_entropy()
    plot_rate_distortion_curves()
    plt.show()
