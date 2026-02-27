"""
Histogram of per-symbol d-tilted information illustrating the CLT
approximation underlying the normal approximation theorem.

For a Bernoulli(p) source with block length n, the per-symbol average
information (1/n) * sum_{i=1}^n j_X(X_i, D) takes finitely many values
(one for each possible number of 1s in the block).  By the CLT, its
distribution is approximately N(R(D), V(D)/n).

This script produces a two-panel figure comparing two distortion levels:
  - Left panel:  D = 0.1  (high rate, large dispersion)
  - Right panel: D = 0.2  (lower rate, smaller dispersion)

Each panel shows:
  1. The exact PMF (bars) of the per-symbol average information.
  2. The Gaussian density N(R(D), V(D)/n) overlaid.
  3. R(D) marked as a vertical dashed line.
  4. R(n, D, epsilon) marked with the epsilon-tail shaded.

Parameters: p=0.3, n=6, epsilon=0.05.
"""

import os
import numpy as np
from scipy.stats import norm
from scipy.special import comb
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Matplotlib: publication-quality defaults (matches other scripts)
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
# Helpers (self-contained to avoid import issues)
# ---------------------------------------------------------------------------

def binary_entropy(p):
    """H(p) in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def rate_distortion_bernoulli(p, D):
    """R(D) = H(p) - H(D) for 0 <= D <= min(p, 1-p), else 0."""
    D_max = min(p, 1 - p)
    if D < 0 or D > D_max:
        return 0.0
    return max(binary_entropy(p) - binary_entropy(D), 0.0)


def d_tilted_info_bernoulli(x, p, D):
    """
    d-tilted information j_X(x, D) in bits for Bernoulli(p), Hamming.

    Uses the simplified form:
        j_X(x, D) = -D * log2((1-D)/D) + log2(1/Z(x))
    with Z(0) = (1-p)/(1-D) and Z(1) = p/(1-D).
    """
    common = -D * np.log2((1 - D) / D)
    if x == 0:
        return common + np.log2((1 - D) / (1 - p))
    else:
        return common + np.log2((1 - D) / p)


def dispersion_bernoulli(p, D):
    """V(D) = p(1-p)(j_X(1,D) - j_X(0,D))^2 in bits^2."""
    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)
    return p * (1 - p) * (j1 - j0) ** 2


def Qinv(epsilon):
    """Inverse Q-function: Q^{-1}(eps) = Phi^{-1}(1 - eps)."""
    return norm.isf(epsilon)


def normal_approx_rate(p, D, epsilon, n):
    """R(n, D, eps) via normal approximation."""
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)
    return RD + np.sqrt(VD / n) * Qinv(epsilon) + 0.5 * np.log2(n) / n


# ---------------------------------------------------------------------------
# Single-panel helper
# ---------------------------------------------------------------------------

def _draw_panel(ax, p, D, n, epsilon):
    """Draw one histogram panel on the given axes."""
    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)

    # --- Exact discrete distribution ---
    k_vals = np.arange(n + 1)
    probs = np.array([comb(n, k, exact=True) * p**k * (1 - p)**(n - k)
                       for k in k_vals])
    info_vals = (k_vals * j1 + (n - k_vals) * j0) / n

    # --- Gaussian approximation N(R(D), V(D)/n) ---
    sigma = np.sqrt(VD / n)
    bar_width = (j1 - j0) / n
    x_min = min(info_vals) - 1.5 * bar_width
    x_max = max(info_vals) + 3.0 * bar_width
    x_dense = np.linspace(x_min, x_max, 500)
    gauss_pdf = norm.pdf(x_dense, loc=RD, scale=sigma)
    gauss_scaled = gauss_pdf * bar_width  # scale to match bar heights

    # --- Threshold ---
    R_threshold = normal_approx_rate(p, D, epsilon, n)

    # Bars: exact PMF
    ax.bar(
        info_vals, probs,
        width=bar_width * 0.8,
        color=COLOR_CYCLE[0], alpha=0.6, edgecolor="white",
        label="Exact PMF",
        zorder=3,
    )

    # Gaussian overlay
    ax.plot(
        x_dense, gauss_scaled,
        color=COLOR_CYCLE[1], linewidth=2.0,
        label=r"Gaussian $\mathcal{N}(R(D),\, V\!/n)$",
        zorder=4,
    )

    # Shade the epsilon-tail
    tail_mask = x_dense >= R_threshold
    if np.any(tail_mask):
        ax.fill_between(
            x_dense[tail_mask], 0, gauss_scaled[tail_mask],
            color=COLOR_CYCLE[1], alpha=0.25,
            zorder=2,
        )

    # Vertical line: R(D)
    ax.axvline(
        RD, color="black", linestyle="--", linewidth=1.5,
        label=rf"$R(D) = {RD:.3f}$",
        zorder=5,
    )

    # Vertical line: R(n, D, epsilon)
    ax.axvline(
        R_threshold, color=COLOR_CYCLE[2], linestyle="-.", linewidth=2.0,
        label=rf"$R(n,D,\varepsilon) = {R_threshold:.3f}$",
        zorder=5,
    )

    # Annotate the shaded tail
    if np.any(tail_mask):
        arrow_x = R_threshold + 0.3 * sigma
        arrow_y = max(gauss_scaled[tail_mask]) * 0.5
        text_x = R_threshold + 1.5 * sigma
        text_y = max(probs) * 0.40
        ax.annotate(
            rf"$\varepsilon = {epsilon}$" + "\ntail",
            xy=(arrow_x, arrow_y),
            xytext=(text_x, text_y),
            fontsize=9,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            zorder=6,
        )

    ax.set_xlabel(
        r"$\frac{1}{n}\sum_{i=1}^{n} \jmath_X(X_i, D)$ (bits)"
    )
    ax.set_title(rf"$D = {D}$", fontsize=13)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(framealpha=0.9, fontsize=7.5, loc="upper right")


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_clt_histogram(
    p: float = 0.3,
    D_values: tuple = (0.1, 0.2),
    n: int = 6,
    epsilon: float = 0.05,
    save: bool = True,
) -> plt.Figure:
    """
    Two-panel figure comparing the CLT histogram at two distortion levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, D in zip(axes, D_values):
        _draw_panel(ax, p, D, n, epsilon)

    axes[0].set_ylabel("Probability")

    fig.suptitle(
        rf"CLT Approximation of Per-Symbol $d$-Tilted Information:"
        rf"  $\mathrm{{Ber}}({p})$, $n={n}$, $\varepsilon={epsilon}$",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "clt_histogram.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_clt_histogram()
    plt.show()
