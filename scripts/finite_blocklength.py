"""
Finite block length rate-distortion bounds for a Bernoulli(p) source
with Hamming distortion.

Computes and plots:
  1. Normal approximation R(n, D, epsilon) vs block length n for several epsilon.
  2. Achievability bound, converse bound, and normal approximation vs n.
  3. Comprehensive comparison: finite-n rate vs distortion alongside R(D).

Mathematical background
-----------------------
  R(D) = H(p) - H(D)                           (rate-distortion function, bits)
  lambda* = ln((1-D)/D)                         (optimal Lagrange multiplier)
  P*_Xhat(1) = p*(1-D) + (1-p)*D               (optimal reproduction distribution)
  j_X(x, D)                                     (d-tilted information, bits)
  V(D) = p*(1-p) * (j_X(1,D) - j_X(0,D))^2    (rate-distortion dispersion, bits^2)

  Normal approximation:
      R(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps) + (1/2)*log2(n)/n

  Achievability (DT) bound:
      R_ach(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps) + log2(n)/n

  Converse bound:
      R_con(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps)
"""

import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from rate_distortion import binary_entropy, rate_distortion_bernoulli

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
# Core functions
# ---------------------------------------------------------------------------

def d_tilted_info_bernoulli(x: int, p: float, D: float) -> float:
    """
    d-tilted information j_X(x, D) for a Bernoulli(p) source with
    Hamming distortion, in bits.

    Parameters
    ----------
    x : int
        Source symbol (0 or 1).
    p : float
        Bernoulli parameter (probability of 1).
    D : float
        Target distortion level, 0 < D < min(p, 1-p).

    Returns
    -------
    float
        j_X(x, D) in bits.
    """
    pxhat1 = p * (1 - D) + (1 - p) * D  # P*_Xhat(1)
    pxhat0 = 1.0 - pxhat1                # P*_Xhat(0)

    # log2((1-D)/D) * D  is the common first term
    common = np.log2((1 - D) / D) * D

    if x == 0:
        # j_X(0, D) = log2((1-D)/D)*D + log2(1 / (pxhat0 + pxhat1 * D/(1-D)))
        inner = pxhat0 + pxhat1 * D / (1 - D)
        return common + np.log2(1.0 / inner)
    else:
        # j_X(1, D) = log2((1-D)/D)*D + log2(1 / (pxhat0 * D/(1-D) + pxhat1))
        inner = pxhat0 * D / (1 - D) + pxhat1
        return common + np.log2(1.0 / inner)


def dispersion_bernoulli(p: float, D: float) -> float:
    """
    Rate-distortion dispersion V(D) for Bernoulli(p) with Hamming
    distortion, in bits^2.

    V(D) = p*(1-p) * (j_X(1,D) - j_X(0,D))^2
    """
    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)
    return p * (1 - p) * (j1 - j0) ** 2


def Qinv(epsilon: float) -> float:
    """
    Inverse Q-function.

    Q(x) = P(Z > x) for Z ~ N(0,1), so Q^{-1}(eps) = Phi^{-1}(1 - eps)
    where Phi is the standard normal CDF.

    Equivalently: norm.isf(epsilon) = norm.ppf(1 - epsilon).
    """
    return norm.isf(epsilon)


def normal_approximation(
    p: float, D: float, epsilon: float, n: int | np.ndarray
) -> float | np.ndarray:
    """
    Normal approximation to the finite block length rate.

    R(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps) + (1/2)*log2(n)/n

    Parameters
    ----------
    p : float
        Bernoulli parameter.
    D : float
        Target distortion.
    epsilon : float
        Excess-distortion probability.
    n : int or array
        Block length(s).

    Returns
    -------
    float or array
        Approximate rate in bits.
    """
    n = np.asarray(n, dtype=float)
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)
    q = Qinv(epsilon)
    return RD + np.sqrt(VD / n) * q + 0.5 * np.log2(n) / n


def achievability_bound(
    p: float, D: float, epsilon: float, n: int | np.ndarray
) -> float | np.ndarray:
    """
    Achievability (DT) bound approximation.

    R_ach(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps) + log2(n)/n

    This is slightly above the normal approximation (which uses 0.5*log2(n)/n
    for the third-order term).
    """
    n = np.asarray(n, dtype=float)
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)
    q = Qinv(epsilon)
    return RD + np.sqrt(VD / n) * q + np.log2(n) / n


def converse_bound(
    p: float, D: float, epsilon: float, n: int | np.ndarray
) -> float | np.ndarray:
    """
    Converse bound approximation.

    R_con(n, D, eps) ~ R(D) + sqrt(V(D)/n) * Q^{-1}(eps)

    No third-order correction, making it slightly below the normal
    approximation for finite n.
    """
    n = np.asarray(n, dtype=float)
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)
    q = Qinv(epsilon)
    return RD + np.sqrt(VD / n) * q


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rate_vs_blocklength(save: bool = True) -> plt.Figure:
    """
    Plot 1: Normal approximation R(n, D, eps) vs block length n
    for p=0.3, D=0.1, and several values of epsilon.

    Shows how the rate converges to R(D) as n grows.
    """
    p, D = 0.3, 0.1
    epsilons = [0.01, 0.05, 0.1]
    n_vals = np.logspace(np.log10(50), np.log10(5000), 500)
    RD = rate_distortion_bernoulli(p, D)

    fig, ax = plt.subplots()

    for i, eps in enumerate(epsilons):
        R_n = normal_approximation(p, D, eps, n_vals)
        ax.plot(
            n_vals,
            R_n,
            color=COLOR_CYCLE[i],
            linewidth=2.0,
            label=rf"$\varepsilon = {eps}$",
        )

    ax.axhline(
        RD, color="black", linestyle="--", linewidth=1.5,
        label=f"$R(D) = {RD:.4f}$ bits",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel("Rate (bits/source symbol)")
    ax.set_title(
        rf"Normal Approximation: $p={p}$, $D={D}$",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "rate_vs_blocklength.pdf")
        fig.savefig(path)
        print(f"Saved {path}")

    return fig


def plot_finite_blocklength_bounds(save: bool = True) -> plt.Figure:
    """
    Plot 2: Achievability bound, converse bound, and normal approximation
    vs block length n, with shading between achievability and converse.

    Parameters: p=0.3, D=0.1, epsilon=0.1.
    """
    p, D, eps = 0.3, 0.1, 0.1
    n_vals = np.logspace(np.log10(50), np.log10(5000), 500)
    RD = rate_distortion_bernoulli(p, D)

    R_ach = achievability_bound(p, D, eps, n_vals)
    R_con = converse_bound(p, D, eps, n_vals)
    R_norm = normal_approximation(p, D, eps, n_vals)

    fig, ax = plt.subplots()

    # Shading between achievability and converse.
    ax.fill_between(
        n_vals, R_con, R_ach,
        color=COLOR_CYCLE[0], alpha=0.15,
        label="Achievability\u2013converse gap",
    )

    ax.plot(
        n_vals, R_ach,
        color=COLOR_CYCLE[1], linewidth=2.0,
        label="Achievability (DT) bound",
    )
    ax.plot(
        n_vals, R_con,
        color=COLOR_CYCLE[2], linewidth=2.0,
        label="Converse bound",
    )
    ax.plot(
        n_vals, R_norm,
        color=COLOR_CYCLE[3], linewidth=2.0, linestyle="-.",
        label="Normal approximation",
    )
    ax.axhline(
        RD, color="black", linestyle="--", linewidth=1.5,
        label=f"$R(D) = {RD:.4f}$ bits",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel("Rate (bits/source symbol)")
    ax.set_title(
        rf"Finite Block Length Bounds: $p={p}$, $D={D}$, $\varepsilon={eps}$",
        fontsize=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9, fontsize=9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "finite_blocklength.pdf")
        fig.savefig(path)
        print(f"Saved {path}")

    return fig


def plot_comprehensive_comparison(save: bool = True) -> plt.Figure:
    """
    Plot 3: Rate vs distortion for finite block lengths alongside R(D).

    For p=0.3, epsilon=0.1, plots the normal approximation curves for
    n in {100, 500, 2000} as well as the asymptotic R(D) curve, all as
    functions of D.
    """
    p, eps = 0.3, 0.1
    n_values = [100, 500, 2000]
    D_vals = np.linspace(0.01, 0.29, 400)

    # Asymptotic R(D).
    RD_curve = rate_distortion_bernoulli(p, D_vals)

    fig, ax = plt.subplots()

    ax.plot(
        D_vals, RD_curve,
        color="black", linewidth=2.5,
        label=r"$R(D) = H(p) - H(D)$",
    )

    for i, n in enumerate(n_values):
        R_n = normal_approximation(p, D_vals, eps, n)
        ax.plot(
            D_vals, R_n,
            color=COLOR_CYCLE[i],
            linewidth=2.0,
            linestyle="--",
            label=rf"$n = {n}$",
        )

    ax.set_xlabel("Distortion $D$")
    ax.set_ylabel("Rate (bits/source symbol)")
    ax.set_title(
        rf"Finite-$n$ Rate vs Distortion: $p={p}$, $\varepsilon={eps}$",
        fontsize=14,
    )
    ax.set_xlim(0.01, 0.29)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "comprehensive_comparison.pdf")
        fig.savefig(path)
        print(f"Saved {path}")

    return fig


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_convergence() -> None:
    """
    Verify that the normal approximation converges to R(D) as n -> infinity.
    """
    p, D, eps = 0.3, 0.1, 0.1
    RD = rate_distortion_bernoulli(p, D)
    VD = dispersion_bernoulli(p, D)

    print("=" * 60)
    print("Verification: normal approximation convergence to R(D)")
    print(f"  p = {p}, D = {D}, epsilon = {eps}")
    print(f"  R(D)   = {RD:.6f} bits")
    print(f"  V(D)   = {VD:.6f} bits^2")
    print(f"  Q^{{-1}}({eps}) = {Qinv(eps):.6f}")
    print("-" * 60)
    print(f"  {'n':>10s}  {'R(n,D,eps)':>14s}  {'gap':>14s}")
    print("-" * 60)

    for n in [100, 500, 1_000, 5_000, 10_000, 100_000, 1_000_000]:
        R_n = normal_approximation(p, D, eps, n)
        gap = R_n - RD
        print(f"  {n:>10d}  {R_n:>14.6f}  {gap:>14.6f}")

    print("=" * 60)
    print("As n -> infinity, the gap vanishes, confirming convergence.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_convergence()
    plot_rate_vs_blocklength()
    plot_finite_blocklength_bounds()
    plot_comprehensive_comparison()
    plt.show()
