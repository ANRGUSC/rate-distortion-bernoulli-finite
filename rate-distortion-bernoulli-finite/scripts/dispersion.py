"""
D-tilted information and rate-distortion dispersion for Bernoulli(p) with
Hamming distortion.

Computes and plots:
  1. d-tilted information j_X(0, D) and j_X(1, D) vs D for p = 0.3.
  2. Dispersion V(D) vs D for several values of p.

The optimal test channel for Bernoulli(p) with Hamming distortion has a BSC(D)
*backward* channel P(x|xhat), inducing an asymmetric forward channel when
p != 0.5.  The optimal reproduction distribution is:

    Q*(1) = (p - D) / (1 - 2D),    Q*(0) = (1 - p - D) / (1 - 2D)

The d-tilted information (Kostina & Verdu, Def. 6) is:

    j_X(x, D) = D_KL(P_{Xhat|X=x} || Q*) + lambda*(E[d(x,Xhat)|X=x] - D)

where lambda* = ln((1-D)/D) is the optimal Lagrange multiplier.

References:
  - Kostina & Verdu, "Fixed-length lossy compression in the finite
    blocklength regime," IEEE Trans. Inf. Theory, 2012.
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
# Helper
# ---------------------------------------------------------------------------

def binary_entropy(p: float | np.ndarray) -> float | np.ndarray:
    """Compute H(p) = -p*log2(p) - (1-p)*log2(1-p), handling p=0 and p=1."""
    p = np.asarray(p, dtype=float)
    h = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    h[mask] = -p[mask] * np.log2(p[mask]) - (1 - p[mask]) * np.log2(1 - p[mask])
    return float(h) if h.ndim == 0 else h


def _kl_binary(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL divergence D(Bern(a) || Bern(b)) in nats, element-wise."""
    kl = np.zeros_like(a)
    m = (a > 0) & (a < 1) & (b > 0) & (b < 1)
    kl[m] = a[m] * np.log(a[m] / b[m]) + (1 - a[m]) * np.log((1 - a[m]) / (1 - b[m]))
    return kl


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def rate_distortion_bernoulli(
    p: float, D: float | np.ndarray
) -> float | np.ndarray:
    """
    Rate-distortion function for Bernoulli(p), Hamming distortion.

    R(D) = H(p) - H(D)   for 0 <= D <= min(p, 1-p)
    R(D) = 0              otherwise
    """
    D = np.asarray(D, dtype=float)
    R = binary_entropy(p) - binary_entropy(D)
    D_max = min(p, 1 - p)
    R = np.where((D >= 0) & (D <= D_max), R, 0.0)
    R = np.maximum(R, 0.0)
    return float(R) if R.ndim == 0 else R


def d_tilted_info_bernoulli(
    x: int, p: float, D: float | np.ndarray
) -> float | np.ndarray:
    """
    D-tilted information j_X(x, D) in bits for a Bernoulli(p) source with
    Hamming distortion, for source symbol x in {0, 1}.

    The optimal backward channel is BSC(D):  P(x | xhat) = 1-D if x == xhat.
    This induces the forward channel:

        P(Xhat=xhat | X=x) = Q*(xhat) * P(x | xhat) / P(x)

    with Q*(1) = (p - D)/(1 - 2D) and Q*(0) = (1-p-D)/(1-2D).

    The d-tilted information is:

        j_X(x, D) = D_KL(P_{Xhat|X=x} || Q*) + lambda*(E[d|X=x] - D)

    in nats, then converted to bits.  lambda* = ln((1-D)/D).
    """
    D = np.asarray(D, dtype=float)

    # Optimal Lagrange multiplier (nats)
    lam = np.log((1 - D) / D)

    # Optimal reproduction distribution
    q1 = (p - D) / (1 - 2 * D)       # Q*(Xhat=1)
    q0 = 1 - q1                       # Q*(Xhat=0)

    if x == 0:
        # Forward channel: P(Xhat=0|X=0) = q0*(1-D)/(1-p)
        #                  P(Xhat=1|X=0) = q1*D/(1-p)
        fwd_0 = q0 * (1 - D) / (1 - p)   # P(Xhat=0 | X=0)
        fwd_1 = q1 * D / (1 - p)          # P(Xhat=1 | X=0)
        # E[d(0, Xhat) | X=0] = P(Xhat=1 | X=0)
        e_dist = fwd_1
    elif x == 1:
        # Forward channel: P(Xhat=0|X=1) = q0*D/p
        #                  P(Xhat=1|X=1) = q1*(1-D)/p
        fwd_0 = q0 * D / p                # P(Xhat=0 | X=1)
        fwd_1 = q1 * (1 - D) / p          # P(Xhat=1 | X=1)
        # E[d(1, Xhat) | X=1] = P(Xhat=0 | X=1)
        e_dist = fwd_0
    else:
        raise ValueError(f"x must be 0 or 1, got {x}")

    # D_KL(P_{Xhat|X=x} || Q*) in nats, using the binary KL helper
    # P_{Xhat|X=x} is Bern(fwd_1), Q* is Bern(q1)
    kl = _kl_binary(fwd_1, q1)

    # d-tilted information in nats, then convert to bits
    j_nats = kl + lam * (e_dist - D)
    j_bits = j_nats / np.log(2)

    return float(j_bits) if j_bits.ndim == 0 else j_bits


def dispersion_bernoulli(
    p: float, D: float | np.ndarray
) -> float | np.ndarray:
    """
    Rate-distortion dispersion V(D) in bits^2 for Bernoulli(p), Hamming
    distortion.

    V(D) = Var[j_X(X, D)] = p(1-p)(j_X(1,D) - j_X(0,D))^2
    """
    D = np.asarray(D, dtype=float)
    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)
    V = p * (1 - p) * (j1 - j0) ** 2
    return float(V) if np.ndim(V) == 0 else V


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dtilted_info(p: float = 0.3, save: bool = True) -> plt.Figure:
    """Plot d-tilted information j_X(0, D) and j_X(1, D) vs D."""
    eps = 1e-6
    D_max = min(p, 1 - p)
    D = np.linspace(eps, D_max - eps, 500)

    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)

    fig, ax = plt.subplots()
    ax.plot(D, j0, color=COLOR_CYCLE[0], linewidth=2.0,
            label=r"$\jmath_X(0,\, D)$")
    ax.plot(D, j1, color=COLOR_CYCLE[1], linewidth=2.0,
            label=r"$\jmath_X(1,\, D)$")

    ax.set_xlabel("Distortion $D$")
    ax.set_ylabel(r"$\jmath_X(x,\, D)$ (bits)")
    ax.set_title(
        rf"$d$-tilted Information: Bernoulli($p={p}$), Hamming",
        fontsize=14,
    )
    ax.set_xlim(0, D_max)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "dtilted_info.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


def plot_dispersion_curves(save: bool = True) -> plt.Figure:
    """Plot V(D) vs D for several Bernoulli(p) sources."""
    p_values = [0.1, 0.2, 0.3, 0.5]
    eps = 1e-6

    fig, ax = plt.subplots()

    for i, p in enumerate(p_values):
        D_max = min(p, 1 - p)
        D = np.linspace(eps, D_max - eps, 500)
        V = dispersion_bernoulli(p, D)
        ax.plot(
            D,
            V,
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            linewidth=2.0,
            label=f"$p = {p}$",
        )

    ax.set_xlabel("Distortion $D$")
    ax.set_ylabel(r"Dispersion $V(D)$ (bits$^2$)")
    ax.set_title(
        "Rate-Distortion Dispersion: Bernoulli Source, Hamming Distortion",
        fontsize=14,
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "dispersion_curves.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(p: float = 0.3, D: float = 0.1) -> None:
    """Check that E[j_X(X, D)] = R(D)."""
    j0 = d_tilted_info_bernoulli(0, p, D)
    j1 = d_tilted_info_bernoulli(1, p, D)
    E_j = (1 - p) * j0 + p * j1
    R = rate_distortion_bernoulli(p, D)

    print(f"Verification (p={p}, D={D}):")
    print(f"  j_X(0, D) = {j0:.6f} bits")
    print(f"  j_X(1, D) = {j1:.6f} bits")
    print(f"  E[j_X(X, D)] = {E_j:.6f} bits")
    print(f"  R(D)          = {R:.6f} bits")
    print(f"  Match: {np.isclose(E_j, R)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify()
    print()
    plot_dtilted_info()
    plot_dispersion_curves()
    plt.show()
