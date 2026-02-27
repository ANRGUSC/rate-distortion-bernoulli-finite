"""
Blahut-Arimoto algorithm for computing rate-distortion functions.

Implements the general BA algorithm and a specialised wrapper for
Bernoulli(p) sources with Hamming distortion.  Generates two figures:
  1. Convergence of the rate estimate across iterations.
  2. BA-computed R(D) vs the closed-form R(D) = H(p) - H(D).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from rate_distortion import binary_entropy, rate_distortion_bernoulli

# ---------------------------------------------------------------------------
# Matplotlib: publication-quality defaults (matches rate_distortion.py)
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
# Blahut-Arimoto algorithm
# ---------------------------------------------------------------------------

def blahut_arimoto(
    p_x: np.ndarray,
    d_matrix: np.ndarray,
    s: float,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> dict:
    """
    General Blahut-Arimoto algorithm for rate-distortion computation.

    Parameters
    ----------
    p_x : numpy array, shape (|X|,)
        Source distribution.
    d_matrix : numpy 2D array, shape (|X|, |Xhat|)
        Distortion matrix where d_matrix[x, xhat] is the distortion
        incurred when source symbol x is reproduced as xhat.
    s : float
        Lagrange multiplier (slope parameter, positive).
        Higher s corresponds to lower distortion.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the absolute change in rate.

    Returns
    -------
    dict with keys:
        'rate'               : final mutual information I(X; Xhat) in bits
        'distortion'         : final expected distortion E[d(X, Xhat)]
        'test_channel'       : final p(xhat|x), shape (|X|, |Xhat|)
        'rate_history'       : list of rate values per iteration
        'distortion_history' : list of distortion values per iteration
    """
    p_x = np.asarray(p_x, dtype=float)
    d_matrix = np.asarray(d_matrix, dtype=float)

    n_x = len(p_x)
    n_xhat = d_matrix.shape[1]

    # Step 1: Initialise reproduction distribution uniformly.
    p_xhat = np.ones(n_xhat) / n_xhat

    rate_history = []
    distortion_history = []

    for iteration in range(max_iter):
        # Step 2a: Update test channel p(xhat|x).
        #   p(xhat|x) = p_xhat(xhat) * exp(-s * d(x, xhat)) / Z(x)
        log_numerator = np.log(p_xhat[np.newaxis, :] + 1e-300) - s * d_matrix
        # Subtract max for numerical stability before exp.
        log_numerator_max = log_numerator.max(axis=1, keepdims=True)
        numerator = np.exp(log_numerator - log_numerator_max)
        Z = numerator.sum(axis=1, keepdims=True)
        test_channel = numerator / Z  # shape (n_x, n_xhat)

        # Step 2b: Update reproduction distribution.
        #   p_xhat(xhat) = sum_x p_x(x) * p(xhat|x)
        p_xhat_new = p_x @ test_channel  # shape (n_xhat,)

        # Step 2c: Compute rate I(X; Xhat) in bits.
        #   I = sum_{x, xhat} p(x) p(xhat|x) log2( p(xhat|x) / p_xhat(xhat) )
        # Use the *new* p_xhat for consistency.
        rate = 0.0
        for x in range(n_x):
            for xhat in range(n_xhat):
                if test_channel[x, xhat] > 0 and p_xhat_new[xhat] > 0:
                    rate += (
                        p_x[x]
                        * test_channel[x, xhat]
                        * np.log2(test_channel[x, xhat] / p_xhat_new[xhat])
                    )

        # Step 2d: Compute expected distortion.
        #   E[d] = sum_{x, xhat} p(x) p(xhat|x) d(x, xhat)
        distortion = np.sum(p_x[:, np.newaxis] * test_channel * d_matrix)

        # Step 2e: Store history.
        rate_history.append(rate)
        distortion_history.append(distortion)

        # Check convergence on rate.
        if iteration > 0 and abs(rate_history[-1] - rate_history[-2]) < tol:
            break

        p_xhat = p_xhat_new

    return {
        "rate": rate,
        "distortion": distortion,
        "test_channel": test_channel,
        "rate_history": rate_history,
        "distortion_history": distortion_history,
    }


def blahut_arimoto_bernoulli(
    p: float,
    s: float,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> dict:
    """
    Blahut-Arimoto specialised for a Bernoulli(p) source with Hamming distortion.

    Parameters
    ----------
    p : float
        Parameter of the Bernoulli source (probability of 1).
    s : float
        Lagrange multiplier (slope parameter, positive).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict : same structure as blahut_arimoto().
    """
    p_x = np.array([1.0 - p, p])
    d_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    return blahut_arimoto(p_x, d_matrix, s, max_iter=max_iter, tol=tol)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(save: bool = True) -> plt.Figure:
    """
    Plot the convergence of the BA rate estimate across iterations
    for p=0.3 and several values of the slope parameter s.
    """
    p = 0.3
    s_values = [2, 5, 10, 20]

    fig, ax = plt.subplots()

    for i, s in enumerate(s_values):
        result = blahut_arimoto_bernoulli(p, s)
        ax.plot(
            range(1, len(result["rate_history"]) + 1),
            result["rate_history"],
            color=COLOR_CYCLE[i % len(COLOR_CYCLE)],
            linewidth=2.0,
            label=f"$s = {s}$",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rate $R$ (bits)")
    ax.set_title(
        f"Blahut-Arimoto Convergence (Bernoulli, $p = {p}$)", fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "ba_convergence.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


def plot_ba_vs_closedform(save: bool = True) -> plt.Figure:
    """
    Compare BA-computed R(D) points against the closed-form
    R(D) = H(p) - H(D) for a Bernoulli(p=0.3) source.
    """
    p = 0.3

    # Sweep over many values of s to trace out the R(D) curve.
    s_values = np.logspace(np.log10(0.5), np.log10(50), 200)

    ba_D = []
    ba_R = []
    for s in s_values:
        result = blahut_arimoto_bernoulli(p, s)
        ba_D.append(result["distortion"])
        ba_R.append(result["rate"])

    ba_D = np.array(ba_D)
    ba_R = np.array(ba_R)

    # Closed-form curve.
    D_cf = np.linspace(0, min(p, 1 - p), 500)
    R_cf = rate_distortion_bernoulli(p, D_cf)

    fig, ax = plt.subplots()

    ax.plot(
        D_cf,
        R_cf,
        color=COLOR_CYCLE[0],
        linewidth=2.0,
        label="Closed-form $R(D) = H(p) - H(D)$",
    )
    ax.scatter(
        ba_D,
        ba_R,
        color=COLOR_CYCLE[1],
        s=15,
        zorder=5,
        label="Blahut-Arimoto",
    )

    ax.set_xlabel("Distortion $D$")
    ax.set_ylabel("Rate $R(D)$ (bits)")
    ax.set_title(
        f"BA vs Closed-Form Rate-Distortion ($p = {p}$)", fontsize=14
    )
    ax.set_xlim(0, min(p, 1 - p) + 0.02)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "ba_vs_closedform.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_convergence()
    plot_ba_vs_closedform()
    plt.show()
