import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, sigma=1.0, length_scale=1.0):
    """
    Computes the RBF (squared-exponential) kernel matrix between two vectors.
    """
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T

    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)

    return sigma**2 * np.exp(-sqdist / (2 * length_scale**2))

def brownian_kernel(x1, x2):
    """
    Computes the covariance matrix for a standard Brownian motion k(s, t) = min(s, t).
    Assumes inputs are non-negative.
    """
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T

    return np.minimum(x1, x2.T)

if __name__ == "__main__":
    np.random.seed(42)

    # --- Domain Setups ---
    x_rbf = np.linspace(-5, 5, 200)
    x_brownian = np.linspace(0, 10, 500)

    # --- Covariance Matrix Generation ---
    K_rbf = rbf_kernel(x_rbf, x_rbf)
    K_brownian = brownian_kernel(x_brownian, x_brownian)

    # Add a tiny noise term to the diagonals for numerical stability (jitter)
    K_rbf += 1e-8 * np.eye(len(x_rbf))
    K_brownian += 1e-8 * np.eye(len(x_brownian))

    # --- Prior Sampling ---
    samples_gaussian = np.random.multivariate_normal(np.zeros(len(x_rbf)), K_rbf, size=5)
    samples_brownian = np.random.multivariate_normal(np.zeros(len(x_brownian)), K_brownian, size=5)

    # --- Visualizing the Sample Paths ---
    plt.figure(figsize=(14, 6))

    # Plot RBF Samples
    plt.subplot(1, 2, 1)
    for s in samples_gaussian:
        plt.plot(x_rbf, s)
    plt.title("GP Prior Samples (Smooth RBF Kernel)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)

    # Plot Brownian Motion Samples
    plt.subplot(1, 2, 2)
    for s in samples_brownian:
        plt.plot(x_brownian, s)
    plt.title("GP Prior Samples (Rough Brownian Kernel)")
    plt.xlabel("t")
    plt.ylabel("W(t)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
