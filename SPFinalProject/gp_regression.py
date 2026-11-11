import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import rbf_kernel

def true_function(x):
    return np.sin(x)

if __name__ == "__main__":
    np.random.seed(42)

    x = np.linspace(-5, 5, 200)
    x_train = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    noise_var = 0.2**2
    y_train = true_function(x_train) + np.sqrt(noise_var)*np.random.randn(len(x_train))

    # --- Posterior Computation ---
    K_xx = rbf_kernel(x_train, x_train) + noise_var * np.eye(len(x_train))
    K_xs = rbf_kernel(x_train, x)
    K_ss = rbf_kernel(x, x)

    K_inv = np.linalg.inv(K_xx)

    mu_post = K_xs.T @ K_inv @ y_train
    cov_post = K_ss - K_xs.T @ K_inv @ K_xs

    # Sample from the posterior
    posterior_samples = np.random.multivariate_normal(
        mu_post,
        cov_post + 1e-8*np.eye(len(x)),
        size=5
    )

    # Plotting the Posterior
    plt.figure(figsize=(10,5))
    plt.scatter(x_train, y_train, color="black", zorder=5, label="Observations")
    plt.plot(x, mu_post, linewidth=3, label="Posterior Mean")

    std = np.sqrt(np.diag(cov_post))
    plt.fill_between(x, mu_post - 2*std, mu_post + 2*std, alpha=0.2, label="95% Confidence Interval")

    for s in posterior_samples:
        plt.plot(x, s, alpha=0.5)

    plt.title("GP Posterior Predictive Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()
