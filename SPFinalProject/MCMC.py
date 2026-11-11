import numpy as np
import matplotlib.pyplot as plt
from gaussian_process import rbf_kernel
from gp_regression import true_function

def log_prior_lengthscale(l):
    """
    Evaluates the log-prior probability density.
    We assume l follows a Gamma distribution: l ~ Gamma(shape=2.0, scale=1.0)
    Density: p(l) \\propto l^(shape-1) * exp(-l/scale) for l > 0
    """
    if l <= 0:
        return -np.inf
    shape, scale = 2.0, 1.0
    return (shape - 1) * np.log(l) - l / scale

def log_posterior_lengthscale(l, x_train, y_train, sigma=1.0, noise_std=0.2):
    """
    Computes the unnormalized log posterior: log pi(l) = log p(y | X, l) + log p(l)
    """
    log_prior = log_prior_lengthscale(l)
    if np.isinf(log_prior):
        return -np.inf

    K = rbf_kernel(x_train, x_train, sigma=sigma, length_scale=l) + noise_std**2 * np.eye(len(x_train))

    try:
        sign, logdet = np.linalg.slogdet(K)
        K_inv = np.linalg.inv(K)

        n = len(y_train)
        log_likelihood = -0.5 * y_train.T @ K_inv @ y_train - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)

        return log_likelihood + log_prior

    except np.linalg.LinAlgError:
        return -np.inf

def metropolis_hastings(n_samples, initial_l, proposal_width=0.15):
    """
    Runs a Metropolis-Hastings sampler to generate draws from the lengthscale posterior.
    """
    samples = []
    current = initial_l
    current_logp = log_posterior_lengthscale(current, x_train, y_train)

    for i in range(n_samples):
        proposal = current + np.random.normal(0, proposal_width)

        proposal_logp = log_posterior_lengthscale(proposal, x_train, y_train)

        log_alpha = proposal_logp - current_logp

        if np.log(np.random.uniform()) < log_alpha:
            current = proposal
            current_logp = proposal_logp

        samples.append(current)

    return np.array(samples)

if __name__ == "__main__":
    np.random.seed(42)

    x_train = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    noise_var = 0.2**2
    y_train = true_function(x_train) + np.sqrt(noise_var)*np.random.randn(len(x_train))

    # Execution
    chain = metropolis_hastings(n_samples=15000, initial_l=0.5, proposal_width=0.15)

    # Diagnostic Visualizations
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(chain, color='teal', lw=0.5)
    plt.axhline(1.0, color='red', linestyle='--', label='True Kernel Setup Value')
    plt.title("Metropolis-Hastings Parameter Trace")
    plt.xlabel("MCMC Iteration")
    plt.ylabel(r"Length scale ($\ell$)")
    plt.legend()

    plt.subplot(1, 2, 2)
    burn_in = int(0.2 * len(chain))
    plt.hist(chain[burn_in:], bins=50, density=True, color='darkblue', alpha=0.7)
    plt.title(r"Empirical Posterior Density $\pi(\ell \mid X, y)$")
    plt.xlabel(r"Length scale ($\ell$)")
    plt.tight_layout()
    plt.show()
