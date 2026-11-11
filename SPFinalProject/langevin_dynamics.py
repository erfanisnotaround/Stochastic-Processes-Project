import numpy as np
import matplotlib.pyplot as plt

# Target Distribution: A mixture of two Gaussians
def log_pi(x):
    return np.log(0.5*np.exp(-0.5*(x+2)**2) + 0.5*np.exp(-0.5*(x-2)**2))

def score(x):
    h = 1e-5
    return (log_pi(x + h) - log_pi(x - h)) / (2 * h)

def langevin_dynamics(n_steps=20000, eps=0.02, x0=0.0):
    x = x0
    samples = []

    for _ in range(n_steps):
        drift = (eps / 2) * score(x)
        noise = np.sqrt(eps) * np.random.randn()
        x = x + drift + noise
        samples.append(x)

    return np.array(samples)

if __name__ == "__main__":
    np.random.seed(42)

    samples = langevin_dynamics(n_steps=20000, eps=0.02, x0=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Plot Trajectory
    ax1.plot(samples[:500], lw=1)
    ax1.set_title("Langevin Dynamics Trajectory (Early Segment)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("X_t")
    ax1.grid(True)

    # Verify Stationary Distribution
    grid = np.linspace(-6,6,300)
    true_density = np.exp([log_pi(x) for x in grid])
    true_density /= np.trapezoid(true_density, grid)

    ax2.hist(samples[5000:], bins=60, density=True, alpha=0.5, label="Langevin Samples (Post Burn-in)")
    ax2.plot(grid, true_density, lw=2, label="True Target Density")
    ax2.set_title("Empirical vs True Distribution")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
