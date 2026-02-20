import numpy as np
import matplotlib.pyplot as plt

# --- Plotting Standards ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6,
    'lines.linewidth': 1.0,
    'figure.titlesize': 10,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

class GradientKuramotoChain:
    def __init__(self, n_nodes=5, dt=0.01, alpha_error=35.0, k_pot=1.0):
        self.N = n_nodes
        self.dt = dt
        self.theta = np.zeros(n_nodes)
        self.gamma = np.ones(n_nodes) * 5.5
        self.omega = np.ones(n_nodes) * 1.0

        self.g_L = 0.5
        self.g_U = 2.5
        self.g_H = 6.0
        self.k_pot = k_pot
        self.alpha_error = alpha_error

    def potential_derivative(self, gamma):
        return self.k_pot * (gamma - self.g_L) * (gamma - self.g_U) * (self.g_H - gamma)

    def step(self, perturbation_input):
        coupling_force = np.zeros(self.N)
        instantaneous_error_sq = np.zeros(self.N)

        for i in range(self.N):
            force = 0
            err_sum = 0
            count = 0
            for neighbor in [i-1, i+1]:
                if 0 <= neighbor < self.N:
                    diff = np.sin(self.theta[neighbor] - self.theta[i])
                    force += diff
                    err_sum += diff**2
                    count += 1

            coupling_force[i] = self.gamma[i] * force
            if count > 0:
                instantaneous_error_sq[i] = err_sum / count

        # Gamma Update
        F_intrinsic = self.potential_derivative(self.gamma)
        F_error = - self.alpha_error * instantaneous_error_sq
        d_gamma = F_intrinsic + F_error
        self.gamma += d_gamma * self.dt
        self.gamma = np.clip(self.gamma, 0.01, 10.0)

        # Theta Update
        drift = self.omega + coupling_force
        # perturbation_input is the noise term (sigma * dW)
        self.theta += drift * self.dt + perturbation_input

        return self.gamma

# --- Parameter Sweep Simulation ---
def generate_figure4():
    print("Generating Figure 4 (Robustness Landscape)...")
    np.random.seed(42)
    dt = 0.01
    target_node = 2

    # Ranges
    alpha_values = np.linspace(5.0, 100.0, 40)
    sigma_values = np.linspace(0.0, 20.0, 40)

    # Grid
    phase_map = np.zeros((len(sigma_values), len(alpha_values)))

    # Simulation Loop
    for i, sigma in enumerate(sigma_values):
        for j, alpha in enumerate(alpha_values):
            model = GradientKuramotoChain(n_nodes=5, dt=dt, alpha_error=alpha, k_pot=1.0)

            # Simulation (20 seconds)
            for _ in range(2000):
                base_noise = np.random.normal(0, 0.1, 5) * np.sqrt(dt)
                target_noise = np.random.normal(0, sigma) * np.sqrt(dt)
                base_noise[target_node] += target_noise
                model.step(base_noise)

            phase_map[i, j] = model.gamma[target_node]

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    X, Y = np.meshgrid(alpha_values, sigma_values)
    cmap = plt.cm.RdYlBu

    # pcolormesh
    mesh = ax.pcolormesh(X, Y, phase_map, shading='auto', cmap=cmap, vmin=0, vmax=6)

    # Colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(r"Steady-State Precision $\gamma$", rotation=270, labelpad=15)

    # Labels & Title
    ax.set_xlabel(r"Error Sensitivity ($\alpha_{error}$)")
    ax.set_ylabel(r"Environmental Uncertainty ($\sigma_{noise}$)")
    ax.set_title(r"Topological Robustness of Precision Collapse", loc='left', fontweight='bold')

    # Annotations
    ax.text(75, 3.0, "Healthy Regime\n(Robust Sync)",
            color='#377EB8',
            ha='center', va='center', fontweight='bold', fontsize=8,

            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#377EB8', boxstyle='round,pad=0.3'))

    # Zero-Efficiency　/ Collapsed Regime
    ax.text(25, 16.0, "Zero-Efficiency Regime\n(Collapse)",
            color='#E41A1C',
            ha='center', va='center', fontweight='bold', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#E41A1C', boxstyle='round,pad=0.3'))

    # Contour Line (Threshold)
    ax.contour(X, Y, phase_map, levels=[2.5], colors='k', linestyles='--', linewidths=1.0)

    plt.tight_layout()
    plt.savefig('Fig4_Robustness.png')
    print("Fig4_Robustness.png Generated.")

if __name__ == "__main__":
    generate_figure4()
