import numpy as np
import matplotlib.pyplot as plt

class GradientKuramotoChain:
    def __init__(self, n_nodes=5, dt=0.01, alpha_error=8.0, k_pot=1.0):
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

        # Gamma Update (Deterministic ODE)
        F_intrinsic = np.array([self.potential_derivative(g) for g in self.gamma])
        F_error = - self.alpha_error * instantaneous_error_sq
        d_gamma = F_intrinsic + F_error
        self.gamma += d_gamma * self.dt
        self.gamma = np.clip(self.gamma, 0.01, 10.0)

        # Theta Update (Stochastic SDE: Euler-Maruyama)
        # drift * dt + diffusion (noise is already scaled by sqrt(dt))
        drift = self.omega + coupling_force
        self.theta += drift * self.dt + perturbation_input
        
        return self.gamma

# --- Parameter Sweep Simulation ---
np.random.seed(42)
dt = 0.01
target_node = 2

# Ranges for the sweep
alpha_values = np.linspace(5.0, 100.0, 25) # Error Sensitivity
sigma_values = np.linspace(0.0, 20.0, 25)  # Noise Magnitude

# Grid to store final state (Steady-state Precision)
phase_map = np.zeros((len(sigma_values), len(alpha_values)))

for i, sigma in enumerate(sigma_values):
    for j, alpha in enumerate(alpha_values):
        model = GradientKuramotoChain(n_nodes=5, dt=dt, alpha_error=alpha, k_pot=1.0)
        
        # Run Simulation
        for _ in range(2000): # 20 seconds
            # noise scales with sqrt(dt) for SDE
            noise = np.random.normal(0, 0.1, 5) * np.sqrt(dt)
            noise[target_node] += np.random.normal(0, sigma) * np.sqrt(dt)
            model.step(noise)
        
        phase_map[i, j] = model.gamma[target_node]

# --- Visualization ---
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(alpha_values, sigma_values)
cmap = plt.cm.RdYlBu 

plt.pcolormesh(X, Y, phase_map, shading='auto', cmap=cmap, vmin=0, vmax=6)
cbar = plt.colorbar()
cbar.set_label(r"Steady-State Precision $\gamma$", rotation=270, labelpad=20)

plt.xlabel(r"Error Sensitivity ($\alpha_{error}$)", fontsize=14)
plt.ylabel(r"Environmental Uncertainty ($\sigma_{noise}$)", fontsize=14)
plt.title(r"$\mathbf{Fig. 3.}$ Topological Robustness of Precision Collapse", fontsize=16, fontweight='bold', loc='left')

plt.text(70, 2.5, "Healthy Regime\n(Robust Sync)", color='white', ha='center', fontweight='bold', fontsize=12)
plt.text(20, 15.0, "PPEN Regime\n(Precision Collapse)", color='white', ha='center', fontweight='bold', fontsize=12)

# Contour line at transition boundary
plt.contour(X, Y, phase_map, levels=[2.5], colors='k', linestyles='--', linewidths=1.5)

plt.tight_layout()
plt.savefig('Fig3_Robustness.png', dpi=300)
plt.show()
