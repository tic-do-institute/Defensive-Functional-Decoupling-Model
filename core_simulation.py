import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# --- PNAS Plotting Standards ---
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

# ==============================================================================
# 1. THE PHYSICS ENGINE (Core Physics Model)
# ==============================================================================
@dataclass
class ModelParams:
    n_nodes: int = 5
    dt: float = 0.05 # Balance between stability and speed
    omega: float = 1.0

    # --- Energy Landscape Parameters ---
    g_L: float = 0.5   # Defensive Well (Low Precision)
    g_U: float = 3.0   # Barrier
    g_H: float = 6.0   # Healthy Well (High Precision)
    k_pot: float = 2.0 # Potential steepness

    # --- Thermodynamics ---
    # [CRITICAL UPDATE] Set error sensitivity to 35.0 to ensure collapse
    alpha_error: float = 35.0

class GCAIKuramotoChain:
    def __init__(self, params: ModelParams):
        self.p = params
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.p.n_nodes)
        self.gamma = np.ones(self.p.n_nodes) * self.p.g_H

    def _potential_force(self, gamma):
        # Force = -dV/dg
        return self.p.k_pot * (gamma - self.p.g_L) * (gamma - self.p.g_U) * (self.p.g_H - gamma)

    def step(self, noise_std_dev, sip_impulse=None):
        N = self.p.n_nodes

        # 1. Phase Coupling
        theta_left = np.roll(self.theta, 1); theta_right = np.roll(self.theta, -1)
        diff_left = np.sin(theta_left - self.theta); diff_left[0] = 0.0
        diff_right = np.sin(theta_right - self.theta); diff_right[-1] = 0.0
        weighted_coupling = self.gamma * (diff_left + diff_right)

        # 2. Local Prediction Error
        neighbor_counts = np.ones(N) * 2; neighbor_counts[0] = 1; neighbor_counts[-1] = 1
        sq_error = (diff_left**2 + diff_right**2) / neighbor_counts

        # 3. Precision Dynamics
        F_intrinsic = self._potential_force(self.gamma)
        F_error     = - self.p.alpha_error * sq_error

        d_gamma = F_intrinsic + F_error

        if sip_impulse is not None:
            d_gamma += sip_impulse

        self.gamma += d_gamma * self.p.dt
        self.gamma = np.clip(self.gamma, 0.01, 10.0)

        # 4. Phase Update
        noise = np.random.normal(0, 1, N) * noise_std_dev * np.sqrt(self.p.dt)
        d_theta = self.p.omega + weighted_coupling
        self.theta += d_theta * self.p.dt + noise

        z = np.mean(np.exp(1j * self.theta))
        return self.theta, self.gamma, np.abs(z)

# ==============================================================================
# 2. GENERATE FIGURE 1 (Time Series)
# ==============================================================================
def generate_figure1():
    print("Generating Figure 1...")
    # Set fine time resolution for time series plotting
    params = ModelParams(n_nodes=5, dt=0.01)
    model = GCAIKuramotoChain(params)

    T = 80.0
    steps = int(T / params.dt)
    time = np.linspace(0, T, steps)

    t_stress_start, t_stress_end = 20.0, 35.0
    t_sip = 60.0
    target_node = 2

    theta_hist = np.zeros((steps, params.n_nodes))
    gamma_hist = np.zeros((steps, params.n_nodes))
    order_hist = np.zeros(steps)

    np.random.seed(42)

    for i, t in enumerate(time):
        base_noise = 0.1
        noise_vec = np.ones(params.n_nodes) * base_noise

        # Stress condition
        if t_stress_start <= t < t_stress_end:
            noise_vec[target_node] += 6.0 # Strong noise to force collapse

        # SIP intervention
        sip_vec = None
        if t_sip <= t < t_sip + 0.15:
            sip_vec = np.zeros(params.n_nodes)
            sip_vec[target_node] = 150.0 # Impulse

        theta, gamma, R = model.step(noise_vec, sip_impulse=sip_vec)
        theta_hist[i] = theta
        gamma_hist[i] = gamma
        order_hist[i] = R

    # Plotting
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.5)

    # Panel A
    ax0 = fig.add_subplot(gs[0])
    for n in range(5):
        is_target = (n == target_node)
        ax0.plot(time, np.sin(theta_hist[:, n]),
                 c='#E41A1C' if is_target else 'gray',
                 alpha=0.9 if is_target else 0.3, lw=1.2 if is_target else 0.5)
    ax0.set_ylabel(r"State ($\sin\theta$)")
    ax0.set_title("A. Phase Dynamics: Emergent Desynchronization", loc='left', fontweight='bold')
    ax0.set_xlim(0, T); ax0.set_ylim(-1.5, 2.0)

    y_txt = 1.6
    ax0.text(10, y_txt, "1. Homeostasis", ha='center', fontsize=7)
    ax0.text(27.5, y_txt, "2. Exposure", ha='center', color='#E41A1C', fontweight='bold', fontsize=7)
    ax0.text(47.5, y_txt, "3. Persistence", ha='center', color='#FF7F00', fontweight='bold', fontsize=7)
    ax0.text(70, y_txt, "4. Resolution", ha='center', color='#377EB8', fontweight='bold', fontsize=7)

    # Panel B
    ax1 = fig.add_subplot(gs[1])
    for n in range(5):
        is_target = (n == target_node)
        ax1.plot(time, gamma_hist[:, n],
                 c='#FF7F00' if is_target else 'gray',
                 alpha=1.0 if is_target else 0.2, lw=1.5 if is_target else 0.5)

    ax1.axhline(params.g_H, c='#4DAF4A', ls=':', lw=1.5, label='Healthy Well')
    ax1.axhline(params.g_L, c='#E41A1C', ls=':', lw=1.5, label='Defensive Well')
    ax1.set_ylabel(r"Precision ($\gamma$)")
    ax1.set_title("B. Precision Dynamics: Metabolic vs. Informational Cost", loc='left', fontweight='bold')
    ax1.set_xlim(0, T); ax1.set_ylim(0, 8.5)
    ax1.legend(loc='upper left', ncol=2, frameon=True)

    # Panel C
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(time, order_hist, c='#984EA3', lw=1.5)
    ax2.set_ylabel(r"Sync Order ($R$)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("C. Systemic Phase Transition", loc='left', fontweight='bold')
    ax2.set_xlim(0, T); ax2.set_ylim(0, 1.15)

    for ax in [ax0, ax1, ax2]:
        ax.axvspan(t_stress_start, t_stress_end, color='#E41A1C', alpha=0.1, lw=0)
        ax.axvline(t_sip, color='#377EB8', ls='--', lw=1.0)

    ax2.text(t_sip+1.5, 0.1, "SIP Impulse", color='#377EB8', fontweight='bold', fontsize=7)

    plt.savefig('Fig1_Gradient_Simulation.png')
    print("Fig1_Gradient_Simulation.png Generated.")

# ==============================================================================
# 3. GENERATE FIGURE 2 (Hysteresis Loop)
# ==============================================================================
def generate_figure2():
    print("Generating Figure 2...")
    params = ModelParams(n_nodes=5, dt=0.1)
    model = GCAIKuramotoChain(params)
    target_node = 2

    sigma_fwd = np.linspace(0.0, 7.0, 50)
    sigma_rev = np.linspace(7.0, 0.0, 50)

    gamma_fwd = []
    gamma_rev = []

    settle_steps = 300
    avg_window = 50

    np.random.seed(42)

    # Forward
    model.gamma[:] = params.g_H
    for s in sigma_fwd:
        gammas = []
        for i in range(settle_steps):
            noise_vec = np.ones(5)*0.1; noise_vec[target_node] += s
            model.step(noise_vec)
            if i >= (settle_steps - avg_window):
                gammas.append(model.gamma[target_node])
        gamma_fwd.append(np.mean(gammas))

    # Reverse
    for s in sigma_rev:
        gammas = []
        for i in range(settle_steps):
            noise_vec = np.ones(5)*0.1; noise_vec[target_node] += s
            model.step(noise_vec)
            if i >= (settle_steps - avg_window):
                gammas.append(model.gamma[target_node])
        gamma_rev.append(np.mean(gammas))

    # Plot
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.plot(sigma_fwd, gamma_fwd, 'o-', c='#377EB8', ms=2.5, lw=1.2, label='Increasing Stress')
    ax.plot(sigma_rev, gamma_rev, 'o--', c='#E41A1C', ms=2.5, lw=1.2, label='Decreasing Stress')

    ax.fill_betweenx([0, 8], 2.8, 4.2, color='purple', alpha=0.1, lw=0)

    ax.text(3.5, 1.0, "Bistable\nRegion",
            ha='center', va='center', color='purple', fontweight='bold', fontsize=7,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    ax.set_xlabel(r"External Uncertainty ($\sigma^2$)", fontweight='bold')
    ax.set_ylabel(r"Steady-state Precision ($\gamma$)", fontweight='bold')

    ax.set_title("Hysteresis Loop: The Thermodynamic Trap", fontweight='bold', fontsize=9, pad=8)

    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 1.5)

    ax.grid(True, ls=':', alpha=0.5, lw=0.5)
    ax.legend(loc='upper left', frameon=True, fontsize=6, borderpad=0.3)

    plt.tight_layout()
    plt.savefig('Fig2_PhaseDiagram_Hysteresis.png')
    print("Fig2_PhaseDiagram_Hysteresis.png Generated.")

if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
