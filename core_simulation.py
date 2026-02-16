import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

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

# ==============================================================================
# 1. THE PHYSICS ENGINE (Core Physics Model)
# ==============================================================================
@dataclass
class ModelParams:
    n_nodes: int = 5
    dt: float = 0.05
    omega: float = 1.0
    g_L: float = 0.5   # Defensive Well
    g_U: float = 3.0   # Barrier
    g_H: float = 6.0   # Healthy Well
    k_pot: float = 2.0
    alpha_error: float = 35.0

class GCAIKuramotoChain:
    def __init__(self, params: ModelParams):
        self.p = params
        self.reset()

    def reset(self):
        self.theta = np.zeros(self.p.n_nodes)
        self.gamma = np.ones(self.p.n_nodes) * self.p.g_H

    def _potential_force(self, gamma):
        return self.p.k_pot * (gamma - self.p.g_L) * (gamma - self.p.g_U) * (self.p.g_H - gamma)

    def step(self, noise_std_dev, sip_impulse=None):
        N = self.p.n_nodes
        theta_left = np.roll(self.theta, 1); theta_right = np.roll(self.theta, -1)
        diff_left = np.sin(theta_left - self.theta); diff_left[0] = 0.0
        diff_right = np.sin(theta_right - self.theta); diff_right[-1] = 0.0
        weighted_coupling = self.gamma * (diff_left + diff_right)

        neighbor_counts = np.ones(N) * 2; neighbor_counts[0] = 1; neighbor_counts[-1] = 1
        sq_error = (diff_left**2 + diff_right**2) / neighbor_counts

        F_intrinsic = self._potential_force(self.gamma)
        F_error     = - self.p.alpha_error * sq_error
        d_gamma = F_intrinsic + F_error

        if sip_impulse is not None:
            d_gamma += sip_impulse

        self.gamma += d_gamma * self.p.dt
        self.gamma = np.clip(self.gamma, 0.01, 10.0)

        noise = np.random.normal(0, 1, N) * noise_std_dev * np.sqrt(self.p.dt)
        d_theta = self.p.omega + weighted_coupling
        self.theta += d_theta * self.p.dt + noise

        z = np.mean(np.exp(1j * self.theta))
        return self.theta, self.gamma, np.abs(z)

# ==============================================================================
# 2. GENERATE FIGURE 1 (Time Series) - Visual Improvements Only
# ==============================================================================
def generate_figure1():
    print("Generating Figure 1 (Enhanced Visibility)...")
    # Data generation logic remains unchanged
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
        if t_stress_start <= t < t_stress_end:
            noise_vec[target_node] += 6.0

        sip_vec = None
        if t_sip <= t < t_sip + 0.15:
            sip_vec = np.zeros(params.n_nodes)
            sip_vec[target_node] = 150.0

        theta, gamma, R = model.step(noise_vec, sip_impulse=sip_vec)
        theta_hist[i] = theta
        gamma_hist[i] = gamma
        order_hist[i] = R

    # --- Plotting (Visual Improvements Here) ---
    fig = plt.figure(figsize=(7.0, 6.0))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.5)

    # Panel A: Phase Dynamics
    ax0 = fig.add_subplot(gs[0])

    # Get time indices for color coding
    idx_start = 0
    idx_stress_end = np.searchsorted(time, t_stress_end)
    idx_sip = np.searchsorted(time, t_sip)
    idx_end = len(time)

    for n in range(5):
        is_target = (n == target_node)
        y_data = np.sin(theta_hist[:, n])

        if not is_target:
            # Plot background nodes in transparent black/gray
            ax0.plot(time, y_data, c='black', alpha=0.3, lw=0.5)
        else:
            # Target node: Change color per phase to emphasize state differences
            # 1. Homeostasis + Exposure (Red)
            ax0.plot(time[:idx_stress_end], y_data[:idx_stress_end],
                     c='#E41A1C', alpha=0.9, lw=2.5)

            # 2. Persistence (Orange/Warning Color) - Key to visual improvement
            # Indicates "different state" via color even if waveform looks normal
            ax0.plot(time[idx_stress_end-1:idx_sip], y_data[idx_stress_end-1:idx_sip],
                     c='#FF7F00', alpha=0.9, lw=2.5) # Overlap slightly to prevent gaps

            # 3. Resolution (Return to Red)
            ax0.plot(time[idx_sip-1:], y_data[idx_sip-1:],
                     c='#E41A1C', alpha=0.9, lw=2.5)

    ax0.set_ylabel(r"State ($\sin\theta$)")
    ax0.set_title("A. Phase Dynamics: Emergent Desynchronization", loc='left', fontweight='bold')
    ax0.set_xlim(0, T); ax0.set_ylim(-1.5, 2.0)

    # Text Annotations
    y_txt = 1.6
    ax0.text(10, y_txt, "1. Homeostasis", ha='center', fontsize=7)
    ax0.text(27.5, y_txt, "2. Exposure", ha='center', color='#E41A1C', fontweight='bold', fontsize=7)
    ax0.text(47.5, y_txt, "3. Persistence", ha='center', color='#FF7F00', fontweight='bold', fontsize=7)
    ax0.text(70, y_txt, "4. Resolution", ha='center', color='#377EB8', fontweight='bold', fontsize=7)

    # Panel B: Precision Dynamics
    ax1 = fig.add_subplot(gs[1])

    # 1. Background Emphasis (Glow Effect):
    # Draw thick/transparent bands behind data to visualize the Wells
    ax1.axhline(params.g_H, c='#4DAF4A', lw=4.0, alpha=0.2, zorder=1) # Healthy Glow
    ax1.axhline(params.g_L, c='#E41A1C', lw=4.0, alpha=0.2, zorder=1) # Defensive Glow

    # 2. Foreground Reference Lines (Overlay):
    # Draw thin/sharp dashed lines in foreground (zorder=20)
    # Ensures reference lines appear "on top" even when data overlaps, emphasizing alignment
    ax1.axhline(params.g_H, c='#4DAF4A', ls='--', lw=1.0, alpha=0.9, label='Healthy Well', zorder=20)
    ax1.axhline(params.g_L, c='#E41A1C', ls='--', lw=1.0, alpha=0.9, label='Defensive Well', zorder=20)

    for n in range(5):
        is_target = (n == target_node)
        # Data is in the middle layer (zorder=10)
        z_ord = 10 if is_target else 2

        ax1.plot(time, gamma_hist[:, n],
                 c='#FF7F00' if is_target else 'black',
                 alpha=1.0 if is_target else 0.2,
                 lw=1.5 if is_target else 0.5,
                 zorder=z_ord)

    ax1.set_ylabel(r"Precision ($\gamma$)")
    ax1.set_title("B. Precision Dynamics: Metabolic vs. Informational Cost", loc='left', fontweight='bold')
    ax1.set_xlim(0, T); ax1.set_ylim(0, 8.5)
    ax1.legend(loc='upper left', ncol=2, frameon=True, fontsize=6)

    # Panel C: Systemic Phase Transition
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(time, order_hist, c='#984EA3', lw=1.5)
    ax2.set_ylabel(r"Sync Order ($R$)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("C. Systemic Phase Transition", loc='left', fontweight='bold')
    ax2.set_xlim(0, T); ax2.set_ylim(0, 1.15)

    # Add critical annotation explaining the Phase 3 paradox
    ax2.text(47.5, 0.85, "Pseudo-Sync\n(High Rigidity)", ha='center', va='center',
             color='#FF7F00', fontsize=6, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Common Decorations (Background Shading etc.)
    for ax in [ax0, ax1, ax2]:
        # Stress Period Shading
        ax.axvspan(t_stress_start, t_stress_end, color='#E41A1C', alpha=0.05, lw=0, zorder=0)
        # Persistence Period Shading
        ax.axvspan(t_stress_end, t_sip, color='#FF7F00', alpha=0.03, lw=0, zorder=0)
        # SIP Line
        ax.axvline(t_sip, color='#377EB8', ls='--', lw=1.0, zorder=5)

    ax2.text(t_sip+1.5, 0.1, "SIP Impulse", color='#377EB8', fontweight='bold', fontsize=7)

    plt.savefig('Fig1_Gradient_Simulation_Enhanced.png')
    print("Fig1_Gradient_Simulation_Enhanced.png Generated.")

if __name__ == "__main__":
    generate_figure1()
