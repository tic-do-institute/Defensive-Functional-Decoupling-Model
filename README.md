# Bistable Precision Dynamics and Informational Reversal
## A Computational Mechanism of Defensive Functional Decoupling

**Official Simulation Code Repository** **Manuscript Status:** Under Review at *PNAS (Proceedings of the National Academy of Sciences)*

This repository contains the source code used to generate the computational results and figures presented in the manuscript: *"Bistable Precision Dynamics and Informational Reversal: A Computational Mechanism of Defensive Functional Decoupling"*.

The code implements the **Gradient-based Coupled Active Inference (G-CAI) Kuramoto model**, simulating how chronic functional disorders emerge as metastable attractors driven by defensive precision collapse and hysteresis.

---

## 📂 Repository Structure

* **`core_simulation.py`** The core physics engine. It simulates the time-series evolution of the system, precision dynamics, and the hysteresis loop.
   * **Generates Figure 1:** Phase dynamics, precision collapse, and recovery via Specific Informational Perturbation (SIP).
   * **Generates Figure 2:** The hysteresis loop showing the bistable region (Defensive vs. Healthy wells).

* **`sensitivity_analysis.py`** Performs a comprehensive parameter sweep to validate the topological robustness of the model.
   * **Generates Figure S1:** Phase diagram (heatmap) of steady-state precision against Error Sensitivity ($\alpha$) and Environmental Uncertainty ($\sigma$).

* **`requirements.txt`** List of Python dependencies required to run the simulations.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8 or higher
* Standard scientific computing libraries (`numpy`, `matplotlib`)

### Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/tic-do-institute/Defensive-Functional-Decoupling-Model.git](https://github.com/tic-do-institute/Defensive-Functional-Decoupling-Model.git)
   cd Defensive-Functional-Decoupling-Model

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```



---

## 💻 Usage

To reproduce the figures from the manuscript, simply run the Python scripts. The figures will be saved as `.png` files in the same directory.

### Reproducing Figure 1 (Dynamics) & Figure 2 (Hysteresis)

```bash
python core_simulation.py

```

*Output:* `Fig1_Gradient_Simulation.png`, `Fig2_PhaseDiagram_Hysteresis.png`

### Reproducing Figure 3 (Robustness Analysis)

```bash
python sensitivity_analysis.py

```

*Output:* `Fig.S1_Robustness.png`

---

## 📝 Citation

If you use this code or model in your research, please cite the accompanying manuscript (citation details will be updated upon publication).

**Current Reference:** Shiga, T. (2026). *Bistable Precision Dynamics and Informational Reversal: A Computational Mechanism of Defensive Functional Decoupling*. TIC-DO Institute. (Submitted)

---

## 📬 Contact

**Takafumi Shiga** Principal Investigator

TIC-DO Institute, Tokyo, Japan

Email: tic.do.institute@proton.me

Web: [https://tic-do-institute.github.io](https://tic-do-institute.github.io)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```

```
