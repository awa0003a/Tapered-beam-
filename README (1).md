# Tapered-Beam Deep Learning Analysis

> **"Deep Learning Approaches for Tapered Beam Structural Dynamics: A Comparative Study of Six Neural Network Architectures"**
>
> Muhammad Zeerak Awan — IT4Innovations National Supercomputing Centre, VSB–Technical University of Ostrava

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0006--7946--8563-brightgreen)](https://orcid.org/0009-0006-7946-8563)

---

## Overview

This repository contains the **complete, fully reproducible codebase** for a comparative study of six deep learning architectures applied to the prediction of nonlinear tapered beam structural dynamics. All results presented in the manuscript — including training, evaluation, cross-validation, robustness testing, statistical significance analysis, and all 30 publication-quality figures — are produced by a single self-contained Python script.

### Governing Equation

The tapered beam is modelled as a nonlinear oscillator:

$$\ddot{u}(t) = -\frac{1 + \alpha\,\dot{u}(t)^2 + \beta\,u(t)^2}{1 + \alpha\,u(t)^2}\,u(t)$$

with default parameters $\alpha = 0.5$, $\beta = 1.0$.

### Models Compared

| Model | Type | Parameters | Inference (ms)* |
|-------|------|-----------|----------------|
| **LSTM** | Bidirectional recurrent | 547,969 | 69.7 ± 1.0 |
| **GRU** | Bidirectional recurrent | 415,105 | 46.0 ± 0.2 |
| **Transformer** | Self-attention | 4,307 | 66.3 ± 0.1 |
| **TCN** | Temporal convolution | 195,521 | 36.6 ± 0.2 |
| **FCNN** | Fully-connected | 471,553 | 18.3 ± 0.2 |
| **PINN** | Physics-informed (ODE residual + IC loss) | 49,921 | 18.3 ± 0.1 |

*NVIDIA A100-SXM4-40 GB, batch size 512, 10 repeated passes.

---

## Key Results

| Model | RMSE (mean ± std) | R² (mean ± std) | Multi-Traj RMSE |
|-------|------------------|----------------|-----------------|
| LSTM | 0.01304 ± 0.00552 | 0.99988 ± 0.00010 | 0.032 ± 0.031 |
| **GRU** | **0.01401 ± 0.00385** | **0.99988 ± 0.00006** | **0.019 ± 0.029** |
| Transformer | 0.01715 ± 0.00231 | 0.99983 ± 0.00005 | 0.029 ± 0.039 |
| TCN | 0.03023 ± 0.00501 | 0.99946 ± 0.00018 | 0.057 ± 0.029 |
| FCNN | 0.14796 ± 0.00355 | 0.98731 ± 0.00061 | 0.222 ± 0.113 |
| PINN | 1.30835 ± 0.00056 | 0.00851 ± 0.00085 | 1.625 ± 0.533 |

All metrics are **mean ± std over 3 independent training runs** (seeds 42, 49, 56).
Multi-Traj RMSE is evaluated over **20 independent test trajectories** with unseen initial conditions.

---

## Repository Structure

```
Tapered-beam-/
│
├── tapered_beam_revised.py   ← MAIN SCRIPT — full reproducible pipeline
├── requirements.txt          ← Exact package versions
├── LICENSE                   ← MIT licence
├── README.md                 ← This file
│
└── results/                  ← Output directory (created on first run)
    ├── 01_predictions.png
    ├── 02_error_time.png
    ├── ...
    └── 30_computational_cost.png
```

---

## Reproducibility

All sources of randomness are fixed at the top of `tapered_beam_revised.py`:

```python
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

| Purpose | Seed |
|---------|------|
| Global (NumPy + PyTorch + CuDNN) | 42 |
| Training trajectory generation | 42 |
| Test trajectory generation | 99 |
| Repeated run 1 | 42 |
| Repeated run 2 | 49 |
| Repeated run 3 | 56 |

---

## Hyperparameters

All hyperparameters are consolidated in the `HP` dictionary (top of script):

```python
HP = dict(
    seq_len           = 50,       # input window length
    batch_size        = 512,
    epochs            = 30,
    lr                = 1e-3,     # AdamW learning rate
    lstm_hidden       = 128,      lstm_layers = 2,  lstm_dropout = 0.3,
    gru_hidden        = 128,      gru_layers  = 2,  gru_dropout  = 0.3,
    transformer_nhead = 3,        transformer_ff = 128,
    transformer_drop  = 0.2,      transformer_nlayers = 4,
    tcn_kernel        = 3,
    fcnn_hidden       = 512,      fcnn_dropout = 0.3,
    pinn_hidden       = 128,      pinn_layers  = 4,
    pinn_lambda_res   = 1.0,      # ODE residual loss weight
    pinn_lambda_ic    = 10.0,     # initial-condition loss weight
    n_train_traj      = 150,
    n_test_traj       = 20,       # independent test trajectories
    k_folds           = 5,        # cross-validation folds
    n_runs            = 3,        # repeated training runs
    noise_levels      = [0.0, 0.01, 0.05],
    beam_alpha_train  = 0.5,      beam_beta_train = 1.0,
    beam_alpha_ood    = [0.3, 0.7],
    beam_beta_ood     = [0.8, 1.2],
)
```

---

## PINN Formulation

The PINN minimises a composite loss incorporating the governing ODE:

$$\mathcal{L} = \lambda_{\text{res}}\,\mathcal{L}_{\text{res}} + \lambda_{\text{ic}}\,\mathcal{L}_{\text{ic}}$$

$$\mathcal{L}_{\text{res}} = \frac{1}{N_r}\sum_{i=1}^{N_r} \left[\ddot{u}_\theta(t_i) + \frac{1 + \alpha\dot{u}_\theta^2 + \beta u_\theta^2}{1 + \alpha u_\theta^2}\,u_\theta(t_i)\right]^2$$

$$\mathcal{L}_{\text{ic}} = [u_\theta(0) - u_0]^2 + [\dot{u}_\theta(0) - \dot{u}_0]^2$$

Derivatives $\dot{u}_\theta$, $\ddot{u}_\theta$ are computed via PyTorch `autograd` (`create_graph=True`) over $N_r = 512$ collocation points sampled uniformly in $[0, T]$ per training step.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/awa0003a/Tapered-beam-.git
cd Tapered-beam-

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Run the full pipeline (GPU recommended)
python tapered_beam_revised.py
```

The script runs automatically through 9 stages:

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Data generation | 150 train + 20 test trajectories |
| 2 | 5-fold cross-validation | CV RMSE table |
| 3 | Repeated training (3 runs) | Trained models, mean ± std metrics |
| 4 | Multi-trajectory evaluation | RMSE over 20 test ICs |
| 5 | Robustness testing | Noise + OOD parameter tables |
| 6 | Computational cost | Parameter counts + inference times |
| 7 | Statistical significance | Paired t-test results |
| 8 | Plot generation | 30 PNG figures (300 DPI) |
| 9 | Summary table | Console output |

**Estimated runtime:** ~25–40 min on NVIDIA A100; ~3–6 hours on CPU.

---

## Generated Figures

The script generates **30 publication-quality figures** (300 DPI, white background):

| # | Figure | Description |
|---|--------|-------------|
| 01 | `01_predictions.png` | All models vs ground truth |
| 02 | `02_error_time.png` | Absolute error over time (log scale) |
| 03 | `03_phase_space.png` | Phase space portrait |
| 04 | `04_fft.png` | Frequency domain (FFT) |
| 05 | `05_energy.png` | Energy conservation test |
| 06 | `06_rmse_bar.png` | RMSE bar chart with error bars |
| 07 | `07_poincare.png` | Poincaré section |
| 08 | `08_attractor_3d.png` | 3D time-delay attractor |
| 09 | `09_psd.png` | Power spectral density (Welch) |
| 10 | `10_r2_comparison.png` | R² comparison |
| 11 | `11_correlation_heatmap.png` | Inter-model prediction correlation |
| 12 | `12_cumulative_error.png` | Cumulative absolute error |
| 13 | `13_error_distribution.png` | Error histograms (all models) |
| 14 | `14_mae_bar.png` | MAE comparison |
| 15 | `15_recurrence_plot.png` | Recurrence plot |
| 16 | `16_spectrogram.png` | Time–frequency spectrogram |
| 17 | `17_velocity.png` | Velocity field comparison |
| 18 | `18_acceleration.png` | Acceleration field comparison |
| 19 | `19_max_error.png` | Maximum absolute error |
| 20 | `20_nrmse.png` | Normalised RMSE |
| 21 | `21_pearson.png` | Pearson correlation coefficient |
| 22 | `22_windowed_rmse.png` | Time-windowed RMSE evolution |
| 23 | `23_residuals.png` | Residual scatter plots |
| 24 | `24_lyapunov.png` | Largest Lyapunov exponent |
| 25 | `25_correlation_dimension.png` | Correlation dimension |
| 26 | `26_multi_traj_boxplot.png` | **[NEW]** Multi-trajectory RMSE box plot |
| 27 | `27_cv_boxplot.png` | **[NEW]** 5-fold CV RMSE box plot |
| 28 | `28_noise_robustness.png` | **[NEW]** Noise robustness |
| 29 | `29_ood_robustness.png` | **[NEW]** OOD parameter robustness |
| 30 | `30_computational_cost.png` | **[NEW]** Parameter count + inference time |

---

## Validation Framework

### Multi-trajectory generalisation (R1.1)
20 independent test trajectories with $(u_0, \dot{u}_0) \sim \mathcal{U}(0.1,3.0) \times \mathcal{U}(-3.0,3.0)$ (seed = 99, broader than training distribution).

### 5-fold cross-validation (R1.2)
Stratified k-fold (k=5, seed=42) on the 150-trajectory training pool.

### Noise robustness (R1.3)
Additive Gaussian noise at $\varepsilon \in \{0.00, 0.01, 0.05\}$ of signal amplitude.

### OOD parameter robustness (R1.3)
Four $(\alpha, \beta)$ combinations unseen during training: $(0.3,0.8)$, $(0.3,1.2)$, $(0.7,0.8)$, $(0.7,1.2)$.

### Statistical significance (R2.5)
Paired two-tailed Student's t-tests on absolute errors, comparing each model against GRU. All comparisons: $p < 0.0001$.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{awan2024taperedbeam,
  title   = {Deep Learning Approaches for Tapered Beam Structural Dynamics:
             A Comparative Study of Six Neural Network Architectures},
  author  = {Awan, Muhammad Zeerak},
  journal = {[Journal Name]},
  year    = {2024},
  note    = {Under review},
  url     = {https://github.com/awa0003a/Tapered-beam-}
}
```

---

## Hardware

Experiments were conducted on the **Karolina HPC cluster** (IT4Innovations National Supercomputing Centre, VSB–Technical University of Ostrava, Czech Republic) using an **NVIDIA A100-SXM4-40 GB** GPU.

---

## Licence

This project is released under the [MIT Licence](LICENSE).

---

## Contact

**Muhammad Zeerak Awan**
Research Assistant — IT4Innovations, VSB–Technical University of Ostrava
ORCID: [0009-0006-7946-8563](https://orcid.org/0009-0006-7946-8563)
