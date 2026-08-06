#!/usr/bin/env python3
"""
DEEP LEARNING TAPERED BEAM ANALYSIS — REVISED VERSION
6 Neural Network Architectures with Dynamical System Analysis

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.model_selection import KFold
import time
import warnings
from math import pi
warnings.filterwarnings('ignore')

# =============================================================================
# REPRODUCIBILITY — FIXED SEEDS (addresses R2.1)
# =============================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark    = False

# =============================================================================
# HYPERPARAMETERS — fully documented (addresses R2.1)
# =============================================================================
HP = dict(
    seq_len          = 50,
    batch_size       = 512,
    epochs           = 30,
    lr               = 1e-3,
    lstm_hidden      = 128,
    lstm_layers      = 2,
    lstm_dropout     = 0.3,
    gru_hidden       = 128,
    gru_layers       = 2,
    gru_dropout      = 0.3,
    transformer_nhead= 3,
    transformer_ff   = 128,
    transformer_drop = 0.2,
    transformer_nlayers = 4,
    tcn_kernel       = 3,
    fcnn_hidden      = 512,
    fcnn_dropout     = 0.3,
    pinn_hidden      = 128,
    pinn_layers      = 4,
    pinn_lambda_res  = 1.0,   # weight on ODE residual loss
    pinn_lambda_ic   = 10.0,  # weight on initial-condition loss
    n_train_traj     = 150,
    n_test_traj      = 20,    # multiple unseen test trajectories (R1.1)
    k_folds          = 5,     # cross-validation folds (R1.2)
    n_runs           = 3,     # repeated runs for statistics (R2.5)
    noise_levels     = [0.0, 0.01, 0.05],  # noise robustness (R1.3)
    beam_alpha_train = 0.5,
    beam_beta_train  = 1.0,
    beam_alpha_ood   = [0.3, 0.7],   # unseen parameter values (R1.3)
    beam_beta_ood    = [0.8, 1.2],
)

# =============================================================================
# PUBLICATION STYLE
# =============================================================================
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':         11,
    'axes.labelsize':    13,
    'axes.titlesize':    15,
    'axes.labelweight':  'bold',
    'axes.titleweight':  'bold',
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'legend.frameon':    True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  'black',
    'legend.fancybox':   True,
    'legend.shadow':     True,
    'figure.titlesize':  16,
    'lines.linewidth':   2.0,
    'axes.linewidth':    1.5,
    'grid.linewidth':    0.8,
    'grid.alpha':        0.4,
    'grid.linestyle':    '--',
    'axes.grid':         True,
    'axes.axisbelow':    True,
    'axes.edgecolor':    'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'text.color':        'black',
    'axes.labelcolor':   'black',
})

COLORS = {
    'LSTM':        'cyan',
    'GRU':         'gold',
    'Transformer': 'hotpink',
    'TCN':         'violet',
    'FCNN':        'orange',
    'PINN':        'red',
}
GROUND_TRUTH_COLOR = 'lime'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# TAPERED BEAM DYNAMICS
# =============================================================================
class TaperedBeam:
    """
    Governing ODE (tapered beam nonlinear oscillator):
        d²u/dt² = -[(1 + α(du/dt)² + β u²) / (1 + α u²)] · u
    Default parameters: α = 0.5, β = 1.0
    """
    def equation(self, y, t, alpha=0.5, beta=1.0):
        u, du = y
        den = 1.0 + alpha * u**2
        num = 1.0 + alpha * du**2 + beta * u**2
        return [du, -(num / den) * u]

    def solve(self, u0=1.0, du0=0.0, T=30.0, steps=10000,
              alpha=0.5, beta=1.0):
        t   = np.linspace(0, T, steps)
        sol = odeint(self.equation, [u0, du0], t, args=(alpha, beta))
        return t, sol[:, 0], sol[:, 1]

beam = TaperedBeam()


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_training_data(n_traj=150, seed=42):
    """150 diverse training trajectories (fixed seed for reproducibility)."""
    rng = np.random.RandomState(seed)
    U = []
    for _ in range(n_traj):
        u0  = rng.uniform(0.3, 2.8)
        du0 = rng.uniform(-2.5, 2.5)
        _, u, _ = beam.solve(u0, du0, T=15, steps=1200,
                             alpha=HP['beam_alpha_train'],
                             beta=HP['beam_beta_train'])
        U.append(u)
    print(f"  Generated {n_traj} training trajectories (seed={seed}).")
    return np.concatenate(U)


def generate_multi_test_trajectories(n_traj=20, seed=99):
    """
    Multiple independent test trajectories with unseen initial conditions.
    Addresses Reviewer 1 Comment 1 and Reviewer 2 Comment 3.
    """
    rng = np.random.RandomState(seed)
    # Sample ICs from regions not covered by training distribution edges
    test_ics = []
    while len(test_ics) < n_traj:
        u0  = rng.uniform(0.1, 3.0)
        du0 = rng.uniform(-3.0, 3.0)
        test_ics.append((u0, du0))

    trajectories = []
    for u0, du0 in test_ics:
        t, u, du = beam.solve(u0, du0, T=30.0, steps=10000,
                              alpha=HP['beam_alpha_train'],
                              beta=HP['beam_beta_train'])
        trajectories.append({'t': t, 'u': u, 'du': du, 'u0': u0, 'du0': du0})
    print(f"  Generated {n_traj} independent test trajectories (seed={seed}).")
    return trajectories


def generate_ood_trajectories():
    """
    Out-of-distribution trajectories: unseen α/β parameter values.
    Addresses Reviewer 1 Comment 3.
    """
    ood = []
    for alpha in HP['beam_alpha_ood']:
        for beta in HP['beam_beta_ood']:
            t, u, du = beam.solve(u0=1.5, du0=0.5, T=30.0, steps=10000,
                                  alpha=alpha, beta=beta)
            ood.append({'t': t, 'u': u, 'du': du,
                        'alpha': alpha, 'beta': beta})
    print(f"  Generated {len(ood)} OOD parameter trajectories.")
    return ood


def add_noise(u, noise_level, seed=0):
    """Additive Gaussian noise for robustness testing (R1.3)."""
    if noise_level == 0.0:
        return u
    rng = np.random.RandomState(seed)
    sigma = noise_level * np.std(u)
    return u + rng.normal(0, sigma, size=u.shape)


# =============================================================================
# DATASET
# =============================================================================
class BeamDataset(Dataset):
    def __init__(self, u, seq_len=50):
        self.seq_len = seq_len
        data = torch.FloatTensor(u).unfold(0, seq_len + 1, 1)
        self.X_seq  = []
        self.X_time = []
        self.y      = []
        for i in range(len(data)):
            seq = data[i, :-1]
            t   = torch.arange(seq_len).float() / seq_len
            du  = torch.diff(seq, prepend=seq[:1])
            X_seq = torch.stack([t, seq, du], dim=1)
            self.X_seq.append(X_seq)
            self.X_time.append(t)
            self.y.append(data[i, -1])

    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X_seq[i], self.X_time[i], self.y[i]


# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, HP['lstm_hidden'], HP['lstm_layers'],
                            batch_first=True,
                            dropout=HP['lstm_dropout'],
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(HP['lstm_hidden'] * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze()


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, HP['gru_hidden'], HP['gru_layers'],
                          batch_first=True,
                          dropout=HP['gru_dropout'],
                          bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(HP['gru_hidden'] * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1]).squeeze()


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos     = nn.Parameter(torch.randn(1, HP['seq_len'], 3))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=3,
            nhead=HP['transformer_nhead'],
            dim_feedforward=HP['transformer_ff'],
            dropout=HP['transformer_drop'],
            batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=HP['transformer_nlayers'])
        self.fc = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x + self.pos
        x = self.transformer(x)
        return self.fc(x[:, -1]).squeeze()


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(0.2)
        self.ds    = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        res = x if self.ds is None else self.ds(x)
        out = out[:, :, :res.shape[2]]
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(3, 64),
            TCNBlock(64, 128, dilation=2),
            TCNBlock(128, 128, dilation=4),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return self.fc(x.squeeze(-1)).squeeze()


class FCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HP['seq_len'] * 3, HP['fcnn_hidden']), nn.GELU(),
            nn.Dropout(HP['fcnn_dropout']),
            nn.Linear(HP['fcnn_hidden'], HP['fcnn_hidden']), nn.GELU(),
            nn.Dropout(HP['fcnn_dropout']),
            nn.Linear(HP['fcnn_hidden'], 256), nn.GELU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x.flatten(1)).squeeze()


# =============================================================================
# REVISED PINN — with proper ODE residual loss and IC loss (R1.4 / R2.4)
# =============================================================================
class PINN(nn.Module):
    """
    Physics-Informed Neural Network for the tapered beam ODE.

    The network u_θ(t) is trained by minimising:
        L = λ_res · L_res  +  λ_ic · L_ic

    where
        L_res = (1/N_r) Σ |d²u_θ/dt² + f(u_θ, du_θ/dt)|²
        L_ic  = |u_θ(0) - u0|² + |u_θ'(0) - du0|²

    and f is the tapered beam right-hand side:
        f = [(1 + α(u')² + β u²) / (1 + α u²)] · u
    """
    def __init__(self, u0_ic=1.8, du0_ic=0.0,
                 alpha=0.5, beta=1.0):
        super().__init__()
        # Store physical parameters and ICs as buffers
        self.register_buffer('u0_ic',  torch.tensor(u0_ic,  dtype=torch.float32))
        self.register_buffer('du0_ic', torch.tensor(du0_ic, dtype=torch.float32))
        self.register_buffer('alpha',  torch.tensor(alpha,  dtype=torch.float32))
        self.register_buffer('beta',   torch.tensor(beta,   dtype=torch.float32))

        # Build network: tanh activations for smooth second-order derivatives
        layers = []
        in_dim = 1
        for _ in range(HP['pinn_layers']):
            layers += [nn.Linear(in_dim, HP['pinn_hidden']), nn.Tanh()]
            in_dim  = HP['pinn_hidden']
        layers += [nn.Linear(HP['pinn_hidden'], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        """
        t : Tensor of shape (N,)  — collocation time points, requires_grad=True
        Returns u_hat of shape (N,)
        """
        t_in  = t.unsqueeze(-1)          # (N,1)
        u_hat = self.net(t_in).squeeze() # (N,)
        return u_hat

    def physics_loss(self, t_coll):
        """
        Compute ODE residual loss over collocation points t_coll.
        Uses autograd to obtain du/dt and d²u/dt².
        """
        t_coll  = t_coll.requires_grad_(True)
        u_hat   = self.forward(t_coll)

        # First derivative via autograd
        grads   = torch.autograd.grad(u_hat.sum(), t_coll,
                                      create_graph=True)[0]
        du_hat  = grads  # (N,)

        # Second derivative
        grads2  = torch.autograd.grad(du_hat.sum(), t_coll,
                                      create_graph=True)[0]
        ddu_hat = grads2  # (N,)

        # Tapered beam residual: d²u/dt² + f(u, du) = 0
        alpha = self.alpha
        beta  = self.beta
        den   = 1.0 + alpha * u_hat**2
        num   = 1.0 + alpha * du_hat**2 + beta * u_hat**2
        rhs   = -(num / den) * u_hat
        residual = ddu_hat - rhs                    # should be ≈ 0

        L_res = torch.mean(residual**2)
        return L_res

    def ic_loss(self, t0=None):
        """
        Initial-condition loss: u(0) = u0_ic,  u'(0) = du0_ic
        """
        if t0 is None:
            t0 = torch.zeros(1, device=self.u0_ic.device)
        t0        = t0.requires_grad_(True)
        u_at_0    = self.forward(t0)
        du_at_0   = torch.autograd.grad(u_at_0.sum(), t0,
                                         create_graph=True)[0]
        L_ic = (u_at_0 - self.u0_ic)**2 + (du_at_0 - self.du0_ic)**2
        return L_ic.squeeze()


# =============================================================================
# TRAINING
# =============================================================================
def build_model(name, u0_ic=1.8, du0_ic=0.0):
    if name == 'LSTM':        return LSTMModel()
    if name == 'GRU':         return GRUModel()
    if name == 'Transformer': return TransformerModel()
    if name == 'TCN':         return TCNModel()
    if name == 'FCNN':        return FCNNModel()
    if name == 'PINN':        return PINN(u0_ic=u0_ic, du0_ic=du0_ic)
    raise ValueError(f"Unknown model: {name}")


def train_model(model, loader, epochs=30, name="", verbose=False):
    model.to(device)
    opt  = optim.AdamW(model.parameters(), lr=HP['lr'], weight_decay=1e-5)
    crit = nn.MSELoss()

    if name == "PINN":
        # PINN training: physics + IC loss (addresses R1.4 / R2.4)
        model.train()
        T_end = 30.0
        for epoch in range(epochs):
            total_loss = 0.0
            for _ in range(len(loader)):      # same number of steps as data-driven models
                # Sample collocation points uniformly in [0, T]
                t_coll = (torch.rand(512) * T_end).to(device)
                t0     = torch.zeros(1).to(device)

                opt.zero_grad()
                L_res  = model.physics_loss(t_coll)
                L_ic   = model.ic_loss(t0)
                loss   = (HP['pinn_lambda_res'] * L_res
                          + HP['pinn_lambda_ic'] * L_ic)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  {name:<12s} Epoch {epoch+1:2d} | "
                      f"L_res={L_res.item():.6f}  L_ic={L_ic.item():.6f}")
    else:
        # Data-driven training
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_seq, _, y_batch in loader:
                X_seq, y_batch = X_seq.to(device), y_batch.to(device)
                opt.zero_grad()
                pred = model(X_seq)
                loss = crit(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  {name:<12s} Epoch {epoch+1:2d} | "
                      f"Loss: {total_loss/len(loader):.6f}")
    return model


# =============================================================================
# INFERENCE
# =============================================================================
def predict(model, name, test_dataset, u_true):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=512)
    preds = []
    with torch.no_grad():
        for X_seq, X_time, _ in test_loader:
            if name == "PINN":
                # PINN takes physical time in [0, T]; scale last time step
                t_phys = (X_time[:, -1] * 30.0).to(device)
                # Use no-grad forward (no autograd needed at inference)
                t_in   = t_phys.unsqueeze(-1)
                pred   = model.net(t_in).squeeze()
            else:
                pred = model(X_seq.to(device))
            preds.extend(pred.cpu().numpy())
    # Prepend the warm-up window with ground truth
    result = np.concatenate([u_true[:HP['seq_len']], np.array(preds)])
    return result


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics['RMSE']      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['MAE']       = float(mean_absolute_error(y_true, y_pred))
    metrics['R2']        = float(r2_score(y_true, y_pred))
    metrics['Max_Error'] = float(max_error(y_true, y_pred))
    metrics['NRMSE']     = metrics['RMSE'] / (y_true.max() - y_true.min())
    metrics['MAPE']      = float(np.mean(np.abs(
                            (y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    metrics['Pearson_r'],  _ = pearsonr(y_true, y_pred)
    metrics['Spearman_r'], _ = spearmanr(y_true, y_pred)
    errors = y_pred - y_true
    metrics['Mean_Error'] = float(np.mean(errors))
    metrics['Std_Error']  = float(np.std(errors))
    return metrics


def aggregate_metrics(list_of_metrics):
    """Mean ± std across repeated runs (addresses R2.5)."""
    keys = list_of_metrics[0].keys()
    agg  = {}
    for k in keys:
        vals = [m[k] for m in list_of_metrics]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return agg


# =============================================================================
# DYNAMICAL SYSTEMS ANALYSIS
# =============================================================================
def compute_lyapunov_exponent(trajectory, dt=0.003):
    n = len(trajectory)
    d = []
    for i in range(min(1000, n - 100)):
        dists = np.abs(trajectory - trajectory[i])
        dists[max(0, i - 10):min(n, i + 10)] = np.inf
        j = np.argmin(dists)
        if j < n - 100:
            for k in range(100):
                if i + k < n and j + k < n:
                    d.append(np.log(
                        np.abs(trajectory[i + k] - trajectory[j + k]) + 1e-10))
    return float(np.mean(d) / (dt * 100)) if d else 0.0


def compute_correlation_dimension(trajectory, max_r=2.0, n_points=20):
    n    = min(5000, len(trajectory))
    traj = trajectory[:n]
    radii       = np.logspace(-2, np.log10(max_r), n_points)
    correlations = []
    for r in radii:
        count   = 0
        samples = min(500, n)
        idx     = np.random.choice(n, samples, replace=False)
        for i in idx:
            dists = np.abs(traj - traj[i])
            count += np.sum(dists < r) - 1
        correlations.append(count / (samples * (samples - 1)))
    correlations = np.array(correlations)
    mask = correlations > 1e-12
    if mask.sum() > 5:
        log_r = np.log(radii[mask])
        log_c = np.log(correlations[mask])
        return float(np.polyfit(log_r, log_c, 1)[0])
    return 0.0


# =============================================================================
# CROSS-VALIDATION (addresses R1.2 / R2.1)
# =============================================================================
def run_cross_validation(U_train, model_names, k=5, seed=42):
    """
    K-fold cross-validation on training data.
    Returns per-model mean ± std RMSE across folds.
    """
    print(f"\n--- {k}-Fold Cross-Validation ---")
    dataset = BeamDataset(U_train, seq_len=HP['seq_len'])
    kf      = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = list(range(len(dataset)))

    cv_results = {name: [] for name in model_names}

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"  Fold {fold+1}/{k}")
        train_sub  = Subset(dataset, train_idx)
        val_sub    = Subset(dataset, val_idx)
        train_load = DataLoader(train_sub, batch_size=HP['batch_size'], shuffle=True)
        val_load   = DataLoader(val_sub,   batch_size=HP['batch_size'])

        for name in model_names:
            if name == 'PINN':
                # PINN CV: train on fold, evaluate ODE residual on held-out times
                model = build_model(name).to(device)
                model = train_model(model, train_load, HP['epochs'], name)
                # Evaluate on val times — extract last time from X_time
                val_preds, val_true = [], []
                model.eval()
                with torch.no_grad():
                    for _, X_time, y in val_load:
                        t_phys = (X_time[:, -1] * 30.0).to(device)
                        t_in   = t_phys.unsqueeze(-1)
                        pred   = model.net(t_in).squeeze()
                        val_preds.extend(pred.cpu().numpy())
                        val_true.extend(y.numpy())
            else:
                model = build_model(name).to(device)
                model = train_model(model, train_load, HP['epochs'], name)
                val_preds, val_true = [], []
                model.eval()
                with torch.no_grad():
                    for X_seq, _, y in val_load:
                        pred = model(X_seq.to(device))
                        val_preds.extend(pred.cpu().numpy())
                        val_true.extend(y.numpy())

            rmse = np.sqrt(mean_squared_error(val_true, val_preds))
            cv_results[name].append(rmse)

    print("\n  CV Results (RMSE):")
    for name in model_names:
        vals = cv_results[name]
        print(f"  {name:<15s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")
    return cv_results


# =============================================================================
# ROBUSTNESS TESTING (addresses R1.3)
# =============================================================================
def run_robustness_test(trained_models, u_true, t_test):
    """
    Test performance under additive Gaussian noise and OOD parameters.
    """
    print("\n--- Noise Robustness Test ---")
    noise_results = {name: {} for name in trained_models}
    for noise_lvl in HP['noise_levels']:
        u_noisy = add_noise(u_true, noise_lvl, seed=7)
        noisy_ds = BeamDataset(u_noisy, seq_len=HP['seq_len'])
        for name, model in trained_models.items():
            pred = predict(model, name, noisy_ds, u_noisy)
            min_len = min(len(u_true), len(pred))
            rmse    = np.sqrt(mean_squared_error(u_true[:min_len], pred[:min_len]))
            noise_results[name][noise_lvl] = rmse
        print(f"  Noise={noise_lvl:.2f}: " +
              " | ".join(f"{n}={noise_results[n][noise_lvl]:.5f}"
                         for n in trained_models))

    print("\n--- OOD Parameter Test ---")
    ood_trajs  = generate_ood_trajectories()
    ood_results = {name: [] for name in trained_models}
    for traj in ood_trajs:
        u_ood = traj['u']
        ood_ds = BeamDataset(u_ood, seq_len=HP['seq_len'])
        for name, model in trained_models.items():
            pred    = predict(model, name, ood_ds, u_ood)
            min_len = min(len(u_ood), len(pred))
            rmse    = np.sqrt(mean_squared_error(u_ood[:min_len], pred[:min_len]))
            ood_results[name].append(rmse)
        print(f"  α={traj['alpha']}, β={traj['beta']}: " +
              " | ".join(f"{n}={ood_results[n][-1]:.5f}"
                         for n in trained_models))

    return noise_results, ood_results


# =============================================================================
# COMPUTATIONAL COST (addresses R2.6)
# =============================================================================
def measure_inference_time(model, name, test_dataset, n_reps=10):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=512)
    times  = []
    with torch.no_grad():
        for _ in range(n_reps):
            t_start = time.perf_counter()
            for X_seq, X_time, _ in loader:
                if name == 'PINN':
                    t_phys = (X_time[:, -1] * 30.0).to(device)
                    model.net(t_phys.unsqueeze(-1))
                else:
                    model(X_seq.to(device))
            times.append(time.perf_counter() - t_start)
    return float(np.mean(times)), float(np.std(times))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# PLOT HELPERS
# =============================================================================
def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("DEEP LEARNING TAPERED BEAM ANALYSIS — REVISED")
    print("=" * 70 + "\n")

    model_names = ["LSTM", "GRU", "Transformer", "TCN", "FCNN", "PINN"]

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print("=== STEP 1: DATA GENERATION ===")
    U_train      = generate_training_data(n_traj=HP['n_train_traj'], seed=42)
    test_trajs   = generate_multi_test_trajectories(
                       n_traj=HP['n_test_traj'], seed=99)

    # Primary test trajectory (u0=1.8, du0=0.0)
    t_test, u_true, du_true = beam.solve(u0=1.8, du0=0.0, T=30.0,
                                         steps=10000,
                                         alpha=HP['beam_alpha_train'],
                                         beta=HP['beam_beta_train'])
    test_dataset = BeamDataset(u_true, seq_len=HP['seq_len'])

    # Full training dataset & loader
    full_dataset = BeamDataset(U_train, seq_len=HP['seq_len'])
    full_loader  = DataLoader(full_dataset, batch_size=HP['batch_size'],
                              shuffle=True)

    # ------------------------------------------------------------------
    # 2. K-Fold Cross-Validation (R1.2 / R2.1)
    # ------------------------------------------------------------------
    print("\n=== STEP 2: CROSS-VALIDATION ===")
    cv_results = run_cross_validation(U_train, model_names,
                                      k=HP['k_folds'], seed=42)

    # ------------------------------------------------------------------
    # 3. Repeated Training Runs for Statistics (R2.5)
    # ------------------------------------------------------------------
    print("\n=== STEP 3: REPEATED RUNS ===")
    run_metrics_all  = {name: [] for name in model_names}
    all_predictions_runs = []

    for run_idx in range(HP['n_runs']):
        run_seed = 42 + run_idx * 7
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        print(f"\n--- Run {run_idx+1}/{HP['n_runs']} (seed={run_seed}) ---")

        trained = {}
        for name in model_names:
            print(f"  Training {name} ...")
            model = build_model(name, u0_ic=1.8, du0_ic=0.0)
            trained[name] = train_model(model, full_loader,
                                        HP['epochs'], name,
                                        verbose=(run_idx == 0))

        # Predict on primary test trajectory
        predictions = {"True": u_true}
        for name, model in trained.items():
            pred = predict(model, name, test_dataset, u_true)
            predictions[name] = pred

        # Align lengths
        min_len = min(len(v) for v in predictions.values())
        for k in predictions:
            predictions[k] = predictions[k][:min_len]
        t  = t_test[:min_len]
        u  = u_true[:min_len]
        du = du_true[:min_len]

        # Metrics
        for name in model_names:
            m = compute_metrics(u, predictions[name])
            run_metrics_all[name].append(m)

        if run_idx == 0:
            # Keep the first-run models and predictions for plots
            trained_final      = trained
            predictions_final  = predictions
            t_final, u_final, du_final = t, u, du

    # Aggregate metrics: mean ± std
    agg_metrics = {name: aggregate_metrics(run_metrics_all[name])
                   for name in model_names}

    # ------------------------------------------------------------------
    # 4. Multi-Trajectory Evaluation (R1.1 / R2.3)
    # ------------------------------------------------------------------
    print("\n=== STEP 4: MULTI-TRAJECTORY EVALUATION ===")
    multi_rmse = {name: [] for name in model_names}
    for traj in test_trajs:
        t_tr, u_tr = traj['t'], traj['u']
        ds_tr = BeamDataset(u_tr, seq_len=HP['seq_len'])
        for name, model in trained_final.items():
            pred    = predict(model, name, ds_tr, u_tr)
            min_l   = min(len(u_tr), len(pred))
            rmse_tr = np.sqrt(mean_squared_error(u_tr[:min_l], pred[:min_l]))
            multi_rmse[name].append(rmse_tr)

    print("\n  Multi-Trajectory RMSE (mean ± std over 20 test ICs):")
    for name in model_names:
        vals = multi_rmse[name]
        print(f"  {name:<15s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    # ------------------------------------------------------------------
    # 5. Robustness Testing (R1.3)
    # ------------------------------------------------------------------
    print("\n=== STEP 5: ROBUSTNESS TESTING ===")
    noise_results, ood_results = run_robustness_test(
        trained_final, u_true, t_test)

    # ------------------------------------------------------------------
    # 6. Computational Costs (R2.6)
    # ------------------------------------------------------------------
    print("\n=== STEP 6: COMPUTATIONAL COSTS ===")
    comp_costs = {}
    for name, model in trained_final.items():
        inf_mean, inf_std = measure_inference_time(
            model, name, test_dataset)
        n_params = count_parameters(model)
        comp_costs[name] = {
            'n_params':    n_params,
            'inf_time_ms': inf_mean * 1000,
            'inf_std_ms':  inf_std  * 1000,
        }
        print(f"  {name:<15s}: params={n_params:,}  "
              f"inf={inf_mean*1000:.2f}±{inf_std*1000:.2f} ms")

    # ------------------------------------------------------------------
    # 7. Statistical Significance Testing (R2.5)
    # ------------------------------------------------------------------
    print("\n=== STEP 7: STATISTICAL SIGNIFICANCE ===")
    # Compare each model against the best by RMSE using paired t-test
    best_name = min(model_names,
                    key=lambda n: agg_metrics[n]['RMSE']['mean'])
    best_errors = np.abs(u_final - predictions_final[best_name])
    sig_results = {}
    for name in model_names:
        if name == best_name:
            sig_results[name] = (None, None)
            continue
        model_errors = np.abs(u_final - predictions_final[name])
        _, pval = ttest_rel(best_errors, model_errors)
        sig_results[name] = (pval, pval < 0.05)
        print(f"  {name} vs {best_name}: p={pval:.4f}  "
              f"{'*sig*' if pval<0.05 else 'n.s.'}")

    # ------------------------------------------------------------------
    # 8. PLOTTING (25 + additional figures)
    # ------------------------------------------------------------------
    print("\n=== STEP 8: GENERATING PLOTS ===")
    t  = t_final
    u  = u_final
    du = du_final
    predictions = predictions_final

    # ── Plot 1: All Model Predictions ──────────────────────────────────
    print("Plot 01/30: All Predictions...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, u, color=GROUND_TRUTH_COLOR, lw=3, label='Ground Truth', zorder=10)
    for name in model_names:
        ax.plot(t, predictions[name], '--', lw=2.5,
                label=name, color=COLORS[name], alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement u(t)")
    ax.set_title("Deep Learning Models vs Ground Truth")
    ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("01_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Error Evolution ─────────────────────────────────────────
    print("Plot 02/30: Error Evolution...")
    fig, ax = plt.subplots(figsize=(14, 7))
    for name in model_names:
        err = np.abs(u - predictions[name])
        ax.semilogy(t, err,
                    label=f"{name} (RMSE={agg_metrics[name]['RMSE']['mean']:.5f}"
                          f"±{agg_metrics[name]['RMSE']['std']:.5f})",
                    color=COLORS[name], lw=2.5, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Error (log scale)")
    ax.set_title("Prediction Error Over Time")
    ax.legend(ncol=2); _despine(ax)
    plt.tight_layout()
    plt.savefig("02_error_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Phase Space ─────────────────────────────────────────────
    print("Plot 03/30: Phase Space...")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.plot(u, du, color=GROUND_TRUTH_COLOR, lw=3.5, label='Ground Truth',
            alpha=0.9, zorder=10)
    for name in model_names:
        du_pred = np.gradient(predictions[name], t)
        ax.plot(predictions[name], du_pred, '--', lw=2.5,
                label=name, color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Displacement u"); ax.set_ylabel("Velocity du/dt")
    ax.set_title("Phase Space Portrait")
    ax.legend(ncol=2); _despine(ax)
    plt.tight_layout()
    plt.savefig("03_phase_space.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 4: FFT ─────────────────────────────────────────────────────
    print("Plot 04/30: FFT...")
    fig, ax = plt.subplots(figsize=(14, 7))
    freq = fftfreq(len(u), t[1] - t[0])[:len(u)//2]
    ax.semilogy(freq, np.abs(fft(u))[:len(u)//2],
                color=GROUND_TRUTH_COLOR, lw=3, label='Ground Truth', alpha=0.9)
    for name in model_names:
        ax.semilogy(freq, np.abs(fft(predictions[name]))[:len(u)//2],
                    '--', lw=2.5, label=name, color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Domain Analysis (FFT)")
    ax.set_xlim(0, 1.5); ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("04_fft.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 5: Energy Conservation ─────────────────────────────────────
    print("Plot 05/30: Energy Conservation...")
    energy_true = 0.5*du**2 + 0.5*u**2 + 0.25*u**4
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, energy_true, color=GROUND_TRUTH_COLOR, lw=3,
            label='Ground Truth', alpha=0.9)
    for name in model_names:
        du_p = np.gradient(predictions[name], t)
        e_p  = 0.5*du_p**2 + 0.5*predictions[name]**2 + 0.25*predictions[name]**4
        ax.plot(t, e_p, '--', lw=2.5, label=name, color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Total Energy")
    ax.set_title("Energy Conservation Test")
    ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("05_energy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 6: RMSE Bar with Error Bars (mean ± std) ───────────────────
    print("Plot 06/30: RMSE Bar...")
    fig, ax = plt.subplots(figsize=(11, 7))
    rmse_means = [agg_metrics[n]['RMSE']['mean'] for n in model_names]
    rmse_stds  = [agg_metrics[n]['RMSE']['std']  for n in model_names]
    bars = ax.bar(model_names, rmse_means, yerr=rmse_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.set_ylabel("RMSE (mean ± std)"); ax.set_title("RMSE — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val, std in zip(bars, rmse_means, rmse_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1e-5,
                f'{val:.5f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("06_rmse_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 7: Poincaré Section ─────────────────────────────────────────
    print("Plot 07/30: Poincaré Section...")
    fig, ax = plt.subplots(figsize=(11, 9))
    idx = np.where(np.diff(np.sign(du)))[0]
    ax.scatter(u[idx], du[idx], c=GROUND_TRUTH_COLOR, s=120,
               label='Ground Truth', alpha=0.9,
               edgecolors='gold', linewidths=2.5, zorder=10)
    for name in model_names:
        du_p   = np.gradient(predictions[name], t)
        idx_p  = np.where(np.diff(np.sign(du_p)))[0]
        ax.scatter(predictions[name][idx_p], du_p[idx_p], s=60,
                   alpha=0.7, label=name, color=COLORS[name],
                   edgecolors='black', linewidths=1.5)
    ax.set_xlabel("Displacement u"); ax.set_ylabel("Velocity du/dt")
    ax.set_title("Poincaré Section (Zero-Crossing Analysis)")
    ax.legend(ncol=2); _despine(ax)
    plt.tight_layout()
    plt.savefig("07_poincare.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 8: 3D Attractor ─────────────────────────────────────────────
    print("Plot 08/30: 3D Attractor...")
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(13, 10))
    ax  = fig.add_subplot(111, projection='3d')
    tau = 20
    ax.plot(u[:-2*tau], u[tau:-tau], u[2*tau:],
            color=GROUND_TRUTH_COLOR, lw=3, alpha=0.9, label='Ground Truth')
    for name in model_names:
        p = predictions[name]
        ax.plot(p[:-2*tau], p[tau:-tau], p[2*tau:],
                '--', lw=2, alpha=0.7, label=name, color=COLORS[name])
    ax.set_xlabel("u(t)"); ax.set_ylabel("u(t+τ)"); ax.set_zlabel("u(t+2τ)")
    ax.set_title("3D Phase Space Reconstruction", pad=20)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("08_attractor_3d.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 9: PSD ──────────────────────────────────────────────────────
    print("Plot 09/30: PSD...")
    fig, ax = plt.subplots(figsize=(14, 7))
    f_w, Sxx = welch(u, fs=1/(t[1]-t[0]), nperseg=1024)
    ax.semilogy(f_w, Sxx, color=GROUND_TRUTH_COLOR, lw=3,
                label='Ground Truth', alpha=0.9)
    for name in model_names:
        f_p, Sp = welch(predictions[name], fs=1/(t[1]-t[0]), nperseg=1024)
        ax.semilogy(f_p, Sp, '--', lw=2.5,
                    label=name, color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density (Welch)")
    ax.set_xlim(0, 1.0); ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("09_psd.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 10: R² Bar ──────────────────────────────────────────────────
    print("Plot 10/30: R² Bar...")
    fig, ax = plt.subplots(figsize=(11, 7))
    r2_means = [agg_metrics[n]['R2']['mean'] for n in model_names]
    r2_stds  = [agg_metrics[n]['R2']['std']  for n in model_names]
    bars = ax.bar(model_names, r2_means, yerr=r2_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.axhline(1.0, color='gold', linestyle='--', lw=2.5, alpha=0.7, label='Perfect')
    ax.set_ylabel("R² (mean ± std)"); ax.set_title("R² — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, r2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.legend(); _despine(ax)
    plt.tight_layout()
    plt.savefig("10_r2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 11: Correlation Heatmap ──────────────────────────────────────
    print("Plot 11/30: Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    pred_matrix  = np.column_stack([predictions[n] for n in model_names])
    corr_matrix  = np.corrcoef(pred_matrix.T)
    im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=0.95, vmax=1.0)
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticklabels(model_names)
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f'{corr_matrix[i,j]:.4f}', ha='center', va='center',
                    color='black', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title("Inter-Model Prediction Correlation")
    plt.tight_layout()
    plt.savefig("11_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 12: Cumulative Error ─────────────────────────────────────────
    print("Plot 12/30: Cumulative Error...")
    fig, ax = plt.subplots(figsize=(14, 7))
    for name in model_names:
        ax.plot(t, np.cumsum(np.abs(u - predictions[name])),
                label=name, color=COLORS[name], lw=2.5, alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cumulative Absolute Error")
    ax.set_title("Cumulative Error Accumulation")
    ax.legend(ncol=3); _despine(ax)
    plt.tight_layout()
    plt.savefig("12_cumulative_error.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 13: Error Distributions ─────────────────────────────────────
    print("Plot 13/30: Error Distributions...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        err = predictions[name] - u
        axes[idx].hist(err, bins=50, color=COLORS[name],
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[idx].axvline(0, color='red', linestyle='--', lw=2.5, alpha=0.8)
        axes[idx].set_title(f"{name} Error Distribution")
        axes[idx].set_xlabel("Error"); axes[idx].set_ylabel("Frequency")
        axes[idx].text(0.05, 0.95,
                       f'μ={np.mean(err):.5f}\nσ={np.std(err):.5f}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10, fontweight='bold')
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("13_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 14: MAE Bar ──────────────────────────────────────────────────
    print("Plot 14/30: MAE Bar...")
    fig, ax = plt.subplots(figsize=(11, 7))
    mae_means = [agg_metrics[n]['MAE']['mean'] for n in model_names]
    mae_stds  = [agg_metrics[n]['MAE']['std']  for n in model_names]
    bars = ax.bar(model_names, mae_means, yerr=mae_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.set_ylabel("MAE (mean ± std)"); ax.set_title("MAE — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, mae_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("14_mae_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 15: Recurrence Plot ──────────────────────────────────────────
    print("Plot 15/30: Recurrence Plot...")
    fig, ax = plt.subplots(figsize=(11, 9))
    n_s    = min(500, len(u))
    u_s    = u[:n_s]
    dist_m = np.abs(u_s[:, None] - u_s[None, :])
    thresh = 0.1 * np.std(u_s)
    ax.imshow(dist_m < thresh, cmap='binary', origin='lower', aspect='auto')
    ax.set_xlabel("Time Index"); ax.set_ylabel("Time Index")
    ax.set_title("Recurrence Plot (Ground Truth)")
    plt.tight_layout()
    plt.savefig("15_recurrence_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 16: Spectrogram ───────────────────────────────────────────────
    print("Plot 16/30: Spectrogram...")
    fig, ax = plt.subplots(figsize=(14, 7))
    f_sg, t_sg, Sxx = spectrogram(u, fs=1/(t[1]-t[0]), nperseg=256)
    im = ax.pcolormesh(t_sg, f_sg, 10*np.log10(Sxx), shading='gouraud', cmap='viridis')
    ax.set_ylabel("Frequency (Hz)"); ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram (Ground Truth)"); ax.set_ylim(0, 1.5)
    plt.colorbar(im, ax=ax, label='Power (dB)')
    plt.tight_layout()
    plt.savefig("16_spectrogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 17: Velocity ──────────────────────────────────────────────────
    print("Plot 17/30: Velocity...")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, du, color=GROUND_TRUTH_COLOR, lw=3, label='Ground Truth',
            alpha=0.9, zorder=10)
    for name in model_names:
        ax.plot(t, np.gradient(predictions[name], t), '--', lw=2.5,
                label=name, color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity du/dt")
    ax.set_title("Velocity Field Comparison")
    ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("17_velocity.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 18: Acceleration ──────────────────────────────────────────────
    print("Plot 18/30: Acceleration...")
    ddu = np.gradient(du, t)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(t, ddu, color=GROUND_TRUTH_COLOR, lw=3, label='Ground Truth',
            alpha=0.9, zorder=10)
    for name in model_names:
        du_p  = np.gradient(predictions[name], t)
        ddu_p = np.gradient(du_p, t)
        ax.plot(t, ddu_p, '--', lw=2.5, label=name,
                color=COLORS[name], alpha=0.75)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration d²u/dt²")
    ax.set_title("Acceleration Field Comparison")
    ax.legend(ncol=4); _despine(ax)
    plt.tight_layout()
    plt.savefig("18_acceleration.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 19: Max Error ─────────────────────────────────────────────────
    print("Plot 19/30: Max Error...")
    fig, ax = plt.subplots(figsize=(11, 7))
    me_means = [agg_metrics[n]['Max_Error']['mean'] for n in model_names]
    me_stds  = [agg_metrics[n]['Max_Error']['std']  for n in model_names]
    bars = ax.bar(model_names, me_means, yerr=me_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.set_ylabel("Max Error (mean ± std)")
    ax.set_title("Maximum Absolute Error — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, me_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("19_max_error.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 20: NRMSE ────────────────────────────────────────────────────
    print("Plot 20/30: NRMSE...")
    fig, ax = plt.subplots(figsize=(11, 7))
    nrmse_means = [agg_metrics[n]['NRMSE']['mean'] for n in model_names]
    nrmse_stds  = [agg_metrics[n]['NRMSE']['std']  for n in model_names]
    bars = ax.bar(model_names, nrmse_means, yerr=nrmse_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.set_ylabel("NRMSE (mean ± std)")
    ax.set_title("Normalised RMSE — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, nrmse_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.5f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("20_nrmse.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 21: Pearson r ─────────────────────────────────────────────────
    print("Plot 21/30: Pearson r...")
    fig, ax = plt.subplots(figsize=(11, 7))
    pr_means = [agg_metrics[n]['Pearson_r']['mean'] for n in model_names]
    pr_stds  = [agg_metrics[n]['Pearson_r']['std']  for n in model_names]
    bars = ax.bar(model_names, pr_means, yerr=pr_stds,
                  color=[COLORS[n] for n in model_names],
                  alpha=0.85, edgecolor='black', linewidth=2,
                  capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    ax.axhline(1.0, color='gold', linestyle='--', lw=2.5, alpha=0.7)
    ax.set_ylabel("Pearson r (mean ± std)")
    ax.set_title("Pearson Correlation — 3 Repeated Runs")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, pr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("21_pearson.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 22: Windowed RMSE ────────────────────────────────────────────
    print("Plot 22/30: Windowed RMSE...")
    fig, ax = plt.subplots(figsize=(14, 7))
    window = 500
    for name in model_names:
        rw, tw = [], []
        for i in range(0, len(u) - window, window//2):
            rw.append(np.sqrt(mean_squared_error(u[i:i+window],
                                                  predictions[name][i:i+window])))
            tw.append(t[i + window//2])
        ax.plot(tw, rw, '-o', label=name, color=COLORS[name],
                lw=2.5, markersize=6, alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Windowed RMSE")
    ax.set_title("Time-Windowed RMSE Evolution")
    ax.legend(ncol=3); _despine(ax)
    plt.tight_layout()
    plt.savefig("22_windowed_rmse.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 23: Residuals ────────────────────────────────────────────────
    print("Plot 23/30: Residuals...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, name in enumerate(model_names):
        residuals = predictions[name] - u
        axes[idx].scatter(predictions[name], residuals, c=COLORS[name],
                          alpha=0.5, s=20, edgecolors='black', linewidths=0.5)
        axes[idx].axhline(0, color='red', linestyle='--', lw=2.5, alpha=0.8)
        axes[idx].set_xlabel("Predicted"); axes[idx].set_ylabel("Residuals")
        axes[idx].set_title(f"{name} Residuals")
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("23_residuals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 24: Lyapunov Exponents ───────────────────────────────────────
    print("Plot 24/30: Lyapunov Exponents...")
    fig, ax = plt.subplots(figsize=(11, 7))
    names_w   = ['Ground Truth'] + model_names
    colors_w  = [GROUND_TRUTH_COLOR] + [COLORS[n] for n in model_names]
    lyap_vals = [compute_lyapunov_exponent(u)] + \
                [compute_lyapunov_exponent(predictions[n]) for n in model_names]
    bars = ax.bar(names_w, lyap_vals, color=colors_w,
                  alpha=0.85, edgecolor='black', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', lw=2, alpha=0.5)
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title("Largest Lyapunov Exponent")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, lyap_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("24_lyapunov.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 25: Correlation Dimension ────────────────────────────────────
    print("Plot 25/30: Correlation Dimension...")
    fig, ax = plt.subplots(figsize=(11, 7))
    cd_vals = [compute_correlation_dimension(u)] + \
              [compute_correlation_dimension(predictions[n]) for n in model_names]
    bars = ax.bar(names_w, cd_vals, color=colors_w,
                  alpha=0.85, edgecolor='black', linewidth=2)
    ax.set_ylabel("Correlation Dimension")
    ax.set_title("Correlation Dimension (Attractor Complexity)")
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, cd_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    _despine(ax)
    plt.tight_layout()
    plt.savefig("25_correlation_dimension.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 26: Multi-Trajectory RMSE Box Plot (NEW — R1.1) ──────────────
    print("Plot 26/30: Multi-Trajectory RMSE Box Plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    data_for_box = [multi_rmse[n] for n in model_names]
    bp = ax.boxplot(data_for_box, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2))
    for patch, name in zip(bp['boxes'], model_names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.75)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model"); ax.set_ylabel("RMSE")
    ax.set_title(f"Multi-Trajectory RMSE Distribution\n"
                 f"({HP['n_test_traj']} Independent Test Trajectories)")
    _despine(ax)
    plt.tight_layout()
    plt.savefig("26_multi_traj_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 27: 5-Fold CV Results (NEW — R1.2) ───────────────────────────
    print("Plot 27/30: CV Results...")
    fig, ax = plt.subplots(figsize=(12, 7))
    cv_data = [cv_results[n] for n in model_names]
    bp = ax.boxplot(cv_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=2),
                    capprops=dict(linewidth=2))
    for patch, name in zip(bp['boxes'], model_names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.75)
    ax.set_xticklabels(model_names)
    ax.set_xlabel("Model"); ax.set_ylabel("Fold RMSE")
    ax.set_title(f"{HP['k_folds']}-Fold Cross-Validation RMSE")
    _despine(ax)
    plt.tight_layout()
    plt.savefig("27_cv_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 28: Noise Robustness (NEW — R1.3) ────────────────────────────
    print("Plot 28/30: Noise Robustness...")
    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = np.arange(len(HP['noise_levels']))
    bar_width   = 0.12
    offsets     = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2,
                               len(model_names)) * bar_width
    for i, name in enumerate(model_names):
        vals = [noise_results[name][nl] for nl in HP['noise_levels']]
        ax.bar(x_positions + offsets[i], vals, bar_width,
               label=name, color=COLORS[name], alpha=0.8,
               edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'σ={nl:.2f}' for nl in HP['noise_levels']])
    ax.set_xlabel("Noise Level (fraction of signal std)")
    ax.set_ylabel("RMSE")
    ax.set_title("Robustness to Additive Gaussian Noise")
    ax.legend(ncol=3); _despine(ax)
    plt.tight_layout()
    plt.savefig("28_noise_robustness.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 29: OOD Parameter Robustness (NEW — R1.3) ───────────────────
    print("Plot 29/30: OOD Parameter Robustness...")
    ood_labels = [f"α={t['alpha']},β={t['beta']}" for t in generate_ood_trajectories()]
    fig, ax    = plt.subplots(figsize=(13, 7))
    x_pos      = np.arange(len(ood_labels))
    offsets    = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2,
                              len(model_names)) * 0.13
    for i, name in enumerate(model_names):
        ax.bar(x_pos + offsets[i], ood_results[name], 0.13,
               label=name, color=COLORS[name], alpha=0.8,
               edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ood_labels, rotation=30, ha='right')
    ax.set_xlabel("Parameter Set (α, β)")
    ax.set_ylabel("RMSE")
    ax.set_title("Out-of-Distribution Parameter Robustness")
    ax.legend(ncol=3); _despine(ax)
    plt.tight_layout()
    plt.savefig("29_ood_robustness.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ── Plot 30: Computational Cost (NEW — R2.6) ──────────────────────────
    print("Plot 30/30: Computational Cost...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    params   = [comp_costs[n]['n_params']    for n in model_names]
    inf_ms   = [comp_costs[n]['inf_time_ms'] for n in model_names]
    inf_std  = [comp_costs[n]['inf_std_ms']  for n in model_names]

    axes[0].bar(model_names, params,
                color=[COLORS[n] for n in model_names],
                alpha=0.85, edgecolor='black', linewidth=2)
    axes[0].set_ylabel("Number of Parameters")
    axes[0].set_title("Model Parameter Count")
    axes[0].tick_params(axis='x', rotation=45)
    for i, (name, val) in enumerate(zip(model_names, params)):
        axes[0].text(i, val, f'{val:,}', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].bar(model_names, inf_ms, yerr=inf_std,
                color=[COLORS[n] for n in model_names],
                alpha=0.85, edgecolor='black', linewidth=2,
                capsize=6, error_kw=dict(elinewidth=2, ecolor='black'))
    axes[1].set_ylabel("Inference Time (ms)")
    axes[1].set_title("Inference Time per Test Set")
    axes[1].tick_params(axis='x', rotation=45)
    for i, (name, val) in enumerate(zip(model_names, inf_ms)):
        axes[1].text(i, val, f'{val:.1f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle("Computational Cost Comparison", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("30_computational_cost.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 9. Summary Table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE METRICS (mean ± std over 3 runs)")
    print("=" * 80)
    metric_names_print = ['RMSE', 'MAE', 'R2', 'Max_Error', 'NRMSE',
                          'MAPE', 'Pearson_r', 'Spearman_r']
    print(f"\n{'Model':<15}", end='')
    for m in metric_names_print:
        print(f"  {m:>20}", end='')
    print()
    print("-" * (15 + 22 * len(metric_names_print)))
    for name in model_names:
        print(f"{name:<15}", end='')
        for m in metric_names_print:
            mn = agg_metrics[name][m]['mean']
            sd = agg_metrics[name][m]['std']
            print(f"  {mn:>9.5f}±{sd:<8.5f}", end='')
        print()

    print(f"\n{'Model':<15}  {'Params':>10}  {'Inf(ms)':>10}  "
          f"{'Multi-RMSE mean':>18}  {'Multi-RMSE std':>15}")
    print("-" * 75)
    for name in model_names:
        mr = np.mean(multi_rmse[name])
        ms = np.std(multi_rmse[name])
        print(f"{name:<15}  {comp_costs[name]['n_params']:>10,}  "
              f"{comp_costs[name]['inf_time_ms']:>10.2f}  "
              f"{mr:>18.6f}  {ms:>15.6f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE — 30 Plots Generated")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
