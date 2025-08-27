import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
# import scprep
# import m_phate
import m_phate.train
import m_phate.data
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from joblib import parallel_config
import keras_tuner as kt
from m_phate import M_PHATE
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import copy
import time
# import sys
import warnings

warnings.filterwarnings('ignore')


# -------------------------
# Load and preprocess data
# -------------------------
df = pd.read_csv('FR_common_sectors.csv')
df = df.sort_values(['sector', 'Year'])

features = ['Value Added [M.EUR]', 'Employment [1000 p.]', 'Energy Carrier Net Total [TJ]']
target = 'GHG emissions [kt CO2 eq.]'

scaler_x = StandardScaler()
scaler_y = StandardScaler()

processed_data = []
for sector in df['sector'].unique():
    sector_mask = df['sector'] == sector
    X = scaler_x.fit_transform(df.loc[sector_mask, features])
    y = scaler_y.fit_transform(df.loc[sector_mask, [target]])
    df.loc[sector_mask, features] = X
    df.loc[sector_mask, target] = y.flatten()
    processed_data.append(df.loc[sector_mask].copy())

df_processed = pd.concat(processed_data)
y_target = np.array(df_processed.loc[:, [target]])
X = np.array(df_processed.loc[:, features])

# -------------------------
# Shuffle before splitting
# -------------------------
X, y_target = shuffle(X, y_target, random_state=42)

# -------------------------
# Split into train/test (70/30)
# -------------------------
test_size = int(0.3 * len(X))
x_train, x_test = X[test_size:], X[:test_size]
y_train, y_test = y_target[test_size:], y_target[:test_size]

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# -------------------------
# Select trace subset from x_test
# -------------------------
np.random.seed(42)
n_trace = 50
trace_idx = np.random.choice(len(x_test), n_trace, replace=False)
x_trace = x_test[trace_idx]


# -------------------------
# KL-divergence transform BEFORE training (RBF-style)
# -------------------------
def kl_divergence_gaussians(mu1, cov1, mu2, cov2, eps=1e-12):
    """
    Batched KL(p1||p2) for batches of Gaussians.
    mu*: (B, d)
    cov*: (B, d, d)
    returns (B,) KL(p1||p2)
    """
    B, D = mu1.shape
    device = mu1.device
    eye = torch.eye(D, device=device).unsqueeze(0).expand(B, D, D)
    cov2_inv = torch.linalg.inv(cov2 + eps * eye)
    trace_term = torch.einsum('bii->b', torch.matmul(cov2_inv, cov1))
    diff = (mu2 - mu1).unsqueeze(-1)  # (B, d, 1)
    quad_term = torch.matmul(torch.matmul(diff.transpose(1, 2), cov2_inv), diff).squeeze()
    log_det_cov1 = torch.logdet(cov1 + eps * eye)
    log_det_cov2 = torch.logdet(cov2 + eps * eye)
    return 0.5 * (log_det_cov2 - log_det_cov1 - D + trace_term + quad_term)


def kl_transform_rbf_intra_inter_fast(data_np, m=2, alpha=2,
                                      intraslice_knn=7, interslice_knn=1,
                                      eps_cov=1e-8, device='cpu'):
    """
    Fast KL-RBF transform with KNN prefiltering for intraslice similarities
    and exact (small) interslice calculation for same-index series.
    Returns sim_matrix[:, :D]
    """
    torch_device = torch.device(device)
    data = torch.tensor(data_np, dtype=torch.float32, device=torch_device)
    N, D = data.shape

    pad = (m - (N % m)) % m
    if pad > 0:
        pad_rows = data[-1:].repeat(pad, 1)
        data = torch.cat([data, pad_rows], dim=0)
    Np = data.shape[0]
    n_per_slice = Np // m

    covs = torch.stack([torch.diag_embed(x ** 2 + eps_cov) for x in data])  # (Np, D, D)

    sim_matrix = torch.zeros((Np, Np), dtype=torch.float32, device=torch_device)

    # Intraslice sims with KNN prefilter
    for r in range(m):
        start = r * n_per_slice
        end = start + n_per_slice
        slice_mu = data[start:end]
        slice_cov = covs[start:end]
        S = slice_mu.shape[0]
        if S <= 1:
            continue

        with torch.no_grad():
            dists = torch.cdist(slice_mu, slice_mu)
            diag_mask = torch.eye(S, device=torch_device) * 1e9
            dists_masked = dists + diag_mask
            K = min(intraslice_knn, S - 1)
            knn_dists, knn_idx = torch.topk(dists_masked, K, largest=False)

        for i in range(S):
            neighbors = knn_idx[i]
            k = neighbors.shape[0]
            if k == 0:
                continue
            mu_i = slice_mu[i].unsqueeze(0).expand(k, D)
            cov_i = slice_cov[i].unsqueeze(0).expand(k, D, D)
            mu_j = slice_mu[neighbors]
            cov_j = slice_cov[neighbors]

            kl_vals = kl_divergence_gaussians(mu_i, cov_i, mu_j, cov_j)
            sigma_val = (knn_dists[i, -1] if knn_dists.shape[1] > 0 else torch.tensor(1.0,
                device=torch_device)).clamp(min=1e-8)
            sigma_alpha = (sigma_val ** alpha)
            sim_vals = torch.exp(- (kl_vals ** (alpha / 2.0)) / sigma_alpha)

            global_i = start + i
            global_js = (start + neighbors)
            sim_matrix[global_i, global_js] = sim_vals
            sim_matrix[global_js, global_i] = sim_vals

        sim_matrix[start:end, start:end].fill_diagonal_(1.0)

    # Interslice sims (per index across slices)
    for idx in range(n_per_slice):
        series_mu = data[idx::n_per_slice]
        series_cov = covs[idx::n_per_slice]
        if series_mu.shape[0] < 2:
            continue
        m_local = series_mu.shape[0]
        D = series_mu.shape[1]
        mu_i = series_mu.unsqueeze(1).repeat(1, m_local, 1).view(-1, D)
        mu_j = series_mu.unsqueeze(0).repeat(m_local, 1, 1).view(-1, D)
        cov_i = series_cov.unsqueeze(1).repeat(1, m_local, 1, 1).view(-1, D, D)
        cov_j = series_cov.unsqueeze(0).repeat(m_local, 1, 1, 1).view(-1, D, D)

        kl_vals = kl_divergence_gaussians(mu_i, cov_i, mu_j, cov_j)
        kl_mat = kl_vals.view(m_local, m_local)
        mask_m = torch.eye(m_local, device=torch_device) * 1e9
        sorted_rows, _ = torch.sort(kl_mat + mask_m, dim=1)
        dk = sorted_rows[:, min(1, m_local - 1)]
        eps_global = float(dk.mean().clamp(min=1e-12))
        sim = torch.exp(- kl_mat / (eps_global ** 2))
        for r in range(m_local):
            for v in range(m_local):
                if r == v:
                    continue
                i = r * n_per_slice + idx
                j = v * n_per_slice + idx
                sim_matrix[i, j] = sim[r, v]

    sim_matrix.fill_diagonal_(1.0)

    if pad > 0:
        sim_matrix = sim_matrix[:N, :N]

    sim_numpy = sim_matrix.cpu().numpy()
    return sim_numpy[:, :data_np.shape[1]]
    #return sim_numpy


# Apply KL-RBF transform
x_train_kl = kl_transform_rbf_intra_inter_fast(x_train, m=2, intraslice_knn=7, device='cpu')
x_test_kl = kl_transform_rbf_intra_inter_fast(x_test, m=2, intraslice_knn=7, device='cpu')
x_trace_kl = kl_transform_rbf_intra_inter_fast(x_trace, m=2, intraslice_knn=7, device='cpu')

n_val = min(1000, len(x_test_kl))
val_idx = np.random.default_rng(42).choice(len(x_test_kl), n_val, replace=False)
#????
#x_val = x_test_kl[val_idx]  # Inputs
#y_val = y_test[val_idx]  # Labels (one-hot encoded)

# Normalize after KL transform
scaler_kl = StandardScaler()
x_train_kl = scaler_kl.fit_transform(x_train_kl)
x_test_kl  = scaler_kl.transform(x_test_kl)
x_trace_kl = scaler_kl.transform(x_trace_kl)

x_val = x_test_kl[val_idx]
y_val = y_test[val_idx] # use the SAME scaler_y


# -------------------------
# Define BNN Builder for AutoML (KerasTuner)
# -------------------------
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


lrelu_alpha_default = 0.1


def build_bnn_model(hp: kt.HyperParameters, return_trace: bool = False):
    """Builds a Keras model with tunable hyperparameters.
    If return_trace=True, returns (training_model, model_trace_with_hidden_outputs).
    """
    input_dim = x_train_kl.shape[1]
    inputs_layer = keras.layers.Input(shape=(input_dim,), dtype='float32', name='inputs')

    # Tunables
    units1 = hp.Int('units_l1', min_value=16, max_value=128, step=16, default=32)
    units2 = hp.Int('units_l2', min_value=16, max_value=128, step=16, default=32)
    units3 = hp.Int('units_l3', min_value=16, max_value=128, step=16, default=32)
    dropout_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2)
    lr = hp.Choice('learning_rate', values=[5e-5, 1e-4, 2e-4, 5e-4], default=1e-4)
    wd = hp.Choice('weight_decay', values=[0.0, 1e-6, 1e-5, 1e-4], default=1e-5)
    lrelu_alpha = hp.Choice('lrelu_alpha', values=[0.05, 0.1, 0.2], default=lrelu_alpha_default)

    lrelu = keras.layers.LeakyReLU(alpha=lrelu_alpha)

    # Hidden stack
    h1 = keras.layers.Dense(units1, name='h1')(inputs_layer)
    h1 = lrelu(h1)
    h1 = MCDropout(dropout_rate)(h1)

    h2 = keras.layers.Dense(units2, name='h2')(h1)
    h2 = lrelu(h2)
    h2 = MCDropout(dropout_rate)(h2)

    h3 = keras.layers.Dense(units3, name='h3')(h2)
    h3 = lrelu(h3)
    h3 = MCDropout(dropout_rate)(h3)

    outputs_layer = keras.layers.Dense(1, activation='linear', name='output_regression')(h3)

    training_model = keras.models.Model(inputs=inputs_layer, outputs=outputs_layer, name='bnn_training_model')
    optimizer = AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0)
    training_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    if return_trace:
        model_trace = keras.models.Model(inputs=inputs_layer, outputs=[h1, h2, h3], name='bnn_trace_model')
        return training_model, model_trace
    return training_model


# -------------------------
# AutoML Search (Bayesian Optimization)
# -------------------------
tuner = kt.BayesianOptimization(
    hypermodel=build_bnn_model,
    objective=kt.Objective('val_mean_squared_error', direction='min'),
    max_trials=50,  # set trail times
    num_initial_points=8,
    directory='automl_kl_bnn',
    project_name='kl_bnn_mphate',
    overwrite=True,
)

es = EarlyStopping(
    monitor='val_mean_squared_error',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

tuner.search(
    x_train_kl, y_train,
    validation_data=(x_test_kl, y_test),
    epochs=200,
    callbacks=[ ],  # Only early stop here, origin is es
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best HP:", best_hp.values)

# -------------------------
# Rebuild BEST model and the trace model with identical HPs
# -------------------------
training_model, model_trace = build_bnn_model(best_hp, return_trace=True)

# Prepare TraceHistory with KL-transformed trace inputs
trace = m_phate.train.TraceHistory(x_trace_kl, model_trace) # call m_phate tracehistory func.
history = keras.callbacks.History()


# Utility to flatten weights
def get_weights_vector(model):
    """Flatten all weights into a single 1D vector."""
    weights = model.get_weights()
    return np.concatenate([w.flatten() for w in weights]).astype(np.float64)

# Callback to log weights after every epoch
weight_path = []

class WeightHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        weight_path.clear()
        weight_path.append(get_weights_vector(self.model))  # initial snapshot

    def on_epoch_end(self, epoch, logs=None):
        weight_path.append(get_weights_vector(self.model))  # snapshot per epoch

weight_history = WeightHistory()

# Simple training progress printout (you had this already)
class ProgressLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1} - "
            f"loss: {logs['loss']:.4f} - "
            f"mse: {logs['mean_squared_error']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_mse: {logs['val_mean_squared_error']:.4f}"
        )

# Main training run
with parallel_config(backend='threading', n_jobs=1):
    training_model.fit(
        x_train_kl, y_train,
        batch_size=64,
        epochs=200,
        verbose=1,
        callbacks=[
            trace,               # For M-PHATE trace data
            history,             # For loss curves
            weight_history,      # ✅ new trajectory tracker
            ProgressLogger(),    # Progress printout
            ReduceLROnPlateau(   # Optional scheduler
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ],
        validation_data=(x_test_kl, y_test)
    )

# Safety check: make sure weight_path has >1 point
if len(weight_path) < 2:
    w0 = get_weights_vector(training_model)
    w1 = w0 + 1e-3 * np.random.randn(*w0.shape)
    weight_path = [w0, w1]

print(f"Training completed with {len(weight_path)} snapshots recorded")

print("\n=== DATA SCALE ANALYSIS ===")
print(f"Original y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"Original y_test range:  [{y_test.min():.4f}, {y_test.max():.4f}]")
print(f"y_val range:            [{y_val.min():.4f}, {y_val.max():.4f}]")
print(f"---")
print(f"x_train_kl range:     [{x_train_kl.min():.4f}, {x_train_kl.max():.4f}]")
print(f"x_val range:          [{x_val.min():.4f}, {x_val.max():.4f}]")
print(f"---")
# Let's also get the mean and std of the KL data to see if the scaler worked
print(f"x_train_kl mean: {x_train_kl.mean():.4f}, std: {x_train_kl.std():.4f}")
print(f"x_val mean:      {x_val.mean():.4f}, std: {x_val.std():.4f}")

# Let's evaluate the final model on the validation set and see the loss
final_val_loss = training_model.evaluate(x_val, y_val, verbose=0)
print(f"---")
print(f"Final model evaluation on (x_val, y_val): Loss = {final_val_loss[0]:.4f}, MSE = {final_val_loss[1]:.4f}")


# 8. Loss surface + optimization paths (robust)
# -------------------------------
def trajectory_aligned_directions(w_start, w_star, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    d1 = (w_star - w_start).astype(np.float64)
    n1 = np.linalg.norm(d1)
    if n1 < 1e-12:
        # try an early snapshot to avoid zero direction
        # (use 10% into training if available)
        idx = max(0, int(0.1 * (len(weight_path) - 1)))
        d1 = (w_star - weight_path[idx]).astype(np.float64)
        n1 = np.linalg.norm(d1)
    if n1 < 1e-12:
        # final fallback: random direction
        d1 = rng.standard_normal(w_star.shape)
        n1 = np.linalg.norm(d1)
    d1 /= (n1 + 1e-12)

    # d2: random, orthogonal to d1
    d2 = rng.standard_normal(w_star.shape)
    d2 -= d1 * np.dot(d1, d2)
    d2 /= (np.linalg.norm(d2) + 1e-12)
    return d1, d2

def make_probe_model(training_model):
    probe = keras.models.clone_model(training_model)
    probe.build(training_model.input_shape)
    probe.set_weights(training_model.get_weights())
    probe.compile(optimizer=training_model.optimizer, loss='mean_squared_error')
    return probe

def set_weights_vector_in_probe(probe_model, vector, template_model):
    new_weights, idx = [], 0
    for w in template_model.get_weights():
        size = np.prod(w.shape)
        new_weights.append(vector[idx:idx + size].reshape(w.shape))
        idx += size
    probe_model.set_weights(new_weights)

#newly added
@tf.function(experimental_relax_shapes=True)
def _mse_once(x, y, model):
    y_pred = model(x, training=True)   # MC dropout on
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(y, y_pred))

def evaluate_loss_mc(model, x, y, T=5):
    vals = [] #loss
    for _ in range(T):
        vals.append(float(_mse_once(x, y, model))) #add.numpy()????
    return float(np.mean(vals))

def evaluate_loss_surface_mc(probe_model, template_model, w_star, d1, d2,
                             alphas, betas, x_data, y_data, T=3):
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float32)
    total = len(alphas) * len(betas)
    k = 0
    t0 = time.time()
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w_new = w_star + a * d1 + b * d2
            set_weights_vector_in_probe(probe_model, w_new, template_model)
            Z[i, j] = evaluate_loss_mc(probe_model, x_data, y_data, T=T)
            k += 1
            if (k % 5 == 0) or (k == total):
                dt = time.time() - t0
                print(f"  Loss surface: {k}/{total} points, elapsed {dt:.1f}s")
    return Z

def project_to_plane(w, w_star, d1, d2):
    delta = (w - w_star).astype(np.float64)
    return np.dot(delta, d1), np.dot(delta, d2)

def line_segment_path(theta_s, theta_l, n_points=30):
    alphas = np.linspace(0.0, 1.0, n_points)
    return [theta_s + a * (theta_l - theta_s) for a in alphas], alphas

print("\nPreparing loss surface visualization...")
w_star  = weight_path[-1]
w_start = weight_path[0]

rng = np.random.default_rng(123)
d1, d2 = trajectory_aligned_directions(w_start, w_star, rng=rng)

# Project recorded trajectory
path_proj = np.array([project_to_plane(w, w_star, d1, d2) for w in weight_path])

# set grid around the observed path
alpha_min, alpha_max = path_proj[:, 0].min() - 0.5, path_proj[:, 0].max() + 0.5
beta_min,  beta_max  = path_proj[:, 1].min() - 0.5, path_proj[:, 1].max() + 0.5
alphas = np.linspace(alpha_min, alpha_max, 35)
betas  = np.linspace(beta_min,  beta_max, 35)

# probe/template
template_model = keras.models.clone_model(training_model)
template_model.build(training_model.input_shape)
template_model.set_weights(training_model.get_weights())
probe_model = make_probe_model(training_model)

print("Computing loss surface ...")
loss_surface = evaluate_loss_surface_mc(
    probe_model, template_model, w_star, d1, d2,
    alphas, betas, x_val, y_val, T=3 #x_val, y_val???
)

print("Projecting actual training path...")
path_z_actual = []
for idx, w in enumerate(weight_path):
    set_weights_vector_in_probe(probe_model, w, template_model)
    path_z_actual.append(evaluate_loss_mc(probe_model, x_val, y_val, T=3))
    if (idx + 1) % 10 == 0 or idx == len(weight_path) - 1:
        print(f"  Processed {idx + 1}/{len(weight_path)} weight points")
path_z_actual = np.array(path_z_actual)
proj_path_actual = path_proj

print("Projecting line segment path...")
line_ws, _ = line_segment_path(w_start, w_star, n_points=30)
proj_path_line = np.array([project_to_plane(w, w_star, d1, d2) for w in line_ws])
path_z_line = []
for idx, w_line in enumerate(line_ws):
    set_weights_vector_in_probe(probe_model, w_line, template_model)
    path_z_line.append(evaluate_loss_mc(probe_model, x_val, y_val, T=3))
path_z_line = np.array(path_z_line)

# --- Plot (use 1D x/y to avoid shape confusion) ---
print("Generating 3D visualization...")
import plotly.graph_objects as go

fig = go.Figure()
# z must be (len(y), len(x)) when x,y are 1D
fig.add_trace(go.Surface(
    x=alphas, y=betas, z=loss_surface.T,
    colorscale='Viridis', opacity=0.8, showscale=True,
    colorbar=dict(title='MC Loss', x=0.98, len=0.7, y=0.5, thickness=15),
    cmin=min(loss_surface.min(), path_z_actual.min()),
    cmax=max(loss_surface.max(), path_z_actual.max()),
    name='MC Loss'
))

fig.add_trace(go.Scatter3d(
    x=proj_path_actual[:, 0], y=proj_path_actual[:, 1], z=path_z_actual,
    mode='lines+markers', marker=dict(size=3),
    line=dict(width=4), name='Actual Path'
))

fig.add_trace(go.Scatter3d(
    x=proj_path_line[:, 0], y=proj_path_line[:, 1], z=path_z_line,
    mode='lines+markers', marker=dict(size=3),
    line=dict(width=4, dash='dash'), name='Line Segment Path (θ^s→θ^l)'
))

fig.update_layout(
    title='BNN Loss Surface with Paths',
    scene=dict(xaxis_title='α (d1)', yaxis_title='β (d2)', zaxis_title='Loss'),
    width=1000, height=700
)
fig.show()

# -------------------------
# Extract activations recorded during training
# -------------------------
trace_data = np.array(trace.trace)
print("trace_data shape:", trace_data.shape)  # (n_epochs, n_neurons_total, n_samples)

# -------------------------
# Apply M-PHATE to visualize activations
# -------------------------
trace_data_tensor = torch.tensor(trace_data, dtype=torch.float32)
mean = trace_data_tensor.mean(dim=2, keepdim=True)
std = trace_data_tensor.std(dim=2, keepdim=True) + 1e-6
standardized_trace = (trace_data_tensor - mean) / std

m_phate_op = M_PHATE(
    n_components=3,
    n_jobs=1,
    interslice_knn=3,  # Try lower values
    intraslice_knn=50   # Explicitly set
)
m_phate_data = m_phate_op.fit_transform(standardized_trace.numpy())

# -------------------------
# Plot training loss (smoothed)
# -------------------------
#loss = history.history['mean_squared_error']
#val_loss = history.history['val_mean_squared_error']


train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])

def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed

plt.figure(figsize=(8, 5))
plt.plot(smooth_curve(train_loss), label="Train Loss (smoothed)")
plt.plot(smooth_curve(val_loss), label="Validation Loss (smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Plot M-PHATE visualization
# -------------------------
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n_epochs, n_neurons, _ = trace_data.shape
epoch_ids = np.repeat(np.arange(n_epochs), n_neurons)
# Infer 3 hidden blocks: sizes can vary per trial, but TraceHistory concatenates; we map indices
# For coloring by layer, we split n_neurons into 3 equal contiguous chunks
chunk = n_neurons // 3 if n_neurons >= 3 else 1
layer_ids = np.concatenate([
    np.full(chunk, 0),
    np.full(chunk, 1),
    np.full(n_neurons - 2 * chunk, 2)
])
layer_ids = np.tile(layer_ids, n_epochs)

fig = make_subplots(rows=1, cols=2,
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    subplot_titles=["Epoch (3D)", "Layer (3D)"])

scatter1 = go.Scatter3d(
    x=m_phate_data[:, 0], y=m_phate_data[:, 1], z=m_phate_data[:, 2],
    mode='markers',
    marker=dict(size=3, color=epoch_ids, colorscale='Viridis', opacity=0.8,
                colorbar=dict(title="Layer IDs", x=0.48, len=0.8, y=0.5, thickness=15)),
    name='Epoch'
)

scatter2 = go.Scatter3d(
    x=m_phate_data[:, 0], y=m_phate_data[:, 1], z=m_phate_data[:, 2],
    mode='markers',
    marker=dict(size=3, color=layer_ids, colorscale='Turbo', opacity=0.8,
                colorbar=dict(title="Layer IDs", x=0.98, len=0.8, y=0.5, thickness=15)),
    name='Layer'
)

fig.add_trace(scatter1, row=1, col=1)
fig.add_trace(scatter2, row=1, col=2)
fig.update_layout(height=600, width=1100,
                  title_text="M-PHATE Visualization: KL-divergenced Activations (Best HP)",
                  margin=dict(l=40, r=40, t=40, b=0))
fig.show()

