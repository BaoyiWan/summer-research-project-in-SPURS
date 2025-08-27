import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import scprep
import torch
import m_phate
import m_phate.train
import m_phate.data
from joblib import parallel_config
import keras_tuner as kt
from tensorflow.keras.optimizers.experimental import AdamW
import copy
import time
import sys

# -------------------------------
# 1. GPU setup
# -------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
    except RuntimeError as e:
        print(e)

# -------------------------------
# 2. Data loading
# -------------------------------
x_train, x_test, y_train, y_test = m_phate.data.load_mnist()

# Create smaller validation subset for faster loss surface computation
val_idx = np.random.choice(len(x_test), 1000, replace=False)
x_val = x_test[val_idx]
y_val = y_test[val_idx]


# -------------------------------
# 3. Model definition (BNN w/ MC dropout)
# -------------------------------
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)  # always apply dropout


lrelu_alpha_default = 0.1


def build_bnn_model(hp: kt.HyperParameters, return_trace: bool = False):
    input_dim = x_train.shape[1]
    inputs_layer = keras.layers.Input(shape=(input_dim,), dtype='float32', name='inputs')

    # Tunables
    units1 = hp.Int('units_l1', 32, 128, step=32, default=32)
    units2 = hp.Int('units_l2', 32, 128, step=32, default=32)
    units3 = hp.Int('units_l3', 32, 128, step=32, default=32)
    dropout_rate = hp.Float('dropout', 0.2, 0.5, step=0.1, default=0.3)
    lr = hp.Choice('learning_rate', [5e-5, 1e-4, 2e-4, 5e-4], default=1e-4)
    wd = hp.Choice('weight_decay', [0.0, 1e-6, 1e-5, 1e-4], default=1e-5)
    lrelu_alpha = hp.Choice('lrelu_alpha', [0.05, 0.1, 0.2], default=lrelu_alpha_default)

    lrelu = keras.layers.LeakyReLU(alpha=lrelu_alpha)

    h1 = keras.layers.Dense(units1, name='h1')(inputs_layer)
    h1 = lrelu(h1)
    h1 = MCDropout(dropout_rate)(h1)

    h2 = keras.layers.Dense(units2, name='h2')(h1)
    h2 = lrelu(h2)
    h2 = MCDropout(dropout_rate)(h2)

    h3 = keras.layers.Dense(units3, name='h3')(h2)
    h3 = lrelu(h3)
    h3 = MCDropout(dropout_rate)(h3)

    outputs_layer = keras.layers.Dense(10, activation='softmax', name='output_all')(h3)

    training_model = keras.models.Model(inputs=inputs_layer, outputs=outputs_layer, name='bnn_training_model')
    optimizer = AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0)
    training_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    if return_trace:
        model_trace = keras.models.Model(inputs=inputs_layer, outputs=[h1, h2, h3], name='bnn_trace_model')
        return training_model, model_trace
    return training_model


# -------------------------------
# 4. Hyperparameter search
# -------------------------------
tuner = kt.Hyperband(
    build_bnn_model,
    objective='val_categorical_accuracy',
    max_epochs=20,
    factor=3,
    directory='automl_dir',
    project_name='bnn_mnist'
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

tuner.search(x_train, y_train, validation_data=(x_test, y_test),
             epochs=200, callbacks=[early_stop], verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best HP:", best_hp.values)

# -------------------------------
# 5. Trace setup
# -------------------------------
np.random.seed(42)
trace_idx = []
for i in range(10):
    trace_idx.append(np.random.choice(np.argwhere(y_test[:, i] == 1).flatten(), 5, replace=False))
trace_idx = np.concatenate(trace_idx)
x_trace = x_test[trace_idx]

training_model, model_trace = build_bnn_model(best_hp, return_trace=True)
trace = m_phate.train.TraceHistory(x_trace, model_trace)
history = keras.callbacks.History()


# -------------------------------
# 6. Weight utilities
# -------------------------------
def get_weights_vector(model):
    weights = model.get_weights()
    return np.concatenate([w.flatten() for w in weights])


weight_path = []


class WeightHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weight_path.append(get_weights_vector(self.model))


weight_history = WeightHistory()


# -------------------------------
# 7. Training with fallback
# -------------------------------
class ProgressLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch + 1} - loss: {logs['loss']:.4f} - acc: {logs['categorical_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f} - val_acc: {logs['val_categorical_accuracy']:.4f}")


min_epochs = 30
max_epochs = 100

# Custom training loop with fallback
for attempt in range(2):  # Try with and without early stopping
    try:
        with parallel_config(backend='threading', n_jobs=1):
            callbacks = [trace, history, weight_history, ProgressLogger()]

            if attempt == 0:
                print("\nTraining with EarlyStopping...")
                callbacks.append(early_stop)
                training_model.fit(
                    x_train, y_train,
                    batch_size=64, epochs=max_epochs,
                    verbose=0,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test)
                )
            else:
                print("\nTraining without EarlyStopping...")
                training_model.fit(
                    x_train, y_train,
                    batch_size=64, epochs=min_epochs,
                    verbose=0,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test)
                )
        break  # Exit loop if successful
    except Exception as e:
        print(f"Training error: {e}")
        if attempt == 1:
            print("Both training attempts failed. Using random weights.")
            weight_path = [get_weights_vector(training_model)] * min_epochs

# Ensure we have at least min_epochs of weight history
if len(weight_path) < min_epochs:
    weight_path.extend([weight_path[-1]] * (min_epochs - len(weight_path)))

print(f"Training completed with {len(weight_path)} epochs recorded")


# -------------------------------
# 8. Loss surface + optimization paths
# -------------------------------
# --- New helper functions ---
def trajectory_aligned_directions(w_start, w_star, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    # First direction = actual optimization trajectory
    d1 = w_star - w_start
    d1 /= (np.linalg.norm(d1) + 1e-12)

    # Second direction = random, orthogonal to d1
    d2 = rng.standard_normal(w_star.shape)
    d2 -= d1 * np.dot(d1, d2)
    d2 /= (np.linalg.norm(d2) + 1e-12)

    return d1, d2


def make_probe_model(training_model):
    probe = keras.models.clone_model(training_model)
    probe.build(training_model.input_shape)
    probe.set_weights(training_model.get_weights())
    probe.compile(optimizer=training_model.optimizer,
                  loss='categorical_crossentropy')
    return probe


def set_weights_vector_in_probe(probe_model, vector, template_model):
    new_weights, idx = [], 0
    for w in template_model.get_weights():
        size = np.prod(w.shape)
        new_weights.append(vector[idx:idx + size].reshape(w.shape))
        idx += size
    probe_model.set_weights(new_weights)


def evaluate_loss_mc(model, x, y, T=5):
    losses = []
    for _ in range(T):
        y_pred = model(x, training=True)  # forward pass w/ MC dropout
        loss_val = tf.keras.losses.categorical_crossentropy(y, y_pred)
        losses.append(tf.reduce_mean(loss_val).numpy())
    return float(np.mean(losses))


def evaluate_loss_surface_mc(probe_model, template_model, w_star, d1, d2,
                             alphas, betas, x_data, y_data, T=3):
    Z = np.zeros((len(alphas), len(betas)), dtype=np.float32)
    total_points = len(alphas) * len(betas)
    processed = 0
    start_time = time.time()

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w_new = w_star + a * d1 + b * d2
            set_weights_vector_in_probe(probe_model, w_new, template_model)
            Z[i, j] = evaluate_loss_mc(probe_model, x_data, y_data, T=T)

            processed += 1
            if processed % 5 == 0 or processed == total_points:
                elapsed = time.time() - start_time
                print(f"  Loss surface: {processed}/{total_points} points - "
                      f"Elapsed: {elapsed:.1f}s - "
                      f"ETA: {elapsed / processed * (total_points - processed):.1f}s")
                sys.stdout.flush()
    return Z


def project_to_plane(w, w_star, d1, d2):
    delta = w - w_star
    return np.dot(delta, d1), np.dot(delta, d2)


def line_segment_path(theta_s, theta_l, n_points=30):
    alphas = np.linspace(0.0, 1.0, n_points)
    return [theta_s + a * (theta_l - theta_s) for a in alphas], alphas


print("\nPreparing loss surface visualization...")
w_star = weight_path[-1]  # Final weights
w_start = weight_path[0]  # Initial weights
rng = np.random.default_rng(123)
d1, d2 = trajectory_aligned_directions(w_start, w_star, rng=rng)

path_proj = [project_to_plane(w, w_star, d1, d2) for w in weight_path]
path_proj = np.array(path_proj)

alpha_min, alpha_max = path_proj[:, 0].min() - 0.5, path_proj[:, 0].max() + 0.5
beta_min, beta_max = path_proj[:, 1].min() - 0.5, path_proj[:, 1].max() + 0.5

alphas = np.linspace(alpha_min, alpha_max, 25)
betas = np.linspace(beta_min, beta_max, 25)

template_model = keras.models.clone_model(training_model)
template_model.build(training_model.input_shape)
template_model.set_weights(training_model.get_weights())
probe_model = make_probe_model(training_model)

print("Computing loss surface (this may take a while)...")
loss_surface = evaluate_loss_surface_mc(
    probe_model, template_model, w_star, d1, d2,
    alphas, betas, x_val, y_val, T=3
)

print("Projecting actual training path...")
proj_path_actual, path_z_actual = [], []
for idx, w in enumerate(weight_path):
    a, b = project_to_plane(w, w_star, d1, d2)
    proj_path_actual.append([a, b])
    set_weights_vector_in_probe(probe_model, w, template_model)
    path_z_actual.append(evaluate_loss_mc(probe_model, x_val, y_val, T=3))

    if (idx + 1) % 10 == 0 or idx == len(weight_path) - 1:
        print(f"  Processed {idx + 1}/{len(weight_path)} weight points")
proj_path_actual = np.array(proj_path_actual)

print("Projecting line segment path...")
line_ws, line_alphas = line_segment_path(w_start, w_star, n_points=30)
proj_path_line, path_z_line = [], []
for idx, w_line in enumerate(line_ws):
    a, b = project_to_plane(w_line, w_star, d1, d2)
    proj_path_line.append([a, b])
    set_weights_vector_in_probe(probe_model, w_line, template_model)
    path_z_line.append(evaluate_loss_mc(probe_model, x_val, y_val, T=3))

    if (idx + 1) % 5 == 0 or idx == len(line_ws) - 1:
        print(f"  Processed {idx + 1}/{len(line_ws)} line points")
proj_path_line = np.array(proj_path_line)

# --- Plot ---
print("Generating 3D visualization...")
import plotly.graph_objects as go

A, B = np.meshgrid(alphas, betas)

fig = go.Figure()
fig.add_trace(go.Surface(
    x=A, y=B, z=loss_surface.T,
    colorscale='Viridis', opacity=0.75,
    showscale=True, colorbar=dict(title='MC Loss'),
    cmin=min(loss_surface.min(), min(path_z_actual)),
    cmax=max(loss_surface.max(), max(path_z_actual))
))
fig.add_trace(go.Scatter3d(x=proj_path_actual[:, 0], y=proj_path_actual[:, 1], z=path_z_actual,
                           mode='lines+markers', marker=dict(size=3),
                           line=dict(width=4), name='Actual Path'))
fig.add_trace(go.Scatter3d(x=proj_path_line[:, 0], y=proj_path_line[:, 1], z=path_z_line,
                           mode='lines+markers', marker=dict(size=3),
                           line=dict(width=4, dash='dash'),
                           name='Line Segment Path (θ^s→θ^l)'))
fig.update_layout(title='BNN Loss Surface with Paths',
                  scene=dict(xaxis_title='α (d1)', yaxis_title='β (d2)', zaxis_title='Loss'),
                  width=1000, height=700)
fig.show()

# -------------------------------
# 9. M-PHATE embedding (3D scatter panes)
# -------------------------------
print("\nComputing M-PHATE embedding...")
trace_data = np.array(trace.trace)
units_per_layer = [int(t.shape[-1]) for t in model_trace.outputs]
n_neurons = int(np.sum(units_per_layer))
layer_ids = np.tile(np.concatenate([np.repeat(i, u) for i, u in enumerate(units_per_layer)]),
                    trace_data.shape[0])
epoch = np.repeat(np.arange(trace_data.shape[0]), n_neurons)
digit_ids = y_test.argmax(1)[trace_idx]

trace_data_tensor = torch.tensor(trace_data, dtype=torch.float32)
mean = trace_data_tensor.mean(dim=0, keepdim=True)
std = trace_data_tensor.std(dim=0, keepdim=True) + 1e-6
standardized_trace = (trace_data_tensor - mean) / std

digit_activity = np.array([np.sum(np.abs(standardized_trace.numpy()[:, :, digit_ids == digit]), axis=2)
                           for digit in np.unique(digit_ids)])
most_active_digit = np.argmax(digit_activity, axis=0).flatten()

print("Running M-PHATE (this may take several minutes)...")
with parallel_config(backend='threading', n_jobs=1):
    m_phate_op = m_phate.M_PHATE(n_components=3, n_jobs=1)
    m_phate_data = m_phate_op.fit_transform(standardized_trace.numpy())
    print("M-PHATE variance:", np.var(m_phate_data, axis=0))


def smooth_curve(points, factor=0.9):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed


plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)
plt.show()

print("Generating M-PHATE visualization...")
from plotly.subplots import make_subplots

# Create figure with adjusted spacing
fig = make_subplots(rows=1, cols=3,
                    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    subplot_titles=["Epoch (3D)", "Layer (3D)", "Most Active Digit (3D)"],
                    horizontal_spacing=0.08)  # Increased horizontal spacing

# Add first trace (Epoch)
fig.add_trace(go.Scatter3d(x=m_phate_data[:, 0], y=m_phate_data[:, 1], z=m_phate_data[:, 2],
                           mode='markers',
                           marker=dict(size=3, color=epoch, colorscale='Viridis',
                                       colorbar=dict(title="Epoch", x=0.28, len=0.25, y=0.5, thickness=15),
                                       showscale=True)),
              row=1, col=1)

# Add second trace (Layer IDs)
fig.add_trace(go.Scatter3d(x=m_phate_data[:, 0], y=m_phate_data[:, 1], z=m_phate_data[:, 2],
                           mode='markers',
                           marker=dict(size=3, color=layer_ids, colorscale='Cividis',
                                       colorbar=dict(title="Layer IDs", x=0.63, len=0.25, y=0.5, thickness=15),
                                       showscale=True)),
              row=1, col=2)

# Add third trace (Most Active Digit)
fig.add_trace(go.Scatter3d(x=m_phate_data[:, 0], y=m_phate_data[:, 1], z=m_phate_data[:, 2],
                           mode='markers',
                           marker=dict(size=3, color=most_active_digit, colorscale='Plasma',
                                       colorbar=dict(title="Most Active Digit", x=0.98, len=0.25, y=0.5, thickness=15),
                                       showscale=True)),
              row=1, col=3)

# Update layout with proper margins and size
fig.update_layout(height=600, width=2000,  # Increased width to accommodate colorbars
                  showlegend=False,
                  margin=dict(l=50, r=100, b=50, t=80))  # Increased right margin for colorbar space

fig.show()

print("All visualizations completed successfully!")
