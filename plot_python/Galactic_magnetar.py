import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fitburst.analysis.model import SpectrumModeler

plt.style.use('pres_sty.mplstyle')

# Parámetros principales
t_tot = 15  # Total pulse time in seconds
n_times = 1000
n_channels = 1024
freqs = np.linspace(400, 800, n_channels, endpoint=False, dtype=np.float32)  # MHz
times = np.linspace(0, t_tot, n_times, dtype=np.float32)

def normalize(arr_data):
    """
    Normalize the input array by subtracting the median and dividing by the
    standard deviation along the last axis.

    Parameters:
        arr_data (numpy.ndarray): Input array to be normalized.

    Returns:
        numpy.ndarray: Normalized array.
    """
    arr = arr_data.copy()
    # Subtract the median along the last axis from each element
    arr -= np.nanmedian(arr, axis=-1)[..., None]

    # Divide by the standard deviation along the last axis
    arr /= np.nanstd(arr, axis=-1)[..., None]

    return arr

burst_parameters = {
    "amplitude": [1, 1],
    "arrival_time": [0.5, 0.54],
    "burst_width": [0.000585, 0.000335],
    "dm": [332.7206, 332.7206],
    "dm_index": [-2., -2.],
    "ref_freq": [800, 800],
    "scattering_index": [-4., -4.],
    "scattering_timescale": [0.759, 0.759],
    "spectral_index": [-5.75, 3.61],
    "spectral_running": [1.0, -19.9],
}
num_components = len(burst_parameters["dm"])
model = SpectrumModeler(freqs, times, num_components=num_components)
model.update_parameters(burst_parameters)
spectrum_model = model.compute_model()

# De-dispersado
times_dedispersed = np.linspace(0.2, 0.8, 50, dtype=np.float32)
burst_parameters_dd = burst_parameters.copy()
burst_parameters_dd["dm"] = [0, 0]
model_dd = SpectrumModeler(freqs, times_dedispersed, num_components=num_components)
model_dd.update_parameters(burst_parameters_dd)
spectrum_model_dd = model_dd.compute_model()

# Añadir ruido
noise_level = 5
spectrum_model += noise_level * np.random.normal(size=spectrum_model.shape)
spectrum_model_dd += noise_level * np.random.normal(size=spectrum_model_dd.shape)

# Normalizar
spectrum_model = normalize(spectrum_model)
spectrum_model_dd = normalize(spectrum_model_dd)

# ========= PLOTTING =========
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], hspace=0.05, wspace=0.15)

# Panel izquierdo: dynamic spectrum
ax_dyn = fig.add_subplot(gs[0, 0])
plim = np.percentile(spectrum_model, 99.9)
im = ax_dyn.pcolormesh(times, freqs, spectrum_model, cmap="viridis", vmin=0, vmax=plim, shading="auto")
ax_dyn.set_xlabel("Time (s)")
ax_dyn.set_ylabel("Frequency (MHz)")

# Panel derecho: espectro de-dispersado
ax_dyn_dd = fig.add_subplot(gs[0, 1])
plim_dd = np.percentile(spectrum_model_dd, 99.9)
im2 = ax_dyn_dd.pcolormesh(times_dedispersed, freqs, spectrum_model_dd, cmap="viridis", vmin=0, vmax=plim, shading="auto")
ax_dyn_dd.set_xlabel("Time (s)")
ax_dyn_dd.tick_params(labelleft=False)

# Crear una barra de color compartida
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label="Amplitude (arb.)")

plt.show()

