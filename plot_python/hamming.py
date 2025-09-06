import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set latin modern as font
mpl.style.use('thesis_sty.mplstyle')

# Parameters
N = 64  # Number of points

# Frequency axis (normalized to bin width)
f = np.linspace(0, 4, 1000)

# FFT (rectangular window) amplitude response
def rect_window_response(f, N):
    # sinc function (normalized)
    return np.abs(np.sinc(f))

# Hamming window amplitude response
def hamming_window_response(f, N):
    # Hamming window coefficients
    n = np.arange(N)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    # Frequency response
    response = np.zeros_like(f)
    for i, fi in enumerate(f):
        response[i] = np.abs(np.sum(w * np.exp(-2j * np.pi * fi * n / N)) / np.sum(w))
    return response

# Compute responses
rect_resp = rect_window_response(f, N)
hamming_resp = hamming_window_response(f, N)

# Convert to dB, avoid log(0) by setting a small floor
eps = 1e-12
rect_resp_db = 20 * np.log10(np.maximum(rect_resp, eps))
hamming_resp_db = 20 * np.log10(np.maximum(hamming_resp, eps))

# Plot
plt.figure(figsize=(5, 3))
plt.plot(f, rect_resp_db, 'k--', label='FFT')
plt.plot(f, hamming_resp_db, 'k-', label='Hamming-windowed')
plt.xlabel('Frequency (Normalized to bin width)')
plt.ylabel('Amplitude [dB]')
plt.legend()
plt.xlim(0, 4)
plt.ylim(-80, 1)
plt.tight_layout()
plt.savefig('../figures/hamming_window_response_db.pdf')
plt.show()