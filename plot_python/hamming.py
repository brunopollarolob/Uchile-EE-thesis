import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis_sty.mplstyle")
# Parameters
N = 16                  # FFT length
P = 4                   # Polyphase taps (window length factor)
M = N * P               # Total window length

# Frequency axis
freq = np.linspace(-4, 4, 2048) * np.pi / N

# Direct FFT (rectangular window)
rect_win = np.ones(N)
rect_fft = np.fft.fftshift(np.fft.fft(rect_win, 2048))
rect_response = 20 * np.log10(np.abs(rect_fft) / np.max(np.abs(rect_fft)))

# Polyphase filter bank (windowed FFT with sinc window)
n = np.arange(M) - (M - 1) / 2
sinc_win = np.sinc(n / N)
win = sinc_win * np.hamming(M)
pfb_fft = np.fft.fftshift(np.fft.fft(win, 2048))
pfb_response = 20 * np.log10(np.abs(pfb_fft) / np.max(np.abs(pfb_fft)))

# Plot
plt.figure(figsize=(6, 3))
plt.plot(np.linspace(-4, 4, 2048), pfb_response, label="PFB")
plt.plot(np.linspace(-4, 4, 2048), rect_response, '--', label="FFT")
plt.ylim(-80, 5)
plt.xlim(-4, 4)
plt.xlabel("Frequency (bins)")
plt.ylabel("Power (dB)")
plt.legend()
plt.savefig("../figures/hamming_window_response_db.pdf")
plt.show()