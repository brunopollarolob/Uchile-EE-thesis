import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.style.use("thesis_sty.mplstyle")

Nfft = 8192
sample_rate = 2457.6 # MHz
full_scale = 122.9 # dBFS
acc_len = 30000 # Accumulation length in samples
df = sample_rate / Nfft # Frequency resolution: 0.30MHz
fbins = np.arange(-Nfft//2, Nfft//2) # [-4096, -4095, ..., 4095]
faxis = fbins * df + (sample_rate / 2 ) # [0.00MHz, 0.30MHz, ..., 2457.60MHz]

spec = np.load("LOs.npy")
spec_dB = 10 * np.log10(spec / acc_len) - full_scale
mean_spec = np.fft.fftshift(spec_dB)
# Peak detection
noise_floor = np.median(mean_spec)
peaks, props = find_peaks(mean_spec, height=noise_floor + 10, distance=40)
peak_freqs = faxis[peaks]
peak_powers = mean_spec[peaks]

# Sort by power
idx = np.argsort(peak_powers)[::-1]

# Identify main tones and top 6 spurious
main_peaks = idx[:4]
spurious_peaks = np.delete(idx, np.arange(0, 4))[:6]

main_freqs = peak_freqs[main_peaks]
main_powers = peak_powers[main_peaks]
spur_freqs = peak_freqs[spurious_peaks]
spur_powers = peak_powers[spurious_peaks]

# Metrics
signal_power = np.max(main_powers)
sfdr = signal_power - np.max(spur_powers)
snr = signal_power - noise_floor

print("Main tones (MHz / dBm):")
for f, p in zip(main_freqs, main_powers):
    print(f"{f:.1f} MHz  →  {p:.2f} dBm")
print("\nSpurious peaks (MHz / dBm):")
for f, p in zip(spur_freqs, spur_powers):
    print(f"{f:.1f} MHz  →  {p:.2f} dBm")
print(f"\nSFDR: {sfdr:.2f} dB")
print(f"SNR:  {snr:.2f} dB")
print(f"Noise floor: {noise_floor:.2f} dBm")

plt.figure(figsize=(7.2, 4))
plt.plot(faxis, mean_spec, color="navy", lw=0.8)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dBm)")
plt.ylim(-95, 5)
plt.xlim(0, 2457.6)

# Annotate main tones
for f, p in zip(main_freqs, main_powers):
    plt.annotate(f"{p:.1f} dBm",
                 xy=(f, p),
                 xytext=(f, p + 6),
                 ha="center",
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', lw=0.6))

# Annotate top 5 spurious
for f, p in zip(spur_freqs, spur_powers):
    plt.annotate(f"{p:.1f} dBm",
                 xy=(f, p),
                 xytext=(f, p + 6),
                 ha="center",
                 fontsize=8,
                 color="darkred",
                 arrowprops=dict(arrowstyle='->', lw=0.5, color="darkred"))

plt.text(0.02, 0.95,
         f"SFDR = {sfdr:.1f} dB\nSNR = {snr:.1f} dB",
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"))

plt.tight_layout()
plt.savefig('../figures/lo_tones_spectrum.pdf')
plt.show()

