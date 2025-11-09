import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.style.use("thesis_sty.mplstyle")

# Read CSV file and extract metadata
with open('tone.csv', 'r') as f:
    lines = f.readlines()

# Parse header information
n_points = int(lines[4].split(',')[1])
start_freq = float(lines[9].split(',')[1])  # Hz
stop_freq = float(lines[10].split(',')[1])   # Hz

# Read trace data (starts at line 35, column 0 = freq, column 1 = power)
data = pd.read_csv('tone.csv', skiprows=35, header=None, usecols=[0, 1])
frequencies = data[0].values  # Hz
power_dBm = data[1].values     # dBm

# Convert frequencies to MHz
faxis = frequencies / 1e6

# Peak detection
noise_floor = np.median(power_dBm)
peaks, props = find_peaks(power_dBm, height=noise_floor + 10, distance=10)
peak_freqs = faxis[peaks]
peak_powers = power_dBm[peaks]

# Sort by power
idx = np.argsort(peak_powers)[::-1]

# Identify main tones (top 4) and spurious (next 10)
main_peaks = idx[:4]
spurious_peaks = np.delete(idx, np.arange(0, 4))[:10]

main_freqs = peak_freqs[main_peaks]
main_powers = peak_powers[main_peaks]
spur_freqs = peak_freqs[spurious_peaks]
spur_powers = peak_powers[spurious_peaks]

# Metrics
signal_power = np.min(main_powers)
sfdr = signal_power - np.max(spur_powers) if len(spur_powers) > 0 else 0
snr = signal_power - noise_floor

print("Main tones (MHz / dBm):")
for f, p in zip(main_freqs, main_powers):
    print(f"{f:.1f} MHz  →  {p:.2f} dBm")
print("\nSpurious peaks (MHz / dBm):")
for f, p in zip(spur_freqs, spur_powers):
    print(f"{f:.1f} MHz  →  {p:.2f} dBm")
print(f"\nSFDR: {sfdr:.2f} dB")
print(f"Noise floor: {noise_floor:.2f} dBm")

# Plot
plt.figure(figsize=(7, 3.5))
plt.plot(faxis, power_dBm, color="navy", lw=0.8)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dBm)")
plt.xlim(0, stop_freq / 1e6)

# Auto y-limit with some margin
y_min = np.min(power_dBm) - 5
y_max = np.max(power_dBm) + 10
plt.ylim(y_min, y_max)

# Annotate main tones
for f, p in zip(main_freqs, main_powers):
    plt.annotate(f"{p:.1f} dBm",
                 xy=(f, p),
                 xytext=(f, p + 6),
                 ha="center",
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', lw=0.6))

spur_offsets = [7, 11, 9, 15, 8, 12]
for i, (f, p) in enumerate(zip(spur_freqs[:6], spur_powers[:6])):
    offset = spur_offsets[i % len(spur_offsets)]
    plt.annotate(f"{p:.1f} dBm",
                 xy=(f, p),
                 xytext=(f, p + offset),
                 ha="center",
                 fontsize=8,
                 color="darkred",
                 arrowprops=dict(arrowstyle='->', lw=0.5, color="darkred"))


plt.tight_layout()
plt.savefig('../figures/tone_spectrum.pdf')
plt.show()
