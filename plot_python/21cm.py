# imports
import numpy as np
import struct
import matplotlib.pyplot as plt

plt.style.use('pres_sty.mplstyle')


# reading .dat file
with open("C:\\Users\\bruno\\Downloads\\cassiopeia.dat", 'rb') as f:
    nf, nt, alt, az = struct.unpack('iiff', f.read(16))
    lf = np.fromfile(f, dtype=np.float32, count=nf)
    lt = np.fromfile(f, dtype=np.float64, count=nt)
    sp = np.fromfile(f, dtype=np.float32, count=nf)
    bl = np.fromfile(f, dtype=np.float32, count=nf)
    wf = np.fromfile(f, dtype=np.float32, count=nf * nt)

# Step 1: Define the region around the peak
peak_index = np.argmax(sp)  # assuming both have the peak at the same index
width = 10  # or whatever width is appropriate
start = max(0, peak_index - width // 2)
end = min(len(sp), peak_index + width // 2 + 1)

# Step 2: Estimate the offset in that region
offset = np.mean(sp[start:end] - bl[start:end])

# Step 3: Apply correction to the baseline
bl_corrected = bl + offset  # or bl_corrected = bl + offset if bl is too low

# Find the second peak after 1420.4 MHz
mask = lf > 1420.4
indices_after_1420_4 = np.where(mask)[0]
spectrum_after_1420_4 = sp[mask] / bl_corrected[mask]
freq_after_1420_4 = lf[mask]

# Find peaks (local maxima)
peaks = []
for i in range(1, len(spectrum_after_1420_4) - 1):
    if spectrum_after_1420_4[i] > spectrum_after_1420_4[i-1] and spectrum_after_1420_4[i] > spectrum_after_1420_4[i+1]:
        peaks.append(i)

# Get the second peak
if len(peaks) >= 2:
    second_peak_idx = peaks[1]
    peak_freq = freq_after_1420_4[second_peak_idx]
    peak_intensity = spectrum_after_1420_4[second_peak_idx]
else:
    # Fallback: use the highest point after 1420.4 MHz
    second_peak_idx = np.argmax(spectrum_after_1420_4)
    peak_freq = freq_after_1420_4[second_peak_idx]
    peak_intensity = spectrum_after_1420_4[second_peak_idx]

# plot averaged spectrum, baseline, and the normalized spectrum
plt.figure(figsize=(4, 3))
plt.plot(lf, sp/bl_corrected)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Intensity (dB arb.)')
plt.xlim(1416, 1424)  # Limit x-axis to 1424 MHz
plt.ylim(1,2.6)

# Add arrow pointing to the second peak
plt.annotate('Línea 21 cm HI', 
             xy=(peak_freq, peak_intensity), 
             xytext=(peak_freq + 1, peak_intensity + 0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=1),
             fontsize=10, color='red')
plt.savefig('21cm_line.pdf')
plt.show()