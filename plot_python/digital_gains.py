import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis_sty.mplstyle")
# Frequency axis in MHz
frequencies = np.linspace(0, 2457.6, 8192, endpoint=False, dtype=np.float32) 
digital_gains = np.load('digital_gains_calibration_coeffs_fine.npy')  # Load digital gains
plt.figure(figsize=(7, 3))
plt.plot(frequencies, 2 * digital_gains, color="black", lw=1)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Digital gain coefficient")
plt.xlim(0, 2457.6)

plt.savefig('../figures/digital_gains_lab.pdf')

plt.show()