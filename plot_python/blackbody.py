import numpy as np
from scipy import constants

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('thesis_sty.mplstyle')

T = 5500  

# Now using frequency instead of wavelength
frequency = np.logspace(11, 16, 1000)  # Hz range covering visible spectrum
# Constants
h = constants.h  # Planck constant
c = constants.c  # Speed of light
k = constants.k  # Boltzmann constant

# Planck's law for spectral brightness in terms of frequency
def planck_law_freq(nu, T):
    return (2.0 * h * nu**3 / c**2) / (np.exp(h * nu / (k * T)) - 1)

# Rayleigh-Jeans approximation in terms of frequency
def rayleigh_jeans_freq(nu, T):
    return 2.0 * nu**2 * k * T / c**2

# Wien approximation in terms of frequency
def wien_approx_freq(nu, T):
    return (2.0 * h * nu**3 / c**2) * np.exp(-h * nu / (k * T))

# Calculate spectral brightness
B_planck = planck_law_freq(frequency, T)
B_rayleigh_jeans = rayleigh_jeans_freq(frequency, T)
B_wien = wien_approx_freq(frequency, T)

plt.figure(figsize=(5, 3))

plt.loglog(frequency, B_planck, 'k--', label='Planck', linewidth=1.2)
plt.loglog(frequency, B_rayleigh_jeans, 'b--', label='Rayleigh-Jeans', linewidth=1)
plt.loglog(frequency, B_wien, 'r--', label='Wien', linewidth=1)

# Visible light range in frequency
visible_min, visible_max = c/750e-9, c/380e-9  # Converting wavelength to frequency
visible_frequencies = np.linspace(visible_min, visible_max, 100)

# Create rainbow gradient for visible spectrum
for i in range(len(visible_frequencies)-1):
    # Map frequency to RGB color (red->violet, reversed from wavelength)
    color = plt.cm.jet(1 - (visible_frequencies[i] - visible_min) / (visible_max - visible_min))
    plt.axvspan(visible_frequencies[i], visible_frequencies[i+1], alpha=0.3, color=color, ec=None)

plt.xlabel('Frequency $\\nu$ (Hz)')
plt.ylabel('Spectral brightness $B_{\\nu}$ (W m$^{-2}$ Hz$^{-1}$)')
plt.title(f'$T$ = {T} K')
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.xlim(1e11, 1e16)
plt.ylim(1e-20, 1e-5)

plt.legend(loc='upper left')

plt.savefig('../figures/blackbody_radiation_frequency.pdf')

plt.show()