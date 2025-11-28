import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis_sty.mplstyle")
frequencies = np.linspace(0, 2457.6, 8192, endpoint=False, dtype=np.float32) 
fdm_chains_spectrum = np.load('spectrum_fdm_load_shift21_gain150.npy')
fdm_chains_spectrum_dB = 10 * np.log10(np.fft.fftshift(fdm_chains_spectrum) )
bands = [(1000, 1672), (1880, 2552), (2768, 3440), (3664, 4336),
         (4552, 5224), (5440, 6112), (6336, 7008), (7216, 7888)]

fig, ax = plt.subplots(figsize=(6.8, 3))
ax.plot(frequencies, fdm_chains_spectrum_dB, lw=1)
# draw shaded bands and labels
for i, (b0, b1) in enumerate(bands):
    b0 = int(b0)
    b1 = int(b1)
    if b0 >= len(frequencies):
        continue
    b1 = min(b1, len(frequencies) - 1)
    f0 = frequencies[b0]
    f1 = frequencies[b1]
    ax.axvspan(f0, f1, color='0.9', zorder=0)
    ax.text((f0 + f1) / 2, 1.02, rf'$\Delta\nu_{{{i}}}$',
            ha='center', va='bottom', transform=ax.get_xaxis_transform(),
            color='0.5', fontsize=10)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Power (dB arb.)")
ax.set_xlim(0, 2457.6)
plt.tight_layout()

plt.savefig('../figures/fdm_chains_spectrum_lab.pdf')
plt.show()