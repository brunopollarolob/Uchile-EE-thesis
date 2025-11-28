import numpy as np
import matplotlib.pyplot as plt
plt.style.use("thesis_sty.mplstyle")

raw_data = np.load('charts8_calibrated_shift22_fine.npy')
def unpack_complex(bytearray):
    real = (bytearray & 0xF0) >> 4
    imag = (bytearray & 0x0F)
    real = real.astype(np.int8)
    imag = imag.astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16

    return real + 1j*imag
complex_data = np.zeros(raw_data.shape, dtype=np.complex64)
for i in range(complex_data.shape[1]):
    complex_data[:, i] = unpack_complex(raw_data[:, i])

complex_data = np.fft.fftshift(complex_data, axes=0)
chain0 = complex_data[1000:1672, :] # 300-500 MHz
chain1 = np.conjugate(np.flip(complex_data[1880:2552, :], axis=0))
chain2 = np.conjugate(np.flip(complex_data[2768:3440, :], axis=0))
chain3 = np.conjugate(np.flip(complex_data[3664:4336, :], axis=0))
chain4 = complex_data[4552:5224, :]
chain5 = complex_data[5440:6112, :]
chain6 = complex_data[6336:7008, :]
chain7 = complex_data[7216:7888, :]

frequencies = np.linspace(300, 501.6, 672, endpoint=False)  

fig = plt.figure(figsize=(6.8, 4))
plt.plot(frequencies, 10 * np.log10(np.abs(chain0)**2 + 1e-3).mean(axis=1), label='Chain 0', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain1)**2 + 1e-3).mean(axis=1), label='Chain 1', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain2)**2 + 1e-3).mean(axis=1), label='Chain 2', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain3)**2 + 1e-3).mean(axis=1), label='Chain 3', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain4)**2 + 1e-3).mean(axis=1), label='Chain 4', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain5)**2 + 1e-3).mean(axis=1), label='Chain 5', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain6)**2 + 1e-3).mean(axis=1), label='Chain 6', alpha=0.7)
plt.plot(frequencies, 10 * np.log10(np.abs(chain7)**2 + 1e-3).mean(axis=1), label='Chain 7', alpha=0.7)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Amplitude (dB arb.)')
plt.ylim(-10, 20)
plt.xlim(300, 501.3)
plt.legend()
plt.tight_layout()
plt.savefig('../figures/amplitude_charts8.pdf')
plt.show()

bands = [chain0, chain1, chain2, chain3, chain4, chain5, chain6, chain7]

correlation_products = {}

for i in range(len(bands)):
    for j in range(len(bands)):
        key = f"corr_{i}{j}"  
        correlation_products[key] = bands[i] * np.conj(bands[j])

fig, axes = plt.subplots(8, 8, figsize=(12, 12))

for i in range(8):
    for j in range(8):
        key = f"corr_{i}{j}"
        phase = np.angle(np.mean(correlation_products[key], axis=1))  

        ax = axes[i, j]
        ax.scatter(frequencies, phase, s=0.25)
        ax.set_ylim(-np.pi, np.pi)  

        if i == 7:
            ax.set_xlabel("Frequency (MHz)", fontsize=12)
            ax.tick_params(axis='x', labelsize=12)
        else:
            ax.set_xticks([])
            
        if j == 0:
            ax.set_ylabel("Phase (rad)", fontsize=12)
            ax.set_yticks([-np.pi/2, 0, np.pi/2])
            ax.set_yticklabels([r"$-\pi/2$", "0", r"$+\pi/2$"])
            ax.tick_params(axis='y', labelsize=12)
        else:
            ax.set_yticks([])
            
plt.tight_layout()
plt.savefig('../figures/corr_charts8.pdf')
plt.show()

plt.show()