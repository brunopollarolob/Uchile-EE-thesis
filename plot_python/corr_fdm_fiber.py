import numpy as np
import matplotlib.pyplot as plt
plt.style.use("thesis_sty.mplstyle")

frequencies = np.linspace(300, 501.6, 672, endpoint=False)  

mean_corr = np.load('mean_corr.npz', allow_pickle=True)

fig, axes = plt.subplots(8, 8, figsize=(12, 12))

for i in range(8):
    for j in range(8):
        key = f"corr_{i}{j}"
        phase = np.angle(mean_corr[key])  

        ax = axes[i, j]
        ax.scatter(frequencies, phase, s=0.25)
        ax.set_ylim(-np.pi, np.pi) 
        ax.set_title(r"$S_{{" + f"{i}{j}" + "} }$")

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
plt.savefig('../figures/corr_fdm_load_fiber.pdf')
plt.show()

plt.show()