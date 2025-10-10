import matplotlib.pyplot as plt
import numpy as np

plt.style.use("thesis_sty.mplstyle")

fig, axes = plt.subplots(4, 1, figsize=(6, 6.5), sharey=True)

# Band edges (MHz → GHz)
band_edges = np.array([300, 500, 566, 766, 833, 1033, 1100, 1300,
                       1366, 1566, 1633, 1833, 1900, 2100, 2166, 2366]) / 1000.0

bands = [(band_edges[i], band_edges[i+1]) for i in range(0, len(band_edges), 2)]
mag = np.array([0, 1, 1, 0])

def draw_trapezoids(ax, bands, shift=0.0, alias=False):
    for (f1, f2) in bands:
        freq = np.array([f1, f1, f2, f2]) + shift

        
        ax.fill_between(freq, 0, mag, color="gray", alpha=0.7)
        ax.plot(freq, mag, color="black", linewidth=1)

        if alias:
            freq_neg = -freq + 2*shift
            ax.fill_between(freq_neg, 0, mag, color="gray", alpha=0.2)
            ax.plot(freq_neg, mag, color="black", linewidth=1)


            freq_alias = 4.9 - freq + 2*shift
            ax.fill_between(freq_alias, 0, mag, color="gray", alpha=0.2)
            ax.plot(freq_alias, mag, color="black", linewidth=1)

def arrowed_spines(ax):
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.annotate("", xy=(ax.get_xlim()[1]+0.1, 0), xytext=(ax.get_xlim()[0], 0),
                arrowprops=dict(arrowstyle="->", color="black"))
    ax.annotate("", xy=(0, ax.get_ylim()[1]+0.1), xytext=(0, ax.get_ylim()[0]),
                arrowprops=dict(arrowstyle="->", color="black"))

# ========== PLOTS ==========
nu_s = 4.9  # Sampling frequency
configs = [
    ("Analog signal from FDM", 0.0, False, [-nu_s/2, 0, nu_s/2, nu_s]),
    ("Digitized signal with aliasing in (-1) and (+2) zones", 0.0, True, [-nu_s/2, 0, nu_s/2, nu_s]),
    (r"NCO mixing -$\nu_s/4$", -1.225, True, [-nu_s/2, 0, nu_s/2, nu_s]),
    ("Decimation by 2", -1.225, False, [-nu_s/4, 0, nu_s/4]),
]

for ax, (title, shift, alias, xticks) in zip(axes, configs):
    ax.set_xlim(-2.6, 5.1)
    ax.set_ylim(0, 1.3)


    draw_trapezoids(ax, bands, shift=shift, alias=alias)

    for x in xticks:
        ax.axvline(x=x, color="black", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$\nu_s$" if x == nu_s else 
                        r"$\nu_s/2$" if x == nu_s/2 else 
                        r"$-\nu_s/2$" if x == -nu_s/2 else
                        r"$-\nu_s/4$" if x == -nu_s/4 else
                        r"$\nu_s/4$" if x == nu_s/4 else
                        r"$0$" if x == 0 else 
                        "" for x in xticks], fontsize=10)
    ax.set_yticks([])

    arrowed_spines(ax)

plt.tight_layout(pad=1.5)
plt.savefig("../figures/quadrature_sampling_rfdc.pdf")
plt.show()
