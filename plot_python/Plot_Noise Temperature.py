import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use('thesis_sty.mplstyle')

def main():
    parser = argparse.ArgumentParser(description='Compare sky temperature between Carén and Calán')
    parser.add_argument('--hot', required=True, help='Path to hot load NPY file')
    parser.add_argument('--cold', required=True, help='Path to cold load NPY file')
    parser.add_argument('--sky_caren', required=True, help='Path to Carén sky NPY file')
    parser.add_argument('--sky_calan', required=True, help='Path to Calán sky NPY file')
    parser.add_argument('--sky_lab', help='Path to Lab sky NPY file (optional)')
    parser.add_argument('--save', help='Directory to save plots (optional)')
    parser.add_argument('--show', action='store_true', help='Show plots')
    args = parser.parse_args()

    # Frequency axis
    Nfft = 8192
    sample_rate = 2457.6  # MHz
    full_scale = 122.9    # Empirical value to adjust data to dBm/MHz
    acc_len = 7500
    df = sample_rate / Nfft
    roll_value = Nfft // 2

    fbins = np.arange(-Nfft//2, Nfft//2)
    faxis = fbins * df + (sample_rate / 2)

    # Load and process hot and cold loads
    def process_data(path):
        data = np.load(path)[:, 10:]
        data = np.roll(data, roll_value, axis=0)
        data = 10 * np.log10((data / acc_len) + 1) - full_scale
        return np.nanmean(data, axis=1)

    power_hot = process_data(args.hot)
    power_cold = process_data(args.cold)
    power_sky_caren = process_data(args.sky_caren)
    power_sky_calan = process_data(args.sky_calan)
    if args.sky_lab:
        power_sky_lab = process_data(args.sky_lab)


    # Calculate receiver noise temperature
    ENRdB = (14.665 + 15.42) / 2
    T_OFF = 290
    T_ON = ((10 ** (ENRdB / 10)) * T_OFF) + T_OFF
    print(f"T_ON: {T_ON}")

    Y_factor = (10 ** (power_hot / 10)) / (10 ** (power_cold / 10))
    T_rx = (T_ON - Y_factor * T_OFF) / (Y_factor - 1)

    # Fit polynomial to noise temperature
    new_freqs = np.linspace(300, 2457.6, 15, endpoint=False)
    new_noise_temperature = np.interp(new_freqs, faxis, T_rx)
    Ta = new_noise_temperature
    ta_freqs = new_freqs

    coef = np.polyfit(ta_freqs, Ta, 5)
    poly = np.poly1d(coef)
    Ta_fit = poly(faxis)

    # Compute sky temperatures
    pow_sky_caren = 10 ** (power_sky_caren / 10)
    pow_sky_calan = 10 ** (power_sky_calan / 10)
    if args.sky_lab:
        pow_sky_lab = 10 ** (power_sky_lab / 10)
    pow_ref_lineal = 10 ** (power_cold / 10)

    ratio_caren = pow_sky_caren / pow_ref_lineal
    ratio_calan = pow_sky_calan / pow_ref_lineal

    Ts_caren = (290 + Ta_fit) * ratio_caren - Ta_fit
    Ts_calan = (290 + Ta_fit) * ratio_calan - Ta_fit
    if args.sky_lab:
        ratio_lab = pow_sky_lab / pow_ref_lineal
        Ts_lab = (290 + Ta_fit) * ratio_lab - Ta_fit

    # --- Plot Sky Temperature Comparison ---
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.plot(faxis, Ts_caren, label="Laguna Carén", color="C0", alpha=0.7)
    ax.plot(faxis, Ts_calan, label="Cerro Calán", color="C1", alpha=0.7)
    if args.sky_lab:
        ax.plot(faxis, Ts_lab, label="Laboratory", color="C2", alpha=0.7)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Temperature (K)")
    ax.set_yscale('log')
    ax.set_xlim(0, 2457.6)
    ax.set_ylim(50, 1e10)
    ax.set_yticks([100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ax.set_xticks(np.arange(0, 2500, 250))
    ax.legend()

    # Inset for 300–500 MHz band
    axins = ax.inset_axes([0.46, 0.5, 0.25, 0.35])  

    idx_min, idx_max = 1000, 1672
    axins.plot(faxis[idx_min:idx_max], Ts_caren[idx_min:idx_max], color="C0", alpha=0.7)
    axins.plot(faxis[idx_min:idx_max], Ts_calan[idx_min:idx_max], color="C1", alpha=0.7)
    if args.sky_lab:
        axins.plot(faxis[idx_min:idx_max], Ts_lab[idx_min:idx_max], color="C2", alpha=0.7)
    axins.set_xlim(300, 500)
    axins.set_yscale('log')
    axins.set_xticks([300, 400, 500])
    axins.set_yticks([1e3, 1e5, 1e7, 1e9])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.tight_layout()

    # fig_t_rx = plt.figure(figsize=(6.8, 3))
    # plt.plot(faxis, T_rx, label=r"Measured $T_{\text{rx}}$", color="C3", alpha=0.7)
    # plt.plot(faxis, Ta_fit, label=r"Fitted $T_{\text{rx}}^{\text{fit}}$", color="C4", alpha=0.7)
    # plt.xlabel("Frequency (MHz)")
    # plt.ylabel("Temperature (K)")
    # plt.legend()
    # plt.xlim(0, 2457.6)
    # plt.ylim(0, 2000)

    # fig_t_rx.tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        plt.savefig(os.path.join(args.save, 'sky_temperature_comparison.pdf'),
                    dpi=250, bbox_inches='tight')

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
