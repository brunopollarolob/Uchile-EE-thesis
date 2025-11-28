import numpy as np
import matplotlib.pyplot as plt

plt.style.use("thesis_sty.mplstyle")

average_corr = np.load('average_corr_sun_2078s.npy')

averaged_corr_12_300mhz = average_corr[2]
averaged_corr_12_400mhz = average_corr[333]
averaged_corr_12_500mhz = average_corr[630]
averaged_phase_300 = np.angle(averaged_corr_12_300mhz)
averaged_phase_400 = np.angle(averaged_corr_12_400mhz)
averaged_phase_500 = np.angle(averaged_corr_12_500mhz)
averaged_phase_300_unwrapped = np.unwrap(averaged_phase_300)
averaged_phase_400_unwrapped = np.unwrap(averaged_phase_400)
averaged_phase_500_unwrapped = np.unwrap(averaged_phase_500)

time_axis = np.arange(averaged_corr_12_300mhz.shape[0]) * (1000 * 1.03e-2)   # in seconds

# constants
omega_earth = 7.2921150e-5     # rad/s (Earth's sidereal angular rate)
deg2rad = np.pi/180.0
delta = -8.5 * deg2rad         # source declination in radians

freqs_mhz = np.array([300.6, 399.9, 489.0])
freqs_hz  = freqs_mhz * 1e6
lambdas   = 3e8 / freqs_hz     

# fit phases to get slope and covariance (slope in rad / s)

(p300, cov300) = np.polyfit(time_axis, averaged_phase_300_unwrapped, 1, cov=True)
slope_300, intercept_300 = p300
slope_err_300 = np.sqrt(cov300[0,0])

(p400, cov400) = np.polyfit(time_axis, averaged_phase_400_unwrapped, 1, cov=True)
slope_400, intercept_400 = p400
slope_err_400 = np.sqrt(cov400[0,0])

(p500, cov500) = np.polyfit(time_axis, averaged_phase_500_unwrapped, 1, cov=True)
slope_500, intercept_500 = p500
slope_err_500 = np.sqrt(cov500[0,0])

# pack slopes and errors
slopes = np.array([slope_300, slope_400, slope_500])
slope_errs = np.array([slope_err_300, slope_err_400, slope_err_500])

# compute b_EW and its uncertainty by simple linear error propagation
b_EW = (slopes * lambdas) / (2*np.pi * omega_earth * np.cos(delta))
b_EW_err = (slope_errs * lambdas) / (2*np.pi * omega_earth * np.cos(delta))

for f_mhz, s, ds, b, db in zip(freqs_mhz, slopes, slope_errs, b_EW, b_EW_err):
    print(f"{f_mhz:7.1f} MHz : slope = {s:.6e} rad/s ± {ds:.2e}  =>  b_EW = {b:.3f} m ± {db:.3f} m")

plt.figure(figsize=(6.8, 2.9))
plt.scatter(time_axis, averaged_phase_300, s=10, c='b', label='300.6 MHz')
plt.scatter(time_axis, averaged_phase_400, s=10, c='r', label='399.9 MHz')
plt.scatter(time_axis, averaged_phase_500, s=10, c='g', label='489 MHz')
plt.legend(loc='lower right')

plt.xlabel("Time (s + 2025-10-15 15:49:49.870 UTC)")
plt.ylabel("Phase (rad)")
plt.yticks([-np.pi/2, 0, np.pi/2], [r"$-\pi/2$", "0", r"$+\pi/2$"])
plt.xlim(0, time_axis[-1])
plt.ylim(-np.pi, np.pi)  
plt.tight_layout()
plt.savefig("../figures/sun_fringes.pdf")
plt.show()

# Unwrapped phase figure with linear fits
plt.figure(figsize=(6.8, 2.9))
plt.plot(time_axis, averaged_phase_300_unwrapped, 'b.', markersize=6, label='300.6 MHz (unwrapped)')
plt.plot(time_axis, slope_300 * time_axis + intercept_300, 'b--', label='Fit 300.6 MHz')
plt.plot(time_axis, averaged_phase_400_unwrapped, 'r.', markersize=6, label='399.9 MHz (unwrapped)')
plt.plot(time_axis, slope_400 * time_axis + intercept_400, 'r--', label='Fit 399.9 MHz')
plt.plot(time_axis, averaged_phase_500_unwrapped, 'g.', markersize=6, label='489 MHz (unwrapped)')
plt.plot(time_axis, slope_500 * time_axis + intercept_500, 'g--', label='Fit 489 MHz')
plt.legend(loc='lower left')
plt.xlabel("Time (s + 2025-10-15 15:49:49.870 UTC)")
plt.ylabel("Unwrapped phase (rad)")
plt.xlim(0, time_axis[-1])
plt.tight_layout()
plt.savefig("../figures/sun_fringes_unwrapped_fit.pdf")
plt.show()

# Real, imaginary and amplitude envelope figure
real_300 = np.real(averaged_corr_12_300mhz)
imag_300 = np.imag(averaged_corr_12_300mhz)
amplitude_300 = np.abs(averaged_corr_12_300mhz)



plt.figure(figsize=(6.8, 3.5))
plt.plot(time_axis, real_300, 'b-', label='Re 300.6 MHz')
plt.plot(time_axis, imag_300, 'b--', label='Im 300.6 MHz')
plt.plot(time_axis, amplitude_300, 'b:', label='|300.6 MHz|')
plt.legend(loc='upper right', fontsize='small')
plt.xlabel("Time (s + 2025-10-15 15:49:49.870 UTC)")
plt.ylabel("Correlation components")
plt.xlim(0, time_axis[-1])
plt.tight_layout()
plt.savefig("../figures/sun_fringes_real_imag_amplitude.pdf")
plt.show()