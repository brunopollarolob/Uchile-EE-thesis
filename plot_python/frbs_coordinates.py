import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Thesis style
plt.style.use("thesis_sty.mplstyle")

# Load the CSV
df = pd.read_csv('FRB_SOURCE_1757202878644.csv')

# Get RA and Dec
ra = df['RA'].values
dec = df['Dec'].values
repeater = df['Repeater'].values

# Convert RA (hh:mm:ss) to degrees
def hms_to_deg(hms):
    h, m, s = [float(x) for x in hms.split(':')]
    return 15 * (h + m/60 + s/3600)
ra_deg = np.array([hms_to_deg(str(r)) for r in ra])

# Convert Dec (dd:mm:ss) to degrees
def dms_to_deg(dms):
    sign = -1 if str(dms).startswith('-') else 1
    parts = str(dms).replace('+','').replace('-','').split(':')
    d, m, s = [float(x) for x in parts]
    return sign * (d + m/60 + s/3600)
dec_deg = np.array([dms_to_deg(str(d)) for d in dec])

# Convert to radians for Mollweide
ra_rad = np.radians(ra_deg)
dec_rad = np.radians(dec_deg)

# Shift RA to [-180, 180] degrees for Mollweide
ra_rad = np.pi - ra_rad

plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='mollweide')

# Separate repeaters and non-repeaters
is_repeater = (repeater == 'Yes')

sc1 = ax.scatter(ra_rad[is_repeater], dec_rad[is_repeater], s=7, c='red', alpha=0.7, label='repeaters')
sc2 = ax.scatter(ra_rad[~is_repeater], dec_rad[~is_repeater], s=5, c='blue', alpha=0.7, label='one-offs')

# # Add Crab pulsar position
# crab_ra_hms = '05:34:31.97'
# crab_dec_dms = '+22:00:52.1'
# crab_ra_deg = hms_to_deg(crab_ra_hms)
# crab_dec_deg = dms_to_deg(crab_dec_dms)
# crab_ra_rad = np.radians(crab_ra_deg)
# crab_dec_rad = np.radians(crab_dec_deg)
# crab_ra_rad = np.pi - crab_ra_rad  # Shift for Mollweide
# ax.scatter(crab_ra_rad, crab_dec_rad, s=40, c='green', marker='*', label='Crab (PSR J0534+2200)')

# # Add Vela pulsar position
# vela_ra_hms = '08:35:20.655'
# vela_dec_dms = '-45:10:34.87'
# vela_ra_deg = hms_to_deg(vela_ra_hms)
# vela_dec_deg = dms_to_deg(vela_dec_dms)
# vela_ra_rad = np.radians(vela_ra_deg)
# vela_dec_rad = np.radians(vela_dec_deg)
# vela_ra_rad = np.pi - vela_ra_rad  # Shift for Mollweide
# ax.scatter(vela_ra_rad, vela_dec_rad, s=40, c='orange', marker='*', label='Vela (PSR J0835-4510)')

ax.grid(True)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.legend(loc='upper right')

# Deactivate ticks on RA axis
ax.set_xticklabels([])

plt.tight_layout()
plt.savefig('../figures/frb_sky_distribution.pdf')
plt.show()
