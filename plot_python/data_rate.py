import numpy as np

import matplotlib.pyplot as plt

plt.style.use('thesis_sty.mplstyle')

# Data (s, data rate in bits/s)
time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60])
data_rate_bps = np.array([1.24168e+10, 1.24145e+10, 1.24714e+10, 1.24987e+10, 1.24989e+10, 1.23371e+10, 1.22496e+10, 1.23877e+10, 1.23949e+10, 1.24638e+10, 1.24225e+10, 1.24859e+10, 1.24849e+10, 1.24208e+10, 1.24277e+10, 1.24969e+10, 1.24182e+10, 1.2451e+10, 1.24429e+10, 1.24185e+10, 1.24211e+10, 1.24759e+10, 1.24301e+10, 1.2347e+10, 1.24345e+10, 1.24677e+10, 1.24084e+10, 1.24312e+10, 1.24974e+10, 1.24986e+10, 1.24391e+10, 1.2499e+10, 1.24986e+10, 1.24803e+10, 1.24123e+10, 1.24708e+10, 1.22862e+10, 1.2499e+10, 1.24244e+10, 1.24712e+10, 1.24994e+10, 1.21781e+10, 1.24824e+10, 1.24509e+10, 1.24561e+10, 1.24296e+10, 1.23214e+10, 1.24087e+10, 1.24595e+10, 1.2354e+10, 1.24689e+10, 1.23445e+10, 1.2386e+10, 1.2474e+10, 1.24993e+10, 1.24563e+10, 1.23041e+10, 1.24484e+10, 1.23965e+10, 1.24717e+10, 1.23627e+10])


pkt_per_second = np.array([283127, 283074, 284372, 284994, 284998, 281309, 279313, 282464, 282627, 284198, 283257, 284703, 284680, 283218, 283375, 284953, 283159, 283906, 283721, 283166, 283225, 284474, 283429, 281534, 283531, 284287, 282934, 283456, 284965, 284992, 283634, 285002, 284992, 284575, 283023, 284357, 280148, 285000, 283301, 284368, 285009, 277684, 284622, 283904, 284022, 283418, 280951, 282941, 284100, 281695, 284314, 281478, 282425, 284431, 285008, 284028, 280557, 283847, 282663, 284378, 281893])

# Convert to Gbit/s
data_rate_gbps = data_rate_bps / 1e9

# Calculate statistics
mean_rate = np.mean(data_rate_gbps)
mean_pkt_rate = np.mean(pkt_per_second)
expected_rate = 13.15
expected_pkt_rate = 300000

# Create plot
fig, ax = plt.subplots(figsize=(6.8, 3))

# Plot data with improved styling
ax.plot(time, data_rate_gbps, marker='o', linestyle='-', markersize=4, 
    linewidth=1.5, color='#2E86AB', markerfacecolor='#2E86AB', 
    markeredgecolor='white', markeredgewidth=0.5, label='Measured data rate')

# Plot reference lines
ax.axhline(y=mean_rate, color='#E63946', linestyle='--', linewidth=2, 
       label=f'Mean = {mean_rate:.2f} Gbps', alpha=0.8)
ax.axhline(y=expected_rate, color='#06A77D', linestyle='--', linewidth=2, 
       label=f'Expected = {expected_rate} Gbps', alpha=0.8)

# Styling
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel(r'Data Rate (\si{\giga\bit\per\second})', fontweight='bold')
ax.set_ylim(12, 13.4)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='best', frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig('../figures/data_rate_plot.pdf')
plt.show()

fig, ax = plt.subplots(figsize=(6.8, 3))
# Plot packet rate data with improved styling
ax.plot(time, pkt_per_second, marker='o', linestyle='-', markersize=4, 
    linewidth=1.5, color='#2E86AB', markerfacecolor='#2E86AB', 
    markeredgecolor='white', markeredgewidth=0.5, label='Measured packet rate')
# Plot reference lines
ax.axhline(y=mean_pkt_rate, color='#E63946', linestyle='--', linewidth=2, 
       label=f'Mean = {mean_pkt_rate:.0f} pkt/s', alpha=0.8)
ax.axhline(y=expected_pkt_rate, color='#06A77D', linestyle='--', linewidth=2, 
       label=f'Expected = {expected_pkt_rate} pkt/s', alpha=0.8)
# Styling
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Packet Rate (pkt/s)', fontweight='bold')
ax.set_ylim(270000, 310000)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
ax.legend(loc='best', frameon=True, fancybox=True)
plt.tight_layout()
plt.savefig('../figures/packet_rate_plot.pdf')
plt.show()