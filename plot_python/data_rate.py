import numpy as np

import matplotlib.pyplot as plt

plt.style.use('thesis_sty.mplstyle')

# Parse the data file
def parse_log_file(filename):
       time = []
       rx_rate = []
       lost_rate = []
       lost_percent = []
       data_rate = []
       
       with open(filename, 'r') as f:
              t = 0
              for line in f:
                     if 'rfsocHandler' in line:
                            parts = line.split('|')
                            
                            # Extract RX packets/s
                            rx = float(parts[2].split(':')[1].strip().split()[0])
                            rx_rate.append(rx)
                            
                            # Extract Lost packets/s and percentage
                            lost_str = parts[3].split(':')[1].strip()
                            lost = float(lost_str.split()[0])
                            lost_pct = float(lost_str.split('(')[1].split('%')[0])
                            lost_rate.append(lost)
                            lost_percent.append(lost_pct)
                            
                            # Extract data rate in Mbps
                            rate = float(parts[5].split(':')[1].strip().split()[0])
                            data_rate.append(rate)
                            
                            time.append(t)
                            t += 1
       
       return np.array(time), np.array(rx_rate), np.array(lost_rate), np.array(lost_percent), np.array(data_rate)

# Theoretical values
theoretical_rate = (5482 * 2457.6e6 / 8192) * 8 * 4 / 1e6  # Convert to Mbps
theoretical_pkt_rate = 1.2e6  # pkt/s

# Load data
time, rx_rate, lost_rate, lost_percent, data_rate = parse_log_file('test_lost.txt')

# Plot 1: Data Rate vs Time
fig1, ax1 = plt.subplots(figsize=(6.8, 3))
ax1.plot(time, data_rate / 1e3, label='Measured', linewidth=1.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel(r'Data rate (\unit{\giga\bit\per\second})')
ax1.axhline(y=theoretical_rate / 1e3, color='r', linestyle='--', label=f'Theoretical ({theoretical_rate/1e3:.2f} ' + r'\unit{\giga\bit\per\second})')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/data_rate_32_antennas.pdf')

# # Plot 2: RX Packet Rate vs Time
# fig2, ax2 = plt.subplots(figsize=(6.8, 3))
# ax2.plot(time, rx_rate / 1e6, label='Measured', linewidth=1.5)
# ax2.axhline(y=theoretical_pkt_rate / 1e6, color='r', linestyle='--', label=f'Theoretical ({theoretical_pkt_rate/1e6:.2f} Mpkt/s)')
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel(r'RX rate (Mpkt\,$s$^{-1}$)')
# ax2.legend()
# ax2.grid(True, alpha=0.3)
# plt.tight_layout()

# Plot 3: Lost Packets vs Time
fig3, ax3 = plt.subplots(figsize=(6.8, 3))
ax3.plot(time, lost_percent, linewidth=1.5, color='orange')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel(r'Lost packets (\%)')
mean_lost_percent = np.mean(lost_percent)
ax3.axhline(y=mean_lost_percent, color='r', linestyle='--', label=f'Mean ({mean_lost_percent:.3f}' + r'\%)')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/lost_packets_32_antennas.pdf')

plt.show()