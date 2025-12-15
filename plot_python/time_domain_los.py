import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("thesis_sty.mplstyle")

# load voltage time series data from CSV
data = pd.read_csv('lo0.csv')
time = data['Time(s)'].values
voltage = data['CH1(V)'].values

# plot time domain signal
plt.figure(figsize=(6.8, 3.5))
plt.plot(time * 1e9, voltage*1e3)  # convert time to ns
plt.xlim(-0.0015*1e3, 0.84*1e3)  
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.tight_layout()
plt.savefig('../figures/lo_osc_time.pdf')
plt.show()