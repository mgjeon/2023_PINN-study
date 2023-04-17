import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
from datetime import datetime, timedelta

# Energy (Colab)
csv_colab = Path('ar_377_colab/energy.csv')
df_colab = pd.read_csv(csv_colab, index_col=False)
bin_colab = 2
cm_per_pixel_colab = 360e5 * bin_colab 
dV_colab = cm_per_pixel_colab**3
energy_colab = (df_colab['energy_density']*dV_colab)/1e33
date_colab = df_colab['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# Energy (Server)
csv_server = Path('ar_377_series_each_eval/energy.csv')
df_server = pd.read_csv(csv_server, index_col=False)
bin_server = 2
cm_per_pixel_server = 360e5 * bin_server 
dV_server = cm_per_pixel_server**3
energy_server = (df_server['energy_density']*dV_server)/1e33
date_server = df_server['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

plt.figure(figsize=(10, 5))
plt.plot(date_colab, energy_colab, label='Colab', color='k')
plt.plot(date_server, energy_server, label='Server', color='r')
plt.title('NOAA 11158 Total Magnetic Energy')
plt.ylabel(r'E ($10^{33}$ erg)')
plt.legend(loc='upper left')
figure_energy_path = '11158_energy.png'
plt.savefig(figure_energy_path, dpi=300)

# Free Energy (Colab)
free_csv_colab = Path('ar_377_colab/free_energy.csv')
free_df_colab = pd.read_csv(free_csv_colab, index_col=False)
free_bin_colab = 2
free_cm_per_pixel_colab = 360e5 * free_bin_colab 
free_dV_colab = free_cm_per_pixel_colab**3
free_energy_colab = (free_df_colab['free_energy_density']*free_dV_colab)/1e32
free_date_colab = free_df_colab['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# Free Energy (Server)
free_csv_server = Path('ar_377_series_each_eval/free_energy.csv')
free_df_server = pd.read_csv(free_csv_server, index_col=False)
free_bin_server = 2
free_cm_per_pixel_server = 360e5 * free_bin_server 
free_dV_server = free_cm_per_pixel_server**3
free_energy_server = (free_df_server['energy_density']*free_dV_server)/1e33
free_date_server = free_df_server['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

plt.figure(figsize=(10, 5))
plt.plot(free_date_colab, free_energy_colab, label='Colab', color='k')
plt.plot(free_date_server, free_energy_server, label='Server', color='r')
plt.title('NOAA 11158 Free Energy')
plt.ylabel(r'E ($10^{32}$ erg)')
plt.legend(loc='upper left')
figure_energy_path = '11158_free_energy.png'
plt.savefig(figure_energy_path, dpi=300)


