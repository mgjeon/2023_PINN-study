import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.dates as mdates

csvmine = 'ar_NF2_NewLoss/ar_series_7115_2017-09-04T00:00:00/energy.csv'
df_mine = pd.read_csv(csvmine, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3
energy_mine = df_mine['energy_density']*dV
date_mine = df_mine['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvKusano = 'Kusano/NOAA12673_Kusano.csv'
dfKusano = pd.read_csv(csvKusano, index_col=False)
bin_Kusano = 2
dx_Mm = dfKusano['dx'][0]
dy_Mm = dfKusano['dy'][0]
dz_Mm = dfKusano['dz'][0]
dx_cm = dx_Mm*1e8
dy_cm = dy_Mm*1e8
dz_cm = dz_Mm*1e8
dV_cm3 = dx_cm*dy_cm*dz_cm
dV_cm3 = dV_cm3*(bin_Kusano**3)
energy_Kusano = dfKusano['energy_density']*dV_cm3
date_Kusano = dfKusano['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


csvmine2 = 'ar_NF2_original/ar_series_7115_2017-09-04T00:00:00/energy.csv'
df_mine2 = pd.read_csv(csvmine2, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3
energy_mine2 = df_mine2['energy_density']*dV
date_mine2 = df_mine2['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


# plt.figure(figsize=(10, 5))
# plt.plot(date_mine, energy_mine, label='Server', color='k')
# plt.plot(date_Kusano, energy_Kusano, label='Kusano', color='g')
# plt.title('NOAA 12673 Total Magnetic Energy')
# plt.ylabel(r'E')
# plt.legend(loc='lower right')
# figure_energy_path = '12673_energy.png'
# plt.savefig(figure_energy_path, dpi=300)

# fig = plt.figure(figsize=(10,5))
# ax = fig.add_subplot(111)
# ax.plot(date_mine, energy_mine, label='Server(New Loss)', color='k')
# ax.plot(date_mine2, energy_mine2, label='Server(Original)', color='r')

# ax.plot(date_Kusano, energy_Kusano, label='Kusano', color='g')
# ax.set_title('NOAA 12673 Total Magnetic Energy')
# ax.set_ylabel(r'E')
# ax.legend(loc='lower right')
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
# fig.autofmt_xdate()
# figure_energy_path = '12673_energy_compare.png'
# fig.savefig(figure_energy_path, dpi=300)


csvmine_free = 'ar_NF2_NewLoss/ar_series_7115_2017-09-04T00:00:00/free_energy.csv'
df_mine_free = pd.read_csv(csvmine_free, index_col=False)
energy_mine_free = df_mine_free['free_energy_density']*dV
date_mine_free = df_mine_free['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvmine_free2 = 'ar_NF2_original/ar_series_7115_2017-09-04T00:00:00/free_energy.csv'
df_mine_free2 = pd.read_csv(csvmine_free2, index_col=False)
energy_mine_free2= df_mine_free2['free_energy_density']*dV
date_mine_free2 = df_mine_free2['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

pot_energy_Kusano = dfKusano['energy_density_pot']*dV_cm3
free_energy_Kusano = energy_Kusano - pot_energy_Kusano

# sever_Kusano = pd.merge(left=df_mine, right=dfKusano, how='inner', on='date')
# sever_Kusano_free = sever_Kusano['energy_density_x']*dV - sever_Kusano['energy_density_pot']*dV_cm3
# sever_Kusano_date = sever_Kusano['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(date_mine_free2, energy_mine_free2, label='E(server) - E_pot(server) Original', color='k')
ax.plot(date_mine_free, energy_mine_free, label='E(server) - E_pot(server) New Loss', color='orange')
# ax.plot(sever_Kusano_date, sever_Kusano_free, label='E(server) - E_pot(Kusano)', color='r')
ax.plot(date_Kusano, free_energy_Kusano, label='E(Kusano) - E_pot(Kusano)', color='g')
ax.set_title('NOAA 12673 Free Energy')
ax.set_ylabel(r'E_free')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
figure_energy_path = '12673_free_energy_compare.png'
fig.savefig(figure_energy_path, dpi=300)

energy_mine_pot = energy_mine - energy_mine_free
energy_mine_pot2 = energy_mine2 - energy_mine_free2

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(date_mine2, energy_mine2, label='E(server) Original', color='k')
ax.plot(date_mine2, energy_mine_pot2, label='E_pot(server) Original', color='k', linestyle='--')
ax.plot(date_mine, energy_mine, label='E(server) New Loss', color='orange')
ax.plot(date_mine, energy_mine_pot, label='E_pot(server) New Loss', color='orange', linestyle='--')
ax.plot(date_Kusano, energy_Kusano, label='E(Kusano)', color='g')
ax.plot(date_Kusano, pot_energy_Kusano, label='E_pot(Kusano)', color='g', linestyle='--')
ax.set_title('NOAA 12673 Energy & Potential Energy')
ax.set_ylabel('E & E_pot')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
figure_energy_path = '12673_pot_energy_compare.png'
fig.savefig(figure_energy_path, dpi=300)
