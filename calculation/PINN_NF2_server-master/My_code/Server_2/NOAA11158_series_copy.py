import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.dates as mdates
csvmine = 'ar_NF2_original/ar_series_377_2011-02-12T00:00:00/energy.csv'
df_mine = pd.read_csv(csvmine, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3
energy_mine = df_mine['energy_density']*dV
date_mine = df_mine['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvmine_new = 'ar_NF2_NewLoss/ar_series_377_2011-02-12T00:00:00/energy.csv'
df_mine_new = pd.read_csv(csvmine_new, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3
energy_mine_new = df_mine_new['energy_density']*dV
date_mine_new = df_mine_new['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvcolab = 'ar_377_colab/energy.csv'
df_colab = pd.read_csv(csvcolab, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3
energy_colab = df_colab['energy_density']*dV
date_colab = df_colab['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvKusano = 'Kusano/NOAA11158_Kusano.csv'
dfKusano = pd.read_csv(csvKusano, index_col=False)
bin_Kusano = 1.86
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


csv01 = 'NF2_author/series_01.csv'
df01 = pd.read_csv(csv01, index_col=False)
energy_01 = df01[' E (10^33 erg)']*1e33
date_01 = df01['TIME (10^9 sec since 1-jan-1979 )'].map(lambda x: datetime(1979, 1, 1) + timedelta(seconds=x*1e9))

csv1 = 'NF2_author/series_1.csv'
df1 = pd.read_csv(csv1, index_col=False)
energy_1 = df1[' E (10^33 erg)']*1e33
date_1 = df1['TIME (10^9 sec since 1-jan-1979 )'].map(lambda x: datetime(1979, 1, 1) + timedelta(seconds=x*1e9))

csvwb1 = 'NF2_author/series_wb1.csv'
dfwb1 = pd.read_csv(csvwb1, index_col=False)
energy_wb1 = dfwb1[' E (10^33 erg)']*1e33
date_wb1 = dfwb1['TIME (10^9 sec since 1-jan-1979 )'].map(lambda x: datetime(1979, 1, 1) + timedelta(seconds=x*1e9))

csvwb2 = 'NF2_author/series_wb2.csv'
dfwb2 = pd.read_csv(csvwb2, index_col=False)
energy_wb2 = dfwb2[' E (10^33 erg)']*1e33
date_wb2 = dfwb2['TIME (10^9 sec since 1-jan-1979 )'].map(lambda x: datetime(1979, 1, 1) + timedelta(seconds=x*1e9))


# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(date_mine, energy_mine, label='Server', color='k')
# ax.plot(date_colab, energy_colab, label='Colab', color='b', linestyle=':')
# # ax.plot(date_Kusano, energy_Kusano, label=r'Kusano et al. (2020) $\times 1.86^3$', color='g')


# ax.plot(date_01, energy_01, label=r'NF2 $\lambda_{div/ff} = 0.1$', color='#d62728')
# # ax.plot(date_1, energy_1, label=r'NF2 $\lambda_{div/ff} = 1$', color='#ff7f0e')
# ax.plot(date_wb1, energy_wb1, label=r'NF2 Wiegelmann et al. (2012) $w_d = 1$', color='#1f77b4')
# # ax.plot(date_wb2, energy_wb2, label=r'NF2 Wiegelmann et al. (2012) $w_d = 2$', color='#e377c2')

# ax.set_title('NOAA 11158 Total Magnetic Energy')
# ax.set_ylabel(r'E')
# ax.legend(loc='lower right')
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
# fig.autofmt_xdate()
# figure_energy_path = '11158_energy.png'
# fig.savefig(figure_energy_path, dpi=300)

csvmine_free = 'ar_NF2_original/ar_series_377_2011-02-12T00:00:00/free_energy.csv'
df_mine_free = pd.read_csv(csvmine_free, index_col=False)
energy_mine_free = df_mine_free['free_energy_density']*dV
date_mine_free = df_mine_free['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvmine_free_new = 'ar_NF2_NewLoss/ar_series_377_2011-02-12T00:00:00/free_energy.csv'
df_mine_free_new = pd.read_csv(csvmine_free_new, index_col=False)
energy_mine_free_new = df_mine_free_new['free_energy_density']*dV
date_mine_free_new = df_mine_free_new['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

csvcolab_free = 'ar_377_colab/free_energy.csv'
df_colab_free = pd.read_csv(csvcolab_free, index_col=False)
energy_colab_free = df_colab_free['free_energy_density']*dV

pot_energy_Kusano = dfKusano['energy_density_pot']*dV_cm3
free_energy_Kusano = energy_Kusano - pot_energy_Kusano

sever_Kusano = pd.merge(left=df_mine, right=dfKusano, how='inner', on='date')
sever_Kusano_free = sever_Kusano['energy_density_x']*dV - sever_Kusano['energy_density_pot']*dV_cm3
sever_Kusano_date = sever_Kusano['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))


free_energy_01 = df01[' Ef (10^33 erg)']*1e33
free_energy_1 = df1[' Ef (10^33 erg)']*1e33
free_energy_wb1 = dfwb1[' Ef (10^33 erg)']*1e33
free_energy_wb2 = dfwb2[' Ef (10^33 erg)']*1e33

pot_energy_wb1 = energy_wb1 - free_energy_wb1

server_wb1_free = energy_mine - pot_energy_wb1

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(date_mine_free, energy_mine_free, label='E(server) - E_pot(server)', color='k')

# ax.plot(date_mine_free_new, energy_mine_free_new, label='New E(server) - E_pot(server)', color='orange')
# plt.plot(date_colab, energy_colab_free, label='Colab', color='b', linestyle=':')

ax.plot(date_01, free_energy_01, label=r'NF2 $\lambda_{div/ff} = 0.1$', color='#d62728')
# ax.plot(date_1, free_energy_1, label=r'NF2 $\lambda_{div/ff} = 1$', color='#ff7f0e')
ax.plot(date_wb1, free_energy_wb1, label=r'NF2 Wiegelmann et al. (2012) $w_d = 1$', color='#1f77b4')
# ax.plot(date_wb2, free_energy_wb2, label=r'NF2 Wiegelmann et al. (2012) $w_d = 2$', color='#e377c2')

# ax.plot(date_wb1, server_wb1_free, label='E(Kusano) - E_pot(wb1)', color='g')

# ax.plot(sever_Kusano_date, sever_Kusano_free, label='E(server) - E_pot(Kusano)', color='r')
ax.plot(date_Kusano, free_energy_Kusano, label='E(Kusano) - E_pot(Kusano)', color='g')
ax.set_title('NOAA 11158 Free Energy')
ax.set_ylabel(r'E_free')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
figure_energy_path = '11158_free_energy.png'
fig.savefig(figure_energy_path, dpi=300)


energy_mine_pot = energy_mine - energy_mine_free
energy_colab_pot = energy_colab - energy_colab_free

energy_mine_pot_new = energy_mine_new - energy_mine_free_new


pot_energy_01 = energy_01 - free_energy_01
pot_energy_1 = energy_1 - free_energy_1

pot_energy_wb2 = energy_wb2 - free_energy_wb2


fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(date_mine, energy_mine, label='E(server)', color='k')
ax.plot(date_mine, energy_mine_pot, label='E_pot(server)', color='k', linestyle='--')

# ax.plot(date_mine_new, energy_mine_new, label='new E(server)', color='orange')
# ax.plot(date_mine_new, energy_mine_pot_new, label='new E_pot(server)', color='orange', linestyle='--')

ax.plot(date_Kusano, energy_Kusano, label='E(Kusano)', color='g')
ax.plot(date_Kusano, pot_energy_Kusano, label='E_pot(Kusano)', color='g', linestyle='--')

ax.plot(date_01, energy_01, label=r'NF2 $\lambda_{div/ff} = 0.1$', color='#d62728')
# ax.plot(date_1, energy_1, label=r'NF2 $\lambda_{div/ff} = 1$', color='#ff7f0e')
ax.plot(date_wb1, energy_wb1, label=r'NF2 Wiegelmann et al. (2012) $w_d = 1$', color='#1f77b4')
# ax.plot(date_wb2, energy_wb2, label=r'NF2 Wiegelmann et al. (2012) $w_d = 2$', color='#e377c2')

ax.plot(date_01, pot_energy_01, label=r'NF2 $\lambda_{div/ff} = 0.1$', color='#d62728', linestyle='--')
# ax.plot(date_1, pot_energy_1, label=r'NF2 $\lambda_{div/ff} = 1$', color='#ff7f0e', linestyle='--')
ax.plot(date_wb1, pot_energy_wb1, label=r'NF2 Wiegelmann et al. (2012) $w_d = 1$', color='#1f77b4', linestyle='--')
# ax.plot(date_wb2, pot_energy_wb2, label=r'NF2 Wiegelmann et al. (2012) $w_d = 2$', color='#e377c2', linestyle='--')

ax.set_title('NOAA 11158 Energy & Potential Energy')
ax.set_ylabel('E & E_pot')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
fig.autofmt_xdate()
figure_energy_path = '11158_pot_energy.png'
fig.savefig(figure_energy_path, dpi=300)
