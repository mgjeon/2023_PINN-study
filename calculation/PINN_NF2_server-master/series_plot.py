import argparse
import json
from datetime import datetime, timedelta

import pandas as pd

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

# JSON
base_path = str(args.base_path)
NOAA = base_path[12:]

energy_path = f"runs_eval/{NOAA}/energy.csv"
# energy_path = "/userhome/jeon_mg/workspace/PINN_NF2/My_results/ar_NF2_original/ar_series_7115_2017-09-04T00:00:00/energy.csv"
energy_df = pd.read_csv(energy_path, index_col=False)
bin = 2
cm_per_pixel = 360e5 * bin 
dV = cm_per_pixel**3
energy_value = energy_df['energy_density']*dV 
energy_date = energy_df['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

K_energy_path = f"data/Kusano/{NOAA}_Kusano.csv"
K_energy_df = pd.read_csv(K_energy_path, index_col=False)
dx_Mm = K_energy_df['dx'][0]
dy_Mm = K_energy_df['dy'][0]
dz_Mm = K_energy_df['dz'][0]
dx_cm = dx_Mm*1e8
dy_cm = dy_Mm*1e8
dz_cm = dz_Mm*1e8
dV_cm3 = (dx_cm*dy_cm*dz_cm)*(2**3)
K_energy_value = K_energy_df['energy_density']*dV_cm3
K_energy_date = K_energy_df['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

plt.figure(figsize=(10, 5))
plt.plot(energy_date, energy_value, label='Server', color='k')
plt.plot(K_energy_date, K_energy_value, label='Kusano', color='g')
plt.title('NOAA 12673 Total Magnetic Energy')
plt.ylabel(r'E')
plt.legend(loc='lower right')
figure_energy_path = f"runs_eval/{NOAA}/energy_plot.png"
plt.savefig(figure_energy_path, dpi=300)