import os
import argparse
import json
from datetime import datetime

import numpy as np

import glob

from nf2.evaluation.unpack import load_cube
from nf2.train.metric import energy
from nf2.evaluation.energy import get_free_mag_energy

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)

# JSON
base_path = str(args.base_path)
NOAA = base_path[12:]

dim = args.dim
bin = args.bin
use_potential_boundary = args.use_potential_boundary
lambda_div = args.lambda_div
lambda_ff = args.lambda_ff

info = 'dim%d_bin%d_pf%s_ld%s_lf%s' % (
        dim, bin, str(use_potential_boundary), lambda_div, lambda_ff)

base_path = os.path.join(base_path, info)

eval_path = f'runs_eval/{NOAA}/{info}'

# Data
nf2_paths = sorted(glob.glob(os.path.join(base_path, '**', 'extrapolation_result.nf2')))

cm_per_pixel = 360e5 * args.bin
dV = cm_per_pixel**3

# Energy
eval_energy_path = f"{eval_path}/energy"
os.makedirs(eval_energy_path, exist_ok=True)
energy_files = []

for path in nf2_paths:
  print(path)
  f = os.path.join(eval_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  print(f)
  if os.path.exists(f): 
    energy_files += [f]
    continue
  b = load_cube(path, progress=True)
  me = energy(b).sum()
  np.save(f, me)
  energy_files += [f]

energy_series_dates = np.array([datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in energy_files])
energy_density = np.array([np.load(f) for f in energy_files])

df = pd.DataFrame({"date":energy_series_dates, "energy_density":energy_density})
csv_path = f"{eval_path}/energy.csv"
df.to_csv(str(csv_path), index=False)  

plt.figure(figsize=(9, 3))
plt.plot(energy_series_dates, energy_density*dV)
plt.title(NOAA)
plt.ylabel('Total magnetic energy')
figure_energy_path = f"{eval_path}/energy.png"
plt.savefig(figure_energy_path, dpi=300)

# Free Energy
eval_free_energy_path = f"{eval_path}/free_energy"
os.makedirs(eval_free_energy_path, exist_ok=True)
free_energy_density_files = []

for path in nf2_paths:
  print(path)
  f = os.path.join(eval_free_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  print(f)
  if os.path.exists(f): 
    free_energy_density_files += [f]
    continue
  b = load_cube(path, progress=True)
  free_me = get_free_mag_energy(b).sum()
  np.save(f, free_me)
  free_energy_density_files += [f]

free_energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in free_energy_density_files]
free_energy_density = np.array([np.load(f) for f in free_energy_density_files])

df = pd.DataFrame({"date":free_energy_series_dates, "free_energy_density":free_energy_density})
csv_path =f"{eval_path}/free_energy.csv"
df.to_csv(str(csv_path), index=False)  

plt.figure(figsize=(9, 3))
plt.plot(free_energy_series_dates, free_energy_density*dV)
plt.title(NOAA)
plt.ylabel('Total free magnetic energy')
figure_free_energy_path = f"{eval_path}/free_energy.png"
plt.savefig(figure_free_energy_path, dpi=300)
