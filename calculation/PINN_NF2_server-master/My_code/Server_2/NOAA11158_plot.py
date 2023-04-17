# general imports
import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
from datetime import datetime
from dateutil.parser import parse
import shutil

# download
import drms
from urllib import request

# data processing
import numpy as np
from astropy.nddata import block_reduce
from sunpy.map import Map

# deep learning
import torch

# NF2
from nf2.train.trainer import NF2Trainer
from nf2.data.download import download_HARP, find_HARP, donwload_ds
from nf2.train.metric import *
from nf2.evaluation.unpack import load_cube
from nf2.evaluation.energy import get_free_mag_energy
from nf2.data.loader import load_hmi_data
from nf2.train.metric import energy

# visualization
from matplotlib import pyplot as plt

from pathlib import Path

jsoc_email = 'mgjeon@khu.ac.kr'
client = drms.Client(email=jsoc_email, verbose=True)

noaa_nums = [11158]
year = 2011
month = 2
day = 12
hour = 0
minute = 0

date = datetime(year, month, day, hour, minute)

sharp_nr = find_HARP(date, noaa_nums, client)
print(sharp_nr)
download_dir = 'AR_377'

bin = 2
spatial_norm = 160 
height = 160 
b_norm = 2500  
d_slice = [66, 658, 9, 377] # crop

dim = 256

lambda_div = 0.1 
lambda_ff = 0.1 
iterations = 10e4 
iterations = int(iterations)
decay_iterations = 5e4 
decay_iterations = int(decay_iterations)
batch_size = 1e4 
batch_size = int(batch_size)
log_interval = 1e4 
log_interval = int(log_interval)
validation_interval = 1e4 
validation_interval = int(validation_interval)
potential = True

base_path = 'ar_%d_%s' % (sharp_nr, date.isoformat('T'))
series_base_path = 'ar_series_%d_%s' % (sharp_nr, date.isoformat('T'))
series_download_dir = 'ar_377_series'
bpp = os.path.join(series_base_path, 'base')

os.makedirs(base_path, exist_ok=True)
os.makedirs(series_base_path, exist_ok=True)
os.makedirs(bpp, exist_ok=True)
os.makedirs(series_download_dir, exist_ok=True)

duration = '120h'

nf2_paths = sorted(glob.glob(os.path.join(bpp, '**', 'extrapolation_result.nf2')))

series_base_path = Path('ar_series_377_2011-02-12T00:00:00')
eval_energy_path = series_base_path / 'eval_energy_with_mag'
os.makedirs(eval_energy_path, exist_ok=True)
energy_files = []

for path in nf2_paths[::10]:
  print(path)
  f = os.path.join(eval_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  fmag = os.path.join(eval_energy_path, os.path.basename(os.path.dirname(path))) + '_mag.npy'
  print(f)
  if os.path.exists(f): 
    energy_files += [f]
    continue
  b = load_cube(path, progress=True)
  np.save(fmag, b)
  B = np.load(fmag)
  me = energy(B).sum()
  np.save(f, me)
  energy_files += [f]

energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in energy_files]

cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3

import pandas as pd
csv_colab = 'ar_377_colab/energy.csv'
df_colab = pd.read_csv(csv_colab, index_col=False)
energy_colab = df_colab['energy_density']*dV
date_colab = df_colab['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

plt.figure(figsize=(9, 3))
plt.scatter(energy_series_dates, [np.load(f)*dV for f in energy_files], color='k', label='Server', s=3, zorder=10)
plt.plot(date_colab, energy_colab, color='r', label='Colab', zorder=0)
plt.title('NOAA 11158')
plt.ylabel('total magnetic energy')
plt.legend()
figure_energy_path = series_base_path / 'energy_dots.png'
plt.savefig(figure_energy_path, dpi=300)

eval_free_energy_path = series_base_path / 'eval_free_energy_with_mag'
os.makedirs(eval_free_energy_path, exist_ok=True)
free_energy_files = []

from nf2.potential.potential_field import get_potential

for path in nf2_paths[::10]:
  print(path)
  f = os.path.join(eval_free_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  fmag = os.path.join(eval_free_energy_path, os.path.basename(os.path.dirname(path))) + '_mag.npy'
  print(f)
  if os.path.exists(f): 
    free_energy_files += [f]
    continue
  b = load_cube(path, progress=True)
  potential = get_potential(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3))
  b_potential = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)
  np.save(fmag, b_potential)
  B_pot = np.load(fmag)
  free_energy = energy(b) - energy(B_pot)
  free_me = free_energy.sum()
  np.save(f, free_me)
  free_energy_files += [f]

free_energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in free_energy_files]

cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3

free_csv_colab = 'ar_377_colab/free_energy.csv'
free_df_colab = pd.read_csv(free_csv_colab, index_col=False)
free_energy_colab = free_df_colab['free_energy_density']*dV
free_date_colab = free_df_colab['date'].map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

plt.figure(figsize=(9, 3))
plt.scatter(free_energy_series_dates, [np.load(f).sum()*dV for f in free_energy_files], color='k', label='Server', s=3, zorder=10)
plt.plot(free_date_colab, free_energy_colab, color='r', label='Colab', zorder=0)
plt.title('NOAA 11158')
plt.ylabel('total free magnetic energy')
plt.legend()
figure_free_energy_path = series_base_path / 'free_energy_dots.png'
plt.savefig(figure_free_energy_path, dpi=300)

