# general imports
import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
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

# base_path = 'ar_%d_%s' % (sharp_nr, date.isoformat('T'))
series_base_path = 'ar_series_%d_%s' % (sharp_nr, date.isoformat('T'))
series_download_dir = 'ar_377_series'
bpp = os.path.join(series_base_path, 'base')

nf2_paths = sorted(glob.glob(os.path.join(bpp, '**', 'extrapolation_result.nf2')))

eval_energy_path = os.path.join(series_base_path, 'eval_energy')
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

cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3

plt.figure(figsize=(9, 3))
plt.plot(energy_series_dates, energy_density*dV)
plt.title('NOAA 11158')
plt.ylabel('total magnetic energy')
figure_energy_path = os.path.join(series_base_path, 'energy.png')
plt.savefig(figure_energy_path, dpi=300)

eval_free_energy_path = os.path.join(series_base_path, 'eval_free_energy')
os.makedirs(eval_free_energy_path, exist_ok=True)
free_energy_files = []

for path in nf2_paths:
  print(path)
  f = os.path.join(eval_free_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  print(f)
  if os.path.exists(f): 
    free_energy_files += [f]
    continue
  b = load_cube(path, progress=True)
  free_me = get_free_mag_energy(b).sum()
  np.save(f, free_me)
  free_energy_files += [f]

free_energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in free_energy_files]

cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3

plt.figure(figsize=(9, 3))
plt.plot(free_energy_series_dates, [np.load(f)*dV for f in free_energy_files])
plt.title('NOAA 11158')
plt.ylabel('total free magnetic energy')
figure_free_energy_path = os.path.join(series_base_path, 'free_energy.png')
plt.savefig(figure_free_energy_path, dpi=300)
