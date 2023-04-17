# general imports
import glob
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
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

noaa_nums = [12673]
year = 2017
month = 9
day = 4
hour = 0
minute = 0

date = datetime(year, month, day, hour, minute)

sharp_nr = find_HARP(date, noaa_nums, client)
print(sharp_nr)

bin = 2
spatial_norm = 160 
height = 160 
b_norm = 2500  
d_slice = None

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
series_download_dir = 'ar_7115_series'
bpp = os.path.join(series_base_path, 'base')

os.makedirs(base_path, exist_ok=True)
os.makedirs(series_base_path, exist_ok=True)
os.makedirs(bpp, exist_ok=True)
os.makedirs(series_download_dir, exist_ok=True)

# scan all data files
hmi_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br.fits')))  # z
err_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp_err.fits')))  # x
err_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt_err.fits')))  # y
err_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br_err.fits')))  # z

hmi_p = hmi_p_files[0]
hmi_t = hmi_t_files[0]
hmi_r = hmi_r_files[0]
err_p = err_p_files[0]
err_t = err_t_files[0]
err_r = err_r_files[0]

filename = os.path.basename(hmi_p)
print(filename)

hmi_cube, error_cube, meta_info = load_hmi_data([hmi_p, err_p, hmi_r, err_r, hmi_t, err_t])

# fig, ax = plt.subplots()
# ax.imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray')
# ax.set_title(f'{hmi_cube[..., 0].transpose().shape} {filename}')
# fig.savefig('./hmi_original.png')

if d_slice is not None:
  hmi_cube = hmi_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]
  error_cube = error_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]

# fig, ax = plt.subplots()
# ax.imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray')
# ax.set_title(f'{hmi_cube[..., 0].transpose().shape} {filename}')
# fig.savefig('./hmi_cropped.png')

if bin > 1:
  hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
  error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)

# fig, ax = plt.subplots()
# ax.imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray')
# ax.set_title(f'{hmi_cube[..., 0].transpose().shape} {filename}')
# fig.savefig('./hmi_cropped_binning.png')

trainer = NF2Trainer(base_path, hmi_cube, error_cube, height, spatial_norm, b_norm,
                     meta_info=meta_info, dim=dim,
                     use_potential_boundary=potential, lambda_div=lambda_div, lambda_ff=lambda_ff,
                     decay_iterations=decay_iterations, meta_path=None)

trainer.train(iterations, batch_size, log_interval, validation_interval, num_workers=os.cpu_count())

new_meta_path = 'ar_7115_2017-09-04T00:00:00/extrapolation_result.nf2'

series_iterations = 2000
series_batch_size = int(1e4)
series_log_interval = 2000
series_validation_interval = -1

# scan all data files
hmi_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br.fits')))  # z
err_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp_err.fits')))  # x
err_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt_err.fits')))  # y
err_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br_err.fits')))  # z

for hmi_p, hmi_t, hmi_r, err_p, err_t, err_r in zip(hmi_p_files, hmi_t_files, hmi_r_files,
                                                    err_p_files, err_t_files, err_r_files):
    file_id = os.path.basename(hmi_p).split('.')[3]
    bp = os.path.join(bpp, file_id)

    # check if finished
    final_model_path = os.path.join(bp, 'final.pt')
    print(final_model_path)
    #print(os.path.exists(final_model_path))
    if os.path.exists(final_model_path):
        new_meta_path = final_model_path
        continue
    
    # data pre-processing; same as for the single extrapolation
    hmi_cube, error_cube, meta_info = load_hmi_data([hmi_p, err_p, hmi_r, err_r, hmi_t, err_t])
    
    if d_slice is not None:
      hmi_cube = hmi_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]
      error_cube = error_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]
    if bin > 1:
      hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
      error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)
    print(hmi_cube.shape)
    trainer = NF2Trainer(bp, hmi_cube, error_cube, height, spatial_norm, b_norm, 
                         meta_info=meta_info, dim=dim, 
                         lambda_div=lambda_div, lambda_ff=lambda_ff,
                         meta_path=new_meta_path, use_potential_boundary=potential)
    trainer.train(series_iterations, series_batch_size, 
                  series_log_interval, series_validation_interval)
    new_meta_path = final_model_path

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

energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in energy_files]

cm_per_pixel = 360e5 * bin
dV = cm_per_pixel**3

plt.figure(figsize=(9, 3))
plt.plot(energy_series_dates, [np.load(f)*dV for f in energy_files])
plt.title('NOAA 12673')
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
plt.title('NOAA 12673')
plt.ylabel('total free magnetic energy')
figure_free_energy_path = os.path.join(series_base_path, 'free_energy.png')
plt.savefig(figure_free_energy_path, dpi=300)

