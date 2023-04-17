#Pytorch GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
print('visible device: ', os.environ["CUDA_VISIBLE_DEVICES"])

#Pytorch device
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('device: ', device)
print('device count: ', torch.cuda.device_count())
print('current device: ', torch.cuda.current_device())

#Parameters for NOAA 11158
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

base_path = 'ar_377_20110212_000000'
series_base_path = 'ar_277_series_20110212_000000'
series_download_dir = 'ar_377_series'
os.makedirs(base_path, exist_ok=True)
os.makedirs(series_base_path, exist_ok=True)
os.makedirs(series_download_dir, exist_ok=True)

# Prepare data for an initial training
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sunpy.map import Map
from astropy.nddata import block_reduce
hmi_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp.fits')))
hmi_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt.fits')))
hmi_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br.fits')))
err_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp_err.fits')))
err_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt_err.fits')))
err_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br_err.fits')))

hmi_p = hmi_p_files[0]
hmi_t = hmi_t_files[0]
hmi_r = hmi_r_files[0]
err_p = err_p_files[0]
err_t = err_t_files[0]
err_r = err_r_files[0]

filename = os.path.basename(hmi_p)
print('Inital traning: ', filename)

hmi_cube = np.stack([Map(hmi_p).data, -Map(hmi_t).data, Map(hmi_r).data]).transpose()
error_cube = np.stack([Map(err_p).data, Map(err_t).data, Map(err_r).data]).transpose()
meta_info = Map(hmi_r).meta 
print('hmi_cube:', np.shape(hmi_cube))

vmin = -1000
vmax = 1000

fig, axes = plt.subplots(3,3, figsize=(15,12))
im = axes[0,0].imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[0,0].set_title(f'{hmi_cube[..., 0].transpose().shape} original Bp')
axes[0,1].imshow(hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[0,1].set_title(f'{hmi_cube[..., 1].transpose().shape} original -Bt')
axes[0,2].imshow(hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[0,2].set_title(f'{hmi_cube[..., 2].transpose().shape} original Br')

if d_slice is not None:
  hmi_cube = hmi_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]
  error_cube = error_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]

axes[1,0].imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[1,0].set_title(f'{hmi_cube[..., 0].transpose().shape} crop Bp')
axes[1,1].imshow(hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[1,1].set_title(f'{hmi_cube[..., 1].transpose().shape} crop -Bt')
axes[1,2].imshow(hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[1,2].set_title(f'{hmi_cube[..., 2].transpose().shape} crop Br')

if bin > 1:
  hmi_cube = block_reduce(hmi_cube, (bin, bin, 1), np.mean)
  error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)

axes[2,0].imshow(hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[2,0].set_title(f'{hmi_cube[..., 0].transpose().shape} binning Bp')
axes[2,1].imshow(hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[2,1].set_title(f'{hmi_cube[..., 1].transpose().shape} binning -Bt')
axes[2,2].imshow(hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
axes[2,2].set_title(f'{hmi_cube[..., 2].transpose().shape} binning Br')

fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, 0.16, 0.5, 0.01])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

fig.suptitle(t=f'{os.path.basename(hmi_p)[:-8]}', x=0.5, y=0.90)
figpath = Path(base_path) / 'hmi_original_crop_binning.png'
fig.savefig(figpath, dpi=300)

# Initial training
