import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sunpy.map import Map
from astropy.nddata import block_reduce

class mgMaker():

    def __init__(self, bin, d_slice, series_download_dir):

        self.bin = bin 
        self.d_slice = d_slice

        self.hmi_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp.fits')))
        self.hmi_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt.fits')))
        self.hmi_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br.fits')))
        self.err_p_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bp_err.fits')))
        self.err_t_files = sorted(glob.glob(os.path.join(series_download_dir, '*Bt_err.fits')))
        self.err_r_files = sorted(glob.glob(os.path.join(series_download_dir, '*Br_err.fits')))

        self.num_files = len(self.hmi_p_files)
        self.series_path = series_download_dir[:-5]
        os.makedirs(self.series_path, exist_ok=True)
        
    def data(self, num, suptitle_y=0.90):
        hmi_p = self.hmi_p_files[num]
        hmi_t = self.hmi_t_files[num]
        hmi_r = self.hmi_r_files[num]
        err_p = self.err_p_files[num]
        err_t = self.err_t_files[num]
        err_r = self.err_r_files[num]
        
        self.hmi_cube = np.stack([Map(hmi_p).data, -Map(hmi_t).data, Map(hmi_r).data]).transpose()
        self.error_cube = np.stack([Map(err_p).data, Map(err_t).data, Map(err_r).data]).transpose()
        
        self.file_info = os.path.basename(hmi_r)[:-8]
        self.meta_info = Map(hmi_r).meta 

        self.result_path = os.path.join(self.series_path, self.file_info[-19:-4])
        os.makedirs(self.result_path, exist_ok=True)

        vmin = -1000
        vmax = 1000

        fig, axes = plt.subplots(3,3, figsize=(15,12))
        im = axes[0,0].imshow(self.hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[0,0].set_title(f'{self.hmi_cube[..., 0].transpose().shape} original Bp')
        axes[0,1].imshow(self.hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[0,1].set_title(f'{self.hmi_cube[..., 1].transpose().shape} original -Bt')
        axes[0,2].imshow(self.hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[0,2].set_title(f'{self.hmi_cube[..., 2].transpose().shape} original Br')

        if self.d_slice is not None:
            self.hmi_cube = self.hmi_cube[self.d_slice[0]:self.d_slice[1], self.d_slice[2]:self.d_slice[3]]
            self.error_cube = self.error_cube[self.d_slice[0]:self.d_slice[1], self.d_slice[2]:self.d_slice[3]]

        axes[1,0].imshow(self.hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[1,0].set_title(f'{self.hmi_cube[..., 0].transpose().shape} crop Bp')
        axes[1,1].imshow(self.hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[1,1].set_title(f'{self.hmi_cube[..., 1].transpose().shape} crop -Bt')
        axes[1,2].imshow(self.hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[1,2].set_title(f'{self.hmi_cube[..., 2].transpose().shape} crop Br')

        if self.bin > 1:
            self.hmi_cube = block_reduce(self.hmi_cube, (self.bin, self.bin, 1), np.mean)
            self.error_cube = block_reduce(self.error_cube, (self.bin, self.bin, 1), np.mean)

        axes[2,0].imshow(self.hmi_cube[..., 0].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[2,0].set_title(f'{self.hmi_cube[..., 0].transpose().shape} binning Bp')
        axes[2,1].imshow(self.hmi_cube[..., 1].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[2,1].set_title(f'{self.hmi_cube[..., 1].transpose().shape} binning -Bt')
        axes[2,2].imshow(self.hmi_cube[..., 2].transpose(), origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        axes[2,2].set_title(f'{self.hmi_cube[..., 2].transpose().shape} binning Br')

        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.25, 0.16, 0.5, 0.01])
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

        fig.suptitle(t=f'{self.file_info}', x=0.5, y=suptitle_y)
        fig_path = Path(self.result_path) / 'hmi_original_crop_binning.png'
        fig.savefig(fig_path, dpi=300)

        return self.hmi_cube, self.error_cube, self.meta_info