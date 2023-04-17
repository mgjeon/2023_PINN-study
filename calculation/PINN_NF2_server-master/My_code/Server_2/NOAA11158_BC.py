import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from Kusano.load_nlfff import nlfff
from pathlib import Path
from sunpy.map import Map

def get_B(npy_path):
    B = np.load(npy_path)
    Bx = B[:, :, :, 0]
    By = B[:, :, :, 1]
    Bz = B[:, :, :, 2]

    return Bx, By, Bz

npy_path_original = 'ar_NF2_original/ar_series_377_2011-02-12T00:00:00/11158_20110215_000000_server.npy'
npy_path_new = 'ar_NF2_NewLoss/ar_series_377_2011-02-12T00:00:00/11158_20110215_000000_server.npy'

Bx_original, By_original, Bz_original = get_B(npy_path_original)
Bx_new, By_new, Bz_new = get_B(npy_path_new)

nc_path = 'Kusano/NOAA11158/11158_20110215_000000.nc'
nlff = nlfff(nc_path)
b = np.stack([nlff.bx, nlff.by, nlff.bz], axis=-1)
bx = b[:, :, :, 0]
by = b[:, :, :, 1]
bz = b[:, :, :, 2]

hmi_p = 'ar_377_series/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bp.fits'
hmi_t = 'ar_377_series/hmi.sharp_cea_720s.377.20110215_000000_TAI.Bt.fits'
hmi_r = 'ar_377_series/hmi.sharp_cea_720s.377.20110215_000000_TAI.Br.fits'

Bx = Map(hmi_p).data
By = -Map(hmi_t).data
Bz = Map(hmi_r).data


fig, axes = plt.subplots(3,4, figsize=(19, 15))

vmin = -3000
vmax = 3000

Bz_z0_original = Bz_original[:, :, 0].transpose()
Bz_z0_new = Bz_new[:, :, 0].transpose()
bz_z0 = bz[:, :, 0].transpose()

server_ny_original, server_nx_original = np.shape(Bz_z0_original)
server_ny_new, server_nx_new = np.shape(Bz_z0_new)
Kusano_ny, Kusano_nx = np.shape(bz_z0)
SHARP_ny, SHARP_nx = np.shape(Bz)

max1 = max(max(Bz_z0_original[:, server_nx_original//2]), max(Bz_z0_new[:, server_nx_new//2]), max(bz_z0[:, Kusano_nx//2]), max(Bz[:, SHARP_nx//2]))
min1 = min(min(Bz_z0_original[:, server_nx_original//2]), min(Bz_z0_new[:, server_nx_new//2]), min(bz_z0[:, Kusano_nx//2]), min(Bz[:, SHARP_nx//2]))

max2 = max(max(Bz_z0_original[server_ny_original//2, :]), max(Bz_z0_new[server_ny_new//2, :]),max(bz_z0[Kusano_ny//2, :]), max(Bz[SHARP_ny//2, :]))
min2 = min(min(Bz_z0_original[server_ny_original//2, :]), min(Bz_z0_new[server_ny_new//2, :]),min(bz_z0[Kusano_ny//2, :]), min(Bz[SHARP_ny//2, :]))

#---------------------------------------------
def draw_server(Bz_z0, i):
    server_ny, server_nx = np.shape(Bz_z0)
    server_x = np.arange(server_nx)
    server_y = np.arange(server_ny)
    server_med_x = (SHARP_nx//2 - 66)//2
    server_med_y = (SHARP_ny//2 - 9)//2

    axes[0,i].plot(server_y, Bz_z0[:, server_med_x], color='blue')
    axes[0,i].plot(server_y, Bz_z0[:, server_med_x], color='blue')
    axes[0,i].set_title(f'Colab [:, (SHARP_nx//2 - 66)//2] = [:, {server_med_x}]')
    axes[0,i].set_ylim((min1, max1))
    axes[0,i].set_ylabel('Colab Bz(z=0) [G]')
    axes[0,i].set_xlabel('server_y')

    axes[1,i].pcolormesh(server_x, server_y, Bz_z0, vmin=vmin, vmax=vmax, cmap='gray', shading='auto')
    axes[1,i].set_aspect('equal')
    axes[1,i].set_title(f'{np.shape(Bz_z0)} Colab Bz(z=0) [G]')
    axes[1,i].axvline(server_x[server_med_x], color='blue')
    axes[1,i].axhline(server_y[server_med_y], color='red')
    axes[1,i].set_xlabel('server_x')
    axes[1,i].set_ylabel('server_y')

    axes[2,i].plot(server_x, Bz_z0[server_med_y, :], color='red')
    axes[2,i].set_title(f'Colab [(SHARP_ny//2 - 9)//2, :] = [{server_med_y}, :]')
    axes[2,i].set_ylim((min2, max2))
    axes[2,i].set_ylabel('Colab Bz(z=0) [G]')
    axes[2,i].set_xlabel('server_x')

draw_server(Bz_z0_original, 0)
draw_server(Bz_z0_new, 3)

#---------------------------------------------
Kusano_x = np.arange(Kusano_nx)
Kusano_y = np.arange(Kusano_ny)

axes[0,1].plot(Kusano_y, bz_z0[:, Kusano_nx//2], color='blue')
axes[0,1].set_title(f'Kusano [:, nx//2] = [:, {Kusano_nx//2}]')
axes[0,1].set_ylim((min1, max1))
axes[0,1].set_ylabel('Kusano Bz(z=0) [G]')
axes[0,1].set_xlabel('Kusano_y')

axes[1,1].pcolormesh(Kusano_x, Kusano_y, bz_z0, vmin=vmin, vmax=vmax, cmap='gray', shading='auto')
axes[1,1].set_aspect('equal')
axes[1,1].set_title(f'{np.shape(bz_z0)} Kusano Bz(z=0) [G]')
axes[1,1].axvline(Kusano_x[Kusano_nx//2], color='blue')
axes[1,1].axhline(Kusano_y[Kusano_ny//2], color='red')
axes[1,1].set_xlabel('Kusano_x')
axes[1,1].set_ylabel('Kusano_y')

axes[2,1].plot(Kusano_x, bz_z0[Kusano_ny//2, :], color='red')
axes[2,1].set_title(f'Kusano [ny//2, :] = [{Kusano_ny//2}, :]')
axes[2,1].set_ylim((min2, max2))
axes[2,1].set_ylabel('Kusano Bz(z=0) [G]')
axes[2,1].set_xlabel('Kusano_x')

#---------------------------------------------
SHARP_x = np.arange(SHARP_nx)
SHARP_y = np.arange(SHARP_ny)

axes[0,2].plot(SHARP_y, Bz[:, SHARP_nx//2], color='blue')
axes[0,2].set_title(f'SHARP [:, nx//2] = [:, {SHARP_nx//2}]')
axes[0,2].set_ylim((min1, max1))
axes[0,2].set_ylabel('Bz [G]')
axes[0,2].set_xlabel('SHARP_y')

im = axes[1,2].pcolormesh(SHARP_x, SHARP_y, Bz, vmin=vmin, vmax=vmax, cmap='gray', shading='auto')
axes[1,2].set_aspect('equal')
axes[1,2].set_title(f'{np.shape(Bz)} SHARP Bz [G]')
axes[1,2].axvline(SHARP_x[SHARP_nx//2], color='blue')
axes[1,2].axhline(SHARP_y[SHARP_ny//2], color='red')
axes[1,2].set_xlabel('SHARP_x')
axes[1,2].set_ylabel('SHARP_y')

axes[2,2].plot(SHARP_x, Bz[SHARP_ny//2, :], color='red')
axes[2,2].set_title(f'SHARP [ny//2, :] = [{SHARP_ny//2}, :]')
axes[2,2].set_ylim((min2, max2))
axes[2,2].set_ylabel('Bz [G]')
axes[2,2].set_xlabel('SHARP_x')


fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.93, 0.44, 0.01, 0.2])
fig.colorbar(im, cax=cbar_ax, orientation='vertical')

fig.suptitle(t='NOAA 11158_20110215_000000', x=0.5, y=0.93)

figpath = '11158_20110215_000000_plot.png'
fig.savefig(figpath, dpi=300)