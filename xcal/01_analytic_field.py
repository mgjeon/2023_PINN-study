import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from astropy.nddata import block_replicate
from tvtk.api import tvtk, write_data
import pyvista as pv 

import numpy as np
from scipy.integrate import solve_bvp

vtk_path = './evaluation.vtk'

mesh = pv.read(vtk_path)
xindmax, yindmax, zindmax = mesh.dimensions
xcenter, ycenter, zcenter = mesh.center

mesh_g = mesh.compute_derivative(scalars='B')

def gradients_to_dict(arr):
    keys = np.array(
        ["dBx/dx", "dBx/dy", "dBx/dz", "dBy/dx", "dBy/dy", "dBy/dz", "dBz/dx", "dBz/dy", "dBz/dz"]
    )
    keys = keys.reshape((3,3))[:, : arr.shape[1]].ravel()
    return dict(zip(keys, mesh_g['gradient'].T))

gradients = gradients_to_dict(mesh_g['gradient'])

curlB_x = gradients['dBz/dy'] - gradients['dBy/dz']
curlB_y = gradients['dBx/dz'] - gradients['dBz/dx']
curlB_z = gradients['dBy/dx'] - gradients['dBx/dy']

curlB = np.vstack([curlB_x, curlB_y, curlB_z]).T

mesh.point_data['curlB'] = curlB

p = pv.Plotter()
p.add_mesh(mesh.outline())

#-----------------------
sargs_B = dict(
    title='Bz [G]',
    title_font_size=15,
    height=0.25,
    width=0.05,
    vertical=True,
    position_x = 0.05,
    position_y = 0.05,
)
dargs_B = dict(
    scalars='B', 
    component=2, 
    clim=(-200, 200), 
    scalar_bar_args=sargs_B, 
    show_scalar_bar=True, 
    lighting=False
)
p.add_mesh(mesh.extract_subset((0, xindmax, 0, yindmax, 0, 0)), 
           cmap='gray', **dargs_B)
#-----------------------

#-----------------------
sargs_J = dict(
    title='J = curl(B)',
    title_font_size=15,
    height=0.25,
    width=0.05,
    vertical=True,
    position_x = 0.9,
    position_y = 0.05,
)
dargs_J = dict(
    scalars='curlB', 
    clim=(0, 20),
    scalar_bar_args=sargs_J, 
    show_scalar_bar=True, 
    lighting=False
)
#-----------------------

def draw_streamlines(pts):
    stream, src = mesh.streamlines(
        return_source=True,
        # source_center=(120, 90, 0),
        # source_radius=5,
        # n_points=100,
        start_position = pts,
        integration_direction='both',
        # progress_bar=False,
        max_time=1000,
        # initial_step_length=0.001,
        # min_step_length=0.001,
        # max_step_length=2.0,
        # max_steps=999999,
        # terminal_speed = 1e-16,
        # max_error = 1e-6,
    )
    p.add_mesh(stream.tube(radius=0.2), 
            cmap='plasma', **dargs_J)
    p.add_mesh(src, point_size=10)

for i in np.arange(8, 64, 8):
    for j in np.arange(8, 64, 8):
        try: 
            draw_streamlines((i, j, 0))
        except:
            print(i, j)


# stream, src = mesh.streamlines(
#         return_source=True,
#         # source_center=(120, 90, 0),
#         source_radius=64,
#         n_points=500,
#         # start_position = pts,
#         integration_direction='both',
#         # progress_bar=False,
#         max_time=1000,
#         # initial_step_length=0.001,
#         # min_step_length=0.001,
#         # max_step_length=2.0,
#         # max_steps=999999,
#         # terminal_speed = 1e-16,
#         # max_error = 1e-6,
#     )
# p.add_mesh(stream.tube(radius=0.2), 
#         cmap='plasma', **dargs_J)

p.camera_position = 'xy'           
# p.camera.azimuth = -100
# p.camera.elevation = -20
# p.camera.zoom(1.0)
p.show_bounds()
p.add_title('LL')
p.show(screenshot='./field.png')