import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tvtk.api import tvtk, write_data
from nf2.potential.potential_field import get_potential
from datetime import datetime
from nf2.evaluation.unpack import load_cube
from nf2.evaluation.vtk import save_vtk

def make_info(nf2_file, vtk_path, npy_path, npy_pot_path, vtk_pot_path):
  nf2_file = str(nf2_file)
  vtk_path = str(vtk_path)
  npy_path = str(npy_path)
  npy_pot_path = str(npy_pot_path)
  vtk_pot_path = str(vtk_pot_path)

  device = torch.device("cuda") if torch.backends.mps.is_available() else "cpu"
  # state = torch.load(str(nf2_file), map_location=device)
  # meta_info = state['meta_info']

  # # Number
  # print(meta_info['harpnum'])
  # print(meta_info['noaa_ar'])
  # # Observation
  # print(meta_info['date-obs'])
  # print(meta_info['t_obs'])
  # print(meta_info['t_rec'])
  # # shape
  # print(state['cube_shape'])
  # print(meta_info['naxis1'], meta_info['naxis2'])

  # nf2 -> b
  b = load_cube(nf2_file, device, progress=True)
  # model = torch.nn.DataParallel(state['model'])
  # cube_shape = state['cube_shape']
  # z = cube_shape[2]
  # coords = np.stack(np.mgrid[:cube_shape[0]:1, :cube_shape[1]:1, :z:1], -1)
  # spatial_norm = 160
  # coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
  # coords_shape = coords.shape
  # cube = []
  # batch_size = 1000
  # coords = coords.view((-1, 3))
  # it = range(int(np.ceil(coords.shape[0] / batch_size)))
  # for k in tqdm(it):
  #     coord = coords[k * batch_size: (k + 1) * batch_size]
  #     coord = coord.to(device)
  #     coord.requires_grad = True
  #     cube += [model(coord).detach().cpu()]
  # cube = torch.cat(cube)
  # cube = cube.view(*coords_shape).numpy()
  # b_norm = 2500
  # b = cube * b_norm    

  # b -> vtk
  save_vtk(b, vtk_path, 'B')
  # bin = 2
  # Mm_per_pix = 360e-3 * bin
  # # Unpack
  # dim = b.shape[:-1]
  # # Generate the grid
  # pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.int64) * Mm_per_pix
  # # reorder the points and vectors in agreement with VTK
  # # requirement of x first, y next and z last.
  # pts = pts.transpose(2, 1, 0, 3)
  # pts = pts.reshape((-1, 3))
  # vectors = b.transpose(2, 1, 0, 3)
  # vectors = vectors.reshape((-1, 3))
  # sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
  # sg.point_data.vectors = vectors
  # sg.point_data.vectors.name = 'B'
  # write_data(sg, str(vtk_path))
  # print(str(vtk_path))

  # b -> npy
  np.save(str(npy_path), b)
  print(str(npy_path))

  # b -> potential b
  potential = get_potential(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3))
  b_potential = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)

  # potential b -> vtk
  save_vtk(b_potential, vtk_pot_path, 'B')

  # potential b -> npy
  np.save(str(npy_pot_path), b_potential)
  print(str(npy_pot_path))

def energy_series_csv(series_base_path):
  energy_path = series_base_path / 'eval_energy'
  energy_files = [x for x in energy_path.glob('**/*.npy')]
  energy_files = sorted(energy_files)
  energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in energy_files]
  energy_density = np.array([np.load(f) for f in energy_files])
  df = pd.DataFrame({"date":energy_series_dates, "energy_density":energy_density})
  csv_path = series_base_path / 'energy.csv'
  df.to_csv(str(csv_path), index=False)  
  print(str(csv_path))

def free_energy_series_csv(series_base_path):
  free_energy_path = series_base_path / 'eval_free_energy'
  free_energy_files = [x for x in free_energy_path.glob('**/*.npy')]
  free_energy_files = sorted(free_energy_files)
  free_energy_series_dates = [datetime.strptime(os.path.basename(f), '%Y%m%d_%H%M%S_TAI.npy') for f in free_energy_files]
  free_energy_density = np.array([np.load(f) for f in free_energy_files])
  df = pd.DataFrame({"date":free_energy_series_dates, "free_energy_density":free_energy_density})
  csv_path = series_base_path / 'free_energy.csv'
  df.to_csv(str(csv_path), index=False)  
  print(str(csv_path))

series_base_path = Path('ar_series_377_2011-02-12T00:00:00')
target = '20110215_000000'
nf2_file = series_base_path / f'base/{target}_TAI/extrapolation_result.nf2'
vtk_path = series_base_path / f'11158_{target}_server.vtk'
npy_path = series_base_path / f'11158_{target}_server.npy'
npy_pot_path = series_base_path / f'11158_{target}_server_pot.npy'
vtk_pot_path = series_base_path / f'11158_{target}_server_pot.vtk'

# make_info(nf2_file, vtk_path, npy_path, npy_pot_path, vtk_pot_path)
energy_series_csv(series_base_path)
free_energy_series_csv(series_base_path)