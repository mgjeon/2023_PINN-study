import os
import json
import argparse
import glob
import pandas as pd
import numpy as np

from tqdm import tqdm

from pinf.analytical_field import get_analytic_b_field
from pinf.unpack import load_cube
from pinf.potential_field import get_potential
from pinf.plot import pv_plot
from pinf.performance_metrics import metrics 

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
args = parser.parse_args()

with open(args.cfg) as config:
    info = json.load(config)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= info['simul']['gpu_id']

def mag_plot(title, mag, vtk_path, plot_path):
    vtk_file = os.path.join(vtk_path, f'{title}.vtk')
    p = pv_plot(B=mag, vtk_path=vtk_file, points=((16, 49, 8), (16, 49, 8)))

    xy_path = os.path.join(plot_path, f'{title}_xy.pdf')
    yz_path = os.path.join(plot_path, f'{title}_yz.pdf')
    xz_path = os.path.join(plot_path, f'{title}_xz.pdf')
    xz_tilted_path = os.path.join(plot_path, f'{title}_xz_tilted.pdf')

    if not os.path.exists(xy_path):
        p.camera_position = 'xy'
        p.save_graphic(xy_path)

    if not os.path.exists(yz_path):
        p.camera_position = 'yz'
        p.save_graphic(yz_path)

    if not os.path.exists(xz_path):
        p.camera_position = 'xz'
        p.save_graphic(xz_path)  

    if not os.path.exists(xz_tilted_path):
        p.camera_position = 'xz'
        p.camera.azimuth = -30
        p.camera.elevation = 25
        p.save_graphic(xz_tilted_path)

def metric_df(B, b, B_potential, iteration):
    metric = metrics(B=B, b=b, B_potential=B_potential)
    iterinfo = {'iter': iteration}
    metric = {**iterinfo, **metric}

    df = pd.DataFrame.from_dict([metric])
    return df

import torch 

def loss_df(file_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loss_path = file_path.replace('fields_', 'loss_')
    state = torch.load(loss_path, map_location=device)
    key = ["BC_loss", "lambda_BC", 'Div_loss', "lambda_div", "FF_loss", "lambda_ff"]
    value = [state['BC_loss'].item(), state['lambda_BC'], state['divergence_loss'].item(), state['lambda_div'], state['force_loss'].item(), state['lambda_ff']]
    df_loss = pd.DataFrame.from_dict([dict(zip(key, value))])
    return df_loss

base_path = info['simul']['base_path']
vtk_path = os.path.join(base_path, info['output']['vtk_path'])
metric_path = os.path.join(base_path, info['output']['metric_path'])
plot_path = os.path.join(base_path, info['output']['plot_path'])

try:
    os.makedirs(vtk_path, exist_ok=False)
    os.makedirs(metric_path, exist_ok=False)
    os.makedirs(plot_path, exist_ok=False)
except:
    print("There are already output files")

n = info['exact']['n']
m = info['exact']['m']
l = info['exact']['l']
psi = eval(info['exact']['psi'])
resolution = info['exact']['resolution']
bounds = info['exact']['bounds']

b = get_analytic_b_field(n=n, m=m, l=l, psi=psi, resolution=resolution, bounds=bounds)

potential = get_potential(b[:, :, 0, 2], b.shape[2], batch_size=1000)
b_potential = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)

mag_plot('LL', b, vtk_path, plot_path)
df_b = metric_df(B=b, b=b, B_potential=b_potential, iteration=-2)

mag_plot('LL_pot', b_potential, vtk_path, plot_path)
df_pot = metric_df(B=b_potential, b=b, B_potential=b_potential, iteration=-1)
df = pd.concat([df_b, df_pot], ignore_index=True)

base_path = os.path.join(info['simul']['base_path'], "run/")
field_files = os.path.join(base_path,'fields_*.nf2')
field_files = sorted(glob.glob(field_files))

metric_file= os.path.join(metric_path, 'metric.csv')

if not os.path.exists(metric_file):
    for file_path in tqdm(field_files, desc='metric: '):
        iters = os.path.basename(file_path).split('.')[0][7:]
        title = 'PINN' + '_' + iters
        B = load_cube(file_path)

        df_new = metric_df(B=B, b=b, B_potential=b_potential, iteration=int(iters))
        df_loss = loss_df(file_path)

        df_new = pd.concat([df_new, df_loss], axis=1)
        
        df = pd.concat([df, df_new], ignore_index=True)
        # print('metric: ', file_path)
        
    df.to_csv(metric_file, index=False)

for file_path in tqdm(field_files, desc='plot, vtk: '):
    iters = os.path.basename(file_path).split('.')[0][7:]
    title = 'PINN' + '_' + iters
    B = load_cube(file_path)
    mag_plot(title, B, vtk_path, plot_path)
    # print('plot: ', file_path)