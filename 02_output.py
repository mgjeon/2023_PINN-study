import os
import json
import argparse
import glob
import pandas as pd

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

vtk_path = info['output']['vtk_path']
metric_path = info['output']['metric_path']
plot_path = info['output']['plot_path']
os.makedirs(vtk_path, exist_ok=True)
os.makedirs(metric_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

n = info['exact']['n']
m = info['exact']['m']
l = info['exact']['l']
psi = eval(info['exact']['psi'])
resolution = info['exact']['resolution']
bounds = info['exact']['bounds']

b = get_analytic_b_field(n=n, m=m, l=l, psi=psi, resolution=resolution, bounds=bounds)

b_potential = get_potential(b[:, :, 0, 2], b.shape[-1])

title = 'LL'
vtk_file = os.path.join(vtk_path, f'{title}.vtk')
p = pv_plot(B=b, vtk_path=vtk_file, points=((16, 49, 8), (16, 49, 8)), title=title)

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

metric = metrics(B=b, b=b, B_potential=b_potential, b_potential=b_potential)
iterinfo = {'iter': -1}
metric = {**iterinfo, **metric}

df = pd.DataFrame.from_dict([metric])
df.to_csv(os.path.join(metric_path, 'metric.csv'), index=False)

field_files = os.path.join(info['simul']['base_path'],'fields_*.nf2')
field_files = sorted(glob.glob(field_files))

for file_path in field_files:
    iters = os.path.basename(file_path).split('.')[0][7:]
    title = 'PINN' + '_' + iters
    B = load_cube(file_path)
    
    metric = metrics(B=B, b=b, B_potential=b_potential, b_potential=b_potential)
    iterinfo = {'iter': int(iters)}
    metric = {**iterinfo, **metric}
    
    df = pd.concat([df, pd.DataFrame([metric])], ignore_index=True)
    print('metric: ', file_path)

df.to_csv(os.path.join(metric_path, 'metric.csv'), index=False)

for file_path in field_files:
    iters = os.path.basename(file_path).split('.')[0][7:]
    title = 'PINN' + '_' + iters
    B = load_cube(file_path)
    
    vtk_file = os.path.join(vtk_path, f'{title}.vtk')
    p = pv_plot(B=B, vtk_path=vtk_file, points=((16, 49, 8), (16, 49, 8)), title=title)

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

    print('plot: ', file_path)