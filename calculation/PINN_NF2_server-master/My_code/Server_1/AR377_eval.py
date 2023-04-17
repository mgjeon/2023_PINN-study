import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"
import glob

from nf2.train.trainer import NF2Trainer
from nf2.data.download import download_HARP, find_HARP, donwload_ds
from nf2.train.metric import *
from nf2.evaluation.unpack import load_cube
from nf2.evaluation.energy import get_free_mag_energy
from nf2.data.loader import load_hmi_data
from nf2.train.metric import energy

from datetime import datetime

from matplotlib import pyplot as plt
from pathlib import Path

result_path = 'ar_377_series_each'

nf2_paths = sorted(glob.glob(os.path.join(result_path, '**', 'extrapolation_result.nf2')))

eval_energy_path = 'ar_377_series_each_eval/eval_energy'
os.makedirs(eval_energy_path, exist_ok=True)
energy_files = []

for path in nf2_paths:
  print(path)
  f = os.path.join(eval_energy_path, os.path.basename(os.path.dirname(path))) + '.npy'
  print(f)
  print(os.path.exists(f))
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
plt.title('AR 377')
plt.ylabel('total magnetic energy')
figure_energy_path = 'energy.png'
plt.savefig(figure_energy_path, dpi=300)

eval_free_energy_path = 'ar_377_series_each_eval/eval_free_energy'
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
plt.plot(free_energy_series_dates, [np.load(f).sum()*dV for f in free_energy_files])
plt.title('AR 377')
plt.ylabel('total free magnetic energy')
figure_free_energy_path = 'free_energy.png'
plt.savefig(figure_free_energy_path, dpi=300)
