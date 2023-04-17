import numpy as np 
from pathlib import Path
from tqdm import tqdm

mag_files = sorted(Path('ar_377_series_eval/magnetic_field').glob('**/*.npy'))
energy_path = Path('ar_377_series_eval/magnetic_energy')
energy_path.mkdir(exist_ok=True)
for mag_file in tqdm(mag_files):
    energy_file = energy_path / mag_file.name
    if energy_file.exists():
        continue
    b = np.load(mag_file)
    energy_density = (b**2).sum(-1) / (8*np.pi)
    energy_density = energy_density.sum()
    np.save(energy_file, energy_density)

pot_mag_files = sorted(Path('ar_377_series_eval/pot_magnetic_field').glob('**/*.npy'))
pot_energy_path = Path('ar_377_series_eval/pot_magnetic_energy')
pot_energy_path.mkdir(exist_ok=True)
for pot_mag_file in tqdm(pot_mag_files):
    pot_energy_file = pot_energy_path / pot_mag_file.name
    if pot_energy_file.exists():
        continue
    b_pot = np.load(pot_mag_file)
    pot_energy_density = (b_pot**2).sum(-1) / (8*np.pi)
    pot_energy_density = pot_energy_density.sum()
    np.save(pot_energy_file, pot_energy_density)

import pandas as pd
from datetime import datetime

energy_csv = Path('ar_377_series_eval') / 'energy.csv'
if energy_csv.exists():
    pass
else:
    energy_files = sorted(energy_path.glob('**/*.npy'))
    energy_series_dates = np.array([datetime.strptime(f.name, '%Y%m%d_%H%M%S.npy') for f in tqdm(energy_files)])
    energy_density = np.array([np.load(f) for f in tqdm(energy_files)])
    energy_df = pd.DataFrame({"date":energy_series_dates, "energy_density":energy_density})
    energy_df.to_csv(str(energy_csv), index=False)  
print(str(energy_csv))

pot_energy_csv = Path('ar_377_series_eval') / 'pot_energy.csv'
if pot_energy_csv.exists():
    pass 
else:
    pot_energy_files = sorted(pot_energy_path.glob('**/*.npy'))
    pot_energy_series_dates = [datetime.strptime(f.name, '%Y%m%d_%H%M%S.npy') for f in tqdm(pot_energy_files)]
    pot_energy_density = np.array([np.load(f) for f in tqdm(pot_energy_files)])
    pot_energy_df = pd.DataFrame({"date":pot_energy_series_dates, "energy_density":pot_energy_density})
    pot_energy_df.to_csv(str(pot_energy_csv), index=False)  
print(str(pot_energy_csv))

free_energy_csv = Path('ar_377_series_eval') / 'free_energy.csv'
if free_energy_csv.exists():
    pass 
else:
    energy_df = pd.read_csv(energy_csv)
    pot_energy_df = pd.read_csv(pot_energy_csv)
    free_energy_series_dates = energy_df['date']
    free_energy_density = energy_df['energy_density'] - pot_energy_df['energy_density']
    free_energy_df = pd.DataFrame({"date":free_energy_series_dates, "energy_density":free_energy_density})
    free_energy_df.to_csv(str(free_energy_csv), index=False)  

print(str(free_energy_csv))
