from datetime import datetime
from pathlib import Path
import netCDF4
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_energy(filename, ncFormat):
        nc=netCDF4.Dataset(filename,'r')

        nc_x = nc.variables['x']
        nc_y = nc.variables['y']
        nc_z = nc.variables['z']
        nc_bx = nc.variables['Bx']
        nc_by = nc.variables['By']
        nc_bz = nc.variables['Bz']
        nc_bxp=nc.variables['Bx_pot']
        nc_byp=nc.variables['By_pot']
        nc_bzp=nc.variables['Bz_pot']
        x = nc_x[:]
        y = nc_y[:]
        z = nc_z[:]
        bx = nc_bx[:].transpose(2,1,0)
        by = nc_by[:].transpose(2,1,0)
        bz = nc_bz[:].transpose(2,1,0)
        bx_pot=nc_bxp[:].transpose(2,1,0)
        by_pot=nc_byp[:].transpose(2,1,0)
        bz_pot=nc_bzp[:].transpose(2,1,0)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        bx = np.array(bx)
        by = np.array(by)
        bz = np.array(bz)
        bx_pot = np.array(bx_pot)
        by_pot = np.array(by_pot)
        bz_pot = np.array(bz_pot)

        # erg / cm^3
        e = (bx**2 + by**2 + bz**2) / (8*np.pi)
        energy_density = e.sum()
        e_pot = (bx_pot**2 + by_pot**2 + bz_pot**2) / (8*np.pi)
        energy_density_pot = e_pot.sum()

    #     # 1 Mm = 1e8 cm
        dx = (x[1]-x[0])
        dy = (y[1]-y[0])
        dz = (z[1]-z[0])

    #     # cm^3
    #     dV = dx*dy*dz

    #     # Energy (10^33 erg)
    #     energy = (energy_density*dV)/1e33

        date = datetime.strptime(filename.name, ncFormat)
        return date, energy_density, energy_density_pot, dx, dy, dz

def Kusano_csv(ncfiles, ncFormat, csvFile):    
    dates = []
    # 1e33
    energys = []
    energys_pot = []
    dxx = []
    dyy = []
    dzz = []

    for ncf in tqdm(ncfiles):
        date, energy, energy_pot, dx, dy, dz = get_energy(ncf, ncFormat)
        dates.append(date)
        energys.append(energy)
        energys_pot.append(energy_pot)
        dxx.append(dx)
        dyy.append(dy)
        dzz.append(dz)

    df = pd.DataFrame({"date":dates, "energy_density":energys, "energy_density_pot":energys_pot, "dx":dxx, "dy":dyy, "dz":dzz})

    df.to_csv(csvFile, index=False)

nc_path = Path('Kusano/NOAA11158')
ncfiles = sorted([x for x in nc_path.glob('**/*.nc')])
ncFormat = '11158_%Y%m%d_%H%M%S.nc'
csvFile = 'Kusano/NOAA11158_Kusano.csv'

Kusano_csv(ncfiles, ncFormat, csvFile)