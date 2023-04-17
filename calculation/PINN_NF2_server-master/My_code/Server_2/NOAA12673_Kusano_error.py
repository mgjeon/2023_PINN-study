import netCDF4

# This is a corrupted file
filename = '/userhome/jeon_mg/workspace/PINN_NF2/data/Kusano/12673_20170906_060000.nc'
nc=netCDF4.Dataset(filename,'r')
print(nc)
"""
<class 'netCDF4._netCDF4.Dataset'>
root group (NETCDF3_CLASSIC data model, file format NETCDF3):
    dimensions(sizes): x(1025), y(513), z(513)
    variables(dimensions): float64 x(x), float64 y(y), float64 z(z), float64 Bx(z, y, x)
    groups: 
"""