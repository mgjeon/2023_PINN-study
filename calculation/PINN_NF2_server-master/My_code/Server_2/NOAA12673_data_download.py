#-----Find HARP number for NOAA 11158 (2011-02-12 00:00:00)
import drms
jsoc_email = 'mgjeon@khu.ac.kr'
client = drms.Client(email=jsoc_email, verbose=True)

import datetime 

noaa_num = 12673
start_time = datetime.datetime(2017, 9, 4, 0, 0, 0)
start_time = start_time.isoformat('_', timespec='seconds')
ar_mapping = client.query('hmi.Mharp_720s[][%sZ]' % start_time, \
                           key=['NOAA_AR', 'HARPNUM'])
harp_num = ar_mapping[ar_mapping['NOAA_AR'] == noaa_num]['HARPNUM']
harp_num = harp_num.iloc[0]
print(f'NOAA {noaa_num}')
print(f'Start Time {start_time}')
print(f'HARPNUM {harp_num}')

#-----Download SHARP fits files
import os
duration = '90h'
#Construct the DRMS query string: "Series[harpnum][timespan]{data segments}"
ds = 'hmi.sharp_cea_720s[%d][%s/%s]{Br, Bp, Bt, Br_err, Bp_err, Bt_err}' % \
     (harp_num, start_time, duration)
print(f'\n{ds}\n')

series_download_dir = 'ar_%d_series' % harp_num
os.makedirs(series_download_dir, exist_ok=True)

result = client.export(ds, method="url", protocol='fits')
print(f"\nRequest URL: {result.request_url}")
print(f"{int(len(result.urls))} file(s) available for download.\n")
result.download(series_download_dir)