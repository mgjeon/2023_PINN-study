import drms
import os
class mgDownloader():

    def __init__(self, jsoc_email, noaa_num, start_time):

        self.jsoc_email = jsoc_email
        self.noaa_num = noaa_num
        self.start_time = start_time.isoformat('_', timespec='seconds')
        
        self.client = drms.Client(email=self.jsoc_email, verbose=True)

    def find_harp_number(self):
        ar_mapping = self.client.query('hmi.Mharp_720s[][%sZ]' % self.start_time, \
                                key=['NOAA_AR', 'HARPNUM'])
        harp_num = ar_mapping[ar_mapping['NOAA_AR'] == self.noaa_num]['HARPNUM']
        self.harp_num = int(harp_num.iloc[0])
        
    def download_fits(self, duration, query=False, download=False):
        self.duration = duration
        self.ds = 'hmi.sharp_cea_720s[%d][%s/%s]{Br, Bp, Bt, Br_err, Bp_err, Bt_err}' % \
                  (self.harp_num, self.start_time, self.duration)
        
        self.series_download_dir = 'ar_%d_series_fits' % self.harp_num
        os.makedirs(self.series_download_dir, exist_ok=True)
        if query:
            result = self.client.export(self.ds, method="url", protocol='fits')
            self.request_url = result.request_url
            self.num_available_files = len(result.urls)
            if download:
                result.download(self.series_download_dir)