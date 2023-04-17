# Find HARP number for NOAA 11158 (2011-02-12 00:00:00)
from mgpinn.downloader import mgDownloader
import datetime 

jsoc_email = 'mgjeon@khu.ac.kr'
noaa_num = 11158
start_time = datetime.datetime(2011, 2, 12, 0, 0, 0)
dl = mgDownloader(jsoc_email, noaa_num, start_time)

dl.find_harp_number()
print(f'NOAA: {dl.noaa_num}')
print(f'Start time: {dl.start_time}')
print(f'HARPNUM: {dl.harp_num}')

# Download fits files
query = False
dl.download_fits(duration='120h', query=query, download=False)
print(f'Duration: {dl.duration}')
print(f'Query: {dl.ds}')
print(f'Destination: {dl.series_download_dir}')
if query:
    print(f'Request URL: {dl.request_url}')
    print(f'# of available files: {dl.num_available_files}')

# Pytorch
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
# import torch
# import glob
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Visible device: ', os.environ["CUDA_VISIBLE_DEVICES"])
# print('Device: ', device)
# print('Device count: ', torch.cuda.device_count())
# print('Current device: ', torch.cuda.current_device())

# Parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--option', type=str, required=True,
                    choices=['initial', 'series', 'mag'])
args = parser.parse_args()

# Data parameters
from mgpinn.maker import mgMaker
bin = 2
d_slice = [66, 658, 9, 377] # crop

# Trainer parameters
from nf2.train.trainer import NF2Trainer
dim = 256
use_potential_boundary = True 
lambda_div = 0.1
lambda_ff = 0.1
height = 320 // bin
spatial_norm = 320 // bin
b_norm = 2500
decay_iterations = int(5e4)
meta_path = None

import warnings
warnings.filterwarnings(action='ignore') 

maker = mgMaker(bin, d_slice, dl.series_download_dir)

if args.option == 'initial':
    # Data
    maker.data(0)

    # Trainer
    trainer = NF2Trainer(maker.result_path, maker.hmi_cube, maker.error_cube, 
                         height, spatial_norm, b_norm,
                         meta_info=maker.meta_info, 
                         dim=dim, use_potential_boundary=use_potential_boundary, 
                         lambda_div=lambda_div, lambda_ff=lambda_ff,
                         decay_iterations=decay_iterations, 
                         meta_path=meta_path)

    iterations = int(10e4)
    batch_size = int(1e4)
    log_interval = int(1e4)
    validation_interval = int(1e4)

    trainer.train(iterations, batch_size, log_interval, validation_interval)
elif args.option == 'series':
    iterations = int(2e3)
    batch_size = int(1e4)
    log_interval = int(2e3)
    validation_interval = int(-1)
    
    for i in range(maker.num_files):
        # Data
        maker.data(i)

        final_model_path = os.path.join(maker.result_path, 'final.pt')
        if os.path.exists(final_model_path):
            meta_path = final_model_path
            print(meta_path)
            continue

        # Trainer
        trainer = NF2Trainer(maker.result_path, maker.hmi_cube, maker.error_cube, 
                            height, spatial_norm, b_norm,
                            meta_info=maker.meta_info, 
                            dim=dim, use_potential_boundary=use_potential_boundary, 
                            lambda_div=lambda_div, lambda_ff=lambda_ff,
                            decay_iterations=decay_iterations, 
                            meta_path=meta_path)

        trainer.train(iterations, batch_size, log_interval, validation_interval)
        meta_path = final_model_path
elif args.option == 'mag':
    # import numpy as np 
    # import torch 
    # from tqdm import tqdm
    
    # def load_cube(path):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     state = torch.load(path, map_location=device)
    #     model = torch.nn.DataParallel(state['model'])
    #     cube_shape = state['cube_shape']
    #     z = cube_shape[2]
    #     coords = np.stack(np.mgrid[:cube_shape[0]:1, :cube_shape[1]:1, :z:1], -1)
    #     coords = torch.tensor(coords / spatial_norm, dtype=torch.float32)
    #     coords_shape = coords.shape
    #     cube = []
    #     batch_size = 1000
    #     coords = coords.view((-1, 3))
    #     it = range(int(np.ceil(coords.shape[0] / batch_size)))
    #     for k in tqdm(it):
    #         coord = coords[k * batch_size: (k + 1) * batch_size]
    #         coord = coord.to(device)
    #         coord.requires_grad = True
    #         cube += [model(coord).detach().cpu()]
    #     cube = torch.cat(cube)
    #     cube = cube.view(*coords_shape).numpy()
    #     b_norm = 2500
    #     b = cube * b_norm
    #     return b
    from nf2.evaluation.unpack import load_cube
    from nf2.potential.potential_field import get_potential

    import glob
    import numpy as np
    result_path = dl.series_download_dir[:-5]
    nf2_paths = sorted(glob.glob(os.path.join(result_path, '**', 'extrapolation_result.nf2')))
    eval_path = result_path+'_eval/'
    mag_path = os.path.join(eval_path, 'magnetic_field')
    pot_mag_path = os.path.join(eval_path, 'pot_magnetic_field')
    os.makedirs(mag_path, exist_ok=True)
    os.makedirs(pot_mag_path, exist_ok=True)

    for path in nf2_paths:
        print(path)
        mag_f = os.path.join(mag_path, os.path.basename(os.path.dirname(path))) + '.npy'
        pot_mag_f = os.path.join(pot_mag_path, os.path.basename(os.path.dirname(path))) + '.npy'
        if os.path.exists(mag_f) and os.path.exists(pot_mag_f): 
            continue
        b = load_cube(path, progress=True)
        potential = get_potential(b[:, :, 0, 2], b.shape[2], batch_size=int(1e3))
        b_potential = - 1 * np.stack(np.gradient(potential, axis=[0, 1, 2], edge_order=2), axis=-1)
        np.save(mag_f, b)
        np.save(pot_mag_f, b_potential)