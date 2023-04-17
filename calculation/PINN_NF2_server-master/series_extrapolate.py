import os
import argparse
import json
import logging
from datetime import datetime

import numpy as np
from astropy.nddata import block_reduce

from nf2.data.loader import load_hmi_data
from nf2.train.trainer import NF2Trainer

import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='config file for the simulation')
args = parser.parse_args()

with open(args.config) as config:
    info = json.load(config)
    for key, value in info.items():
        args.__dict__[key] = value

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)

# JSON
base_path = args.base_path
data_path = args.data_path
meta_path = args.meta_path

d_slice = args.d_slice
bin = args.bin

height = args.height 
spatial_norm = args.spatial_norm
b_norm = args.b_norm

meta_info = args.meta_info 
dim = args.dim
positional_encoding = args.positional_encoding
use_potential_boundary = args.use_potential_boundary
potential_strides = args.potential_strides
use_vector_potential = args.use_vector_potential
lambda_div = args.lambda_div
lambda_ff = args.lambda_ff
decay_iterations = args.decay_iterations
device = args.device
work_directory = args.work_directory

total_iterations = args.total_iterations
batch_size = args.batch_size
log_interval = args.log_interval
validation_interval = args.validation_interval
num_workers = args.num_workers

# find files
hmi_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp.fits')))  # x
hmi_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt.fits')))  # y
hmi_r_files = sorted(glob.glob(os.path.join(data_path, '*Br.fits')))  # z
err_p_files = sorted(glob.glob(os.path.join(data_path, '*Bp_err.fits')))  # x
err_t_files = sorted(glob.glob(os.path.join(data_path, '*Bt_err.fits')))  # y
err_r_files = sorted(glob.glob(os.path.join(data_path, '*Br_err.fits')))  # z

# init logging
os.makedirs(base_path, exist_ok=True)
log = logging.getLogger()
log.setLevel(logging.INFO)
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")))  # set the new file handler
log.addHandler(logging.StreamHandler())  # set the new console handler

# create series
start_time = datetime.now()
for hmi_p, hmi_t, hmi_r, err_p, err_t, err_r in tqdm(zip(hmi_p_files, hmi_t_files, hmi_r_files,
                                                    err_p_files, err_t_files, err_r_files), desc='Series', total=len(hmi_p_files)):
    file_id = os.path.basename(hmi_p).split('.')[3]
    series_base_path = os.path.join(base_path, 'dim%d_bin%d_pf%s_ld%s_lf%s' % (
        dim, bin, str(use_potential_boundary), lambda_div, lambda_ff))
    
    series_base_path = os.path.join(series_base_path, f'{file_id}')

    # print(series_base_path)

    log.info(f'START: {file_id}')

    # check if finished
    final_model_path = os.path.join(series_base_path, 'final.pt')
    if os.path.exists(final_model_path):
        meta_path = final_model_path
        continue

    # load data cube
    b_cube, error_cube, meta_info = load_hmi_data([hmi_p, err_p, hmi_r, err_r, hmi_t, err_t])

    if d_slice is not None:
        b_cube = b_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]
        error_cube = error_cube[d_slice[0]:d_slice[1], d_slice[2]:d_slice[3]]

    if bin > 1:
        b_cube = block_reduce(b_cube, (bin, bin, 1), np.mean)
        error_cube = block_reduce(error_cube, (bin, bin, 1), np.mean)

    trainer = NF2Trainer(series_base_path, b_cube, error_cube, height, spatial_norm, b_norm, 
                     meta_info=meta_info, dim=dim, positional_encoding=positional_encoding, 
                     meta_path=meta_path, use_potential_boundary=use_potential_boundary, 
                     potential_strides=potential_strides, use_vector_potential=use_vector_potential,
                     lambda_div=lambda_div, lambda_ff=lambda_ff, decay_iterations=decay_iterations,
                     device=device, work_directory=work_directory)
    
    # Train
    trainer.train(total_iterations, batch_size, 
                log_interval=log_interval, validation_interval=validation_interval, 
                num_workers=num_workers)
    meta_path = final_model_path

log.info(f'TOTAL RUNTIME: {datetime.now() - start_time}')