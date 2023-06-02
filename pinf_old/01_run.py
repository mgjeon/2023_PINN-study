import os
import json
import argparse
import logging
import numpy as np
from datetime import datetime

from pinf.analytical_field import get_analytic_b_field
from pinf.trainer import NF2Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
args = parser.parse_args()

with open(args.cfg) as config:
    info = json.load(config)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= info['simul']['gpu_id']

n = info['exact']['n']
m = info['exact']['m']
l = info['exact']['l']
psi = eval(info['exact']['psi'])
resolution = info['exact']['resolution']
bounds = info['exact']['bounds']

b = get_analytic_b_field(n=n, m=m, l=l, psi=psi, resolution=resolution, bounds=bounds)

base_path = os.path.join(info['simul']['base_path'], "run/")
meta_path = info['simul']['meta_path']

bin = info['simul']['bin']

height = info['simul']['height']
spatial_norm = info['simul']['spatial_norm']
b_norm = info['simul']['b_norm']

meta_info = info['simul']['meta_info']
dim = info['simul']['dim']
positional_encoding = info['simul']['positional_encoding']
use_potential_boundary = info['simul']['use_potential_boundary']
potential_strides = info['simul']['potential_strides']
use_vector_potential = info['simul']['use_vector_potential']
lambda_div = info['simul']['lambda_div']
lambda_ff = info['simul']['lambda_ff']
decay_iterations = info['simul']['decay_iterations']
device = info['simul']['device']
work_directory = info['simul']['work_directory']

total_iterations = info['simul']['total_iterations']
batch_size = info['simul']['batch_size']
log_interval = info['simul']['log_interval']
validation_interval = info['simul']['validation_interval']
num_workers = info['simul']['num_workers']

# init logging
os.makedirs(base_path, exist_ok=False)
log = logging.getLogger()
log.setLevel(logging.INFO)
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")))  # set the new file handler
log.addHandler(logging.StreamHandler())  # set the new console handler

# base_path = os.path.join(base_path, 'dim%d_bin%d_pf%s_ld%s_lf%s' % (
#         dim, bin, str(use_potential_boundary), lambda_div, lambda_ff))

b_cube = b[:, :, 0, :]

trainer = NF2Trainer(base_path, b_cube, height, spatial_norm, b_norm, 
                     meta_info=meta_info, dim=dim, positional_encoding=positional_encoding, 
                     meta_path=meta_path, use_potential_boundary=use_potential_boundary, 
                     potential_strides=potential_strides, use_vector_potential=use_vector_potential,
                     lambda_div=lambda_div, lambda_ff=lambda_ff, decay_iterations=decay_iterations,
                     device=device, work_directory=work_directory)

start_time = datetime.now()
trainer.train(total_iterations, batch_size, 
              log_interval=log_interval, validation_interval=validation_interval, 
              num_workers=num_workers)
log.info(f'TOTAL RUNTIME: {datetime.now() - start_time}')