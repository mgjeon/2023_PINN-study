import os
import logging
import numpy as np
from datetime import datetime

from pinf.analytical_field import get_analytic_b_field
from pinf.trainer import NF2Trainer

b = get_analytic_b_field(n=1, m=1, l=0.3, psi=np.pi/2, resolution=64, bounds=[-1, 1, -1, 1, 0, 2])

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

base_path = './run2'
meta_path = None

bin = 1

height = 64
spatial_norm = 32
b_norm = 100

meta_info = None
dim = 256
positional_encoding = False
use_potential_boundary = True
potential_strides = 1
use_vector_potential = False
lambda_div = 1e-1
lambda_ff = 1e-1
decay_iterations = 25000
device = None
work_directory = None

total_iterations = 50000
batch_size = 10000
log_interval = 100
validation_interval = 100
num_workers = 8

# init logging
os.makedirs(base_path, exist_ok=True)
log = logging.getLogger()
log.setLevel(logging.INFO)
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(logging.FileHandler("{0}/{1}.log".format(base_path, "info_log")))  # set the new file handler
log.addHandler(logging.StreamHandler())  # set the new console handler

base_path = os.path.join(base_path, 'dim%d_bin%d_pf%s_ld%s_lf%s' % (
        dim, bin, str(use_potential_boundary), lambda_div, lambda_ff))

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