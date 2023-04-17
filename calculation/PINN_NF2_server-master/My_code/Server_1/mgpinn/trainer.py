import torch
import numpy as np

class mgTrainer():
    def __init__(self, result_path,
                hmi_cube, error_cube, 
                bin, meta_info,
                dim=256, use_potential_boundary=True,
                lambda_div=0.1, lambda_ff=0.1,
                decay_iterations=int(5e4),
                final_model_path=None):
        self.result_path = result_path
        self.hmi_cube = hmi_cube
        self.error_cube = error_cube
        self.bin = bin  
        self.meta_info = meta_info

        self.dim = dim 
        self.use_potential_boundary = use_potential_boundary
        self.lambda_div = lambda_div 
        self.lambda_ff = lambda_ff 
        self.decay_iterations = decay_iterations
        self.final_model_path = final_model_path

        self.spatial_norm = 320 // self.bin
        self.height = 320 // self.bin
        self.b_norm = 2500  

        mf_coords = np.mgrid[:self.hmi_cube.shape[0], :self.hmi_cube.shape[1], :1]
        mf_coords = np.stack(mf_coords, -1)
        mf_coords = mf_coords.reshape((-1, 3))
        mf_values = self.hmi_cube.reshape((-1, 3))
        mf_err = self.error_cube.reshape((-1, 3))
        pf_coords, pf_err, pf_values = load_potential_field_data(b_cube, height, potential_strides=4)


        print(np.shape(mf_coords))


        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

