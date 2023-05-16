from tvtk.api import tvtk, write_data
import pyvista as pv 
import os
import numpy as np

def pv_plot(B, vtk_path='./evaluation.vtk', points=((7, 64, 8), (7, 64, 8)), title='LL', overwrite=False):

    if not os.path.exists(vtk_path) or overwrite:
        dim = B.shape[:-1]
        pts = np.stack(np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]], -1).astype(np.float32)
        pts = pts.transpose(2, 1, 0, 3)
        pts = pts.reshape((-1, 3))
        vectors = B.transpose(2, 1, 0, 3)
        vectors = vectors.reshape((-1, 3))
        sg = tvtk.StructuredGrid(dimensions=dim, points=pts)
        sg.point_data.vectors = vectors
        sg.point_data.vectors.name = 'B'
        write_data(sg, str(vtk_path))

    mesh = pv.read(vtk_path)
    xindmax, yindmax, zindmax = mesh.dimensions
    xcenter, ycenter, zcenter = mesh.center

    mesh_g = mesh.compute_derivative(scalars='B')

    def gradients_to_dict(arr):
        keys = np.array(
            ["dBx/dx", "dBx/dy", "dBx/dz", "dBy/dx", "dBy/dy", "dBy/dz", "dBz/dx", "dBz/dy", "dBz/dz"]
        )
        keys = keys.reshape((3,3))[:, : arr.shape[1]].ravel()
        return dict(zip(keys, mesh_g['gradient'].T))

    gradients = gradients_to_dict(mesh_g['gradient'])

    curlB_x = gradients['dBz/dy'] - gradients['dBy/dz']
    curlB_y = gradients['dBx/dz'] - gradients['dBz/dx']
    curlB_z = gradients['dBy/dx'] - gradients['dBx/dy']

    curlB = np.vstack([curlB_x, curlB_y, curlB_z]).T

    mesh.point_data['curlB'] = curlB

    p = pv.Plotter()
    p.add_mesh(mesh.outline())

    #-----------------------
    sargs_B = dict(
        title='Bz [G]',
        title_font_size=15,
        height=0.25,
        width=0.05,
        vertical=True,
        position_x = 0.05,
        position_y = 0.05,
    )
    dargs_B = dict(
        scalars='B', 
        component=2, 
        clim=(-100, 100), 
        scalar_bar_args=sargs_B, 
        show_scalar_bar=True, 
        lighting=False
    )
    p.add_mesh(mesh.extract_subset((0, xindmax, 0, yindmax, 0, 0)), 
            cmap='gray', **dargs_B)
    #-----------------------

    #-----------------------
    sargs_J = dict(
        title='J = curl(B)',
        title_font_size=15,
        height=0.25,
        width=0.05,
        vertical=True,
        position_x = 0.9,
        position_y = 0.05,
    )
    dargs_J = dict(
        scalars='curlB', 
        clim=(0, 20),
        scalar_bar_args=sargs_J, 
        show_scalar_bar=True, 
        lighting=False
    )
    #-----------------------

    def draw_streamlines(pts):
        stream, src = mesh.streamlines(
            return_source=True,
            # source_center=(120, 90, 0),
            # source_radius=5,
            # n_points=100,
            start_position = pts,
            integration_direction='both',
            # progress_bar=False,
            max_time=1000,
            # initial_step_length=0.001,
            # min_step_length=0.001,
            # max_step_length=2.0,
            # max_steps=999999,
            # terminal_speed = 1e-16,
            # max_error = 1e-6,
        )
        p.add_mesh(stream.tube(radius=0.2), 
                cmap='plasma', **dargs_J)
        p.add_mesh(src, point_size=10)

    xrange = points[0]
    yrange = points[1]
    for i in np.arange(*xrange):
        for j in np.arange(*yrange):
            try: 
                draw_streamlines((i, j, 0))
            except:
                print(i, j)


    # stream, src = mesh.streamlines(
    #         return_source=True,
    #         # source_center=(120, 90, 0),
    #         source_radius=64,
    #         n_points=500,
    #         # start_position = pts,
    #         integration_direction='both',
    #         # progress_bar=False,
    #         max_time=1000,
    #         # initial_step_length=0.001,
    #         # min_step_length=0.001,
    #         # max_step_length=2.0,
    #         # max_steps=999999,
    #         # terminal_speed = 1e-16,
    #         # max_error = 1e-6,
    #     )
    # p.add_mesh(stream.tube(radius=0.2), 
    #         cmap='plasma', **dargs_J)

    p.camera_position = 'xy'           
    # p.camera.azimuth = -100
    # p.camera.elevation = -20
    # p.camera.zoom(1.0)
    p.show_bounds()
    p.add_title(title)
    p.show(jupyter_backend='static')