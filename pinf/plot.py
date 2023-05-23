from tvtk.api import tvtk, write_data
import pyvista as pv 
import os
import numpy as np

def pv_plot(B, vtk_path='./evaluation.vtk', points=((7, 64, 8), (7, 64, 8)), overwrite=False):

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

    p = pv.Plotter()
    p.add_mesh(mesh.outline())

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
        clim=(-150, 150), 
        scalar_bar_args=sargs_B, 
        show_scalar_bar=False, 
        lighting=False
    )
    p.add_mesh(mesh.extract_subset((0, xindmax, 0, yindmax, 0, 0)), 
            cmap='gray', **dargs_B)

    def draw_streamlines(pts):
        stream, src = mesh.streamlines(
            return_source=True,
            start_position = pts,
            integration_direction='both',
            max_time=1000,
        )
        # print(pts)
        key = pts[0]*pts[1] + (pts[0]//pts[1]) + (pts[0] - pts[1])
        # print(key)
        np.random.seed(key)
        colors = np.random.rand(3)
        # if pts[0] == 16 and pts[1] == 48:
        #     colors = 'white'
        # print(colors)
        p.add_mesh(stream.tube(radius=0.2), lighting=False, color=colors)
        p.add_mesh(src, point_size=7, color=colors)

    xrange = points[0]
    yrange = points[1]
    for i in np.arange(*xrange):
        for j in np.arange(*yrange):
            try: 
                draw_streamlines((i, j, 0))
            except:
                print(i, j)

    p.camera_position = 'xy'
    p.show_bounds()
    # p.add_title(title)

    return p