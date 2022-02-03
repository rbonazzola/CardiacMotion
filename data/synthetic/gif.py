import os
import vedo
import numpy as np
import trimesh
from copy import copy
from scipy.special import sph_harm
import pyvista as pv
from IPython import embed
import re
import random


def generate_gif(mesh4D, mesh_connectivity, filename, camera_position='xy', show_edges=False, **kwargs):
    
    '''
    Produces a gif file representing the motion of the input mesh.
    
    params:
      mesh4D: a sequence of Trimesh mesh objects.
      mesh_connectivity: faces in a PyVista-friendly format
      filename: the name of the output gif file.
      camera_position:
      
    return:
      None, only produces the gif file.
    '''
    
    pv.set_plot_theme("document")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # plotter = pv.Plotter(shape=(1, len(camera_positions)), notebook=False, off_screen=True)
    plotter = pv.Plotter(notebook=False, off_screen=True)
        
    # Open a gif
    plotter.open_gif(filename) 

    kk = pv.PolyData(np.array(mesh4D[0]), mesh_connectivity)
    # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
    plotter.add_mesh(kk, show_edges=show_edges) 
    
    for t, _ in enumerate(mesh4D):
        kk = pv.PolyData(np.array(mesh4D[t]), mesh_connectivity)
        plotter.camera_position = camera_position
        plotter.update_coordinates(kk.points, render=False)
        plotter.render()             
        plotter.write_frame()
    
    plotter.close()
    
    
def generate_gif_population(population, mesh_connectivity, N_gifs=10, output_dir=".", camera_positions=['xy', 'yz', 'xz'], verbose=False, **kwargs):
    
    '''
      population: population of moving meshes.
      N_gifs: number of gifs to be generated.
    
    '''
    
    for i, moving_mesh in enumerate(population):
        if verbose: print(i)
        for camera_position in camera_positions:        
            filename = os.path.join(output_dir, "gifs/subject{}_{}.gif".format(i, camera_position))
            generate_gif(moving_mesh, mesh_connectivity, filename, camera_position, **kwargs)        
        if i == N_gifs:
            break