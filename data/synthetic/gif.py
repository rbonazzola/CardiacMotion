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


def generate_gif(mesh4D, mesh_connectivity, filename, camera_position='xy'):
    
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

    kk = pv.PolyData(np.array(mesh4D[0].vertices), mesh_connectivity)
    # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
    plotter.add_mesh(kk, show_edges=True) 
    
    for t, _ in enumerate(mesh4D):
        kk = pv.PolyData(np.array(mesh4D[t].vertices), mesh_connectivity)
        plotter.camera_position = camera_position
        plotter.update_coordinates(kk.points, render=False)
        plotter.render()             
        plotter.write_frame()
    
    plotter.close()
    
    
def generate_gif_population(population, mesh_connectivity, N_gifs=10, camera_positions=['xy', 'yz', 'xz'], verbose=False):
    
    '''
      population: population of moving meshes.
      N_gifs: number of gifs to be generated.
    
    '''
    
    for i, moving_mesh in enumerate(population):
        if verbose: print(i)
        for camera_position in camera_positions:        
            filename = "gifs/subject{}_{}.gif".format(i, camera_position)
            generate_gif(moving_mesh, mesh_connectivity, filename, camera_position)        
        if i == N_gifs:
            break