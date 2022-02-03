import os
import vedo
import numpy as np
import trimesh
from copy import copy
from scipy.special import sph_harm
import ipywidgets as widgets
from ipywidgets import interact
import pyvista as pv
from IPython import embed
import re
import pickle as pkl
from utils import appendSpherical_np, cache_base_functions, generate_population
import gif
import json
import argparse

def save_population_as_pkl(population, ofile):

   '''
   Saves a population of moving 3D as a pickle file containing a dictionary.
   '''

   with open(ofile, "wb") as ff:
       pkl.dump(population, ff)


def main(params, ofile, generate_gif):
   
   avg_meshes, moving_meshes, coefs, ref_shape = generate_population(**params)
   
   population = {
     "params": params, "time_avg_mesh": avg_meshes, "moving_mesh": moving_meshes, "coefficients": coefs, "template_mesh": ref_shape
   }
        
   os.makedirs(os.path.dirname(ofile), exist_ok=True)
   save_population_as_pkl(population, ofile)
   
   if generate_gif:
       # connectivity
       conn = ref_shape.faces
       conn = np.c_[3 * np.ones(conn.shape[0]), conn].astype(int)  # add column of 3 to make it PyVista-compatible
       gif.generate_gif_population(
         moving_meshes, 
         output_dir=os.path.dirname(ofile), 
         mesh_connectivity=conn, 
         show_edges=False
       )


if __name__ == "__main__":

    def overwrite_config_items(config, args):
        for attr, value in args.__dict__.items():
            if attr in config.keys() and value is not None:
                config[attr] = value
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", help="path of config file", default="config.json")
    parser.add_argument("--output_dir", help="folder of the output file", default=None)
    parser.add_argument("--output_pkl_file", help="name of the output PKL file", default=None)
    parser.add_argument("--generate_gif", help="Generate files with GIF animations", action="store_true")

    params = json.load(open("config.json"))
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v))
    args = parser.parse_args()
    overwrite_config_items(params, args)

    print(params)

    FILE_PATTERN = "synthetic_population__N_{N}__T_{T}_" + \
    "_sigma_c_{amplitude_static_max}__sigma_s_{amplitude_dynamic_max}_" + \
    "_lmax_{l_max}__nmax_{freq_max}__res_{mesh_resolution}__seed_{random_seed}.pkl" 

    data_hash = hash(params.values()) % 1000000

    if args.output_pkl_file is None:
        ofile = os.path.join(f"synthetic_{data_hash}", FILE_PATTERN.format(**params))
    if args.output_dir:
        ofile = os.path.join(args.output_dir, ofile)        
    
    main(params, ofile, args.generate_gif)
