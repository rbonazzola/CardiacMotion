import os
import numpy as np
from SyntheticMeshPopulation import SyntheticMeshPopulation
import json
import argparse

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

    params = json.load(open("config.json")); 
    popu = SyntheticMeshPopulation(**params, from_cache_if_exists=False)
    # popu.generate_gif_population(show_edges=False)    
    popu.generate_gif_population()    

    main(params, ofile, args.generate_gif)
