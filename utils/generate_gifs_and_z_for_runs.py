import pandas as pd
import mlflow
import ipywidgets as widgets
from ipywidgets import interact
import os, sys
import glob
import torch
from pprint import pprint
from easydict import EasyDict
import pickle as pkl

import numpy as np
import torch
import yaml
from IPython import embed
from utils.image_helpers import generate_gif, merge_gifs_horizontally

os.environ['HOME'] = "/home/user"
os.environ['CARDIAC_MOTION_REPO'] = os.environ["HOME"] + "/01_repos/CardiacMotion"
os.chdir(os.environ['CARDIAC_MOTION_REPO'])

sys.path.append(os.environ['CARDIAC_MOTION_REPO'])

MLFLOW_TRACKING_URI = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Choose runs with good performance
df = mlflow.search_runs(experiment_ids=[str(i) for i in range(2,9)])
df = df[(df["metrics.val_rec_ratio_to_time_mean"] < 1) & (df["params.dataset_n_timeframes"] == '10')]
df["partition"] = df.experiment_id.apply(lambda expid: mlflow.get_experiment(expid).name)
df = df.set_index("run_id")

print(df)

########################################################################

from main_autoencoder_cardiac import *
from config.load_config import load_yaml_config

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence 
from models.Model4D import EncoderTemporalSequence, AutoencoderTemporalSequence
from lightning.ComaLightningModule import CoMA_Lightning
from models.lightning.EncoderLightningModule import TemporalEncoderLightning
from models.TemporalAggregators import TemporalAggregator, FCN_Aggregator

POLYNOMIAL_DEGREE = 10

config = load_yaml_config("config_folded_c_and_s.yaml")
config.network_architecture.pooling.parameters.downsampling_factors = [3, 3, 2, 2] # * 4
config.network_architecture.latent_dim_c = 8 
config.network_architecture.latent_dim_s = 8

from fuzzywuzzy import fuzz, process

######################################################################

ID = "1000511"
fhm_mesh = Cardiac3DMesh(
   filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{ID}/models/FHM_res_0.1_time001.npy",
   faces_filename="/home/user/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
   subpart_id_filename="/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
)

df = df[df['metrics.test_rec_ratio_to_time_mean'] < 0.8] 

df = df.sort_values("partition")

print(df.shape)

print(df.head())

for runid, row in df.iterrows():

    if row.status == "RUNNING":
        print(f"Skipping {runid} because it's still running...")
        continue

    partition = df.loc[runid, ["partition"]].item()
    PARTITION = process.extractOne(partition, partitions.keys())[0]
    
    expid = row.experiment_id

    #### CHECKPOINT
    ckpt_dir = f"{os.environ['HOME']}/01_repos/CardiacMotion/{expid}/{runid}/checkpoints"
    ckpt_path = f"{ckpt_dir}/{os.listdir(ckpt_dir)[0]}"

    # model_weights = torch.load(ckpt_path, map_location=torch.device('cpu'))["state_dict"]
    model_weights = torch.load(ckpt_path)["state_dict"]
    print(f"Loaded weights from checkpoint:\n {ckpt_path}")
    # model_weights = EasyDict(model_weights)
    model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})
    
    try:
        model_weights["encoder.encoder_3d_mesh.layers.layer_2.graph_conv.lins.11.weight"]
        POLYNOMIAL_DEGREE = 12
    except:
        POLYNOMIAL_DEGREE = 10

    config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
    ################################################
    
    FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
    MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
    PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
    SUBSETTING_MATRIX_FILE = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 
    
    subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
    template = EasyDict({
      "v": np.load(MEAN_ACROSS_CYCLE_FILE),
      "f": fhm_mesh[partitions[PARTITION]].f
    })
    
    ################################################
    
    NT = 10 # config.dataset.parameters.T
    cardiac_dataset = CardiacMeshPopulationDataset(
        root_path="data/cardio/Results", 
        procrustes_transforms=PROCRUSTES_FILE,
        faces=template.f,
        subsetting_matrix=subsetting_matrix,
        template_mesh= template,
        N_subj=None,
        phases_filter=1+(50/NT)*np.array(range(NT))
    )
    
    print(f"Length of dataset: {len(cardiac_dataset)}") 
    
    mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=16)
    mesh_dm.setup()
    
    x = EasyDict(next(iter(mesh_dm.train_dataloader())))
    
    mesh_template = mesh_dm.dataset.template_mesh
    coma_args = get_coma_args(config)
    coma_matrices = get_coma_matrices(config, mesh_template, PARTITION)
    coma_args.update(coma_matrices)
    
    enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
    encoder = Encoder3DMesh(**enc_config)
    
    enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 
    
    h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)
    
    z_aggr = FCN_Aggregator(
        features_in = NT*h.shape[-1],
        features_out= enc_config.latent_dim
    )
    
    t_encoder = EncoderTemporalSequence(
        encoder3d = encoder,
        z_aggr_function=z_aggr
    )
    
    decoder_config_c = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_C_ARGS })
    decoder_config_s = EasyDict({ k:v for k,v in coma_args.items() if k in DECODER_S_ARGS })    
    decoder_content = DecoderContent(decoder_config_c)
    decoder_style = DecoderStyle(decoder_config_s, phase_embedding_method="exp_v1")
    t_decoder = DecoderTemporalSequence(decoder_content, decoder_style)
        
    t_ae = AutoencoderTemporalSequence(
        encoder=t_encoder,
        decoder=t_decoder
    )
    
    
    t_ae.load_state_dict(model_weights)
    t_ae = t_ae.to("cuda:0")
    
    mesh_dl = torch.utils.data.DataLoader(cardiac_dataset, batch_size=128, num_workers=16)
    
    MLRUNS_DIR = "/mnt/data/workshop/workshop-user1/output/CardiacMotion/mlruns"
    # RUN_ID = "8c1ffa20cacc4b6c88e18159e01867b4"
    ZFILE = f"{MLRUNS_DIR}/{expid}/{runid}/artifacts/latent_vector.csv"
    
    x["s_t"] = x["s_t"].to("cuda:0")
    output = t_ae(x["s_t"])
    s_t, s_hat_t = x["s_t"], output[2]
    
    if not os.path.exists(ZFILE):    
        
        torch.cuda.empty_cache()
    
        zs = []
        
        for i, x in tqdm(enumerate(mesh_dl)):
            
            # if (i % 10) == 0:
            # print(i)
                
            if i < (len(zs)-1):
                continue
            
            x['s_t'] = x['s_t'].to("cuda:0")
            z = t_ae.encoder(x['s_t'])
            z = z['mu'].detach().cpu().numpy()
            zs.append(z)
            
            
            # zs.append(z)
            torch.cuda.empty_cache() 
        
        zs_concat = np.concatenate(zs)
        z_df = pd.DataFrame(zs_concat, index=cardiac_dataset.ids)
        del zs_concat, zs
        
        # colnames before: 0, 1, 2, 3
        z_df.columns = [ f"z{str(i).zfill(3)}" for i in range(16) ]
        # colnames after: z000, z001, z002, z003
        
        z_df = z_df.reset_index().rename({"index": "ID"}, axis=1)
        z_df.head()
        
        z_df.to_csv(ZFILE, index=False)
        print(ZFILE)
    
    subj_ids = list(range(2))
    faces = template.f
    
    ODIR = f"{os.environ['HOME']}/01_repos/CardiacMotion/mlruns/{expid}/{runid}/artifacts/output/gif"            
   
    if os.path.exists(ODIR) and (len(os.listdir(ODIR)) == 0):
       continue

    for subj_id in tqdm(subj_ids):
    
        for camera in ["xz"]: # , "xy", "yz"]:    
            for suffix, st in {"original": s_t, "reconstruction": s_hat_t}.items():
                mesh4D = st.detach().cpu().numpy().astype("float32")[subj_id]        
                gifpath = generate_gif(
                    mesh4D,
                    faces, 
                    camera_position=camera,
                    filename=f"{ODIR}/id{subj_id}_{suffix}_{camera}.gif", 
                )
            
            merge_gifs_horizontally(
                f"{ODIR}/id{subj_id}_original_{camera}.gif", 
                f"{ODIR}/id{subj_id}_reconstruction_{camera}.gif", 
                f"{ODIR}/id{subj_id}_{PARTITION}_{camera}.gif"
            )
            
            os.remove(f"{ODIR}/id{subj_id}_original_{camera}.gif")
            os.remove(f"{ODIR}/id{subj_id}_reconstruction_{camera}.gif")
