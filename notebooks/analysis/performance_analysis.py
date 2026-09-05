# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas

# %%
import mlflow
import ipywidgets as widgets
from ipywidgets import interact
import os
import glob
import torch
from pprint import pprint
from easydict import EasyDict

import numpy as np
import torch
import yaml

# %%
MLFLOW_TRACKING_URI = "/home/home01/scrb/01_repos/CardiacMotion/mlruns/"
EXPERIMENT_NAME = "Synthetic data"
EXPERIMENT_NAME = "test"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# %%
MLFLOW_URI = mlflow.tracking.get_tracking_uri()
EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

PREFIX = f"{MLFLOW_URI}/{EXPERIMENT_ID}"

# %%
meta_yamls = [f"{PREFIX}/{run}/meta.yaml" for run in os.listdir(PREFIX) if os.path.exists(f"{PREFIX}/{run}/meta.yaml")]

count = 0
for meta_yaml_path in meta_yamls:
    meta_yaml = yaml.safe_load(open(meta_yaml_path))    
    if meta_yaml['experiment_id'] != EXPERIMENT_ID:
        meta_yaml['experiment_id'] = EXPERIMENT_ID
        count += 1
        yaml.dump(meta_yaml, open(meta_yaml_path, "wt"))
        
if count != 0:
    print(f"{count} runs's experiments were fixed to match the experiment of the parent folder")

# %%
df = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID])
df = df[df["metrics.test_rec_ratio_to_time_mean"] < 0.9]
df

# %% [markdown]
# ___

# %%
### Select runs with $T=1$

# row_index = (df["params.dataset_n_timeframes"] == "1")
# 
# # Select runs with performance better than random
# row_index &= (df["metrics.test_rec_ratio_to_pop_mean"] < 1)

# %%
# columns
normalized_metrics = df.columns[df.columns.str.startswith("metrics.test_") & df.columns.str.contains("ratio")]

dataset_params = df.columns[df.columns.str.startswith("params.dataset_")].to_list()

params = df.columns[df.columns.str.startswith("params.")].to_list()

arch_params = [
    'params.n_channels_enc', 
    'params.n_channels_dec_c', 
    'params.n_channels_dec_s', 
    'params.latent_dim_c', 
    'params.latent_dim_s',
    'params.z_aggr_function',
    'params.reduction_factors',
]

# To not display columns that have the same value for all rows
# https://stackoverflow.com/questions/57365283/how-to-show-columns-that-have-different-values-in-rows
def diff_cols(df):
    my_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=False) > 1:
            my_cols.append(col)
    return df[my_cols].copy()


# %%
columns = ["run_id"] + params + normalized_metrics.tolist()

diff_cols(
    df[columns].reset_index(drop=True)
)

# %%
good_runs = df.run_id.values
run_w = widgets.Select(options=good_runs)

# %%
# @interact
# def get_ckptpath(run=run_w):
#     
#     # chkpt_dir = f"{MLFLOW_TRACKING_URI}/{EXPERIMENT_ID}/{RUNID}/"
#     
#     REPO_DIR = "/root/Rodrigo_repos/CardiacMotion"
#     global ckpt_path, model_weights    
#     ckpt_path = glob.glob(f"{REPO_DIR}/1/{run}/checkpoints/*ckpt")    
#     if len(ckpt_path) == 1:
#       ckpt_path = ckpt_path[0]
#     elif len(ckpt_path) == 0:
#       ckpt_path = None
#     
#     model_weights = torch.load(ckpt_path)["state_dict"]
#     model_weights = EasyDict(model_weights)
#     
#     print(mlflow.get_run(run).data.metrics["test_rec_ratio_to_pop_mean"])
#     return ckpt_path

# %%
n_params = 0

for module_name, weights in model_weights.items():
    module_name = module_name.replace("model.", "")
    n_params += np.prod(weights.shape)
    # print(f'{module_name}: {weights.shape}')
    
n_params

# %% [markdown]
# ### Get run IDs based on metrics 

# %%
# run_w = widgets.Select(options=["6a4d73fb59f24d97b37764afdedd4185"])
run_w

# %% [markdown]
# ### Load weights

# %%
ckpt_dir = f"/home/home01/scrb/01_repos/CardiacMotion/mlruns/1//{run_w.value}/checkpoints"
ckpt_path = f"{ckpt_dir}/{os.listdir(ckpt_dir)[0]}"

# %%
# ckpt_path = f"/root/Rodrigo_repos/CardiacMotion/2/{run_w.value}/checkpoints/epoch=334-step=38524.ckpt"
model_weights = torch.load(ckpt_path)["state_dict"]
print(f"Loaded weights from checkpoint:\n {ckpt_path}")
model_weights = EasyDict(model_weights)

# %% [markdown]
# ### Initialize weights

# %%
import sys
import pickle as pkl
import os

os.environ["HOME"] = "/root"
os.environ['CARDIAC_MOTION_REPO'] = os.environ["HOME"] + "/Rodrigo_repos/CardiacMotion"
os.chdir(os.environ['CARDIAC_MOTION_REPO'])

sys.path.append(os.environ['CARDIAC_MOTION_REPO'])

from main_autoencoder_cardiac import *
from config.load_config import load_yaml_config

# %% [markdown]
# #### Synthetic dataset

# %%
from utils.helpers import get_coma_args

# %%
config = load_yaml_config("config_files/config_folded_c_and_s.yaml")

# %%
from data.SyntheticDataModules import SyntheticMeshesDataset

# %%
from config.load_config import load_yaml_config
config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
config.dataset.parameters.N = 1280
config.dataset.parameters.T = 20
config.dataset.random_seed = 135

mesh_ds = SyntheticMeshesDataset(config.dataset.parameters, config.dataset.preprocessing)
mesh_dl = DataLoader(mesh_ds, batch_size=16)

# %%
config.network_architecture.latent_dim_c = 9
config.network_architecture.latent_dim_s = 36

POLYNOMIAL_DEGREE = 10
DOWNSAMPLING = 2
config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
config.network_architecture.pooling.parameters.downsampling_factors = [DOWNSAMPLING] * 4

coma_args = get_coma_args(config, mesh_dl.dataset)
x = EasyDict(next(iter(mesh_dl)))
mesh_template = mesh_ds.mesh_popu.template # mesh_dm.dataset.template_mesh

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence 
from models.Model4D import EncoderTemporalSequence, AutoencoderTemporalSequence
from lightning.ComaLightningModule import CoMA_Lightning

from models.lightning.EncoderLightningModule import TemporalEncoderLightning
from models.TemporalAggregators import TemporalAggregator, FCN_Aggregator

enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
encoder = Encoder3DMesh(**enc_config)

enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 

h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)

NT = 20 # config.dataset.parameters.T
    
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

# %%
model_weights_aliased = {k.replace("model.", ""):v for k,v in model_weights.items()}

# %%
t_ae.load_state_dict(model_weights_aliased)

# %%
torch.device = "cuda:0"

# %%
t_ae = t_ae.cuda()

# %%
z_hat = []
for i, batch in enumerate(mesh_dl):
    # print(i)
    x = batch["s_t"].cuda()
    z = t_ae(x)[0]['mu'].cpu()
    z_hat.append(z)        

# %%
z_hat = torch.concat(z_hat)

# %%
z_c_list = []
z_s_list = []

for i, batch in enumerate(mesh_dl):
    z_c = batch['z_c']
    z_s = batch['z_s']
    z_c = torch.concat([z.unsqueeze(0) for z in z_c])
    z_s = torch.concat([z.unsqueeze(0) for z in z_s])
    z_c_list.append(z_c)
    z_s_list.append(z_s)
    
z_c = torch.concat(z_c_list, axis=1).T
z_s = torch.concat(z_s_list, axis=1).T

# %%
from sklearn.cross_decomposition import CCA
from ipywidgets import interact
import matplotlib.pyplot as plt

# %%
z = torch.concat([z_c, z_s], axis=1)


# %%
@interact
def scatter_cca(component=widgets.IntSlider(min=0,max=44), which_plot=widgets.Select(options=["correlation", "weights"])):
    
    X, Y = z.detach().numpy(), z_hat.detach().numpy()
    Y = Y[:, component]
    cca = CCA(n_components=1) 
    real_z, pred_z = cca.fit_transform(X, Y)
    
    if which_plot == "correlation":
      plt.scatter(real_z, pred_z);
      plt.xlabel("real z")
      plt.ylabel("best linear combination of predicted z's")
    
    elif which_plot == "weights":
      plt.plot(cca.x_weights_);


# %%
kk[1].shape

# %%
cca.fit_transform

# %%
z_s.shape

# %%
z_hat

# %%
z_hat = torch.concat(z_hat).shape

# %% [markdown]
# ___

# %% [markdown]
# #### Cardiac dataset

# %%
from main_autoencoder_cardiac import *

# %%
POLYNOMIAL_DEGREE = 6
DOWNSAMPLING = 2

config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
config.network_architecture.convolution.parameters.polynomial_degree = [POLYNOMIAL_DEGREE] * 4
config.network_architecture.pooling.parameters.downsampling_factors = [DOWNSAMPLING] * 4

# %%
faces = EasyDict(
        pkl.load(open("utils/VTKHelpers/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl", "rb"))
).new_faces

template = EasyDict({
  "v": np.load(f"{os.environ['CARDIAC_MOTION_REPO']}/data/LV_shape_mean_across_timepoints.npy"),
  "f": faces
})


cardiac_dataset = CardiacMeshPopulationDataset(
    root_path="data/cardio/Results", 
    procrustes_transforms="utils/VTKHelpers/data/procrustes_transforms_FHM_35k.pkl",
    faces=faces,
    template_mesh=template,        
)

mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=32)

# datamodule = get_datamodule(config.dataset, batch_size=config.batch_size)

config.network_architecture.latent_dim_c = 8 
config.network_architecture.latent_dim_s = 16

mesh_dm.setup()
x = EasyDict(next(iter(mesh_dm.train_dataloader())))

mesh_template = mesh_dm.dataset.template_mesh
coma_args = get_coma_args(config)
coma_matrices = get_coma_matrices(config, mesh_template)
coma_args.update(coma_matrices)

from models.Model3D import Encoder3DMesh, Decoder3DMesh
from models.Model4D import DECODER_C_ARGS, DECODER_S_ARGS, ENCODER_ARGS
from models.Model4D import DecoderStyle, DecoderContent, DecoderTemporalSequence 
from models.Model4D import EncoderTemporalSequence, AutoencoderTemporalSequence
from lightning.ComaLightningModule import CoMA_Lightning

from models.lightning.EncoderLightningModule import TemporalEncoderLightning
from models.TemporalAggregators import TemporalAggregator, FCN_Aggregator

enc_config = EasyDict({k: v for k, v in coma_args.items() if k in ENCODER_ARGS})
encoder = Encoder3DMesh(**enc_config)

enc_config.latent_dim = config.network_architecture.latent_dim_c + config.network_architecture.latent_dim_s 

h = encoder.forward_conv_stack(x.s_t, preserve_graph_structure=False)

NT = 50 # config.dataset.parameters.T
    
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

# %%
model_weights = EasyDict({k.replace("model.", ""): v for k, v in model_weights.items()})

# %%
t_ae.load_state_dict(model_weights)

# %%
lit_module = CoMA_Lightning(
    model=t_ae, 
    loss_params=config.loss, 
    optimizer_params=config.optimizer,
    additional_params=config,
    mesh_template=mesh_template
)

# %% [markdown]
# ### Load input meshes

# %%
x = next(iter(mesh_dm.val_dataloader()))

# %%
output = t_ae(x["s_t"])

# %% [markdown]
# ### Generate animation

# %%
s_t, s_hat_t = x["s_t"], output[2]


# %%
# subj_idx_w = widgets.IntSlider(min=1, max=len(cardiac_dataset))

def generate_gif(mesh4D, faces, filename, camera_position='xy', show_edges=False, **kwargs):
        
        '''
        Produces a gif file representing the motion of the input mesh.
        
        params:
          ::mesh4D:: a sequence of Trimesh mesh objects.
          ::faces:: array of F x 3 containing the indices of the mesh's triangular faces.
          ::filename:: the name of the output gif file.
          ::camera_position:: camera position for pyvista plotter (check relevant docs)
          
        return:
          None, only produces the gif file.
        '''

        import pyvista as pv
        
        connectivity = np.c_[np.ones(faces.shape[0]) * 3, faces].astype(int)
                
        pv.set_plot_theme("document")
        os.makedirs(os.path.dirname("./"+filename) , exist_ok=True)
        
        # plotter = pv.Plotter(shape=(1, len(camera_positions)), notebook=False, off_screen=True)
        pv.start_xvfb()
        plotter = pv.Plotter(notebook=False, off_screen=True)
            
        # Open a gif
        plotter.open_gif(filename)

        try:
            # if mesh3D is torch.Tensor, this your should run OK
            mesh4D = mesh4D.cpu().numpy()[0].astype("float32")
        except AttributeError:
            pass

        kk = pv.PolyData(mesh4D[0], connectivity)
        # plotter.add_mesh(kk, smooth_shading=True, opacity=0.5 )#, show_edges=True)
        plotter.add_mesh(kk, show_edges=show_edges, opacity=0.5, color="red") 
        
        for t, _ in tqdm(enumerate(mesh4D)):
            # print(t)
            kk = pv.PolyData(mesh4D[t], connectivity)
            plotter.camera_position = camera_position
            plotter.update_coordinates(kk.points, render=False)
            plotter.render()             
            plotter.write_frame()
        
        plotter.close()
        
        return filename


# %%
from tqdm.notebook import trange, tqdm

# %%
import numpy as np
from PIL import Image
import imageio

def merge_gifs_horizontally(gif_file1, gif_file2, output_file):
    # Create reader object for the gif
    gif1 = imageio.get_reader(gif_file1)
    gif2 = imageio.get_reader(gif_file2)

    # Create writer object
    new_gif = imageio.get_writer(output_file)

    for frame_number in range(gif1.get_length()):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()


# %% [markdown]
# ### Generate gif's

# %%
subj_ids = list(range(64))

# %%
for subj_id in subj_ids:
        
    mesh4D = s_hat_t.detach().cpu().numpy().astype("float32")[subj_id]    
    for camera in ["xz", "xy", "yz"]:    
        gifpath = generate_gif(
            mesh4D,
            faces, 
            camera_position=camera,
            filename=f"id{subj_id}_reconstruction_{camera}.gif"
        )
        
       #b64 = base64.b64encode(
       #  open(gifpath,'rb').read()
       #).decode('ascii')
        
                
    mesh4D = s_t.detach().cpu().numpy().astype("float32")[subj_id]        
    for camera in ["xz", "xy", "yz"]:    
        gifpath = generate_gif(
            mesh4D,
            faces, 
            camera_position=camera,
            filename=f"id{subj_id}_original_{camera}.gif", 
        )
    
        #b64 = base64.b64encode(
        #    open(gifpath,'rb').read()
        #).decode('ascii')
    
    for camera in ["xz", "xy", "yz"]: 
        
        merge_gifs_horizontally(
            f"id{subj_id}_original_{camera}.gif", 
            f"id{subj_id}_reconstruction_{camera}.gif", 
            f"id{subj_id}_{camera}.gif"
        )
        
        os.remove(f"id{subj_id}_original_{camera}.gif")
        os.remove(f"id{subj_id}_reconstruction_{camera}.gif")


# %%
@interact
def show_gif(subj_id=widgets.IntSlider(min=0,max=63)):
    
    from IPython.display import HTML
    import base64
   
    
    
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))

# %%
subj_id = "15"

