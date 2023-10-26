import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from mlflow.tracking import MlflowClient
from torch import Tensor
from tqdm import tqdm

from config.cli_args import CLI_args, overwrite_config_items
from config.load_config import load_yaml_config, to_dict, flatten_dict, rsetattr, rgetattr

from utils.helpers import *
from utils.mlflow_helpers import get_mlflow_parameters, get_mlflow_dataset_params
from IPython import embed

import os
import argparse
import pprint

from utils.CardioMesh.CardiacMesh import Cardiac3DMesh

os.environ["CARDIAC_MOTION_REPO"] = f"{os.environ['HOME']}/01_repos/CardiacMotion"
# repo_dir = os.environ.get("CARDIAC_MOTION_REPO", "kk")
repo_dir = os.environ["CARDIAC_MOTION_REPO"]
os.chdir(repo_dir)

import sys
import glob
import re
import pickle as pkl
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler
from typing import Union, List, Optional

from utils.CardioMesh.CardiacMesh import transform_mesh

from easydict import EasyDict
import random

# from typing import Namespace

partitions = {
  "left_atrium" : ("LA", "MVP", "PV1", "PV2", "PV3", "PV4", "PV5"),
  "right_atrium" : ("RA", "TVP", "PV6", "PV7"),
  "left_ventricle" : ("LV", "AVP", "MVP"),
  "right_ventricle" : ("RV", "PVP", "TVP"),
  "biventricle" : ("LV", "AVP", "MVP", "RV", "PVP", "TVP"),
  "aorta" : ("aorta",)
}

# class CardiacMeshPopulationDataset(TensorDataset):
#     
#     def __init__(
#         self, 
#         root_path: str, 
#         faces: Union[np.array, str],
#         N_subj: Union[int, None] = None,
#         procrustes_transforms: Union[str, None] = None,
#         subsetting_matrix: Union[np.array, None] = None,
#         template_mesh = None,
#         phases_filter = None
#         ):
#         
#         '''
#           root_dir: root directory where all the mesh data is located.
#           N_subj: if None, all the subjects are utilized.
#           faces: F x 3 array (F is the number of faces)
#           procrustes_transforms: Mapping from IDs to transforms ("rotation" and "traslation")
#         '''
#         
#         self._root_path = root_path
#         self._paths = self._get_paths(N=N_subj, phases_filter=phases_filter)
#         
#         self.procrustes_transforms = pkl.load(open(procrustes_transforms, "rb"))
#         
#         self.ids = set(self._get_ids()).intersection(set(self.procrustes_transforms.keys()))
#         self.ids = list(self.ids)
#         
#         self._paths = { k: v for k, v in self._paths.items() if k in self.ids}
#         
#         self.subsetting_matrix = subsetting_matrix
#         self.faces = faces
#         self.template_mesh = template_mesh
#         
#     
#     def _get_paths(self, N=None, phases_filter=None):
#             
#         ids = sorted(os.listdir(self._root_path))
#         
#         if N is not None:
#             ids = ids[:N]
#         # regex = re.compile(f"{self._root_path}/.*/models/LV__5220_vertices__time0(\d\d)(_interpolated)?.npy")
#         regex = re.compile(f"{self._root_path}/.*/models/FHM_res_0.1_time0(\d\d).npy")
#         
#         dd = {}
#         
#         for id in tqdm(ids):
#             
#             # paths = sorted(glob.glob(f"{self._root_path}/{id}/models/LV*.npy"))    
#             paths = sorted(glob.glob(f"{self._root_path}/{id}/models/FHM*.npy"))    
#             # print(paths)
#             
#             paths_filtered, phases = [], []
#             
#             for path in paths:
#                 
#                 if regex.match(path) is not None:
#                     phase = regex.match(path).group(1)
#                     if (phases_filter is None) or (int(phase) in phases_filter):
#                         phases.append(phase)          
#                         paths_filtered.append(path)
#                         
#             phases = [ int(phase) for phase in phases ]
#             
#             if ((phases_filter is None) and len(phases) == 50) or ((phases_filter is not None) and len(phases) == len(phases_filter)):
#                 dd[id] = paths_filtered
#         
#         return dd            
#        
#         
#     def _get_ids(self):  
#         
#         return list(self._paths.keys())
#     
#     
#     def __getitem__(self, idx):
#         
#         id = self.ids[idx]
#         
#         procrustes_transforms = self.procrustes_transforms[id]
#         
#         s_t = []
#         for p in self._paths[id]:
#             s = np.load(p, allow_pickle=True)
#             if self.subsetting_matrix is not None:
#                 s = self.subsetting_matrix * s
#             s = transform_mesh(s, **procrustes_transforms) 
#             s_t.append(s)
#               
#         s_t = Tensor(np.array(s_t))
#         
#         s_t_avg = s_t.mean(axis=0)
#         
#         dev_from_tmp_avg = np.array([ mse(s_t[j], s_t_avg) for j, _ in enumerate(s_t) ])
#         dev_from_tmp_avg = Tensor(dev_from_tmp_avg)
#         
#         if self.template_mesh is not None:
#             dev_from_sphere = np.array([ mse(s_t[j], self.template_mesh.v) for j, _ in enumerate(s_t) ])
#             dev_from_sphere = Tensor(dev_from_sphere)
#         else:
#             dev_from_sphere = None
#         
#         dd = {
#           "s_t": s_t,
#           "time_avg_s": s_t_avg,
#           "d_content": dev_from_tmp_avg,  
#           "d_style": dev_from_sphere
#         }
#         
#         return EasyDict(dd)
#     
#     
#     def __len__(self):
#         
#         return len(self.ids)


##########################################################################################

from data.DataModules import CardiacMeshPopulationDM, CardiacMeshPopulationDataset
    
# class CardiacMeshPopulationDM(pl.LightningDataModule):    
#     
#     '''
#     PyTorch datamodule wrapping the CardiacMeshPopulation class
#     '''
#     
#     def __init__(self, 
#         dataset: TensorDataset,
#         batch_size: int = 16,
#         split_lengths: Union[None, List[int]]=None,
#         num_workers=8                 
#     ):
# 
#         '''
#         params:
#             data_dir:
#             batch_size:
#             split_lengths:
#         '''
#         
#         super().__init__()
#                 
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.split_lengths = None
#         self.num_workers=num_workers
# 
# 
#     def setup(self, stage: Optional[str] = None):
# 
#         indices = list(range(len(self.dataset)))
#         random.shuffle(indices)
# 
# 
#         if self.split_lengths is None:
#             
#             train_len = int(0.1 * len(self.dataset))
#             test_len = int(0.02 * len(self.dataset))
#             val_len = int(0.02 * len(self.dataset))            
#             # val_len = len(self.dataset) - train_len - test_len
#             self.split_lengths = [train_len, val_len, test_len]
#         elif self.split_lengths[0] < 1:
#             logger.info("")
# 
#         self.train_indices = indices[:self.split_lengths[0]]
#         self.val_indices = indices[self.split_lengths[0]:self.split_lengths[0]+self.split_lengths[1]]
#         self.test_indices = indices[self.split_lengths[0]+self.split_lengths[1]:self.split_lengths[0]+self.split_lengths[1]++self.split_lengths[2]]
# 
#         #  self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split_lengths)        
# 
# 
#     def train_dataloader(self):
#         return DataLoader(self.dataset, sampler=SubsetRandomSampler(self.train_indices), batch_size=self.batch_size, num_workers=self.num_workers)
# 
#     def val_dataloader(self):
#         return DataLoader(self.dataset, sampler=SubsetRandomSampler(self.val_indices), batch_size=self.batch_size, num_workers=self.num_workers)
# 
#     def test_dataloader(self):
#         return DataLoader(self.dataset, sampler=SubsetRandomSampler(self.test_indices), batch_size=1, num_workers=self.num_workers)
    
    
    
def mlflow_startup(mlflow_config):
    
    '''
      Starts MLflow run, using
      
      mlflow_config: Namespace including run_id, experiment_name, run_name, artifact_location            
    
    '''
    
    mlflow.pytorch.autolog(log_models=True)
 
    if mlflow_config.tracking_uri is not None:
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    
    try:
        exp_id = mlflow.create_experiment(mlflow_config.experiment_name, artifact_location=mlflow_config.artifact_location)
    except:
      # If the experiment already exists, we can just retrieve its ID
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        print(experiment)
        exp_id = experiment.experiment_id
    run_info = {
        "run_id": trainer.logger.run_id,
        "experiment_id": exp_id,
        "run_name": mlflow_config.run_name,
        #"tags": config.additional_mlflow_tags
    }
    
    mlflow.start_run(**run_info)
    
        
def mlflow_log_additional_params(config):
    
    '''
    Log additional parameters, extracted from config
    '''
        
    mlflow_params = get_mlflow_parameters(config)
    mlflow_dataset_params = get_mlflow_dataset_params(config)
    mlflow_params.update(mlflow_dataset_params)
    mlflow.log_params(mlflow_params)    
        
        
def log_computational_graph(model, x):
    
    from torchviz import make_dot
    yhat = model(x)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("comp_graph_network", format="png")
    mlflow.log_figure("comp_graph_network.png")
        
        
def get_coma_args(config: Mapping):

    '''
      Arguments:
        - config: Namespace with all the necessary configuraton.
        - mesh_dataset: torch Dataset with attributes mesh_popu.template with a template Mesh.
    '''

    net = config.network_architecture

    convs = net.convolution
    coma_args = {
        "num_features": net.n_features,
        "n_layers": len(convs.channels_enc),  # REDUNDANT
        "num_conv_filters_enc": convs.channels_enc,
        "num_conv_filters_dec_c": convs.channels_dec_c,
        "num_conv_filters_dec_s": convs.channels_dec_s,
        "cheb_polynomial_order": convs.parameters.polynomial_degree,
        "latent_dim_content": net.latent_dim_c,
        "latent_dim_style": net.latent_dim_s,
        "is_variational": config.loss.regularization.weight != 0,
        "mode": "testing",
        "n_timeframes": config.dataset.parameters.T,
        "phase_input": net.phase_input,
        "z_aggr_function": net.z_aggr_function
    }

    return EasyDict(coma_args)


def get_coma_matrices(config, template, partition, from_cached=True, cache=True):
    
    
    '''
    :param downsample_factors: list of downsampling factors, e.g. [2, 2, 2, 2]
    :param template: a Namespace with attributes corresponding to vertices and faces
    :param cache: if True, will cache the matrices in a pkl file, unless this file already exists.
    :param from_cached: if True, will try to fetch the matrices from a previously cached pkl file.
    :return: a dictionary with keys "downsample_matrices", "upsample_matrices", "adjacency_matrices" and "n_nodes",
    where the first three elements are lists of matrices and the last is a list of integers.
    '''

    # mesh_popu = dm.train_dataset.dataset.mesh_popu
    downsample_factors = config.network_architecture.pooling.parameters.downsampling_factors

    matrices_hash = hash((
        hash("1000215"), 
        hash(tuple(downsample_factors)), 
        hash(partition)
    )) % 1000000

    cached_file = f"data/cached/matrices/{matrices_hash}.pkl"

    if from_cached and os.path.exists(cached_file):
        A_t, D_t, U_t, n_nodes = pkl.load(open(cached_file, "rb"))
    else:
        template_mesh = Mesh(template.v, template.f)
        M, A, D, U = mesh_operations.generate_transform_matrices(
            template_mesh, downsample_factors,
        )
        n_nodes = [len(M[i].v) for i in range(len(M))]
        A_t, D_t, U_t = ([scipy_to_torch_sparse(x).float() for x in X] for X in (A, D, U))
        if cache:
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            with open(cached_file, "wb") as ff:
                pkl.dump((A_t, D_t, U_t, n_nodes), ff)

    return {
        "downsample_matrices": D_t,
        "upsample_matrices": U_t,
        "adjacency_matrices": A_t,
        "n_nodes": n_nodes,
        "template": template
    }


###
def main(model, datamodule, trainer, mlflow_config=None):

    '''
      config (Namespace):       
      trainer_args (Namespace):
      mlflow_config (Namespace):
      
      Example:
      
    '''
            
    if mlflow_config:
        mlflow_config.run_id = trainer.logger.run_id
        mlflow_startup(mlflow_config)             
        mlflow_log_additional_params(config)
    
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path='best') # Generates metrics for the full test dataset
    # trainer.predict(ckpt_path='best', datamodule=datamodule) # Generates figures for a few samples

    mlflow.end_run()

##########################################################################################
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pytorch Trainer for Convolutional Mesh Autoencoders",
        argument_default=argparse.SUPPRESS
    )

    my_args = parser.add_argument_group("model")

    #to avoid a little bit of boilerplate
    for k, v in CLI_args.items():
        my_args.add_argument(*k, **v)
        
    my_args.add_argument("--partition", type=str, default="left_ventricle")
        
    # adding arguments specific to the PyTorch Lightning trainer.
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    ### Load configuration
    if not os.path.exists(args.yaml_config_file):
        logger.error("Config not found" + args.yaml_config_file)

    ref_config = load_yaml_config(args.yaml_config_file)

    try:
        config_to_replace = args.config
        config = overwrite_config_items(ref_config, config_to_replace)
    except AttributeError:
        # If there are no elements to replace
        config = ref_config
        pass

    # https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
    arg_groups = {}
    
    for group in parser._action_groups:
        group_dict = { a.dest: rgetattr(args, a.dest, None) for a in group._group_actions }
        arg_groups[group.title] = EasyDict(group_dict)
    
    trainer_args = arg_groups["pl.Trainer"]
        
    config.log_computational_graph = args.log_computational_graph
    if args.disable_mlflow_logging:
        config.mlflow = None

    print(config.mlflow)

    if config.mlflow:

        if config.mlflow.experiment_name is None:
            config.mlflow.experiment_name = f"{self.partition}"

        exp_info = {
            "experiment_name": config.mlflow.experiment_name,
            "artifact_location": config.mlflow.artifact_location
        }

        trainer_args.logger = MLFlowLogger(
            tracking_uri=config.mlflow.tracking_uri,
            **exp_info,
        )

        mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    else:
        trainer_args.logger = None

    # config.dataset.parameters.N = 5120   
    # params = { 
    #   "N": 100, "T": 20, "mesh_resolution": 10,
    #   "l_max": 2, "freq_max": 2, 
    #   "amplitude_static_max": 0.3, "amplitude_dynamic_max": 0.1, 
    #   "random_seed": 144
    # }
    
    # preproc_params = EasyDict({"center_around_mean": False})
    
    # mesh_ds = SyntheticMeshesDataset(config.dataset.parameters, config.dataset.preprocessing)
    # mesh_dm = SyntheticMeshesDM(mesh_ds)
    
    FACES_FILE = "utils/CardioMesh/data/faces_and_downsampling_mtx_frac_0.1_LV.pkl"
    # MEAN_ACROSS_CYCLE_FILE = f"{os.environ['CARDIAC_MOTION_REPO']}/data/LV_shape_mean_across_timepoints.npy"
    
    PARTITION = args.partition # "left_ventricle"
    MEAN_ACROSS_CYCLE_FILE = f"utils/CardioMesh/data/cached/mean_shape_time_avg__{PARTITION}.npy"
    # f"{os.environ['CARDIAC_MOTION_REPO']}/data/mean_shape_across_timepoints__{PARTITION}.npy"
    # PROCRUSTES_FILE = "utils/CardioMesh/data/procrustes_transforms_FHM_35k.pkl"
    PROCRUSTES_FILE = f"utils/CardioMesh/data/cached/procrustes_transforms_{PARTITION}.pkl"    
    # "utils/CardioMesh/data/procrustes_transforms_LVED_all_Rodrigos_segmentation_60728_subjects.pkl"
    SUBSETTING_MATRIX_FILE = f"/home/user/01_repos/CardioMesh/data/cached/subsetting_matrix_{PARTITION}.pkl" 
    
    subsetting_matrix = pkl.load(open(SUBSETTING_MATRIX_FILE, "rb"))
    
    # faces = EasyDict(
    #     pkl.load(open(FACES_FILE, "rb"))
    # ).new_faces
    
    ID = "1000511"
    fhm_mesh = Cardiac3DMesh(
       filename=f"/mnt/data/workshop/workshop-user1/datasets/meshes/Results_Yan/{ID}/models/FHM_res_0.1_time001.npy",
       faces_filename="/home/user/01_repos/CardioMesh/data/faces_fhm_10pct_decimation.csv",
       subpart_id_filename="/home/user/01_repos/CardioMesh/data/subpartIDs_FHM_10pct.txt"
    )

    mean_shape = np.load(MEAN_ACROSS_CYCLE_FILE)
    faces = fhm_mesh[partitions[PARTITION]].f
    
    template = EasyDict({
      "v": mean_shape,
      "f": faces
    })

    # cardiac_dataset = CardiacMeshPopulationDataset(
    #     root_path="data/cardio/Results", 
    #     procrustes_transforms=PROCRUSTES_FILE,
    #     faces=faces,
    #     template_mesh=template,        
    # )
    
    NT = 10 # config.dataset.parameters.T
    cardiac_dataset = CardiacMeshPopulationDataset(
        root_path="data/cardio/Results", 
        procrustes_transforms=PROCRUSTES_FILE,
        faces=faces,
        subsetting_matrix=subsetting_matrix,
        template_mesh= EasyDict({
            "v": mean_shape,
            "f": faces
        }),
        N_subj=10000,
        phases_filter=1+(50/NT)*np.array(range(NT))
    )
    
    mesh_dm = CardiacMeshPopulationDM(cardiac_dataset, batch_size=8)
    
    # datamodule = get_datamodule(config.dataset, batch_size=config.batch_size)
    
    config.network_architecture.latent_dim_c = 8 
    config.network_architecture.latent_dim_s = 8 
   
    mesh_dm.setup()
    x = EasyDict(next(iter(mesh_dm.train_dataloader())))
    
    mesh_template = mesh_dm.dataset.template_mesh
    coma_args = get_coma_args(config)
    coma_matrices = get_coma_matrices(config, mesh_template, PARTITION)
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
   
    lit_module = CoMA_Lightning(
        model=t_ae, 
        loss_params=config.loss, 
        optimizer_params=config.optimizer,
        additional_params=config,
        mesh_template=mesh_template
    )
        
    trainer = get_lightning_trainer(trainer_args)

    if args.show_config or args.dry_run:
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        pp.pprint(to_dict(config))
        if args.dry_run:
            exit()
    
    main(lit_module, mesh_dm, trainer, config.mlflow)
