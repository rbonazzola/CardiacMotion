import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
import pickle as pkl
from synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation
from PIL import Image

from IPython import embed

NUM_WORKERS = 4

def mse(s1, s2):
    return ((s1-s2)**2).sum(-1).mean(-1)

class SyntheticMeshesDataset(Dataset):
    
    '''
      PyTorch dataset representing a population of synthetic 3D moving meshes
    '''

    def  __init__(self, data_params, preprocessing_params):
        '''
        params:
          params (dict): parameters used by SyntheticMeshPopulation to build the population of meshes
          center_around_mean (boolean): whether to subtract the coordinates of the reference shape at each node  
        '''

        self.mesh_popu = SyntheticMeshPopulation(**data_params)                
        self.preprocessing_params = preprocessing_params

    def __getitem__(self, index):
        
        ''' 
        return:
          the set of moving meshes for an individual, 
          the time-averaged mesh, 
          the MSE of the moving meshes with respect to the time-averaged mesh,
          the MSE of the moving meshes with respect to the reference mesh (a unit sphere)
        '''

        ref_shape = np.array(self.mesh_popu.template.vertices)
        
        moving_meshes = self.mesh_popu.moving_meshes[index]
        time_avg_mesh = self.mesh_popu.time_avg_meshes[index]
        dev_from_tmp_avg = np.array([ mse(moving_meshes[j], time_avg_mesh) for j, _ in enumerate(moving_meshes) ])
        dev_from_sphere = np.array([ mse(moving_meshes[j], ref_shape) for j, _ in enumerate(moving_meshes) ])
        z = self.mesh_popu.__dict__.get("coefficients", None)

        try:
            z_c = list(z[index][0].values())
            z_s = list(z[index][1]['sin'].values()) + list(z[index][1]['cos'].values())
        except:
            pass

        if self.preprocessing_params.center_around_mean:
            for j, _ in enumerate(moving_meshes): 
                moving_meshes[j] -= ref_shape
            time_avg_mesh -= ref_shape

        dd = {
            "s_t": moving_meshes,
            "s_avg": time_avg_mesh,
            "d_content": dev_from_tmp_avg,
            "d_style": dev_from_sphere,
            "z_c": z_c,
            "z_s": z_s
        }

        return dd
    

    def __len__(self):
        return len(self.mesh_popu.moving_meshes)
        
    
#TODO: determine whether this new class is necessary
#Probably better to replace the CardiacMeshDM for a more generic class that
#handles synthetic data as well, maybe called RegisteredMeshDM

class SyntheticMeshesDM(pl.LightningDataModule):
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self,
        data_params,
        preprocessing_params,
        batch_size=32,
        split_lengths: Union[None, List[int]]=None,
        split_fractions: Union[None, List[float]]=None,

    ):

        '''
        params:
            pkl_file: path to pickle file containing a dictionary with keys "population_meshes" and "coefficients"            
            batch_size: batch size for training.
            split_lengths: number of samples for training, validation and testing (in that order). 
                           The last (# sample in testing) is not used and is computed as the difference.
            split_fractions: same as split_lengths but with fractions instead of number of samples. 
                             The previous argument takes precedence over this if both are provided.                             
        '''
        
        super().__init__()                
        
        self.batch_size = batch_size        
        self.split_lengths = split_lengths
        self.data_params = data_params
        self.preprocessing_params = preprocessing_params

        if self.split_lengths is None:
            if split_fractions is not None:
                self.split_fractions = split_fractions 
            else:
                self.split_fractions = [0.7, 0.2, 0.1]


    def setup(self, stage: Optional[str] = None):

        popu = SyntheticMeshesDataset(self.data_params, self.preprocessing_params)

        if self.split_lengths is None:
            train_len = int(self.split_fractions[0] * len(popu))
            val_len = int(self.split_fractions[1] * len(popu))            
            test_len = len(popu) - train_len - val_len
            self.split_lengths = [train_len, val_len, test_len]

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=NUM_WORKERS)

    def predict_dataloader(self):
        predict_len = 16 if len(self.test_dataset) >= 16 else len(self.test_dataset)
        predict_dataset = torch.utils.data.Subset(self.test_dataset, range(predict_len))
        return DataLoader(predict_dataset, batch_size=16, num_workers=NUM_WORKERS)
