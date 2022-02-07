import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
import pickle as pkl
from SyntheticMeshPopulation import SyntheticMeshPopulation

class SyntheticMeshesDataset(Dataset):
    
    '''

    '''

    def  __init__(self, **params):        

        self.mesh_popu = SyntheticMeshPopulation(
            N, T, l_max, freq_max,
            amplitude_static_max, amplitude_dynamic_max,
            mesh_resolution, random_seed
        )

    def __getitem__(self, index):
        return self.moving_meshes.meshes[index]
        
    def __len__(self):
        return len(self.moving_meshes.meshes)
        
    
#TODO: determine whether this new class is necessary
#Probably better to replace the CardiacMeshDM for a more generic class that handles synthetic data as well, maybe RegisteredMeshDM
class SyntheticMeshesDM(pl.LightningDataModule):
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self, 
        **params, 
        # mesh_population: Union[Mapping, CardiacMeshPopulation, None] = None, 
        batch_size: int = 32,
        split_lengths: Union[None, List[int]]=None,
        split_fractions: Union[None, List[float]]=None                 
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

        if self.split_lengths is None:
            if split_fractions is not None:
                self.split_fractions = split_fractions   
            else:
                self.split_fractions = [0.6, 0.2, 0.2]


    def setup(self, stage: Optional[str] = None):

        popu = SyntheticMeshesDataset(**params)

        if self.split_lengths is None:
            train_len = int(self.split_fractions[0] * len(popu))
            val_len = int(self.split_fractions[1] * len(popu))            
            test_len = len(popu) - train_len - val_len
            
            self.split_lengths = [train_len, val_len, test_len]

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=2, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=8)
