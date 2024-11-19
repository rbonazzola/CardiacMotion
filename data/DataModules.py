import torch
import os, sys
import numpy as np
import re
import glob
from utils.CardioMesh.CardiacMesh import CardiacMeshPopulation, Cardiac3DMesh
from torch.utils.data import TensorDataset, DataLoader, random_split
from typing import * # Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from copy import copy
import pickle as pkl
import pytorch_lightning as pl

from utils.CardioMesh.CardiacMesh import transform_mesh
from torch import Tensor

from easydict import EasyDict


def mse(s1, s2):
    return ((s1-s2)**2).sum(-1).mean(-1)


class CardiacMeshPopulationDataset(TensorDataset):
    
    def __init__(
        self, 
        root_path: str, 
        faces: Union[np.ndarray, str],
        N_subj: Union[int, None] = None,
        procrustes_transforms: Union[str, None] = None,
        subsetting_matrix: Union[np.ndarray, None] = None,        
        template_mesh = None,
        phases_filter = None,
        static_shape: Literal["end_diastole", "temporal_mean", "end_systole"] = "end_diastole"
        ):
        
        '''
          root_dir: root directory where all the mesh data is located.
          N_subj: if None, all the subjects are utilized.
          faces: F x 3 array (F is the number of faces)
          procrustes_transforms: Mapping from IDs to transforms ("rotation" and "traslation")
        '''
        
        self._root_path = root_path
        self._paths = self._get_paths(N=N_subj, phases_filter=phases_filter)
        
        self.procrustes_transforms = pkl.load(open(procrustes_transforms, "rb"))
        
        self.ids = set(self._get_ids()).intersection(set(self.procrustes_transforms.keys()))
        self.ids = list(self.ids)
        
        self._paths = { k: v for k, v in self._paths.items() if k in self.ids}
        
        self.subsetting_matrix = subsetting_matrix
        self.faces = faces
        self.template_mesh = template_mesh
        self.static_shape = static_shape
        
    
    def _get_paths(self, N=None, phases_filter=None):
            
        ids = sorted(os.listdir(self._root_path))
        
        if N is not None:
            ids = ids[:N]
        # regex = re.compile(f"{self._root_path}/.*/models/LV__5220_vertices__time0(\d\d)(_interpolated)?.npy")
        regex = re.compile(f"{self._root_path}/.*/models/FHM_res_0.1_time0(\d\d).npy")
        
        dd = {}
        
        for id in ids:
            
            # paths = sorted(glob.glob(f"{self._root_path}/{id}/models/LV*.npy"))    
            paths = sorted(glob.glob(f"{self._root_path}/{id}/models/FHM*.npy"))    
            # print(paths)
            
            paths_filtered, phases = [], []
            
            for path in paths:
                
                if regex.match(path) is not None:
                    phase = regex.match(path).group(1)
                    if (phases_filter is None) or (int(phase) in phases_filter):
                        phases.append(phase)          
                        paths_filtered.append(path)
                        
            phases = [ int(phase) for phase in phases ]
                        
            if ((phases_filter is None) and len(phases) == 50) or ((phases_filter is not None) and len(phases) == len(phases_filter)):
                dd[id] = paths_filtered
            
        return dd            
       
        
    def _get_ids(self):  
        
        return list(self._paths.keys())
    
    
    def __getitem__(self, idx):
        
        id = self.ids[idx]
        
        procrustes_transforms = self.procrustes_transforms[id]
        
        s_t = []
        for p in self._paths[id]:
            
            try:
                s = np.load(p, allow_pickle=True)
            except ValueError as e:
                
                continue
            
            if self.subsetting_matrix is not None:
                s = self.subsetting_matrix * s

            s = transform_mesh(s, **procrustes_transforms) 
            s_t.append(s)
              
        s_t = Tensor(np.array(s_t))
        
        if self.static_shape == "end_diastole":
            s_t_avg = s_t[0]
        elif self.static_shape == "temporal_mean":
            s_t_avg = s_t.mean(axis=0)
        else:
            raise NotImplemented
        
        dev_from_tmp_avg = np.array([ mse(s_t[j], s_t_avg) for j, _ in enumerate(s_t) ])
        dev_from_tmp_avg = Tensor(dev_from_tmp_avg)
        
        if self.template_mesh is not None:
            dev_from_sphere = np.array([ mse(s_t[j], self.template_mesh.v) for j, _ in enumerate(s_t) ])
            dev_from_sphere = Tensor(dev_from_sphere)
        else:
            dev_from_sphere = None
        
        dd = {
          "s_t": s_t,
          "time_avg_s": s_t_avg,
          "d_content": dev_from_tmp_avg,  
          "d_style": dev_from_sphere
        }
        
        return EasyDict(dd)
    
    
    def __len__(self):
        
        return len(self.ids)


class CardiacMeshPopulationDM(pl.LightningDataModule):    
    
    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''
    
    def __init__(self, 
        dataset: TensorDataset,
        batch_size: int = 16,
        split_lengths: Union[None, List[int]]=None,
        num_workers=3
    ):
    
        '''
        params:
            dataset:
            batch_size:
            split_lengths:
            num_workers:
        '''
        
        super().__init__()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.split_lengths = self._get_split_lengths(split_lengths)
        self.num_workers=num_workers
        
        
    def _get_split_lengths(self, split_lengths):
        
        _split_lengths = copy(split_lengths)        
        
        if _split_lengths is None:
            train_len = int(0.6 * len(self.dataset))
            test_len = int(0.2 * len(self.dataset))
            val_len = len(self.dataset) - train_len - test_len
            self.split_lengths = [train_len, val_len, test_len]
        
        elif all([l >= 1 for l in _split_lengths]):                        
            try:
                train_len = _split_lengths[0]
                test_len = _split_lengths[1]
                val_len = _split_lengths[2]
            except IndexError:
                raise IndexError(f"split_lengths should have length three, instead is {_split_lengths}")

        elif all([l < 1 for l in _split_lengths]):
            train_len = int(_split_lengths[0] * len(self.dataset))
            test_len = int(_split_lengths[1] * len(self.dataset))
            if len(_split_lengths) == 2:
                val_len = len(self.dataset) - train_len - test_len
            elif len(_split_lengths) == 3:            
                val_len = int(_split_lengths[2] * len(self.dataset))
            else:
                raise ValueError("Bad values for split lengths. Expecting 2 or 3 fractions/integers.")
                
        return [train_len, val_len, test_len] #, predict_len] 
    
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)
        

    def setup(self, stage: Optional[str] = None):

        # popu = CardiacMeshPopulationDataset(
        #     root_dir=self.data_dir, 
        #     cardiac_population=self.cardiac_population
        # )
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split_lengths)
        
        # indices = list(range(sum(self.split_lengths)))

        # self.train_indices = indices[:self.split_lengths[0]]
        # self.val_indices = indices[self.split_lengths[0]: self.split_lengths[0]+self.split_lengths[1]]
        # self.test_indices = indices[self.split_lengths[0]+self.split_lengths[1]:self.split_lengths[0]+self.split_lengths[1]+self.split_lengths[2]]        
 
    def train_dataloader(self):
        # return DataLoader(self.train_dataset, sampler=SubsetRandomSampler(self.train_indices), batch_size=self.batch_size, num_workers=self.num_workers)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # return DataLoader(self.train_dataset, sampler=SubsetRandomSampler(self.val_indices), batch_size=self.batch_size, num_workers=self.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # return DataLoader(self.train_dataset, sampler=SubsetRandomSampler(self.test_indices), batch_size=self.batch_size, num_workers=self.num_workers)
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
