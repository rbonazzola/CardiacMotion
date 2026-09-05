import os, sys
import logging
import time
import numpy as np
import re
import glob
import pickle as pkl
from easydict import EasyDict
from typing import Optional, List, Union, Literal
from copy import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as pl

from cardio_mesh import (
    CardiacMeshPopulation, 
    Cardiac3DMesh,    
)

from cardio_mesh.procrustes import transform_mesh

logger = logging.getLogger(__name__)

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
        static_shape: Literal["end_diastole", "temporal_mean", "end_systole"] = "end_diastole",
        center_around_mean: bool = False,
        ):

        '''
          root_dir: root directory where all the mesh data is located.
          N_subj: if None, all the subjects are utilized.
          faces: F x 3 array (F is the number of faces)
          procrustes_transforms: Mapping from IDs to transforms ("rotation" and "traslation")
          center_around_mean: if True, subtract the population mean shape (template_mesh.v)
            from all meshes. Reduces data scale to small residuals, which stabilizes training.
        '''

        if center_around_mean and template_mesh is None:
            raise ValueError("center_around_mean=True requires template_mesh to be provided.")

        start = time.perf_counter()
        self._root_path = root_path
        logger.info(
            "Indexing cardiac meshes: root=%s, N_subj=%s, phases_filter=%s",
            root_path,
            N_subj,
            phases_filter,
        )
        self._paths = self._get_paths(N=N_subj, phases_filter=phases_filter)

        logger.info("Loading Procrustes transforms from %s", procrustes_transforms)
        self.procrustes_transforms = pkl.load(open(procrustes_transforms, "rb"))
        logger.info("Loaded %d Procrustes transforms", len(self.procrustes_transforms))

        self.ids = set(self._get_ids()).intersection(set(self.procrustes_transforms.keys()))
        self.ids = list(self.ids)

        self._paths = { k: v for k, v in self._paths.items() if k in self.ids}
        self.ids = sorted(self._paths.keys())

        self.subsetting_matrix = subsetting_matrix
        self.faces = faces
        self.template_mesh = template_mesh
        self.static_shape = static_shape
        self.center_around_mean = center_around_mean
        n_frames = len(next(iter(self._paths.values()))) if self._paths else 0
        logger.info(
            "Cardiac dataset indexed: subjects=%d, frames_per_subject=%d, center_around_mean=%s, elapsed=%.2fs",
            len(self.ids),
            n_frames,
            self.center_around_mean,
            time.perf_counter() - start,
        )
        
    
    def _get_paths(self, N=None, phases_filter=None):
            
        ids = sorted(os.listdir(self._root_path))
        n_available_ids = len(ids)
        
        if N is not None:
            ids = ids[:N]
        # regex = re.compile(f"{self._root_path}/.*/models/LV__5220_vertices__time0(\d\d)(_interpolated)?.npy")
        regex = re.compile(f"{self._root_path}/.*/models/FHM_res_0.1_time0(\d\d).npy")
        
        dd = {}
        
        logger.info("Scanning %d/%d subject folders for complete mesh time series", len(ids), n_available_ids)
        skipped_incomplete = 0
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
            else:
                skipped_incomplete += 1
            
        logger.info(
            "Mesh scan complete: usable_subjects=%d, skipped_incomplete=%d",
            len(dd),
            skipped_incomplete,
        )
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
                logger.warning("Could not load mesh file %s: %s", p, e)
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
            raise NotImplementedError
        
        dev_from_tmp_avg = mse(s_t, s_t_avg.unsqueeze(0))

        if self.template_mesh is not None:
            template_v = Tensor(self.template_mesh.v).unsqueeze(0)
            dev_from_sphere = mse(s_t, template_v)
        else:
            dev_from_sphere = None

        if self.center_around_mean:
            # Shift meshes to residuals w.r.t. population mean.
            # d_content and d_style are invariant to this shift so they stay as-is.
            mean_v = Tensor(self.template_mesh.v)
            s_t     = s_t     - mean_v
            s_t_avg = s_t_avg - mean_v

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
        logger.info(
            "CardiacMeshPopulationDM initialized: dataset=%d, batch_size=%d, split_lengths=%s, num_workers=%d",
            len(self.dataset),
            self.batch_size,
            self.split_lengths,
            self.num_workers,
        )
        
        
    def _get_split_lengths(self, split_lengths):
        
        _split_lengths = copy(split_lengths)        
        
        if _split_lengths is None:
            train_len = int(0.6 * len(self.dataset))
            test_len = int(0.2 * len(self.dataset))
            val_len = len(self.dataset) - train_len - test_len
        
        elif all([l >= 1 for l in _split_lengths]):
            try:
                train_len = _split_lengths[0]
                val_len   = _split_lengths[1]
                test_len  = _split_lengths[2]
            except IndexError:
                raise IndexError(f"split_lengths should have length three, instead is {_split_lengths}")

        elif all([l < 1 for l in _split_lengths]):
            train_len = int(_split_lengths[0] * len(self.dataset))
            val_len   = int(_split_lengths[1] * len(self.dataset))
            if len(_split_lengths) == 2:
                test_len = len(self.dataset) - train_len - val_len
            elif len(_split_lengths) == 3:
                test_len = int(_split_lengths[2] * len(self.dataset))
            else:
                raise ValueError("Bad values for split lengths. Expecting 2 or 3 fractions/integers.")
                
        return [train_len, val_len, test_len] #, predict_len] 
    
        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(popu, self.split_lengths)
        

    def setup(self, stage: Optional[str] = None):

        # popu = CardiacMeshPopulationDataset(
        #     root_dir=self.data_dir, 
        #     cardiac_population=self.cardiac_population
        # )
        
        start = time.perf_counter()
        logger.info("Splitting dataset: lengths=%s", self.split_lengths)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.split_lengths)
        logger.info(
            "Dataset split complete: train=%d, val=%d, test=%d, elapsed=%.2fs",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
            time.perf_counter() - start,
        )
        
        # indices = list(range(sum(self.split_lengths)))

        # self.train_indices = indices[:self.split_lengths[0]]
        # self.val_indices = indices[self.split_lengths[0]: self.split_lengths[0]+self.split_lengths[1]]
        # self.test_indices = indices[self.split_lengths[0]+self.split_lengths[1]:self.split_lengths[0]+self.split_lengths[1]+self.split_lengths[2]]        
 
    def train_dataloader(self):
        logger.info("Creating train dataloader: batch_size=%d, num_workers=%d", self.batch_size, self.num_workers)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        logger.info("Creating val dataloader: batch_size=%d, num_workers=%d", self.batch_size, self.num_workers)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        logger.info("Creating test dataloader: batch_size=%d, num_workers=%d", self.batch_size, self.num_workers)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=torch.cuda.is_available())


class GenericDataModule(pl.LightningDataModule):
    """
    A generic PyTorch Lightning DataModule for handling datasets.

    Args:
        dataset: A PyTorch dataset (e.g., TensorDataset).
        batch_size: Batch size for dataloaders.
        split_lengths: A list of lengths or fractions for splitting the dataset into train, val, and test sets.
                      If None, defaults to [0.6, 0.2, 0.2].
        num_workers: Number of workers for dataloaders.
        shuffle_train: Whether to shuffle the training dataloader.
        shuffle_val: Whether to shuffle the validation dataloader.
        shuffle_test: Whether to shuffle the test dataloader.
    """

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 16,
        split_lengths: Optional[Union[List[int], List[float]]] = None,
        num_workers: int = 3,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
    ):
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.split_lengths = self._get_split_lengths(split_lengths)
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test

    def _get_split_lengths(self, split_lengths: Optional[Union[List[int], List[float]]]) -> List[int]:
        """
        Validates and computes the lengths for splitting the dataset.

        Args:
            split_lengths: A list of lengths or fractions for splitting the dataset.

        Returns:
            A list of integers representing the lengths of the train, val, and test sets.
        """
        if split_lengths is None:
            # Default split: 60% train, 20% val, 20% test
            train_len = int(0.6 * len(self.dataset))
            test_len = int(0.2 * len(self.dataset))
            val_len = len(self.dataset) - train_len - test_len
            return [train_len, val_len, test_len]

        if all(isinstance(l, int) for l in split_lengths):
            # If lengths are provided as integers
            if len(split_lengths) != 3:
                raise ValueError("split_lengths must have exactly 3 elements for train, val, and test sets.")
            return split_lengths

        if all(isinstance(l, float) for l in split_lengths):
            # If lengths are provided as fractions
            if len(split_lengths) not in [2, 3]:
                raise ValueError("split_lengths must have 2 or 3 elements when using fractions.")
            train_len = int(split_lengths[0] * len(self.dataset))
            test_len = int(split_lengths[1] * len(self.dataset))
            val_len = len(self.dataset) - train_len - test_len if len(split_lengths) == 2 else int(split_lengths[2] * len(self.dataset))
            return [train_len, val_len, test_len]

        raise ValueError("split_lengths must be a list of integers or floats.")

    def setup(self, stage: Optional[str] = None):
        """
        Splits the dataset into train, val, and test sets.
        """
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, self.split_lengths
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
