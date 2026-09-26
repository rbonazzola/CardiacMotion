import os, sys
import logging
import time
import numpy as np
import pandas as pd
import re
import glob
import pickle as pkl
import h5py
from easydict import EasyDict
from typing import Optional, List, Union, Literal, Tuple, NamedTuple
from copy import copy

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as pl

from cardio_mesh import (
    CardiacMeshPopulation, 
    Cardiac3DMesh,    
)

from cardio_mesh import paths as cardio_mesh_paths
from cardio_mesh.procrustes import transform_mesh, transform_mesh_torch
from cardio_mesh.pdm_reconstruction import reconstruct_shapes_from_bvalues, reconstruct_shapes_from_bvalues_torch

logger = logging.getLogger(__name__)


def _cuda_usable() -> bool:
    """torch.cuda.is_available() only checks the driver/build support a CUDA
    device exists - it stays True even when that device is busy/unusable
    (common on shared HPC nodes), which then crashes DataLoader pin_memory.
    Probe with a real (tiny) allocation instead."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


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
        center_around_own_mean: bool = False,
        ):

        '''
          root_dir: root directory where all the mesh data is located.
          N_subj: if None, all the subjects are utilized.
          faces: F x 3 array (F is the number of faces)
          procrustes_transforms: Mapping from IDs to transforms ("rotation" and "traslation")
          center_around_mean: if True, subtract the population mean shape (template_mesh.v)
            from all meshes. Reduces data scale to small residuals, which stabilizes training.
          center_around_own_mean: if True, subtract each subject's own centroid (mean position
            over time and vertices) from their own sequence. See the comment at its use site
            below for why this differs from center_around_mean.
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
        self.center_around_own_mean = center_around_own_mean
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

        if self.center_around_own_mean:
            # Subtract THIS subject's own centroid (mean position over both
            # time and vertices) -- unlike center_around_mean (a single fixed
            # population template subtracted from everyone), this removes
            # each subject's own absolute-position offset while leaving their
            # shape and real within-cycle motion untouched (same constant
            # vector subtracted from every vertex/frame of that subject).
            own_centroid = s_t.mean(dim=(0, 1))
            s_t     = s_t     - own_centroid
            s_t_avg = s_t_avg - own_centroid

        dd = {
          "s_t": s_t,
          "time_avg_s": s_t_avg,
          "d_content": dev_from_tmp_avg,
          "d_style": dev_from_sphere
        }
        
        return EasyDict(dd)
    
    
    def __len__(self):
        
        return len(self.ids)


def _load_end_systole_frames(end_systole_frames):
    """{subject_id: 1-based end-systolic frame} from a mapping or a scripts/compute_lv_volumes.py table."""
    if end_systole_frames is None:
        raise ValueError("static_shape='end_systole' requires end_systole_frames (e.g. the table from scripts/compute_lv_volumes.py)")
    if isinstance(end_systole_frames, (str, os.PathLike)):
        path = str(end_systole_frames)
        table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype={"subject_id": str})
        return dict(zip(table.subject_id.astype(str), table.es_frame.astype(int)))
    return {str(k): int(v) for k, v in end_systole_frames.items()}


class CardiacMeshFromBValuesDataset(TensorDataset):
    '''
    Reconstructs meshes from PDM b-values + the per-frame rigid/scale transform
    predicted alongside them, instead of loading pre-reconstructed full-resolution
    meshes from disk. See cardio_mesh/pdm_reconstruction.py for the composed
    affine map (PCA reconstruction, already collapsed at decimation time to the
    target partition, followed by the per-frame rigid inverse transform and the
    per-subject population Procrustes transform).

    Unlike a typical Dataset, the actual reconstruction does NOT happen in
    __getitem__ (that would repeat the same deterministic PCA + Procrustes
    computation, and the same per-subject .npz disk read, on every access, in
    every epoch -- pure redundant work, since nothing here is randomized).
    Instead:
      - __init__ reads every subject's .npz ONCE and keeps the raw b-values/
        pose/Procrustes parameters as in-memory tensors (small: a few hundred
        MB at most, regardless of mesh resolution, since this never holds
        reconstructed vertices).
      - __getitem__ is just an index into those tensors -- cheap, no I/O, safe
        to call from any number of DataLoader workers.
      - decode_batch() does the actual PCA + rigid + Procrustes reconstruction,
        batched, on whatever device the batch already lives on. It's meant to
        be called from CardiacMeshPopulationDM.on_after_batch_transfer, i.e.
        once per batch, on GPU, after the raw params have already been moved
        there -- not once per subject, not on CPU.

    decode_batch's output has the same EasyDict shape (s_t/time_avg_s/
    d_content/d_style) the old per-item __getitem__ used to return, so
    CardiacMeshPopulationDM and the training step don't need to change.
    '''

    def __init__(
        self,
        params_dir: str,
        partition: str,
        procrustes_transforms: Union[str, None] = None,
        N_subj: Union[int, None] = None,
        phases_filter=None,
        template_mesh=None,
        static_shape: Literal["end_diastole", "temporal_mean", "end_systole"] = "end_diastole",
        center_around_mean: bool = False,
        center_around_own_mean: bool = False,
        end_systole_frames=None,
        ):

        '''
          params_dir: EITHER a directory with one <subject_id>.npz per subject
            (holding "bvals" (T, n_components), "translation" (T, 3),
            "qrotation" (T, 4) and "scale" (T,)), OR the path to a single
            consolidated .h5/.hdf5 file with the same fields as top-level
            datasets of shape (N, T, ...) plus a "subject_ids" (N,) string
            dataset (see scripts/maintenance/npz_to_hdf5.py) -- avoids one
            filesystem open per subject, ~95,950 of them for the full
            cohort (~66s sequential vs. a handful of seconds for one file).
          partition: which cached, pre-decimated PCA basis to reconstruct into
            (see cardio_mesh.paths.get_pca_components / get_pca_mean).
          procrustes_transforms: Mapping from IDs to transforms ("rotation" and "traslation"),
            same population-alignment transform used by CardiacMeshPopulationDataset.
          end_systole_frames: required with static_shape="end_systole": each subject's end-systolic
            frame (1-based, over all frames, not only phases_filter's), as a {subject_id: frame}
            mapping or the path to the table written by scripts/compute_lv_volumes.py (es_frame
            column). That frame is loaded even when phases_filter doesn't include it.
        '''

        if center_around_mean and template_mesh is None:
            raise ValueError("center_around_mean=True requires template_mesh to be provided.")

        start = time.perf_counter()
        self._params_dir = params_dir
        self.partition = partition

        logger.info("Loading PCA basis for partition=%s", partition)
        self.pca_components = Tensor(cardio_mesh_paths.get_pca_components(partition))
        self.pca_mean = Tensor(cardio_mesh_paths.get_pca_mean(partition))
        self._pca_device = None  # lazily-cached (components, mean) on whatever device decode_batch last saw

        logger.info("Loading Procrustes transforms from %s", procrustes_transforms)
        with open(procrustes_transforms, "rb") as f:
            all_procrustes_transforms = pkl.load(f)

        self._is_hdf5 = os.path.isfile(self._params_dir) and self._params_dir.endswith((".h5", ".hdf5"))

        if self._is_hdf5:
            with h5py.File(self._params_dir, "r") as hf:
                raw_ids = hf["subject_ids"][:]
            hdf5_id_to_row = {
                (sid.decode("utf-8") if isinstance(sid, bytes) else sid): i
                for i, sid in enumerate(raw_ids) if len(sid) > 0
            }
            available_ids = set(hdf5_id_to_row.keys())
        else:
            available_ids = {f[:-len(".npz")] for f in os.listdir(self._params_dir) if f.endswith(".npz")}

        self.ids = sorted(available_ids.intersection(all_procrustes_transforms.keys()))
        if N_subj is not None:
            self.ids = self.ids[:N_subj]

        self.template_mesh = template_mesh
        self.static_shape = static_shape
        self._es_frame_idx = None  # 0-based end-systolic frame per subject (static_shape="end_systole")
        if static_shape == "end_systole":
            es_frames = _load_end_systole_frames(end_systole_frames)
            missing = [id for id in self.ids if id not in es_frames]
            if missing:
                raise ValueError(f"No end-systolic frame for {len(missing)} of {len(self.ids)} subjects (e.g. {missing[:5]})")
            self._es_frame_idx = np.array([es_frames[id] - 1 for id in self.ids])
        self.center_around_mean = center_around_mean
        self.center_around_own_mean = center_around_own_mean

        frame_idx = None
        if phases_filter is not None:
            # frames in the .npz/.h5 are 0-indexed; phases_filter follows the
            # 1-indexed convention used by CardiacMeshPopulationDataset's
            # filename regex.
            frame_idx = [phase - 1 for phase in phases_filter]

        logger.info("Preloading b-values/pose for %d subjects into memory...", len(self.ids))
        load_start = time.perf_counter()

        if self._is_hdf5:
            # self.ids is sorted the same way rows were written (both are
            # sorted subject-id order), so this index array is guaranteed
            # increasing -- a single efficient fancy-indexed read per field,
            # instead of one np.load() per subject.
            rows = [hdf5_id_to_row[id] for id in self.ids]
            with h5py.File(self._params_dir, "r") as hf:
                bvals_all = hf["bvals"][rows]
                translation_all = hf["translation"][rows]
                qrotation_all = hf["qrotation"][rows]
                scale_all = hf["scale"][rows]
            self._keep_end_systole(bvals_all, translation_all, qrotation_all, scale_all)
            if frame_idx is not None:
                bvals_all = bvals_all[:, frame_idx]
                translation_all = translation_all[:, frame_idx]
                qrotation_all = qrotation_all[:, frame_idx]
                scale_all = scale_all[:, frame_idx]
            self.bvals = Tensor(bvals_all)
            self.translation = Tensor(translation_all)
            self.qrotation = Tensor(qrotation_all)
            self.scale = Tensor(scale_all)
            rotation_list = [all_procrustes_transforms[id]["rotation"] for id in self.ids]
            traslation_list = [all_procrustes_transforms[id]["traslation"] for id in self.ids]
            self.proc_rotation = Tensor(np.stack(rotation_list))
            self.proc_traslation = Tensor(np.stack(traslation_list))
        else:
            bvals_list, translation_list, qrotation_list, scale_list = [], [], [], []
            rotation_list, traslation_list, es_list = [], [], []
            for id in self.ids:
                d = np.load(os.path.join(self._params_dir, f"{id}.npz"))
                bvals, translation, qrotation, scale = d["bvals"], d["translation"], d["qrotation"], d["scale"]
                if self._es_frame_idx is not None:
                    es = self._es_frame_idx[len(bvals_list)]
                    es_list.append((bvals[es], translation[es], qrotation[es], scale[es]))
                if frame_idx is not None:
                    bvals, translation, qrotation, scale = (
                        bvals[frame_idx], translation[frame_idx], qrotation[frame_idx], scale[frame_idx]
                    )
                bvals_list.append(bvals)
                translation_list.append(translation)
                qrotation_list.append(qrotation)
                scale_list.append(scale)
                pr = all_procrustes_transforms[id]
                rotation_list.append(pr["rotation"])
                traslation_list.append(pr["traslation"])

            self.bvals = Tensor(np.stack(bvals_list))              # (N, T, n_components)
            self.translation = Tensor(np.stack(translation_list))  # (N, T, 3)
            self.qrotation = Tensor(np.stack(qrotation_list))      # (N, T, 4)
            self.scale = Tensor(np.stack(scale_list))               # (N, T)
            self.proc_rotation = Tensor(np.stack(rotation_list))    # (N, 3, 3)
            self.proc_traslation = Tensor(np.stack(traslation_list))  # (N, 3), from the Procrustes pkl
            if self._es_frame_idx is not None:
                self._keep_end_systole(*[np.stack(x)[:, None] for x in zip(*es_list)], frame_idx=np.zeros(len(es_list), int))

        logger.info(
            "B-values dataset indexed: subjects=%d, partition=%s, n_components=%d, n_verts=%d, "
            "preload=%.1fs, total=%.2fs",
            len(self.ids),
            partition,
            self.pca_components.shape[0],
            self.pca_mean.shape[0] // 3,
            time.perf_counter() - load_start,
            time.perf_counter() - start,
        )


    def _keep_end_systole(self, bvals, translation, qrotation, scale, frame_idx=None):
        """Keeps each subject's end-systolic frame parameters (from the unfiltered (N, T_all, ...) arrays)."""
        if self._es_frame_idx is None:
            return
        rows = np.arange(len(bvals))
        es = self._es_frame_idx if frame_idx is None else frame_idx
        self.es_bvals = Tensor(np.asarray(bvals)[rows, es])            # (N, n_components)
        self.es_translation = Tensor(np.asarray(translation)[rows, es])  # (N, 3)
        self.es_qrotation = Tensor(np.asarray(qrotation)[rows, es])      # (N, 4)
        self.es_scale = Tensor(np.asarray(scale)[rows, es])              # (N,)


    def __len__(self):

        return len(self.ids)


    def __getitem__(self, idx):
        # Deliberately cheap: no I/O, no reconstruction. See class docstring --
        # the real work happens in decode_batch, once per batch, on GPU.
        return {
            "bvals": self.bvals[idx],
            "translation": self.translation[idx],
            "qrotation": self.qrotation[idx],
            "scale": self.scale[idx],
            "proc_rotation": self.proc_rotation[idx],
            "proc_traslation": self.proc_traslation[idx],
            **({"es_bvals": self.es_bvals[idx], "es_translation": self.es_translation[idx],
                "es_qrotation": self.es_qrotation[idx], "es_scale": self.es_scale[idx]}
               if self._es_frame_idx is not None else {}),
        }


    def _pca_on(self, device):
        if self._pca_device != device:
            self._pca_components_dev = self.pca_components.to(device)
            self._pca_mean_dev = self.pca_mean.to(device)
            self._pca_device = device
        return self._pca_components_dev, self._pca_mean_dev


    def decode_batch(self, batch):
        '''
        batch: dict with bvals/translation/qrotation/scale/proc_rotation/
        proc_traslation, already collated and moved to the target device (by
        CardiacMeshPopulationDM.on_after_batch_transfer). Returns the same
        EasyDict(s_t/time_avg_s/d_content/d_style) shape the old per-item
        __getitem__ used to return, batched over the leading dimension.
        '''
        device = batch["bvals"].device
        pca_components, pca_mean = self._pca_on(device)

        s_t = reconstruct_shapes_from_bvalues_torch(
            batch["bvals"], batch["translation"], batch["qrotation"], batch["scale"],
            pca_components, pca_mean,
        )  # (B, T, n_verts, 3)
        s_t = transform_mesh_torch(s_t, batch["proc_rotation"], batch["proc_traslation"])

        if self.static_shape == "end_diastole":
            s_t_avg = s_t[:, 0]
        elif self.static_shape == "temporal_mean":
            s_t_avg = s_t.mean(dim=1)
        elif self.static_shape == "end_systole":
            # the subject's end-systolic frame, reconstructed like any other frame
            s_es = reconstruct_shapes_from_bvalues_torch(
                batch["es_bvals"].unsqueeze(1), batch["es_translation"].unsqueeze(1),
                batch["es_qrotation"].unsqueeze(1), batch["es_scale"].unsqueeze(1),
                pca_components, pca_mean,
            )
            s_t_avg = transform_mesh_torch(s_es, batch["proc_rotation"], batch["proc_traslation"])[:, 0]
        else:
            raise NotImplementedError(f"static_shape={self.static_shape!r}")

        dev_from_tmp_avg = mse(s_t, s_t_avg.unsqueeze(1))

        if self.template_mesh is not None:
            template_v = torch.as_tensor(self.template_mesh.v, dtype=s_t.dtype, device=device)
            dev_from_sphere = mse(s_t, template_v.unsqueeze(0).unsqueeze(0))
        else:
            template_v = None
            dev_from_sphere = None

        if self.center_around_mean:
            s_t = s_t - template_v
            s_t_avg = s_t_avg - template_v

        if self.center_around_own_mean:
            # Subtract each subject's own centroid (mean position over time
            # and vertices) -- see CardiacMeshPopulationDataset.__getitem__
            # for why this differs from center_around_mean.
            own_centroid = s_t.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, 3)
            s_t     = s_t     - own_centroid
            s_t_avg = s_t_avg - own_centroid.squeeze(1)

        return EasyDict({
            "s_t": s_t,
            "time_avg_s": s_t_avg,
            "d_content": dev_from_tmp_avg,
            "d_style": dev_from_sphere,
        })


class StageConfig(NamedTuple):
    batch_size: int
    grad_accum_steps: int


class BatchSizeScheduler:
    """Resolves batch_size and grad_accum_steps for a given epoch.

    Ported from ~/repos/delphi's data/dataset.py -- same string format/
    semantics, so schedules are portable between the two codebases.

    Schedule format: list of (n_epochs, batch_size, grad_accum_steps) tuples.
    The last entry may use n_epochs=None to mean "for all remaining epochs".

    Use BatchSizeScheduler.from_string() to build from a CLI-friendly string:

        "10:32,10:64,*:256x4"

    Each stage is "n_epochs:batch_size" or "n_epochs:batch_sizexgrad_accum".
    Use "*" as n_epochs for the last open-ended stage.

    Example:
        scheduler = BatchSizeScheduler.from_string("10:32,10:64,*:256x4")
        # epochs 0-9   -> batch_size=32,  grad_accum=1  (effective batch=32)
        # epochs 10-19 -> batch_size=64,  grad_accum=1  (effective batch=64)
        # epoch 20+    -> batch_size=256, grad_accum=4  (effective batch=1024)
    """

    def __init__(self, schedule: List[Tuple[Optional[int], int, int]]):
        if not schedule:
            raise ValueError("schedule must have at least one entry")
        for i, (n, _, _) in enumerate(schedule):
            if n is None and i != len(schedule) - 1:
                raise ValueError("Only the last schedule entry may have n_epochs=None")
        self._schedule = list(schedule)

    def step(self, epoch: int) -> StageConfig:
        """Return the StageConfig (batch_size, grad_accum_steps) for the given epoch."""
        elapsed = 0
        for n_epochs, batch_size, grad_accum in self._schedule:
            if n_epochs is None or epoch < elapsed + n_epochs:
                return StageConfig(batch_size, grad_accum)
            elapsed += n_epochs
        n, bs, ga = self._schedule[-1]
        return StageConfig(bs, ga)

    def stage_boundaries(self) -> dict:
        """{stage_start_epoch: grad_accum_steps} for every stage -- feeds
        pytorch_lightning.callbacks.GradientAccumulationScheduler directly."""
        boundaries = {}
        epoch = 0
        for n_epochs, _, grad_accum in self._schedule:
            boundaries[epoch] = grad_accum
            if n_epochs is None:
                break
            epoch += n_epochs
        return boundaries

    def __str__(self) -> str:
        parts = []
        for n_epochs, batch_size, grad_accum in self._schedule:
            n_str = "*" if n_epochs is None else str(n_epochs)
            bs_str = f"{batch_size}x{grad_accum}" if grad_accum > 1 else str(batch_size)
            parts.append(f"{n_str}:{bs_str}")
        return ",".join(parts)

    @classmethod
    def from_string(cls, s: str) -> "BatchSizeScheduler":
        """Parse a schedule string.

        Format: comma-separated "n_epochs:batch_size[xgrad_accum]" pairs.

        Examples:
            "5:32,10:64,*:128"       # no accumulation
            "10:32,10:64,*:256x4"    # last stage: batch=256, accum=4
        """
        schedule = []
        for part in s.split(","):
            part = part.strip()
            n_str, rest = part.split(":")
            n = None if n_str.strip() == "*" else int(n_str.strip())
            if "x" in rest:
                bs_str, ga_str = rest.split("x")
                grad_accum = int(ga_str.strip())
            else:
                bs_str, grad_accum = rest, 1
            schedule.append((n, int(bs_str.strip()), grad_accum))
        return cls(schedule)


class CardiacMeshPopulationDM(pl.LightningDataModule):

    '''
    PyTorch datamodule wrapping the CardiacMeshPopulation class
    '''

    def __init__(self,
        dataset: TensorDataset,
        batch_size: int = 16,
        split_lengths: Union[None, List[int]]=None,
        num_workers=3,
        batch_size_scheduler: Optional[BatchSizeScheduler] = None,
    ):

        '''
        params:
            dataset:
            batch_size: used as-is when batch_size_scheduler is None; otherwise
                only used before training starts (e.g. sanity checks), since
                train/val/test_dataloader() resolve the per-epoch batch size
                from the scheduler instead.
            split_lengths:
            num_workers:
            batch_size_scheduler: optional BatchSizeScheduler -- when set,
                requires reload_dataloaders_every_n_epochs=1 on the Trainer
                so *_dataloader() actually gets called (and re-resolves
                batch_size) at the start of every epoch. The underlying
                train/val/test split is only computed once, in setup() --
                only the DataLoader's batch_size changes between epochs.
        '''

        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.split_lengths = self._get_split_lengths(split_lengths)
        self.num_workers=num_workers
        self.batch_size_scheduler = batch_size_scheduler
        logger.info(
            "CardiacMeshPopulationDM initialized: dataset=%d, batch_size=%d, split_lengths=%s, num_workers=%d%s",
            len(self.dataset),
            self.batch_size,
            self.split_lengths,
            self.num_workers,
            f", batch_size_schedule={batch_size_scheduler}" if batch_size_scheduler is not None else "",
        )

    def _current_batch_size(self) -> int:
        if self.batch_size_scheduler is None:
            return self.batch_size
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        return self.batch_size_scheduler.step(epoch).batch_size
        
        
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
        bs = self._current_batch_size()
        logger.info("Creating train dataloader: batch_size=%d, num_workers=%d", bs, self.num_workers)
        return DataLoader(self.train_dataset, batch_size=bs, num_workers=self.num_workers, pin_memory=_cuda_usable())

    def val_dataloader(self):
        bs = self._current_batch_size()
        logger.info("Creating val dataloader: batch_size=%d, num_workers=%d", bs, self.num_workers)
        return DataLoader(self.val_dataset, batch_size=bs, num_workers=self.num_workers, pin_memory=_cuda_usable())

    def test_dataloader(self):
        bs = self._current_batch_size()
        logger.info("Creating test dataloader: batch_size=%d, num_workers=%d", bs, self.num_workers)
        return DataLoader(self.test_dataset, batch_size=bs, num_workers=self.num_workers, pin_memory=_cuda_usable())

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Runs once per batch, after Lightning has already moved it to the
        # target device (GPU). CardiacMeshFromBValuesDataset defers the PCA +
        # rigid + Procrustes reconstruction to exactly this point (see its
        # decode_batch docstring) instead of doing it per-subject, per-epoch,
        # on CPU inside __getitem__. CardiacMeshPopulationDataset (the disk
        # path) already returns fully-decoded batches, so it has no
        # decode_batch and this is a no-op for it.
        decode_batch = getattr(self.dataset, "decode_batch", None)
        if decode_batch is not None:
            return decode_batch(batch)
        return batch


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
            pin_memory=_cuda_usable(),
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
            pin_memory=_cuda_usable(),
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
            pin_memory=_cuda_usable(),
        )
