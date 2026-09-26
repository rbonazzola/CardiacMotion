#!/usr/bin/env python3
"""
LV cavity volume of every frame of every subject, and each subject's end-systolic frame (minimum
volume), from the b-values dataset. Volumes are convex hulls of the endocardial vertices (see
cardiac_motion/utils/lv_volumes.py).

Meshes are decoded in chunks exactly as for training (PCA + pose + Procrustes); hulls are computed
in parallel across subjects.

Usage (from the repo root):
    python scripts/compute_lv_volumes.py --bvalues_dir <params.h5> --output data/lv_volumes.csv --n_workers 48
    python scripts/compute_lv_volumes.py ... --n_subjects 200          # quick check
"""
import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from torch.utils.data import default_collate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cardiac_motion"))

import cardio_mesh
from data.DataModules import CardiacMeshFromBValuesDataset
from utils.helpers import get_n_equispaced_timeframes
from utils.lv_volumes import endocardial_volumes, volume_table

logger = logging.getLogger("compute_lv_volumes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bvalues_dir", required=True, help="b-values file (params.h5) or directory")
    parser.add_argument("--output", required=True, help="output table (.csv or .parquet)")
    parser.add_argument("--partition", default="left_ventricle")
    parser.add_argument("--n_subjects", type=int, default=None, help="only the first N subjects (default: all)")
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())
    parser.add_argument("--chunk_size", type=int, default=500, help="subjects decoded at a time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    frames = get_n_equispaced_timeframes(50)  # all frames, 1-based
    mean_shape = cardio_mesh.paths.get_mean_shape(args.partition)
    endo = cardio_mesh.get_lv_wall_labels(args.partition) == "endo"
    dataset = CardiacMeshFromBValuesDataset(
        params_dir=args.bvalues_dir, partition=args.partition,
        procrustes_transforms=cardio_mesh.paths.get_procrustes_file(args.partition),
        N_subj=args.n_subjects, phases_filter=frames,
        template_mesh=EasyDict({"v": mean_shape, "f": None}),
    )
    n = len(dataset)
    logger.info("%d subjects x %d frames, %d endocardial vertices, %d workers", n, len(frames), endo.sum(), args.n_workers)

    torch.set_num_threads(max(1, args.n_workers))
    volumes, t0 = [], time.time()
    with Pool(args.n_workers) as pool:
        for start in range(0, n, args.chunk_size):
            batch = default_collate([dataset[i] for i in range(start, min(start + args.chunk_size, n))])
            s_t = dataset.decode_batch(batch)["s_t"].numpy()  # (B, T, V, 3), mm
            volumes.extend(pool.map(endocardial_volumes, list(s_t[:, :, endo]), chunksize=8))
            done = start + len(s_t)
            logger.info("%d/%d subjects (%.0f s elapsed, ~%.0f s left)", done, n, time.time() - t0,
                        (time.time() - t0) / done * (n - done))

    table = volume_table(dataset.ids, np.stack(volumes), frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    (table.to_parquet if args.output.endswith(".parquet") else table.to_csv)(args.output, index=False)
    logger.info("Wrote %s | EDV median %.0f ml, ESV %.0f ml, EF %.2f | ES frame median %d (IQR %d-%d)",
                args.output, table.edv_ml.median(), table.esv_ml.median(), table.ef.median(), table.es_frame.median(),
                table.es_frame.quantile(0.25), table.es_frame.quantile(0.75))


if __name__ == "__main__":
    main()
