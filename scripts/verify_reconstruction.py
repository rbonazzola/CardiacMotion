#!/usr/bin/env python3
"""
Checks that a trained run, rebuilt the way scripts/extract_latents.py rebuilds it (model config from
the run's MLflow params + best checkpoint), reconstructs in inference (eval) mode as it did during
training: recomputes the validation metrics and compares them with the ones logged at the restored
(best) epoch, and evaluates the test split that was actually held out.

Which subjects were held out: CardiacMeshPopulationDM.setup() re-runs random_split with the global
RNG, and Lightning calls it again on every fit/validate/test. So in main_autoencoder_cardiac.py the
split whose hash is logged ("test subjects hash") is NOT the one used: training used the second
split (made by trainer.fit, after building the model), and the logged test_* metrics were computed
on a third one (made by trainer.test) that shares ~60% of its subjects with the training set. This
script reproduces the second split (seed -> dataset -> setup -> model -> setup again in
trainer.validate), freezes it, and computes both val and test on it. The val_* metrics are the
check (they must match the logged ones); the test_* metrics are the honest held-out test ones and
are NOT expected to match the logged test_* (contaminated) values.

The partition comes from the run's `partition` tag (runs without it were trained on the default,
left_ventricle). The number of subjects isn't logged: it's --n_subjects (default 20000), checked against
the global step of the run's first validation.

Usage (from anywhere; paths are resolved against the repo root):
    python scripts/verify_reconstruction.py 7cfddede
    python scripts/verify_reconstruction.py 7cfddede 90c7221d --output /tmp/verify.csv
"""
import argparse
import hashlib
import logging
import os
import sys

import pandas as pd
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from mlflow.tracking import MlflowClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cardiac_motion"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import cardio_mesh
from cardiac_motion import AutoencoderTemporalSequence
from data.DataModules import CardiacMeshFromBValuesDataset, CardiacMeshPopulationDM
from lightning_modules.ComaLightningModule import CoMA_Lightning, build_wall_thickness_pairs
from utils.helpers import get_n_equispaced_timeframes
from utils.pretrained import load_pretrained, resolve_checkpoint, _model_state_dict_from_checkpoint
from extract_latents import DEFAULT_BVALUES, _param, build_config, resolve_run_id

logger = logging.getLogger("verify_reconstruction")


def verify(client, run_id, args):
    run = client.get_run(run_id)
    params, tags, logged = run.data.params, run.data.tags, run.data.metrics
    partition = tags.get("partition", "left_ventricle")
    n_subjects = args.n_subjects
    # The number of subjects isn't logged: check it against the global step of the first validation
    # (= train batches per epoch - 1)
    # (each validation is logged twice: step = epoch, and step = global step; the first global step is
    # far above the number of epochs, i.e. above half the number of entries)
    history = client.get_metric_history(run_id, "val_recon_loss_s_shape")
    val_steps = sorted(m.step for m in history if m.step > len(history))
    n_train = int(float(params["n_training"]) * n_subjects) if float(params["n_training"]) < 1 else int(params["n_training"])
    expected = -(-n_train // int(params["batch_size"])) - 1
    if val_steps and val_steps[0] != expected:
        raise RuntimeError(f"first validation at global step {val_steps[0]}, but {n_subjects} subjects would give "
                           f"{expected}: pass the run's --n_subjects")
    n_timeframes = _param(params, "n_timeframes")
    static = _param(params, "static_representative", "temporal_mean")
    seed = _param(params, "seed")

    # Same order as main_autoencoder_cardiac.py: global seed first, then dataset and random_split
    pl.seed_everything(seed, workers=True)
    mean_shape = cardio_mesh.paths.get_mean_shape(partition)
    faces = cardio_mesh.load_fhm_topology()[cardio_mesh.close_chamber(partition)].f
    mesh_template = EasyDict({"v": mean_shape, "f": faces})
    dataset = CardiacMeshFromBValuesDataset(
        params_dir=args.bvalues_dir, partition=partition,
        procrustes_transforms=cardio_mesh.paths.get_procrustes_file(partition),
        N_subj=n_subjects, phases_filter=get_n_equispaced_timeframes(n_timeframes),
        template_mesh=mesh_template, static_shape=static,
        center_around_mean=_param(params, "dataset_center_around_mean", False),
        center_around_own_mean=_param(params, "center_around_own_mean", False),
        end_systole_frames=_param(params, "end_systole_file") if static == "end_systole" else None,
    )
    dm = CardiacMeshPopulationDM(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup()  # first split, as in main (the one whose hash main logs)

    checkpoint = resolve_checkpoint(run_id, args.tracking_uri)
    config = build_config(params, _model_state_dict_from_checkpoint(checkpoint))
    config.loss.reconstruction_s.weight = _param(params, "w_s", config.loss.reconstruction_s.weight)
    config.loss.smoothness.weight = _param(params, "w_smooth", 0.0)
    config.loss.thickness.weight = _param(params, "w_thickness", 0.0)
    model = AutoencoderTemporalSequence.build_from_config(config, mesh_template, partition, n_timeframes)
    load_pretrained(model, checkpoint, n_timeframes, args.tracking_uri)

    thickness_pairs = None
    if config.loss.thickness.weight > 0:
        thickness_pairs = build_wall_thickness_pairs(mesh_template.v, cardio_mesh.get_lv_wall_labels(partition))
    lit = CoMA_Lightning(model=model, loss_params=config.loss, optimizer_params=config.optimizer,
                         additional_params=config, mesh_template=mesh_template, thickness_pairs=thickness_pairs)
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, precision=32,
                         enable_progress_bar=False, enable_model_summary=False)
    # trainer.validate re-splits (second split = the one training used) and evaluates its val set
    recomputed = trainer.validate(lit, datamodule=dm, verbose=False)[0]
    dm.setup = lambda stage=None: None  # freeze it: evaluate that split's test set, not a new random one
    recomputed.update(trainer.test(lit, datamodule=dm, verbose=False)[0])
    test_ids = sorted(dataset.ids[k] for k in dm.test_dataset.indices)
    test_hash = hashlib.md5(",".join(map(str, test_ids)).encode()).hexdigest()[:12]

    # Logged val metrics at the restored (best) epoch: each validation is logged with step = epoch
    best_epoch = int(logged["restored_epoch"])
    rows = []
    for name, value in sorted(recomputed.items()):
        if name.startswith("val_"):
            hist = [m for m in client.get_metric_history(run_id, name) if m.step == best_epoch]
            if not hist:
                continue
            ref = sorted(hist, key=lambda m: m.timestamp)[-1].value
            kind = "check"
        elif name.startswith("test_") and name in logged:
            ref, kind = logged[name], "test (logged = contaminated split)"
        else:
            continue
        rows.append(dict(run=run_id[:8], metric=name, kind=kind, logged=ref, recomputed=value,
                         rel_diff=abs(value - ref) / max(abs(ref), 1e-12)))
    table = pd.DataFrame(rows)
    logger.info("Run %s: partition=%s, %d subjects, static=%s, seed=%s, best epoch=%d, held-out test hash=%s, checkpoint=%s",
                run_id[:8], partition, n_subjects, static, seed, best_epoch, test_hash, os.path.basename(checkpoint))
    return table, test_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+", help="MLflow run IDs (or unique prefixes)")
    parser.add_argument("--output", default=None, help="write the comparison table here (.csv)")
    parser.add_argument("--tracking_uri", default=os.path.join(REPO_ROOT, "mlruns"))
    parser.add_argument("--bvalues_dir", default=DEFAULT_BVALUES)
    parser.add_argument("--n_subjects", type=int, default=20000,
                        help="the runs' --n_subjects (not logged in MLflow; checked against the training steps)")
    parser.add_argument("--batch_size", type=int, default=16, help="as in training (batch norm is in eval mode, "
                        "so it only matters for the per-batch averaging of the metrics)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--matmul_precision", default="medium", choices=["medium", "high", "highest"],
                        help="torch float32 matmul precision; the runs were trained with main's default, medium")
    parser.add_argument("--tolerance", type=float, default=0.01, help="max relative difference of the val metrics to call a run OK")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    torch.set_float32_matmul_precision(args.matmul_precision)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    os.chdir(REPO_ROOT)  # COMA matrices cache and end_systole_file paths are relative to the repo root
    tables = []
    for spec in args.run_ids:
        run_id = resolve_run_id(client, spec, args.tracking_uri)
        table, test_hash = verify(client, run_id, args)
        table["held_out_test_hash"] = test_hash
        tables.append(table)
        check = table[table.kind == "check"]
        worst = check.sort_values("rel_diff").iloc[-1]
        status = "OK" if worst.rel_diff <= args.tolerance else "MISMATCH"
        print(f"{run_id[:8]}  {status}  (val vs logged at best epoch: {len(check)} metrics, max rel diff "
              f"{worst.rel_diff:.2e} in {worst.metric}; held-out test hash {test_hash})")
        key = table.set_index("metric")
        for m in ["recon_loss_s_shape", "recon_loss_s_translation", "recon_loss_c", "mean_vertex_dev"]:
            for split in ["val", "test"]:
                k = f"{split}_{m}"
                if k in key.index:
                    label = "logged" if split == "val" else "logged*"
                    print(f"    {k:32s} {label:7s} {key.loc[k, 'logged']:9.4f}   recomputed {key.loc[k, 'recomputed']:9.4f}")
        print("    (* logged test_* were computed on a re-shuffled split that overlaps the training set)")
    if args.output:
        pd.concat(tables).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
