#!/usr/bin/env python3
"""
Latent representation of every subject from a trained run: the posterior means of the content
(z_c) and style (z_s) latents, one row per subject.

The model is rebuilt from the run's MLflow params and loaded from its best checkpoint; meshes are
decoded from the b-values exactly as for training (PCA + pose + Procrustes, same frames).

Output columns: ID, zc01..zcNN, zs01..zsMM. The table is logged as an artifact of the run itself,
latents/latents.csv (latents/latents_first<N>.csv with --n_subjects N); --output also keeps a local copy.

The mesh partition comes from the run's `partition` tag (runs without it were trained on the default,
left_ventricle); --partition overrides it.

Usage (from anywhere; paths are resolved against the repo root):
    python scripts/extract_latents.py 90c7221d08164792923da5d03b674d86
    python scripts/extract_latents.py --run_id 90c7221d   # same, as a named option (for sarray_params TSVs)
    python scripts/extract_latents.py 90c7221d --n_subjects 200 --no_log_artifact --output /tmp/check.csv   # quick check
"""
import argparse
import ast
import logging
import os
import sys
import tempfile
import time

import pandas as pd
import torch
from easydict import EasyDict
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cardiac_motion"))

import cardio_mesh
from cardiac_motion import AutoencoderTemporalSequence
from config.load_config import load_yaml_config
from data.DataModules import CardiacMeshFromBValuesDataset
from utils.helpers import get_n_equispaced_timeframes
from utils.pretrained import load_pretrained, resolve_checkpoint, _model_state_dict_from_checkpoint

logger = logging.getLogger("extract_latents")

DEFAULT_BVALUES = os.path.join(os.path.dirname(REPO_ROOT), "CardiacSegmentation", ".cache", "params.h5")
BASE_CONFIG = os.path.join(REPO_ROOT, "config_files", "config_folded_c_and_s.yaml")


def resolve_run_id(client, spec, tracking_uri):
    '''A full MLflow run ID, or a unique prefix of one (e.g. the 8 characters in the sweep tables).'''
    if len(spec) == 32:
        return spec
    matches = [r.info.run_id for exp in client.search_experiments()
               for r in client.search_runs([exp.experiment_id], max_results=50000)
               if r.info.run_id.startswith(spec)]
    if len(matches) != 1:
        raise ValueError(f"Run ID prefix {spec!r} matches {len(matches)} runs in {tracking_uri}")
    return matches[0]


def _param(params, key, default=None):
    '''MLflow params are strings: parse lists/numbers/bools/None back, leave the rest as str.'''
    if key not in params:
        return default
    try:
        return ast.literal_eval(params[key])
    except (ValueError, SyntaxError):
        return params[key]


def polynomial_degree_from_checkpoint(state_dict, n_layers):
    '''ChebConv order K per encoder layer (one `lins.k` per order): not logged as an MLflow param.'''
    degrees = []
    for i in range(n_layers):
        prefix = f"encoder.encoder_3d_mesh.layers.layer_{i}.graph_conv.lins."
        degrees.append(len({k[len(prefix):].split(".")[0] for k in state_dict if k.startswith(prefix)}))
    if 0 in degrees:
        raise ValueError(f"Couldn't infer the Chebyshev polynomial degree from the checkpoint ({degrees})")
    return degrees


def build_config(params, checkpoint_state_dict):
    config = load_yaml_config(BASE_CONFIG)
    net = config.network_architecture
    net.latent_dim_c = _param(params, "latent_dim_c")
    net.latent_dim_s = _param(params, "latent_dim_s")
    net.convolution.channels_enc = _param(params, "n_channels_enc")
    net.convolution.channels_dec_c = _param(params, "n_channels_dec_c")
    net.convolution.channels_dec_s = _param(params, "n_channels_dec_s")
    net.pooling.parameters.downsampling_factors = _param(params, "reduction_factors")
    net.convolution.parameters.polynomial_degree = polynomial_degree_from_checkpoint(
        checkpoint_state_dict, len(net.convolution.channels_enc))
    net.z_aggr_function = _param(params, "z_aggr_function", net.z_aggr_function)
    net.phase_input = _param(params, "phase_input", False)
    for key in ["d_model", "n_heads", "n_layers", "d_ff", "dropout"]:
        net.transformer[key] = _param(params, f"transformer_{key}", net.transformer[key])
    net.encoder_n_timeframes = _param(params, "encoder_n_timeframes")
    net.translation_head = _param(params, "translation_head", False)
    net.translation_head_hidden = _param(params, "translation_head_hidden", [])
    net.n_harmonics = _param(params, "n_harmonics", 1)
    net.batch_norm = _param(params, "batch_norm", "all")
    net.batch_norm_harmonics = _param(params, "batch_norm_harmonics", 3)
    # w_kl = 0 means a non-variational encoder (no log-variance head)
    config.loss.regularization.weight = _param(params, "w_kl")
    config.dataset.parameters.T = _param(params, "dataset_n_timeframes", config.dataset.parameters.T)
    return config


def move_matrices_to_device(model, device):
    '''The COMA sampling/adjacency matrices are plain lists of tensors, not buffers: .to() skips them.'''
    enc, dec = model.encoder.matrices, model.decoder.matrices
    for i in range(len(enc["downsample"])):
        enc["downsample"][i] = enc["downsample"][i].to(device)
        dec["upsample"][i] = dec["upsample"][i].to(device)
    for i in range(len(enc["A_edge_index"])):
        enc["A_edge_index"][i] = enc["A_edge_index"][i].to(device)
        enc["A_norm"][i] = enc["A_norm"][i].to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", nargs="?", default=None, help="MLflow run ID (or a unique prefix of it)")
    parser.add_argument("--run_id", dest="run_id_opt", default=None, help="same as the positional run_id")
    parser.add_argument("--output", default=None, help="also write the table here (.csv or .parquet)")
    parser.add_argument("--no_log_artifact", action="store_true", help="don't log the table as an artifact of the run")
    parser.add_argument("--tracking_uri", default=os.path.join(REPO_ROOT, "mlruns"))
    parser.add_argument("--bvalues_dir", default=DEFAULT_BVALUES, help="b-values file (params.h5) or directory")
    parser.add_argument("--partition", default=None,
                        help="mesh partition (default: the run's `partition` tag, else left_ventricle)")
    parser.add_argument("--n_subjects", type=int, default=None,
                        help="only the first N subjects, sorted by ID (default: all). Runs trained with "
                             "--n_subjects N saw only these first N subjects.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if (args.run_id is None) == (args.run_id_opt is None):
        parser.error("give the run ID once: positional or --run_id")
    client = MlflowClient(tracking_uri=args.tracking_uri)
    run_id = resolve_run_id(client, args.run_id or args.run_id_opt, args.tracking_uri)
    if args.no_log_artifact and not args.output:
        parser.error("--no_log_artifact needs --output")
    output = os.path.abspath(args.output) if args.output else None
    os.chdir(REPO_ROOT)  # the COMA matrices cache (data/cached/matrices) is relative to the repo root

    run = client.get_run(run_id)
    params = run.data.params
    partition = args.partition or run.data.tags.get("partition", "left_ventricle")
    if _param(params, "center_around_mean", False) or _param(params, "dataset_center_around_mean", False):
        logger.warning("Run was trained with --center_around_mean: make sure the dataset flags below match")
    n_timeframes = _param(params, "n_timeframes")
    checkpoint = resolve_checkpoint(run_id, args.tracking_uri)
    config = build_config(params, _model_state_dict_from_checkpoint(checkpoint))
    latent_dim_c, latent_dim_s = config.network_architecture.latent_dim_c, config.network_architecture.latent_dim_s
    logger.info("Run %s: partition=%s, z_c=%d, z_s=%d, %d frames, batch_norm=%s, checkpoint=%s",
                run_id, partition, latent_dim_c, latent_dim_s, n_timeframes, config.network_architecture.batch_norm, checkpoint)

    mean_shape = cardio_mesh.paths.get_mean_shape(partition)
    faces = cardio_mesh.load_fhm_topology()[cardio_mesh.close_chamber(partition)].f
    mesh_template = EasyDict({"v": mean_shape, "f": faces})

    model = AutoencoderTemporalSequence.build_from_config(config, mesh_template, partition, n_timeframes)
    load_pretrained(model, checkpoint, n_timeframes, args.tracking_uri)
    model.to(args.device).eval()
    move_matrices_to_device(model, args.device)

    # The static representative only changes the decoders' targets, not the encoder's input (s_t)
    dataset = CardiacMeshFromBValuesDataset(
        params_dir=args.bvalues_dir, partition=partition,
        procrustes_transforms=cardio_mesh.paths.get_procrustes_file(partition),
        N_subj=args.n_subjects, phases_filter=get_n_equispaced_timeframes(n_timeframes),
        template_mesh=mesh_template, static_shape="temporal_mean",
        center_around_own_mean=_param(params, "center_around_own_mean", False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Progress bar in a terminal; periodic log lines otherwise (e.g. SLURM logs)
    interactive = sys.stderr.isatty()
    progress = tqdm(total=len(dataset), unit="subj", desc="Encoding", disable=not interactive)
    mus, t0, done = [], time.time(), 0
    with torch.no_grad(), progress:
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            s_t = dataset.decode_batch(batch)["s_t"]
            mus.append(model.encoder(s_t)["mu"].cpu())  # all frames, in order: no frame subsampling
            done += len(s_t)
            progress.update(len(s_t))
            if not interactive:
                logger.info("%d/%d subjects (%.0f s elapsed)", done, len(dataset), time.time() - t0)
    mu = torch.cat(mus).numpy()

    columns = [f"zc{i:02d}" for i in range(1, latent_dim_c + 1)] + [f"zs{i:02d}" for i in range(1, latent_dim_s + 1)]
    table = pd.DataFrame(mu, columns=columns)
    table.insert(0, "ID", dataset.ids)
    logger.info("%d subjects x (%d z_c + %d z_s)", len(table), latent_dim_c, latent_dim_s)

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        (table.to_parquet if output.endswith(".parquet") else table.to_csv)(output, index=False)
        logger.info("Wrote %s", output)
    if not args.no_log_artifact:
        name = "latents.csv" if args.n_subjects is None else f"latents_first{args.n_subjects}.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, name)
            table.to_csv(path, index=False)
            client.log_artifact(run_id, path, artifact_path="latents")
        logger.info("Logged artifact latents/%s to run %s (%s)", name, run_id,
                    os.path.join(client.get_run(run_id).info.artifact_uri, "latents", name))


if __name__ == "__main__":
    main()
