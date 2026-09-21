"""
Profiles a few real training steps (forward + backward + optimizer step) of
the actual AutoencoderTemporalSequence, on GPU, with torch.profiler -- to see
where time really goes instead of guessing from parameter counts or FLOP
estimates. Prints the top ops/modules by CUDA time.

Run via sbatch (needs a real GPU allocation, not the interactive session's
cgroup-limited shell) -- see runs/submit_profile_model.sh.
"""
import os
import sys

REPO_ROOT = "/net/scratch/t19767rb/src/CardiacMotion"
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cardiac_motion"))
os.chdir(REPO_ROOT)

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from easydict import EasyDict

import cardio_mesh
from cardio_mesh import paths as cardio_mesh_paths
from data.DataModules import CardiacMeshFromBValuesDataset
from models.Model4D import AutoencoderTemporalSequence
from lightning_modules.ComaLightningModule import CoMA_Lightning
from config.load_config import load_yaml_config
from utils.helpers import get_n_equispaced_timeframes

PARTITION = "left_ventricle"
PARAMS_DIR = "/net/scratch/t19767rb/src/CardiacSegmentation/.cache/params"
N_TIMEFRAMES = 10
BATCH_SIZE = 64
REDUCTION_FACTORS_LIST = [[2, 2, 2, 2], [2, 2, 3, 3]]


def build(reduction_factors, device):
    config = load_yaml_config("config_files/config_folded_c_and_s.yaml")
    config.network_architecture.latent_dim_c = 16
    config.network_architecture.latent_dim_s = 16
    config.loss.regularization.weight = 1e-4
    config.network_architecture.pooling.parameters.downsampling_factors = reduction_factors

    template_fhm_mesh = cardio_mesh.load_fhm_topology()
    closed_chamber = cardio_mesh.close_chamber(PARTITION)
    faces = template_fhm_mesh[closed_chamber].f
    mean_shape = cardio_mesh_paths.get_mean_shape(PARTITION)
    mesh_template = EasyDict({"v": mean_shape, "f": faces})
    phases_filter = get_n_equispaced_timeframes(N_TIMEFRAMES)

    dataset = CardiacMeshFromBValuesDataset(
        params_dir=PARAMS_DIR, partition=PARTITION,
        procrustes_transforms=cardio_mesh_paths.get_procrustes_file(PARTITION),
        N_subj=BATCH_SIZE * 2, phases_filter=phases_filter, template_mesh=mesh_template,
        center_around_mean=False,
    )

    model = AutoencoderTemporalSequence.build_from_config(config, mesh_template, PARTITION, N_TIMEFRAMES)
    lit_module = CoMA_Lightning(
        model=model, loss_params=config.loss, optimizer_params=config.optimizer,
        additional_params=config, mesh_template=mesh_template,
    ).to(device)
    lit_module.model.set_mode("training")
    for i, _ in enumerate(lit_module.model.encoder.matrices["downsample"]):
        lit_module.model.encoder.matrices["downsample"][i] = lit_module.model.encoder.matrices["downsample"][i].to(device)
        lit_module.model.decoder.matrices["upsample"][i] = lit_module.model.decoder.matrices["upsample"][i].to(device)
    for i, _ in enumerate(lit_module.model.encoder.matrices["A_edge_index"]):
        lit_module.model.encoder.matrices["A_edge_index"][i] = lit_module.model.encoder.matrices["A_edge_index"][i].to(device)
        lit_module.model.encoder.matrices["A_norm"][i] = lit_module.model.encoder.matrices["A_norm"][i].to(device)
    if lit_module.laplacian is not None:
        lit_module.laplacian = lit_module.laplacian.to(device)
    if lit_module.smooth_mask is not None:
        lit_module.smooth_mask = lit_module.smooth_mask.to(device)

    batch = torch.utils.data.default_collate([dataset[i] for i in range(BATCH_SIZE)])
    batch = {k: v.to(device) for k, v in batch.items()}
    decoded = dataset.decode_batch(batch)
    s_t = decoded.s_t
    time_avg_s = decoded.time_avg_s

    optimizer = torch.optim.Adam(lit_module.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    return lit_module, optimizer, s_t, time_avg_s, n_params


def run_steps(lit_module, optimizer, s_t, time_avg_s, n_warmup=3, n_profile=5):
    def step():
        optimizer.zero_grad(set_to_none=True)
        bottleneck, time_avg_shat, shat_t = lit_module(s_t)
        recon_loss_c = lit_module.rec_loss(time_avg_s, time_avg_shat)
        recon_loss_s = lit_module.rec_loss(s_t, shat_t)
        loss = recon_loss_c + lit_module._effective_w_s() * recon_loss_s
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(n_profile):
            with record_function("train_step"):
                step()
        torch.cuda.synchronize()

    return prof


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    for reduction_factors in REDUCTION_FACTORS_LIST:
        print(f"\n{'='*70}\nreduction_factors={reduction_factors}\n{'='*70}")
        lit_module, optimizer, s_t, time_avg_s, n_params = build(reduction_factors, device)
        print(f"trainable params: {n_params:,}")

        prof = run_steps(lit_module, optimizer, s_t, time_avg_s)

        print("\n-- top 15 by CUDA total time --")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        del lit_module, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
