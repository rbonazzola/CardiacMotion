import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
import imageio
import numpy as np

from argparse import Namespace

from models.Model4D import AutoencoderTemporalSequence, LOG_VAR_MAX, LOG_VAR_MIN, clamp_log_var

logger = logging.getLogger(__name__)

def per_vertex_mse(real, recon):
    """
    Squared Euclidean distance per vertex (summed over xyz), averaged over vertices, frames and
    subjects -- i.e. 3x F.mse_loss, which averages over coordinates too. real/recon: (..., V, 3).
    """
    return ((real - recon) ** 2).sum(-1).mean()


losses_menu = {
  "l1": F.l1_loss,
  "mse": per_vertex_mse
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)


def translation_shape_split_loss(rec_loss_fn, real, recon):
    """
    Splits a reconstruction MSE into a per-frame rigid-translation term and a
    shape (translation-invariant) term. real/recon: (..., V, 3). With per_vertex_mse, the
    translation term is the squared distance between centroids.

    Exact decomposition: since real-recon = (shape_real-shape_recon) +
    (centroid_real-centroid_recon) and the shape residual sums to zero over
    V by construction, the cross term vanishes and
    rec_loss_fn(real, recon) == rec_loss_fn(centroid_real, centroid_recon) +
    rec_loss_fn(shape_real, shape_recon) exactly -- so translation_weight=
    shape_weight=1 reproduces the plain MSE precisely; weighting them
    differently gives translation error its own gradient signal instead of
    it competing, diluted, inside one flat per-vertex MSE.
    """
    centroid_real = real.mean(dim=-2, keepdim=True)
    centroid_recon = recon.mean(dim=-2, keepdim=True)
    loss_translation = rec_loss_fn(centroid_real, centroid_recon)
    loss_shape = rec_loss_fn(real - centroid_real, recon - centroid_recon)
    return loss_translation, loss_shape


def safe_mean_ratio(numerator, denominator):
    eps = torch.finfo(denominator.dtype).eps
    valid = denominator.abs() > eps
    if not valid.any():
        return torch.tensor(float("nan"), device=numerator.device, dtype=numerator.dtype)
    return (numerator[valid] / denominator[valid]).mean()


def pooled_ratio(outputs):
    """
    Ratio of totals over an epoch: sum of per-frame reconstruction errors / sum of per-frame
    deviations from the static shape, i.e. total model error relative to the total error of
    predicting the static shape for every frame (see _shared_eval_step).
    """
    rec_err = torch.stack([x["rec_err_sum"] for x in outputs]).sum()
    dev_static = torch.stack([x["dev_static_sum"] for x in outputs]).sum()
    if dev_static.abs() <= torch.finfo(dev_static.dtype).eps:
        return torch.tensor(float("nan"), dtype=rec_err.dtype)
    return rec_err / dev_static


def pooled_mean_vertex_dev(outputs):
    """Mean per-vertex Euclidean reconstruction distance over all vertices/frames/subjects of an epoch."""
    total = torch.stack([x["vertex_dev_sum"] for x in outputs]).sum()
    count = torch.stack([x["vertex_count"] for x in outputs]).sum()
    return total / count


def build_wall_thickness_pairs(vertices, labels) -> torch.Tensor:
    """
    (epi, endo) vertex index pairs across the left-ventricular wall, as close as possible: each epi
    vertex paired with its nearest endo vertex and each endo vertex with its nearest epi vertex
    (duplicates removed), so every vertex of both surfaces is in at least one pair. Computed once
    on a template (e.g. the mean shape): meshes are in vertex correspondence, so the same indices
    apply to every subject and frame. vertices: (V, 3); labels: (V,) with "epi"/"endo" entries.
    """
    vertices = torch.as_tensor(np.asarray(vertices), dtype=torch.float64)
    labels = np.asarray(labels)
    epi = torch.as_tensor(np.flatnonzero(labels == "epi"))
    endo = torch.as_tensor(np.flatnonzero(labels == "endo"))
    if len(epi) == 0 or len(endo) == 0:
        raise ValueError("wall thickness pairs need both epi and endo vertices")
    distances = torch.cdist(vertices[epi], vertices[endo])  # (n_epi, n_endo)
    pairs = torch.cat([
        torch.stack([epi, endo[distances.argmin(dim=1)]], dim=1),
        torch.stack([epi[distances.argmin(dim=0)], endo], dim=1),
    ])
    return torch.unique(pairs, dim=0)


def wall_thickness_distances(x: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Euclidean epi-endo distance of each pair. x: (..., V, 3); pairs: (P, 2) -> (..., P)."""
    return torch.linalg.vector_norm(x[..., pairs[:, 0], :] - x[..., pairs[:, 1], :], dim=-1)


def aggregate_thickness(outputs):
    """
    Epoch-level wall thickness terms: mean of the per-batch losses (like the other losses), and the
    mean absolute thickness error over every pair/frame/subject (NaN when there are no pairs).
    """
    loss = torch.stack([x["thickness_loss"] for x in outputs]).mean()
    count = torch.stack([x["thickness_count"] for x in outputs]).sum()
    abs_err = torch.stack([x["thickness_abs_err_sum"] for x in outputs]).sum()
    mae = abs_err / count if count > 0 else torch.tensor(float("nan"))
    return loss, mae


def build_laplacian(faces: np.ndarray, n_verts: int) -> torch.Tensor:
    """
    Row-stochastic graph Laplacian L = I - A from mesh faces, sparse (n_verts,
    n_verts): (L @ v)[i] = v[i] - mean(neighbors(v)[i]). Same convention as the
    Laplacian smoothing already used elsewhere in this project's pipeline
    (CardiacSegmentation's mesh_render_utils._laplacian_operator), rebuilt here
    as a torch sparse tensor so it's usable inside the training loop.
    """
    edges = set()
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        edges.update({(a, b), (b, a), (b, c), (c, b), (c, a), (a, c)})
    edges = np.array(sorted(edges), dtype=np.int64)
    row, col = edges[:, 0], edges[:, 1]

    deg = np.bincount(row, minlength=n_verts).astype(np.float64)
    deg[deg == 0] = 1.0  # isolated vertices (shouldn't occur post close_chamber fix): no-op row
    vals = (1.0 / deg[row]).astype(np.float32)

    idx_a = torch.tensor(np.stack([row, col]), dtype=torch.long)
    A = torch.sparse_coo_tensor(idx_a, torch.from_numpy(vals), (n_verts, n_verts))

    diag = torch.arange(n_verts, dtype=torch.long)
    idx_i = torch.stack([diag, diag])
    I = torch.sparse_coo_tensor(idx_i, torch.ones(n_verts, dtype=torch.float32), (n_verts, n_verts))

    return (I - A).coalesce()


def laplacian_penalty(x: torch.Tensor, L: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Mean squared Laplacian magnitude of a batch of meshes -- penalizes each
    vertex for sitting away from its neighbors' average, independent of the
    mesh's absolute position/error. x: (B, T, V, 3). L: sparse (V, V).

    mask: optional (V,) float tensor, 1.0 = penalize this vertex, 0.0 = don't
    (see build_smoothness_mask). Reduction is the mean over only the
    masked-in (vertex, batch, frame) entries, so the loss stays on a
    comparable scale regardless of how much of the mesh is masked out.
    """
    B, T, V, C = x.shape
    flat = x.reshape(B * T, V, C).permute(1, 0, 2).reshape(V, B * T * C)
    # torch.sparse.mm has no CUDA kernel for float16, so under mixed
    # precision (autocast) this would otherwise crash -- upcast just for
    # this matmul, then cast the result back.
    if flat.dtype == torch.float16:
        lap = torch.sparse.mm(L.float(), flat.float()).to(torch.float16)
    else:
        lap = torch.sparse.mm(L, flat)
    lap = lap.reshape(V, B * T, C).permute(1, 0, 2)  # (B*T, V, C)
    sq = (lap ** 2).sum(-1)  # (B*T, V)
    if mask is None:
        return sq.mean()
    sq = sq * mask.unsqueeze(0)
    denom = mask.sum() * sq.shape[0]
    return sq.sum() / denom.clamp(min=1.0)


def build_smoothness_mask(template_v: np.ndarray, L: torch.Tensor, percentile: float = 100.0) -> torch.Tensor:
    """
    Per-vertex mask (1.0 = apply the smoothness penalty there, 0.0 = skip),
    computed from the population TEMPLATE's own Laplacian magnitude -- not
    from any single reconstruction. Some regions of the real anatomy aren't
    smooth by construction (partition cut boundaries -- e.g. the aorta's open
    end -- valve annuli, etc.); penalizing the network for matching that real
    roughness would fight genuine anatomy instead of removing reconstruction
    noise. percentile=100 (default) masks nothing -- every vertex is
    penalized, matching behavior before this existed. percentile=90 excludes
    the roughest 10% of *template* vertices (a fixed, structural set -- this
    does not look at any particular subject's reconstruction).
    """
    if percentile >= 100.0:
        return None
    v = torch.as_tensor(template_v, dtype=torch.float32)
    lap = torch.sparse.mm(L, v)  # (V, 3)
    magnitude = lap.norm(dim=-1)  # (V,)
    threshold = torch.quantile(magnitude, percentile / 100.0)
    return (magnitude < threshold).float()


class ConvergenceRamp:
    """
    Tracks a metric's plateau (same idea as ReduceLROnPlateau's patience
    counter) and derives a value that sits at `start` until the tracked
    metric has improved by less than `min_delta` (relative to its
    best-so-far value) for `patience` consecutive epochs, then ramps
    linearly from `start` to `target` over the following `ramp_epochs`
    epochs. Latches once triggered -- doesn't un-trigger if the metric later
    improves again. ramp_epochs<=0 disables all of this: value() always
    returns `target`, matching plain constant-weight behavior.
    """

    def __init__(self, start: float, target: float, ramp_epochs: int, patience: int, min_delta: float):
        self.start = start
        self.target = target
        self.ramp_epochs = ramp_epochs
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._plateau_epochs = 0
        self._converged_at_epoch = None

    def update(self, value: float, epoch: int):
        if self.ramp_epochs <= 0 or self._converged_at_epoch is not None:
            return
        improvement = (self._best - value) / max(self._best, 1e-8)
        if value < self._best:
            self._best = value
        self._plateau_epochs = self._plateau_epochs + 1 if improvement < self.min_delta else 0
        if self._plateau_epochs >= self.patience:
            self._converged_at_epoch = epoch

    def is_complete(self, epoch: int) -> bool:
        """True once value(epoch) has reached `target` (always, if the ramp is disabled)."""
        if self.ramp_epochs <= 0:
            return True
        return self._converged_at_epoch is not None and epoch - self._converged_at_epoch >= self.ramp_epochs

    def value(self, epoch: int) -> float:
        if self.ramp_epochs <= 0:
            return self.target
        if self._converged_at_epoch is None:
            return self.start
        progress = min(1.0, (epoch - self._converged_at_epoch) / self.ramp_epochs)
        return self.start + (self.target - self.start) * progress


class CoMA_Lightning(pl.LightningModule):


    def __init__(self, 
                 model: AutoencoderTemporalSequence, 
                 loss_params: Namespace, 
                 optimizer_params: Namespace,
                 additional_params: Namespace,
                 mesh_template=None,
                 thickness_pairs=None
                ):

        '''
        :param model: PyTorch model.
        :param params: a Namespace with additional parameters
        
        Example:
        from Easydict import EasyDict
        
        loss_params = EasyDict({
          "reconstruction_c.weight": ,
          "reconstruction_s.weight": ,
          "regularization.weight":          
        })          
        '''

        super(CoMA_Lightning, self).__init__()
        self.model = model
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params
        self.additional_params = additional_params
        self.mesh_template = mesh_template 

        self.rec_loss = self.get_rec_loss()

        # Wall thickness term: squared difference between real and reconstructed epi-endo distances
        # over the (epi, endo) vertex pairs from build_wall_thickness_pairs. Non-persistent buffer:
        # follows the module's device, but isn't saved in checkpoints.
        thickness = getattr(self.loss_params, "thickness", None)
        self.w_thickness = getattr(thickness, "weight", 0.0) if thickness is not None else 0.0
        pairs = torch.as_tensor(thickness_pairs, dtype=torch.long) if thickness_pairs is not None else None
        self.register_buffer("thickness_pairs", pairs, persistent=False)
        if self.w_thickness > 0 and self.thickness_pairs is None:
            raise ValueError("loss.thickness.weight > 0 requires thickness_pairs")

        self.laplacian = None
        self.smooth_mask = None
        if mesh_template is not None:
            self.laplacian = build_laplacian(mesh_template.f, mesh_template.v.shape[0])
            self.smooth_mask = build_smoothness_mask(mesh_template.v, self.laplacian, self.smooth_mask_percentile)

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.kl_warmup_epochs = getattr(self.loss_params.regularization, "warmup_epochs", 5)
        self._kl_warning_count = 0

        # w_s ramps up once content (val_recon_loss_c) plateaus; w_smooth ramps
        # up once style (val_recon_loss_s) plateaus -- i.e. smoothing only
        # kicks in once reconstruction itself has stabilized, so it polishes
        # instead of fighting the network while it's still learning shape/motion.
        self._w_s_ramp = ConvergenceRamp(
            self.w_s_start, self.w_s, self.w_s_ramp_epochs, self.w_s_content_patience, self.w_s_content_min_delta,
        )
        self._w_smooth_ramp = ConvergenceRamp(
            self.w_smooth_start, self.w_smooth, self.w_smooth_ramp_epochs,
            self.w_smooth_style_patience, self.w_smooth_style_min_delta,
        )


    def get_rec_loss(self):

        self.w_s = self.loss_params.reconstruction_s.weight
        self.w_s_start = getattr(self.loss_params.reconstruction_s, "start_weight", self.w_s)
        self.w_s_ramp_epochs = getattr(self.loss_params.reconstruction_s, "ramp_epochs", 0)
        self.w_s_content_patience = getattr(self.loss_params.reconstruction_s, "content_patience", 5)
        self.w_s_content_min_delta = getattr(self.loss_params.reconstruction_s, "content_min_delta", 0.02)
        # Splitting recon_loss_s into translation + shape (see
        # translation_shape_split_loss): defaults of 1/1 reproduce the plain
        # MSE exactly; raise translation_weight to prioritize getting the
        # per-frame rigid position right (found to dominate the raw error).
        self.w_translation = getattr(self.loss_params.reconstruction_s, "translation_weight", 1.0)
        self.w_shape = getattr(self.loss_params.reconstruction_s, "shape_weight", 1.0)
        self.w_kl = self.loss_params.regularization.weight

        smoothness = getattr(self.loss_params, "smoothness", None)
        self.w_smooth = smoothness.weight if smoothness is not None else 0.0
        self.w_smooth_start = getattr(smoothness, "start_weight", self.w_smooth) if smoothness is not None else 0.0
        self.w_smooth_ramp_epochs = getattr(smoothness, "ramp_epochs", 0) if smoothness is not None else 0
        self.w_smooth_style_patience = getattr(smoothness, "style_patience", 5) if smoothness is not None else 5
        self.w_smooth_style_min_delta = getattr(smoothness, "style_min_delta", 0.02) if smoothness is not None else 0.02
        self.smooth_mask_percentile = getattr(smoothness, "mask_percentile", 100.0) if smoothness is not None else 100.0

        return losses_menu[self.loss_params.reconstruction_c.type.lower()]


    def _recon_loss_s_split(self, s_t, shat_t):
        loss_translation, loss_shape = translation_shape_split_loss(self.rec_loss, s_t, shat_t)
        recon_loss_s = self.w_translation * loss_translation + self.w_shape * loss_shape
        return recon_loss_s, loss_translation, loss_shape


    def KL_div(self, mu, log_var):
        log_var_has_nonfinite = not torch.isfinite(log_var).all().item()
        log_var_was_clamped = ((log_var < LOG_VAR_MIN).any() or (log_var > LOG_VAR_MAX).any()).item()
        log_var = clamp_log_var(log_var)

        kld = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var)
        kld_has_nonfinite = not torch.isfinite(kld).all().item()

        if (log_var_has_nonfinite or log_var_was_clamped or kld_has_nonfinite) and self._kl_warning_count < 5:
            logger.warning(
                "KL stabilization applied: log_var_range=(%.2f, %.2f), clamped_to=(%.1f, %.1f), nonfinite_log_var=%s, nonfinite_kld=%s",
                self._finite_min(log_var),
                self._finite_max(log_var),
                LOG_VAR_MIN,
                LOG_VAR_MAX,
                bool(log_var_has_nonfinite),
                bool(kld_has_nonfinite),
            )
            self._kl_warning_count += 1

        kld = torch.nan_to_num(kld, nan=0.0, posinf=1e6, neginf=0.0)
        return kld.mean(dim=1).mean(dim=0)

    @staticmethod
    def _finite_min(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return finite.min().detach().cpu().item() if finite.numel() else float("nan")

    @staticmethod
    def _finite_max(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return finite.max().detach().cpu().item() if finite.numel() else float("nan")

    def _is_variational(self):
        return bool(getattr(self.model, "is_variational", False))

    def _effective_w_kl(self):
        if not self._is_variational():
            return 0.0
        if self.current_epoch <= self.kl_warmup_epochs:
            return 0.0
        return self.w_kl

    def _effective_w_s(self):
        return self._w_s_ramp.value(self.current_epoch)

    def _wall_thickness_terms(self, s_t, shat_t):
        """(loss, sum of absolute thickness errors, number of distances); zeros without pairs."""
        if self.thickness_pairs is None:
            zero = torch.zeros((), device=s_t.device, dtype=s_t.dtype)
            return zero, zero, zero
        error = wall_thickness_distances(shat_t, self.thickness_pairs) - wall_thickness_distances(s_t, self.thickness_pairs)
        return (error ** 2).mean(), error.abs().sum(), torch.tensor(float(error.numel()), device=s_t.device)

    def loss_weights_are_final(self) -> bool:
        """
        True once every scheduled loss weight (w_s and w_smooth ramps, KL warm-up) has reached its
        final value at the current epoch. Before that, val_loss is computed with weights that are
        still changing, so it isn't comparable across epochs: early stopping and checkpointing
        (see RampAwareEarlyStopping / MLflowArtifactCheckpoint) wait until this is True.
        """
        epoch = self.current_epoch
        kl_final = not self._is_variational() or epoch > self.kl_warmup_epochs
        return self._w_s_ramp.is_complete(epoch) and self._w_smooth_ramp.is_complete(epoch) and kl_final

    def _effective_w_smooth(self):
        return self._w_smooth_ramp.value(self.current_epoch)

    def _update_ramps(self, val_recon_loss_c: float, val_recon_loss_s: float):
        """Called once per (non-sanity-check) validation epoch. w_s watches
        content (recon_loss_c); w_smooth watches style (recon_loss_s) -- so
        smoothing only ramps up once the network has actually learned to
        reconstruct the motion, not while it's still fighting for accuracy."""
        was_s_converged = self._w_s_ramp._converged_at_epoch is not None
        was_smooth_converged = self._w_smooth_ramp._converged_at_epoch is not None

        self._w_s_ramp.update(val_recon_loss_c, self.current_epoch)
        self._w_smooth_ramp.update(val_recon_loss_s, self.current_epoch)

        if not was_s_converged and self._w_s_ramp._converged_at_epoch is not None:
            logger.info(
                "Content (recon_loss_c) considered converged at epoch %d -- starting w_s ramp %.3f -> %.3f over %d epochs.",
                self.current_epoch, self._w_s_ramp.start, self._w_s_ramp.target, self._w_s_ramp.ramp_epochs,
            )
        if not was_smooth_converged and self._w_smooth_ramp._converged_at_epoch is not None:
            logger.info(
                "Style (recon_loss_s) considered converged at epoch %d -- starting w_smooth ramp %.3f -> %.3f over %d epochs.",
                self.current_epoch, self._w_smooth_ramp.start, self._w_smooth_ramp.target, self._w_smooth_ramp.ramp_epochs,
            )

    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)
    
    
    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device 
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for i, _ in enumerate(self.model.encoder.matrices['downsample']):
            self.model.encoder.matrices['downsample'][i] = self.model.encoder.matrices['downsample'][i].to(self.device)            
            # self.model.encoder.matrices['adjacency_matrices'][i] = self.model.encoder.matrices['adjacency_matrices'][i].to(self.device)
            self.model.decoder.matrices['upsample'][i] = self.model.decoder.matrices['upsample'][i].to(self.device)

        for i, _ in enumerate(self.model.encoder.matrices['A_edge_index']):
            self.model.encoder.matrices['A_edge_index'][i] = self.model.encoder.matrices['A_edge_index'][i].to(self.device)
            self.model.encoder.matrices['A_norm'][i] = self.model.encoder.matrices['A_norm'][i].to(self.device)

        if self.laplacian is not None:
            self.laplacian = self.laplacian.to(self.device)
        if self.smooth_mask is not None:
            self.smooth_mask = self.smooth_mask.to(self.device)

        # embed()
        # self.precision_to_set  = self.trainer.precision
        # self.batch_size_to_set = self.trainer.train_dataloader.batch_size
        
        
    def on_train_epoch_start(self):
        self.model.set_mode("training")

        
    def training_step(self, batch, batch_idx):

        s_t, time_avg_s = batch["s_t"], batch["time_avg_s"]
        bottleneck, time_avg_shat, shat_t = self(s_t)

        # print(f"{bottleneck.mu.mean()=}\n\n{bottleneck.log_var.mean()}\n\n{time_avg_shat.mean()=}\n\n{shat_t.mean()=}")
        # print(f"{bottleneck.mu.mean()=}\n\n{time_avg_shat.mean()=}\n\n{shat_t.mean()=}")
        
        recon_loss_c = self.rec_loss(time_avg_s, time_avg_shat)
        recon_loss_s, recon_loss_s_translation, recon_loss_s_shape = self._recon_loss_s_split(s_t, shat_t)

        effective_w_s = self._effective_w_s()
        self.log("w_s_effective", effective_w_s, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        recon_loss = recon_loss_c + effective_w_s * recon_loss_s

        if self._is_variational():
            bottleneck = self.model.decoder._partition_z(bottleneck["mu"], bottleneck["log_var"])            
            self.mu_c = bottleneck["mu_c"]
            self.log_var_c = bottleneck["log_var_c"]
            self.mu_s = bottleneck["mu_s"]
            self.log_var_s = bottleneck["log_var_s"]            
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)
        else:
            kld_loss_c = kld_loss_s = torch.zeros_like(recon_loss)

        kld_loss = kld_loss_c + kld_loss_s

        smooth_loss = laplacian_penalty(shat_t, self.laplacian, self.smooth_mask) if self.laplacian is not None else torch.zeros_like(recon_loss)

        effective_w_smooth = self._effective_w_smooth()
        self.log("w_smooth_effective", effective_w_smooth, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        thickness_loss, _, _ = self._wall_thickness_terms(s_t, shat_t)
        train_loss = recon_loss + self._effective_w_kl() * kld_loss + effective_w_smooth * smooth_loss \
                     + self.w_thickness * thickness_loss
        
        # print(f"{recon_loss_c.item()=} + {recon_loss_s.item()=}")
        # print(f"{train_loss.item()=} = {recon_loss.item()=} + {self.w_kl} * {kld_loss.item()=}")

        loss_dict = {
           "training_kld_loss": kld_loss,
           "training_recon_loss": recon_loss,
           "training_recon_loss_c": recon_loss_c,
           "training_recon_loss_s": recon_loss_s,
           "training_recon_loss_s_translation": recon_loss_s_translation,
           "training_recon_loss_s_shape": recon_loss_s_shape,
           "training_smooth_loss": smooth_loss,
           "training_thickness_loss": thickness_loss,
           "loss": train_loss
        }

        self.train_outputs.append({k: v.detach().cpu() for k, v in loss_dict.items()})

        # self.train_outputs.append(loss_dict)
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict

    def on_train_epoch_end(self):

        # Aggregate metrics from each batch

        avg_kld_loss = torch.stack([x["training_kld_loss"] for x in self.train_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["training_recon_loss_c"] for x in self.train_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["training_recon_loss_s"] for x in self.train_outputs]).mean()
        avg_recon_loss_s_translation = torch.stack([x["training_recon_loss_s_translation"] for x in self.train_outputs]).mean()
        avg_recon_loss_s_shape = torch.stack([x["training_recon_loss_s_shape"] for x in self.train_outputs]).mean()
        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in self.train_outputs]).mean()
        avg_smooth_loss = torch.stack([x["training_smooth_loss"] for x in self.train_outputs]).mean()
        avg_thickness_loss = torch.stack([x["training_thickness_loss"] for x in self.train_outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()

        self.log_dict({
            "training_kld_loss": avg_kld_loss,
            "training_recon_loss": avg_recon_loss,
            "training_recon_loss_c": avg_recon_loss_c,
            "training_recon_loss_s": avg_recon_loss_s,
            "training_recon_loss_s_translation": avg_recon_loss_s_translation,
            "training_recon_loss_s_shape": avg_recon_loss_s_shape,
            "training_smooth_loss": avg_smooth_loss,
            "training_thickness_loss": avg_thickness_loss,
            "training_loss": avg_loss
          },
          # on_epoch=False,
          on_epoch=True,
          prog_bar=True,
          logger=True,
        )

        self.train_outputs.clear()
        # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#logging
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
    def on_validation_epoch_start(self):
        self.model.set_mode("testing")
        

    # def on_training_epoch_start(self):
    #     
    #     if self.current_epoch < 1:
    #         self.batch_size = 32
    #         self.precision = 32
    #         self.trainer.precision = 32
    #         self.trainer.train_dataloader.batch_size = self.batch_size  # Update batch size
    #     else:
    #         print(f"Chaning batch size (to {self.precision_to_set}) and batch_size (to {self.batch_size_to_set})")
    #         self.batch_size = self.batch_size_to_set
    #         self.precision = self.precision_to_set
    #         self.trainer.precision = self.precision
    #         self.trainer.train_dataloader.batch_size = self.batch_size  # Update batch size

    
    def _shared_eval_step(self, batch, batch_idx):

        '''
        The validation and testing steps are similar, only the names of the logged quantities differ.
        The common part is performed here.
        '''

        s_t = batch["s_t"]        
        time_avg_s = batch["time_avg_s"]        
        mse_mesh_to_tmp_mean = batch["d_content"]        
        mse_mesh_to_pop_mean = batch["d_style"]
        
        bottleneck, time_avg_s_hat, shat_t = self(s_t)

        # content
        recon_loss_c = self.rec_loss(time_avg_s, time_avg_s_hat)
        recon_loss_s, _, _ = self._recon_loss_s_split(s_t, shat_t)
        recon_loss = recon_loss_c + self._effective_w_s() * recon_loss_s

        smooth_loss = laplacian_penalty(shat_t, self.laplacian, self.smooth_mask) if self.laplacian is not None else torch.zeros_like(recon_loss)

        if self._is_variational():
            
            bottleneck = self.model.decoder._partition_z(bottleneck["mu"], bottleneck["log_var"])
            
            self.mu_c = bottleneck["mu_c"]
            self.log_var_c = bottleneck["log_var_c"]
            self.mu_s = bottleneck["mu_s"]
            self.log_var_s = bottleneck["log_var_s"]
            
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)
            kld_loss = kld_loss_c + kld_loss_s
            loss = recon_loss + self._effective_w_kl() * kld_loss + self._effective_w_smooth() * smooth_loss
        else:
            kld_loss = kld_loss_c = kld_loss_s = torch.zeros_like(recon_loss)
            loss = recon_loss + self._effective_w_smooth() * smooth_loss

        thickness_loss, thickness_abs_err_sum, thickness_count = self._wall_thickness_terms(s_t, shat_t)
        loss = loss + self.w_thickness * thickness_loss

        recon_error = mse(s_t, shat_t)

        # N-dimensional
        rec_ratio_to_pop_mean_c = safe_mean_ratio(mse(time_avg_s, time_avg_s_hat), mse(time_avg_s))
        rec_ratio_to_time_mean = safe_mean_ratio(recon_error, mse_mesh_to_tmp_mean)
        # Sums for the pooled version (ratio of totals over the whole epoch, computed at epoch end):
        # unlike the mean of per-frame ratios above, it isn't dominated by frames close to the
        # static frame, whose tiny denominators blow the ratio up.
        # Mean Euclidean distance between reconstructed and real vertices (square root taken per
        # vertex, before averaging), over every frame of the sequence: summed here, averaged at epoch end.
        vertex_dev = torch.linalg.vector_norm(s_t - shat_t, dim=-1)
        self._pooled_sums = {
            "rec_err_sum": recon_error.sum(), "dev_static_sum": mse_mesh_to_tmp_mean.sum(),
            "vertex_dev_sum": vertex_dev.sum(), "vertex_count": torch.tensor(float(vertex_dev.numel())),
            "thickness_loss": thickness_loss,
            "thickness_abs_err_sum": thickness_abs_err_sum, "thickness_count": thickness_count,
        }

        # Only computable when template_mesh was provided to the dataset
        if mse_mesh_to_pop_mean is not None:
            rec_ratio_to_pop_mean = safe_mean_ratio(recon_error, mse_mesh_to_pop_mean)
        else:
            rec_ratio_to_pop_mean = torch.tensor(float("nan"))
               
        return loss,\
               recon_loss, recon_loss_c, recon_loss_s, smooth_loss,\
               kld_loss_c, kld_loss_s, kld_loss,\
               rec_ratio_to_time_mean,\
               rec_ratio_to_pop_mean,\
               rec_ratio_to_pop_mean_c

    
    def validation_step(self, batch, batch_idx):

        # loss_dict = self._shared_eval_step(batch, batch_idx)
        # loss_dict = { "val_"+k: v for k, v in loss_dict.items() }

        loss, recon_loss, recon_loss_c, recon_loss_s, smooth_loss, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean, rec_ratio_to_pop_mean_c = self._shared_eval_step(batch, batch_idx)

        loss_dict = {
          "val_kld_loss": kld_loss,
          "val_recon_loss": recon_loss,
          "val_recon_loss_c": recon_loss_c,
          "val_recon_loss_s": recon_loss_s,
          "val_smooth_loss": smooth_loss,
          "val_loss": loss,
          "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "val_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }
        
        self.val_outputs.append({k: v.detach().cpu() for k, v in {**loss_dict, **self._pooled_sums}.items()})
        self.log_dict(loss_dict)
        return loss_dict

    
    def on_validation_epoch_end(self):

        #TODO: iterate over keys of the elements of `outputs`

        avg_kld_loss = torch.stack([x["val_kld_loss"] for x in self.val_outputs]).mean()
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in self.val_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["val_recon_loss_c"] for x in self.val_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["val_recon_loss_s"] for x in self.val_outputs]).mean()
        avg_smooth_loss = torch.stack([x["val_smooth_loss"] for x in self.val_outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["val_rec_ratio_to_time_mean"] for x in self.val_outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["val_rec_ratio_to_pop_mean"] for x in self.val_outputs]).mean()
        rec_ratio_to_pop_mean_c = torch.stack([x["val_rec_ratio_to_pop_mean_c"] for x in self.val_outputs]).mean()
        rec_ratio_to_time_mean_pooled = pooled_ratio(self.val_outputs)
        mean_vertex_dev = pooled_mean_vertex_dev(self.val_outputs)
        thickness_loss, thickness_mae = aggregate_thickness(self.val_outputs)

        if not self.trainer.sanity_checking:
            self._update_ramps(avg_recon_loss_c.item(), avg_recon_loss_s.item())
            weights_final = self.loss_weights_are_final()
            if weights_final and not getattr(self, "_logged_weights_final", False):
                logger.info("Loss weights reached their final values at epoch %d -- early stopping and "
                            "checkpointing now monitor val_loss.", self.current_epoch)
                self._logged_weights_final = True
            self.log("loss_weights_final", float(weights_final), on_step=False, on_epoch=True, logger=True)

        self.log_dict({
            "val_kld_loss": avg_kld_loss, 
            "val_recon_loss": avg_recon_loss,
            "val_recon_loss_c": avg_recon_loss_c,
            "val_recon_loss_s": avg_recon_loss_s,
            "val_smooth_loss": avg_smooth_loss,
            "val_loss": avg_loss,
            "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
            "val_rec_ratio_to_time_mean_pooled": rec_ratio_to_time_mean_pooled,
            "val_mean_vertex_dev": mean_vertex_dev,
            "val_thickness_loss": thickness_loss,
            "val_thickness_mae": thickness_mae,
            "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
            "val_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
          },
          on_epoch=True,
          prog_bar=True,
          logger=True
        )

        self.val_outputs.clear()


    def on_test_epoch_start(self):
        self.model.set_mode("testing")
        

    def test_step(self, batch, batch_idx):
                
        loss, recon_loss, recon_loss_c, recon_loss_s, smooth_loss, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean, rec_ratio_to_pop_mean_c = self._shared_eval_step(batch, batch_idx)

        loss_dict = {
          "test_kld_loss": kld_loss, 
          "test_recon_loss": recon_loss,
          "test_recon_loss_c": recon_loss_c,
          "test_recon_loss_s": recon_loss_s,
          "test_smooth_loss": smooth_loss,
          "test_loss": loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "test_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }

        self.test_outputs.append({k: v.detach().cpu() for k, v in {**loss_dict, **self._pooled_sums}.items()})
        self.log_dict(loss_dict)
        return loss_dict

    def on_test_epoch_end(self):

       # outputs =self.outputs
        avg_kld_loss = torch.stack([x["test_kld_loss"] for x in self.test_outputs]).mean()
        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in self.test_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["test_recon_loss_c"] for x in self.test_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["test_recon_loss_s"] for x in self.test_outputs]).mean()
        avg_smooth_loss = torch.stack([x["test_smooth_loss"] for x in self.test_outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["test_rec_ratio_to_time_mean"] for x in self.test_outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["test_rec_ratio_to_pop_mean"] for x in self.test_outputs]).mean()
        rec_ratio_to_pop_mean_c = torch.stack([x["test_rec_ratio_to_pop_mean_c"] for x in self.test_outputs]).mean()
        rec_ratio_to_time_mean_pooled = pooled_ratio(self.test_outputs)
        mean_vertex_dev = pooled_mean_vertex_dev(self.test_outputs)
        thickness_loss, thickness_mae = aggregate_thickness(self.test_outputs)
        
        loss_dict = {
          "test_kld_loss": avg_kld_loss, 
          "test_recon_loss": avg_recon_loss,
          "test_recon_loss_c": avg_recon_loss_c,
          "test_recon_loss_s": avg_recon_loss_s,
          "test_smooth_loss": avg_smooth_loss,
          "test_loss": avg_loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_time_mean_pooled": rec_ratio_to_time_mean_pooled,
          "test_mean_vertex_dev": mean_vertex_dev,
          "test_thickness_loss": thickness_loss,
          "test_thickness_mae": thickness_mae,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "test_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }

        self.log_dict(loss_dict)        
        self.test_outputs.clear()

        
    def predict_step(self, batch, batch_idx):

        s_t = batch["s_t"]        
        time_avg_s = batch["time_avg_s"]        
        mse_mesh_to_tmp_mean = batch["d_content"]        
        mse_mesh_to_pop_mean = batch["d_style"]
        
        bottleneck, time_avg_s_hat, s_hat_t = self(s_t)


        ### IMAGES OF TEMPORAL AVERAGE
        if self.additional_params.dataset.preprocessing.center_around_mean:
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s[0].cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"temporal_avg_mesh_{batch_idx}_orig.png")
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s_hat[0].cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"temporal_avg_mesh_{batch_idx}_rec.png")
        else:
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s[0], self.mesh_template.faces,
                                                       f"temporal_avg_mesh_{batch_idx}_orig.png")
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s_hat[0], self.mesh_template.faces,
                                                       f"temporal_avg_mesh_{batch_idx}_rec.png")

        merge_pngs_horizontally(f"temporal_avg_mesh_{batch_idx}_orig.png", f"temporal_avg_mesh_{batch_idx}_rec.png", f"temporal_avg_mesh_{batch_idx}.png")

        self.logger.experiment.log_artifact(
            local_path=f"temporal_avg_mesh_{batch_idx}.png",
            artifact_path="images", run_id=self.logger.run_id
        )

        ### ANIMATIONS OF MOVING MESH
        if self.additional_params.dataset.preprocessing.center_around_mean:
            SyntheticMeshPopulation._generate_gif(s_t.cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"moving_mesh_{batch_idx}_orig.gif")
            SyntheticMeshPopulation._generate_gif(s_hat_t.cpu() + self.mesh_template.vertices, self.mesh_template.faces, f"moving_mesh_{batch_idx}_rec.gif")
        else:
            SyntheticMeshPopulation._generate_gif(s_t, self.mesh_template.faces, f"moving_mesh_{batch_idx}_orig.gif")
            SyntheticMeshPopulation._generate_gif(s_hat_t, self.mesh_template.faces, f"moving_mesh_{batch_idx}_rec.gif")

        merge_gifs_horizontally(f"moving_mesh_{batch_idx}_orig.gif", f"moving_mesh_{batch_idx}_rec.gif", f"moving_mesh_{batch_idx}.gif")
        self.logger.experiment.log_artifact(
            local_path=f"moving_mesh_{batch_idx}.gif",
            artifact_path="animations", run_id=self.logger.run_id
        )
        
        return 1 # to prevent warning messages


    # TODO: Select optimizer from menu (dict)

    def configure_optimizers(self):

        algorithm = self.optimizer_params.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.optimizer_params.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


def merge_pngs_horizontally(png1, png2, output_png):
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    #Read the two images
    image1 = Image.open(png1)
    image2 = Image.open(png2)
    #resize, first image
    image1_size = image1.size
    # image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(output_png, "PNG")

    
def merge_gifs_horizontally(gif_file1, gif_file2, output_file):

    #Create reader object for the gif
    gif1 = imageio.get_reader(gif_file1)
    gif2 = imageio.get_reader(gif_file2)

    #Create writer object
    new_gif = imageio.get_writer(output_file)

    for frame_number in range(gif1.get_length()):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()
