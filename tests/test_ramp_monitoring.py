'''
Early stopping and checkpoint selection must wait until the loss weights (w_s / w_smooth ramps,
KL warm-up) have reached their final values: while w_s ramps up, val_loss rises mechanically and
plain EarlyStopping would stop training (and keep a pre-ramp "best" model).
'''
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "cardiac_motion")
from lightning_modules.ComaLightningModule import ConvergenceRamp
from utils.lightning_helpers import MLflowArtifactCheckpoint, RampAwareEarlyStopping

# val_loss per epoch: low while the weights are ramping (epochs 0-5), then on the final objective
VAL_LOSSES = [1.0, 0.5, 0.6, 0.7, 0.8, 0.9] + [3.0, 2.0, 2.5, 2.6, 2.7, 2.8, 2.9]
FINAL_FROM = 6


class _ScheduledModule(pl.LightningModule):

    def __init__(self, final_from=FINAL_FROM):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.final_from = final_from

    def loss_weights_are_final(self):
        return self.current_epoch >= self.final_from

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", torch.tensor(VAL_LOSSES[min(self.current_epoch, len(VAL_LOSSES) - 1)]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def _loader():
    return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=4)


def _fit(tmp_path, early_stopping, module=None, max_epochs=len(VAL_LOSSES)):
    checkpoint = MLflowArtifactCheckpoint(save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=max_epochs, logger=False, callbacks=[early_stopping, checkpoint], default_root_dir=tmp_path,
        accelerator="cpu", enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module or _ScheduledModule(), _loader(), _loader())
    return trainer, checkpoint


def test_plain_early_stopping_stops_during_ramp(tmp_path):
    # The problem being fixed: val_loss rises after epoch 1 while weights ramp -> stops at epoch 3
    trainer, _ = _fit(tmp_path, EarlyStopping(monitor="val_loss", mode="min", patience=2))
    assert trainer.current_epoch == 4  # stopped after epoch 3


def test_early_stopping_waits_for_final_weights(tmp_path):
    early_stopping = RampAwareEarlyStopping(monitor="val_loss", mode="min", patience=2)
    trainer, _ = _fit(tmp_path, early_stopping)
    # best on the final objective is 2.0 (epoch 7); stops after 2 epochs without improvement (epoch 9)
    assert early_stopping.best_score.item() == 2.0
    assert early_stopping.stopped_epoch == 9


def test_best_checkpoint_selected_on_final_objective(tmp_path):
    _, checkpoint = _fit(tmp_path, RampAwareEarlyStopping(monitor="val_loss", mode="min", patience=2))
    # not epoch 1 (val_loss 0.5, computed with ramping weights)
    assert os.path.basename(checkpoint.best_model_path) == "epoch7__valloss_2.0000.ckpt"
    assert checkpoint.best_model_score.item() == 2.0


def test_fallback_checkpoint_when_weights_never_final(tmp_path):
    _, checkpoint = _fit(tmp_path, RampAwareEarlyStopping(monitor="val_loss", mode="min", patience=2),
                         module=_ScheduledModule(final_from=100), max_epochs=3)
    assert checkpoint.best_model_path and os.path.exists(checkpoint.best_model_path)
    link = os.path.join(os.path.dirname(checkpoint.best_model_path), "best_model.ckpt")
    assert os.readlink(link) == os.path.basename(checkpoint.best_model_path)


def test_convergence_ramp_is_complete():
    assert ConvergenceRamp(0.1, 1.0, ramp_epochs=0, patience=5, min_delta=0.02).is_complete(0)

    ramp = ConvergenceRamp(0.1, 1.0, ramp_epochs=20, patience=5, min_delta=0.02)
    assert not ramp.is_complete(100)  # never converged
    ramp._converged_at_epoch = 61
    assert not ramp.is_complete(80) and ramp.value(80) < 1.0
    assert ramp.is_complete(81) and ramp.value(81) == 1.0


def test_coma_lightning_loss_weights_are_final():
    from test_training import build_model, load_fixture, make_lit

    lit = make_lit(build_model(*load_fixture()))
    assert lit.loss_weights_are_final()  # no ramps configured

    lit._w_s_ramp = ConvergenceRamp(0.1, 1.0, ramp_epochs=20, patience=5, min_delta=0.02)
    assert not lit.loss_weights_are_final()  # waiting for content to converge
    lit._w_s_ramp._converged_at_epoch = -10  # current_epoch is 0: ramp half-way
    assert not lit.loss_weights_are_final()
    lit._w_s_ramp._converged_at_epoch = -20
    assert lit.loss_weights_are_final()
