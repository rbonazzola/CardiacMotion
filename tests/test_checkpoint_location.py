'''
Checkpoints must be written to the MLflow run's artifact store (<artifact_uri>/checkpoints/),
with a best_model.ckpt symlink to the best one -- not to ./<experiment_id>/<run_id>/checkpoints/
under the cwd, which is what Lightning does when the tracking URI has no "file:" prefix.
'''
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "cardiac_motion")
from utils.lightning_helpers import MLflowArtifactCheckpoint


class _TinyModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.layer(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader():
    torch.manual_seed(0)
    x = torch.randn(32, 2)
    return DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=8)


def _fit(tmp_path, monkeypatch, save_top_k=1, max_epochs=3):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir()

    # Plain absolute path, no "file:" prefix -- as produced by prepare_mlflow_config
    mlflow_logger = MLFlowLogger(tracking_uri=str(tracking_dir), experiment_name="test_ckpt")
    checkpoint = MLflowArtifactCheckpoint(save_top_k=save_top_k)
    trainer = pl.Trainer(
        max_epochs=max_epochs, logger=mlflow_logger, callbacks=[checkpoint],
        accelerator="cpu", enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(_TinyModule(), _loader(), _loader())
    run = mlflow_logger.experiment.get_run(mlflow_logger.run_id)
    return cwd, tracking_dir, run, checkpoint, trainer


def test_checkpoints_go_to_run_artifacts(tmp_path, monkeypatch):
    cwd, tracking_dir, run, checkpoint, _ = _fit(tmp_path, monkeypatch)

    expected_dir = tracking_dir / run.info.experiment_id / run.info.run_id / "artifacts" / "checkpoints"
    assert os.path.dirname(checkpoint.best_model_path) == str(expected_dir)
    assert os.listdir(cwd) == [], f"stray files in cwd: {os.listdir(cwd)}"


def test_best_model_symlink_points_to_best(tmp_path, monkeypatch):
    _, _, _, checkpoint, _ = _fit(tmp_path, monkeypatch, save_top_k=2)

    ckpt_dir = os.path.dirname(checkpoint.best_model_path)
    link = os.path.join(ckpt_dir, "best_model.ckpt")
    assert os.path.islink(link)
    assert os.readlink(link) == os.path.basename(checkpoint.best_model_path)  # relative
    assert os.path.basename(checkpoint.best_model_path).startswith("epoch")
    assert "__valloss_" in checkpoint.best_model_path
    ckpts = [f for f in os.listdir(ckpt_dir) if f != "best_model.ckpt"]
    assert len(ckpts) == 2


def test_best_checkpoint_loads_for_test(tmp_path, monkeypatch):
    _, _, _, _, trainer = _fit(tmp_path, monkeypatch)
    trainer.validate(ckpt_path="best", dataloaders=_loader(), verbose=False)
