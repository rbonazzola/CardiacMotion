import logging
import math
import os
import time
from urllib.parse import urlparse, unquote

import torch
import pytorch_lightning as pl

from data.DataModules import CardiacMeshPopulationDM

from typing import Union, Mapping, Sequence

logger = logging.getLogger(__name__)

def get_lightning_module(config: Mapping, dm: pl.LightningDataModule):
    
    '''
      Arguments:
        - config:
        - dm:    
    '''

    # Initialize PyTorch model
    coma_args = get_coma_args(config, dm)

    if config.only_decoder:

        from models.Model4D import DecoderTemporalSequence, DECODER_C_ARGS, DECODER_S_ARGS
        from models.lightning.DecoderLightningModule import TemporalDecoderLightning

        dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
        dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}

        decoder = DecoderTemporalSequence(
            dec_c_config, dec_s_config,
            phase_embedding_method="exp",
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalDecoderLightning(decoder, config)

    elif config.only_encoder:

        from models.Model4D import EncoderTemporalSequence, ENCODER_ARGS
        from models.lightning.EncoderLightningModule import TemporalEncoderLightning

        enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}

        encoder = EncoderTemporalSequence(
            enc_config, z_aggr_function=config.network_architecture.z_aggr_function,
            n_timeframes=config.dataset.parameters.T
        )

        model = TemporalEncoderLightning(encoder, config)

    else:
        from models.Model4D import AutoencoderTemporalSequence
        from models.lightning.ComaLightningModule import CoMA_Lightning
        from models.Model4D import EncoderTemporalSequence, ENCODER_ARGS
        from models.Model4D import DecoderTemporalSequence, DECODER_C_ARGS, DECODER_S_ARGS
        
        enc_config = {k: v for k, v in coma_args.items() if k in ENCODER_ARGS}
        dec_c_config = {k: v for k,v in coma_args.items() if k in DECODER_C_ARGS}
        dec_s_config = {k: v for k,v in coma_args.items() if k in DECODER_S_ARGS}
        
        autoencoder = AutoencoderTemporalSequence(
            enc_config, 
            dec_c_config, 
            dec_s_config,
            z_aggr_function=coma_args.z_aggr_function,
            n_timeframes=coma_args.n_timeframes        
        )
        
        # Initialize PyTorch Lightning module
        model = CoMA_Lightning(autoencoder, config)

    return model


##########################################################################################

# PyTorch Lightning Callbacks

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichModelSummary

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)


class MLflowArtifactCheckpoint(ModelCheckpoint):
    '''
    ModelCheckpoint that saves into the MLflow run's artifact store, under
    <artifact_uri>/checkpoints/, and keeps a best_model.ckpt symlink pointing to the best one
    (same layout as delphi's MLFlowLogger.save_model).

    Without this, Lightning derives the directory from MLFlowLogger.save_dir, which is None
    unless the tracking URI starts with "file:" -- so checkpoints ended up in the cwd,
    as ./<experiment_id>/<run_id>/checkpoints/.
    '''

    BEST_MODEL_LINK = "best_model.ckpt"

    def __init__(self, monitor="val_loss", filename="epoch{epoch}__valloss_{val_loss:.4f}",
                 auto_insert_metric_name=False, **kwargs):
        super().__init__(monitor=monitor, filename=filename,
                         auto_insert_metric_name=auto_insert_metric_name, **kwargs)

    def setup(self, trainer, pl_module, stage):
        if self.dirpath is None:
            self.dirpath = self._mlflow_checkpoint_dir(trainer)
        super().setup(trainer, pl_module, stage)

    @staticmethod
    def _mlflow_checkpoint_dir(trainer):
        mlflow_logger = trainer.logger
        if not isinstance(mlflow_logger, MLFlowLogger):
            return None
        artifact_uri = mlflow_logger.experiment.get_run(mlflow_logger.run_id).info.artifact_uri
        parsed = urlparse(artifact_uri)
        if parsed.scheme not in ("", "file"):
            logger.warning("Artifact URI %s is not local; falling back to Lightning's default checkpoint dir.", artifact_uri)
            return None
        return os.path.join(unquote(parsed.path), "checkpoints")

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero and self.best_model_path:
            link = os.path.join(os.path.dirname(self.best_model_path), self.BEST_MODEL_LINK)
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.basename(self.best_model_path), link)  # relative symlink


model_checkpoint = MLflowArtifactCheckpoint(save_top_k=1)

rich_model_summary = RichModelSummary(max_depth=-1)

progress_bar = RichProgressBar(
  theme=RichProgressBarTheme(
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
  )
)

class MemoryUsageCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        print(f'Memory allocated: {torch.cuda.memory_allocated()} bytes')
        print(f'Memory cached: {torch.cuda.memory_reserved()} bytes')


class EpochMetricsTableCallback(pl.Callback):
    def __init__(self, max_rows=20):
        self.max_rows = max(1, int(max_rows))
        self.rows = []
        self.best_val_loss = None
        self.epoch_start_time = None
        self._last_logged_epoch = None
        self._last_printed_epoch = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._print_if_needed()
        self.epoch_start_time = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        if self._last_logged_epoch == epoch:
            return

        metrics = trainer.callback_metrics
        if not metrics:
            return

        val_loss = self._to_float(self._metric(metrics, "val_loss", "val_loss_epoch"))
        improved = False
        if val_loss is not None and (self.best_val_loss is None or val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            improved = True

        epoch_secs = None
        if self.epoch_start_time is not None:
            epoch_secs = time.perf_counter() - self.epoch_start_time

        row = (
            str(epoch),
            self._fmt(self._metric(metrics, "training_loss", "loss", "loss_epoch")),
            self._fmt(self._metric(metrics, "val_loss", "val_loss_epoch")),
            self._fmt(self._metric(metrics, "val_recon_loss", "val_recon_loss_epoch")),
            self._fmt(self._metric(metrics, "val_rec_ratio_to_time_mean", "val_rec_ratio_to_time_mean_epoch"), digits=3),
            self._fmt_lr(trainer),
            self._fmt_time(epoch_secs),
            "*" if improved else "",
        )
        self.rows.append(row)
        self.rows = self.rows[-self.max_rows:]
        self._last_logged_epoch = epoch

    def on_fit_end(self, trainer, pl_module):
        self._print_if_needed()

    @staticmethod
    def _metric(metrics, *keys):
        for key in keys:
            if key in metrics:
                return metrics[key]
        return None

    def _print_if_needed(self):
        if self._last_logged_epoch is None:
            return
        if self._last_printed_epoch == self._last_logged_epoch:
            return
        self._print_table()
        self._last_printed_epoch = self._last_logged_epoch

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().float().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value

    @classmethod
    def _fmt(cls, value, digits=4):
        value = cls._to_float(value)
        if value is None:
            return ""
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
            return f"{value:.2e}"
        return f"{value:.{digits}f}"

    @staticmethod
    def _fmt_lr(trainer):
        if not trainer.optimizers:
            return ""
        try:
            return f"{trainer.optimizers[0].param_groups[0]['lr']:.2e}"
        except (KeyError, IndexError):
            return ""

    @staticmethod
    def _fmt_time(seconds):
        if seconds is None:
            return ""
        if seconds >= 60:
            return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
        return f"{seconds:.1f}s"

    def _print_table(self):
        headers = (
            "ep",
            "train",
            "val",
            "recon",
            "ratio_t",
            "lr",
            "time",
            "best",
        )

        try:
            from rich import box
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            logger.info("Epoch metrics: %s", dict(zip(headers, self.rows[-1])))
            return

        table = Table(
            title="Epoch metrics",
            box=box.SIMPLE_HEAD,
            show_edge=False,
            header_style="bold",
        )
        table.add_column("ep", justify="right", style="cyan", width=4)
        table.add_column("train", justify="right", width=8)
        table.add_column("val", justify="right", width=8)
        table.add_column("recon", justify="right", width=8)
        table.add_column("ratio_t", justify="right", width=8)
        table.add_column("lr", justify="right", width=8)
        table.add_column("time", justify="right", width=6)
        table.add_column("best", justify="center", width=4)

        for row in reversed(self.rows):
            table.add_row(*row)

        Console().print(table)


class ModelCheckpointWithThreshold(ModelCheckpoint):

    def __init__(self, monitor, threshold, mode='min', *args, **kwargs):
        super().__init__(monitor=monitor, mode=mode, *args, **kwargs)
        self.threshold = threshold

    def _should_save_checkpoint(self, trainer):
        
        current = trainer.callback_metrics.get(self.monitor)
        
        if current is None:
            return False
        
        if self.mode == 'min':
            return current < self.threshold
        else:
            return current > self.threshold


def get_lightning_trainer(trainer_args: Mapping):

    '''
      trainer_args:
    '''
    
    # trainer
    trainer_kwargs = {
        "callbacks": [ early_stopping, model_checkpoint, rich_model_summary, progress_bar ],
        # "gpus": trainer_args.gpus,
        "devices": trainer_args.devices,
        "accelerator": trainer_args.accelerator,
        # "auto_select_gpus": trainer_args.auto_select_gpus,
        "min_epochs": trainer_args.min_epochs, "max_epochs": trainer_args.max_epochs,
        # "auto_scale_batch_size": trainer_args.auto_scale_batch_size,
        "logger": trainer_args.logger,
        "precision": trainer_args.precision,
        "overfit_batches": trainer_args.overfit_batches,
        "limit_test_batches": trainer_args.limit_test_batches
    }

    try:
        trainer = pl.Trainer(**trainer_kwargs)
    except:
        trainer_kwargs["gpus"] = None
        trainer = pl.Trainer(**trainer_kwargs)
    return trainer
