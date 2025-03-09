import torch
import pytorch_lightning as pl

from data.DataModules import CardiacMeshPopulationDM

from typing import Union, Mapping, Sequence

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
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import RichModelSummary

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=10)

model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1)

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
