import pytorch_lightning as pl
import torch

class CoMA(pl.LightningModule):

    def __init__(self, model, params):
        super(CoMA, self).__init__()
        self.model = model
        self.params = params

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        out = self(batch, mode="training")  #
        train_loss = loss_function(out, batch)

        # Why not just passing
        self.logger.experiment.log({
            key: val.item() for key, val in train_loss.items()
        })

        return losses

    def training_epoch_end(self, outputs):
        # Aggregate metrics from each batch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"train_loss": avg_loss})
        pass

    def validation_step(self, batch, batch_idx):
        out = self(batch, mode="training")  #
        train_loss = loss_function(out, batch)

        # Why not just passing train_loss here?
        self.logger.experiment.log({
            key: val.item() for key, val in train_loss.items()
        })

    def validation_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"val_loss": avg_loss})
        pass

    # def test_step(self):

    def test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"val_loss": avg_loss})
        pass

    # TODO: Select optimizer from menu (dict)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
        return optimizer