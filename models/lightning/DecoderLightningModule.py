import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import Namespace

from typing import List, Mapping
from IPython import embed # uncomment for debugging
from models.Model4D import  DecoderTemporalSequence
# from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)


class TemporalDecoderLightning(pl.LightningModule):

    def __init__(self, model: DecoderTemporalSequence, params: Namespace):

        """
        :param model: provide the PyTorch model.
        :param params: a Namespace with additional parameters
        """

        super(TemporalDecoderLightning, self).__init__()
        self.model = model
        self.params = params

        self.rec_loss = self.get_rec_loss()


    def get_rec_loss(self):

        self.w_s = self.params.loss.reconstruction_s.weight
        # self.w_kl = self.params.loss.regularization.weight
        return losses_menu[self.params.loss.reconstruction_c.type.lower()]


    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)


    ########### COMMON BEHAVIOR
    def _unpack_data_from_batch(self, batch):

        s_avg, s_t, z_c, z_s = (batch[k] for k in ["s_avg", "s_t", "z_c", "z_s"])
        z = z_c + z_s
        z = torch.stack(z).transpose(0, 1).type_as(s_t)  # to get N_batches x latent_dim
        return s_avg, s_t, z


    def _shared_step(self, batch, batch_idx):

        s_avg, s_t, z = self._unpack_data_from_batch(batch)
        z = {"mu": z, "log_var": None}
        s_avg_hat, shat_t = self(z)

        recon_loss_c = self.rec_loss(s_avg, s_avg_hat)
        recon_loss_s = self.rec_loss(s_t, shat_t)
        recon_loss = recon_loss_c + self.w_s * recon_loss_s

        loss_dict = {
            "recon_loss_c": recon_loss_c,
            "recon_loss_s": recon_loss_s,
            "loss": recon_loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict


    def _average_over_batches(self, outputs: List[Mapping[str, torch.Tensor]], prefix: str = ""):

        keys = outputs[0].keys()
        loss_dict = {}
        for k in keys:
            avg_loss = torch.stack([x[k] for x in outputs]).mean().detach()
            loss_dict[prefix + k] = avg_loss
        return loss_dict


    ########### TRAINING
    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for matrix_type in ["upsample", "A_edge_index", "A_norm"]:
             for i, _ in enumerate(self.model.matrices["upsample"]):
                self.model.matrices[matrix_type][i] = self.model.matrices[matrix_type][i].to(self.device)

    def on_train_epoch_start(self):
        self.model.set_mode("training")


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)


    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch
        loss_dict = self._average_over_batches(outputs, prefix="training_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    def _shared_eval_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx)


    ########### VALIDATION
    def on_validation_start(self):
        self.model.set_mode("testing")


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def validation_epoch_end(self, outputs):
        loss_dict = self._average_over_batches(outputs, prefix="val_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)


    ########### TESTING
    def on_test_start(self):
        self.model.set_mode("testing")


    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        loss_dict = self._average_over_batches(outputs, prefix="test_")
        self.log_dict(loss_dict, on_epoch=True, prog_bar=True, logger=True)