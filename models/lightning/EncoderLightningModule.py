import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
import imageio
import numpy as np

from IPython import embed # uncomment for debugging
# from models.Model4D import  EncoderTemporalSequence
# from data.synthetic.SyntheticMeshPopulation import SyntheticMeshPopulation

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)


class CineComaEncoder(pl.LightningModule):

    def __init__(self, model, params):

        """
        :param model: provide the PyTorch model.
        :param params: a Namespace with additional parameters
        """

        super(CineComaEncoder, self).__init__()
        self.model = model
        self.params = params

        self.rec_loss = self.get_rec_loss()


    def get_rec_loss(self):

        self.w_s = self.params.loss.reconstruction_s.weight
        self.w_kl = self.params.loss.regularization.weight
        return losses_menu[self.params.loss.reconstruction_c.type.lower()]


    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for i, _ in enumerate(self.model.downsample_matrices):
            self.model.downsample_matrices[i] = self.model.downsample_matrices[i].to(self.device)
            # self.model.upsample_matrices[i] = self.model.upsample_matrices[i].to(self.device)
            self.model.adjacency_matrices[i] = self.model.adjacency_matrices[i].to(self.device)

        for i, _ in enumerate(self.model.A_edge_index):
            self.model.A_edge_index[i] = self.model.A_edge_index[i].to(self.device)
            self.model.A_norm[i] = self.model.A_norm[i].to(self.device)


    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)


    def on_train_epoch_start(self):
        self.model.set_mode("training")


    def training_step(self, batch, batch_idx):

        # data, ids = batch
        # TODO: change dataset/datamodule accordingly
        s_t, z_c, z_s = batch["s_t"], batch["z_c"], batch["z_s"]

        z = z_c + z_s  # list concatenation
        z = torch.stack(z).transpose(0, 1).type_as(s_t)  # to get N_batches x latent_dim

        bottleneck = self(s_t)
        z_hat = bottleneck["mu"]
        recon_loss = self.rec_loss(z, z_hat)

        loss_dict = {
           "training_recon_loss": recon_loss,
           "loss": recon_loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict


    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch

        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log_dict({
            "training_recon_loss": avg_recon_loss.detach(),
            "training_loss": avg_loss.detach()
          },
          on_epoch=True,
          prog_bar=True,
          logger=True,
        )

    def on_validation_epoch_start(self):
        self.model.set_mode("testing")


    def _shared_eval_step(self, batch, batch_idx):

        '''
        The validation and testing steps are similar, only the names of the logged quantities differ.
        The common part is performed here.
        '''


        s_t, z_c, z_s = batch["s_t"], batch["z_c"], batch["z_s"]

        z = z_c + z_s # list concatenation
        z = torch.stack(z).transpose(0,1).type_as(s_t) # to get N_batches x latent_dim

        bottleneck = self(s_t)
        z_hat = bottleneck["mu"]

        recon_loss = self.rec_loss(z, z_hat)
        loss = recon_loss

        # TODO: compute normalized metrics
        # rec_ratio_to_pop_mean_c = mse(time_avg_s, time_avg_s_hat) / mse(time_avg_s)
        # rec_ratio_to_pop_mean = mse(s_t, shat_t) / mse_mesh_to_pop_mean
        # rec_ratio_to_time_mean = mse(s_t, shat_t) / mse_mesh_to_tmp_mean

        return loss,\
               recon_loss


    def validation_step(self, batch, batch_idx):

        loss, recon_loss = self._shared_eval_step(batch, batch_idx)
        loss_dict = { "val_recon_loss": recon_loss, "val_loss": loss }
        self.log_dict(loss_dict)
        return loss_dict


    def validation_epoch_end(self, outputs):

        #TODO: iterate over keys of the elements of `outputs`

        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log_dict(
          {"val_recon_loss": avg_recon_loss.detach(), "val_loss": avg_loss.detach()},
          on_epoch=True,
          prog_bar=True,
          logger=True
        )


    def test_step(self, batch, batch_idx):

        loss, recon_loss = self._shared_eval_step(batch, batch_idx)
        loss_dict = { "test_recon_loss": recon_loss, "test_loss": loss }
        self.log_dict(loss_dict)
        return loss_dict


    def test_epoch_end(self, outputs):

        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()

        loss_dict = {
            "test_recon_loss": avg_recon_loss.detach(),
            "test_loss": avg_loss.detach(),
        }

        self.log_dict(loss_dict)


    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer