import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from IPython import embed # uncomment for debugging
from models.model_c_and_s import Coma4D_C_and_S

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

class CoMA(pl.LightningModule):


    def __init__(self, model, params):

        """
        :param model: provide the PyTorch model.
        :param params: a Namespace with additional parameters
        """

        super(CoMA, self).__init__()
        self.model = model
        self.params = params

        # TOFIX: decide it from parameters

        self.rec_loss = self.get_rec_loss()


    def get_rec_loss(self):

        self.w_s = self.params.loss.reconstruction_s.weight
        self.w_kl = self.params.loss.regularization.weight
        return losses_menu[self.params.loss.reconstruction_c.type.lower()]

    # def get_loss(self):
    #     return
        
    def KL_div(self, mu, log_var):
        return -0.5 * torch.mean(torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device 
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for i, _ in enumerate(self.model.downsample_matrices):
            self.model.downsample_matrices[i] = self.model.downsample_matrices[i].to(self.device)
            self.model.upsample_matrices[i] = self.model.upsample_matrices[i].to(self.device)
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
        moving_meshes, time_avg_mesh, _, _ = batch

        if self.model._is_variational:
            bottleneck, s_avg, s_t = self(moving_meshes)
            self.mu_c, self.log_var_c, self.mu_s, self.log_var_s = bottleneck
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)

        kld_loss = kld_loss_c + kld_loss_s
        recon_loss_c = self.rec_loss(s_t, moving_meshes)
        recon_loss_s = self.rec_loss(s_avg, time_avg_mesh)

        recon_loss = recon_loss_c + self.w_s * recon_loss_s
        train_loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {
           "training_kld_loss": kld_loss,
           "training_recon_loss": recon_loss,
           "training_recon_loss_c": recon_loss_c,
           "training_recon_loss_s": recon_loss_s,
           "loss": train_loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict

    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch

        avg_kld_loss = torch.stack([x["training_kld_loss"] for x in outputs]).mean()
        avg_recon_loss_c = torch.stack([x["training_recon_loss_c"] for x in outputs]).mean()
        avg_recon_loss_s = torch.stack([x["training_recon_loss_s"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log_dict({
            "training_kld_loss": avg_kld_loss,
            "training_recon_loss": avg_recon_loss,
            "training_recon_loss_c": avg_recon_loss_c,
            "training_recon_loss_s": avg_recon_loss_s,
            "training_loss": avg_loss
          },
          on_epoch=True,
          prog_bar=True,
          logger=True,
        )

        # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#logging
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self):
        self.model.set_mode("training")

    def _shared_eval_step(self, batch, batch_idx):

        '''
        The validation and testing steps are similar, only the names of the logged quantities differ.
        The common part is performed here.
        '''

        moving_meshes, time_avg_mesh, mse_mesh_to_tmp_mean, mse_mesh_to_pop_mean = batch

        bottleneck, s_avg, s_t = self(moving_meshes)

        # content
        recon_loss_c = self.rec_loss(s_t, moving_meshes)
        recon_loss_s = self.rec_loss(s_avg, time_avg_mesh)
        recon_loss = recon_loss_c + self.w_s * recon_loss_s

        if self.model._is_variational:
            self.mu_c, self.log_var_c, self.mu_s, self.log_var_s = bottleneck
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)
            kld_loss = kld_loss_c + kld_loss_s
            loss = recon_loss + self.w_kl * kld_loss
        else:
            kld_loss = None
            loss = recon_loss

        mse_per_subj_per_time = ((s_t-moving_meshes)**2).sum(axis=-1).mean(axis=-1)
        
        rec_ratio_to_time_mean = mse_per_subj_per_time / mse_mesh_to_tmp_mean
        rec_ratio_to_pop_mean = mse_per_subj_per_time / mse_mesh_to_pop_mean

        return loss,\
               recon_loss, recon_loss_c, recon_loss_s,\
               kld_loss_c, kld_loss_s, kld_loss,\
               rec_ratio_to_time_mean, rec_ratio_to_pop_mean

    def validation_step(self, batch, batch_idx):

        loss, recon_loss, recon_loss_c, recon_loss_s, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean = self._shared_eval_step(batch, batch_idx)
        
        loss_dict = {
          "val_kld_loss": kld_loss, 
          "val_recon_loss": recon_loss,
          "val_recon_loss_c": recon_loss_c,
          "val_recon_loss_s": recon_loss_s,
          "val_loss": loss, 
          "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean, 
          "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean
        }

        self.log_dict(loss_dict)
        return loss_dict

    def validation_epoch_end(self, outputs):

        #TODO: iterate over keys of the elements of `outputs`

        avg_kld_loss = torch.stack([x["val_kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_recon_loss_c = torch.stack([x["val_recon_loss_c"] for x in outputs]).mean()
        avg_recon_loss_s = torch.stack([x["val_recon_loss_s"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["val_rec_ratio_to_time_mean"] for x in outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["val_rec_ratio_to_pop_mean"] for x in outputs]).mean()

        self.log_dict({
            "val_kld_loss": avg_kld_loss, 
            "val_recon_loss": avg_recon_loss,
            "val_recon_loss_c": avg_recon_loss_c,
            "val_recon_loss_s": avg_recon_loss_s,
            "val_loss": avg_loss,
            "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
            "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean
          },
          on_epoch=True,
          prog_bar=True,
          logger=True
        )

    def test_step(self, batch, batch_idx):
                
        loss, recon_loss, recon_loss_c, recon_loss_s, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean = self._shared_eval_step(batch, batch_idx)

        loss_dict = {
          "test_kld_loss": kld_loss, 
          "test_recon_loss": recon_loss,
          "test_recon_loss_c": recon_loss_c,
          "test_recon_loss_s": recon_loss_s,
          "test_loss": loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean
        }
        self.log_dict(loss_dict)
        return loss_dict

    def test_epoch_end(self, outputs):

        avg_kld_loss = torch.stack([x["test_kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in outputs]).mean()
        avg_recon_loss_c = torch.stack([x["test_recon_loss_c"] for x in outputs]).mean()
        avg_recon_loss_s = torch.stack([x["test_recon_loss_s"] for x in outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["test_rec_ratio_to_time_mean"] for x in outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["test_rec_ratio_to_pop_mean"] for x in outputs]).mean()
        
        loss_dict = {
          "test_kld_loss": avg_kld_loss, 
          "test_recon_loss": avg_recon_loss,
          "test_recon_loss_c": avg_recon_loss_c,
          "test_recon_loss_s": avg_recon_loss_s,
          "test_loss": avg_loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean
        }

        self.log_dict(loss_dict)            


    # TODO: Select optimizer from menu (dict)

    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


# class CoMA_C_and_S(CoMA):

