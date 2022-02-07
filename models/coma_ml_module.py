import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# from IPython import embed # uncomment for debugging

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

class CoMA(pl.LightningModule):
    """
    model: provide the PyTorch model.
    params: 
    """

    def __init__(self, model, params):

        super(CoMA, self).__init__()
        self.model = model
        self.params = params

        self.w_kl = self.params.loss.regularization_loss.weight

        # TOFIX: decide it from parameters

        self.rec_loss_function = losses_menu[params.loss.reconstruction.type.lower()]
        

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
        data = batch

        if self.model._is_variational:
            (mu, log_var), out = self(batch)
            kld_loss = -0.5 * torch.mean(
                torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )

        recon_loss = self.rec_loss_function(
            out, data
        )  # .reshape(-1, self.model.filters[0]))
        train_loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {"training_kld_loss": kld_loss, "training_recon_loss": recon_loss, "loss": train_loss}

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict


    def training_epoch_end(self, outputs):

        # Aggregate metrics from each batch

        avg_kld_loss = torch.stack([x["training_kld_loss"] for x in outputs]).mean()

        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in outputs]).mean()

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log_dict(
            {"training_kld_loss": avg_kld_loss, "training_recon_loss": avg_recon_loss, "training_loss": avg_loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#logging
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self):
        self.model.set_mode("training")

    def validation_step(self, batch, batch_idx):

        # data, ids = batch
        data = batch

        if self.model._is_variational:
            (mu, log_var), out = self(batch)
            kld_loss = -0.5 * torch.mean(
                torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )

        recon_loss = self.rec_loss_function(out, data)  # .reshape(-1, self.model.filters[0]))
        train_loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {"val_kld_loss": kld_loss, "val_recon_loss": recon_loss, "val_loss": train_loss}
        self.log_dict(loss_dict)
        return loss_dict


    def validation_epoch_end(self, outputs):

        #TODO: iterate over keys of the elements of `outputs`

        avg_kld_loss = torch.stack([x["val_kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log_dict(
            {"val_kld_loss": avg_kld_loss, "val_recon_loss": avg_recon_loss, "val_loss": avg_loss},
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        
        # data, ids = batch
        data = batch

        if self.model._is_variational:
            (mu, log_var), out = self(batch)
            kld_loss = -0.5 * torch.mean(
                torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
            )

        recon_loss = self.rec_loss_function(out, data)  # .reshape(-1, self.model.filters[0]))
        train_loss = recon_loss + self.w_kl * kld_loss

        loss_dict = {"test_kld_loss": kld_loss, "test_recon_loss": recon_loss, "test_loss": train_loss}
        self.log_dict(loss_dict)
        return loss_dict



    def test_epoch_end(self, outputs):

        avg_kld_loss = torch.stack([x["test_kld_loss"] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        
        self.log_dict(
            {"test_kld_loss": avg_kld_loss, "test_recon_loss": avg_recon_loss, "test_loss": avg_loss}
        )


    # TODO: Select optimizer from menu (dict)
    def configure_optimizers(self):

        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer
