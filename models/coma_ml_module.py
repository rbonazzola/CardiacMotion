import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from IPython import embed

class CoMA(pl.LightningModule):
    '''
    model: provide 
    params: 
    '''
    
    def __init__(self, model, params):
        
        super(CoMA, self).__init__()
        self.model = model
        
        self.params = params

        #TOFIX: decide it from parameters        
        self.rec_loss_function = F.mse_loss
        self.w_kl = self.params.loss.regularization_loss.weight
                
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)
    
    def on_train_epoch_start(self):
        self.model.set_mode("training")
    
    def training_step(self, batch, batch_idx):
        
        # data, ids = batch
        data = batch

        if self.model._is_variational:
            (mu, log_var), out = self(batch)
            kld_loss = -0.5 * torch.mean(torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        recon_loss = self.rec_loss_function(out, data)#.reshape(-1, self.model.filters[0]))        
        train_loss = recon_loss + self.w_kl * kld_loss

        
        loss_dict = {
            "kld_loss": kld_loss,
            "recon_loss": recon_loss,
            "loss": train_loss
        }

        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)  
        return loss_dict

    
    def training_epoch_end(self, outputs):
        
        # Aggregate metrics from each batch    

        avg_kld_loss = torch.stack(
            [x['kld_loss'] for x in outputs]            
        ).mean()

        avg_recon_loss = torch.stack(
            [x['recon_loss'] for x in outputs]            
        ).mean()

        avg_loss = torch.stack(
            [x['loss'] for x in outputs]            
        ).mean()

        self.log_dict({
            "kld_loss": avg_kld_loss,
            "recon_loss": avg_recon_loss,
            "loss": avg_loss
        }, on_epoch=True, prog_bar=True, logger=True)
            
        # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#logging
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    
    def on_validation_epoch_start(self):
        self.model.set_mode("training")

        
    def validation_step(self, batch, batch_idx):                
        
        # data, ids = batch
        data = batch

        if self.model._is_variational:
            (mu, log_var), out = self(batch)
            kld_loss = -0.5 * torch.mean(torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


        recon_loss = self.rec_loss_function(out, data)#.reshape(-1, self.model.filters[0]))        
        train_loss = recon_loss + self.w_kl * kld_loss

        #Why not just passing
        loss_dict = {
            "kld_loss": kld_loss,
            "recon_loss": recon_loss,
            "loss": train_loss
        }

        self.log_dict(loss_dict)  
        return loss_dict
        
    
    def validation_epoch_end(self, outputs):
        # embed()

        avg_kld_loss = torch.stack(
            [x['kld_loss'] for x in outputs]            
        ).mean()

        avg_recon_loss = torch.stack(
            [x['recon_loss'] for x in outputs]            
        ).mean()

        avg_loss = torch.stack(
            [x['loss'] for x in outputs]            
        ).mean()

        self.log_dict({
            "kld_loss": avg_kld_loss,
            "recon_loss": avg_recon_loss,
            "loss": avg_loss
        }, on_epoch=True, prog_bar=True, logger=True)

    
    # def test_step(self):
    
    def test_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log({"loss":avg_loss})
        pass
    
    
    #TODO: Select optimizer from menu (dict)
    def configure_optimizers(self):
        
        algorithm = self.params.optimizer.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.params.optimizer.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)      
        return optimizer    