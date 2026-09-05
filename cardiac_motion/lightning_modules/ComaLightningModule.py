import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from PIL import Image
import imageio
import numpy as np

from argparse import Namespace

from models.Model4D import AutoencoderTemporalSequence, LOG_VAR_MAX, LOG_VAR_MIN, clamp_log_var

logger = logging.getLogger(__name__)

losses_menu = {
  "l1": F.l1_loss,
  "mse": F.mse_loss
}

def mse(s1, s2=None):
    if s2 is None:
        s2 = torch.zeros_like(s1)
    return ((s1-s2)**2).sum(-1).mean(-1)


def safe_mean_ratio(numerator, denominator):
    eps = torch.finfo(denominator.dtype).eps
    valid = denominator.abs() > eps
    if not valid.any():
        return torch.tensor(float("nan"), device=numerator.device, dtype=numerator.dtype)
    return (numerator[valid] / denominator[valid]).mean()

class CoMA_Lightning(pl.LightningModule):


    def __init__(self, 
                 model: AutoencoderTemporalSequence, 
                 loss_params: Namespace, 
                 optimizer_params: Namespace,
                 additional_params: Namespace,
                 mesh_template=None
                ):

        '''
        :param model: PyTorch model.
        :param params: a Namespace with additional parameters
        
        Example:
        from Easydict import EasyDict
        
        loss_params = EasyDict({
          "reconstruction_c.weight": ,
          "reconstruction_s.weight": ,
          "regularization.weight":          
        })          
        '''

        super(CoMA_Lightning, self).__init__()
        self.model = model
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params
        self.additional_params = additional_params
        self.mesh_template = mesh_template 

        self.rec_loss = self.get_rec_loss()

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        self.kl_warmup_epochs = getattr(self.loss_params.regularization, "warmup_epochs", 5)
        self._kl_warning_count = 0


    def get_rec_loss(self):

        self.w_s = self.loss_params.reconstruction_s.weight
        self.w_kl = self.loss_params.regularization.weight
        return losses_menu[self.loss_params.reconstruction_c.type.lower()]

            
    def KL_div(self, mu, log_var):
        log_var_has_nonfinite = not torch.isfinite(log_var).all().item()
        log_var_was_clamped = ((log_var < LOG_VAR_MIN).any() or (log_var > LOG_VAR_MAX).any()).item()
        log_var = clamp_log_var(log_var)

        kld = 0.5 * (mu.pow(2) + log_var.exp() - 1 - log_var)
        kld_has_nonfinite = not torch.isfinite(kld).all().item()

        if (log_var_has_nonfinite or log_var_was_clamped or kld_has_nonfinite) and self._kl_warning_count < 5:
            logger.warning(
                "KL stabilization applied: log_var_range=(%.2f, %.2f), clamped_to=(%.1f, %.1f), nonfinite_log_var=%s, nonfinite_kld=%s",
                self._finite_min(log_var),
                self._finite_max(log_var),
                LOG_VAR_MIN,
                LOG_VAR_MAX,
                bool(log_var_has_nonfinite),
                bool(kld_has_nonfinite),
            )
            self._kl_warning_count += 1

        kld = torch.nan_to_num(kld, nan=0.0, posinf=1e6, neginf=0.0)
        return kld.mean(dim=1).mean(dim=0)

    @staticmethod
    def _finite_min(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return finite.min().detach().cpu().item() if finite.numel() else float("nan")

    @staticmethod
    def _finite_max(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return finite.max().detach().cpu().item() if finite.numel() else float("nan")

    def _is_variational(self):
        return bool(getattr(self.model, "is_variational", False))

    def _effective_w_kl(self):
        if not self._is_variational():
            return 0.0
        if self.current_epoch <= self.kl_warmup_epochs:
            return 0.0
        return self.w_kl

    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)
    
    
    def on_fit_start(self):

        #TODO: check of alternatives since .to(device) is not recommended
        #This is the most elegant way I found so far to transfer the tensors to the right device 
        #(if this is run within __init__, I get self.device=="cpu" even when I use a GPU, so it doesn't work there)

        for i, _ in enumerate(self.model.encoder.matrices['downsample']):
            self.model.encoder.matrices['downsample'][i] = self.model.encoder.matrices['downsample'][i].to(self.device)            
            # self.model.encoder.matrices['adjacency_matrices'][i] = self.model.encoder.matrices['adjacency_matrices'][i].to(self.device)
            self.model.decoder.matrices['upsample'][i] = self.model.decoder.matrices['upsample'][i].to(self.device)

        for i, _ in enumerate(self.model.encoder.matrices['A_edge_index']):
            self.model.encoder.matrices['A_edge_index'][i] = self.model.encoder.matrices['A_edge_index'][i].to(self.device)
            self.model.encoder.matrices['A_norm'][i] = self.model.encoder.matrices['A_norm'][i].to(self.device)

        # embed()
        # self.precision_to_set  = self.trainer.precision
        # self.batch_size_to_set = self.trainer.train_dataloader.batch_size
        
        
    def on_train_epoch_start(self):
        self.model.set_mode("training")

        
    def training_step(self, batch, batch_idx):

        s_t, time_avg_s = batch["s_t"], batch["time_avg_s"]
        bottleneck, time_avg_shat, shat_t = self(s_t)

        # print(f"{bottleneck.mu.mean()=}\n\n{bottleneck.log_var.mean()}\n\n{time_avg_shat.mean()=}\n\n{shat_t.mean()=}")
        # print(f"{bottleneck.mu.mean()=}\n\n{time_avg_shat.mean()=}\n\n{shat_t.mean()=}")
        
        recon_loss_c = self.rec_loss(time_avg_s, time_avg_shat)
        recon_loss_s = self.rec_loss(s_t, shat_t)

        recon_loss = recon_loss_c + self.w_s * recon_loss_s

        if self._is_variational():
            bottleneck = self.model.decoder._partition_z(bottleneck["mu"], bottleneck["log_var"])            
            self.mu_c = bottleneck["mu_c"]
            self.log_var_c = bottleneck["log_var_c"]
            self.mu_s = bottleneck["mu_s"]
            self.log_var_s = bottleneck["log_var_s"]            
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)
        else:
            kld_loss_c = kld_loss_s = torch.zeros_like(recon_loss)

        kld_loss = kld_loss_c + kld_loss_s

        train_loss = recon_loss + self._effective_w_kl() * kld_loss
        
        # print(f"{recon_loss_c.item()=} + {recon_loss_s.item()=}")
        # print(f"{train_loss.item()=} = {recon_loss.item()=} + {self.w_kl} * {kld_loss.item()=}")

        loss_dict = {
           "training_kld_loss": kld_loss,
           "training_recon_loss": recon_loss,
           "training_recon_loss_c": recon_loss_c,
           "training_recon_loss_s": recon_loss_s,
           "loss": train_loss
        }

        self.train_outputs.append({k: v.detach().cpu() for k, v in loss_dict.items()})

        # self.train_outputs.append(loss_dict)
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log-dict
        self.log_dict(loss_dict)
        return loss_dict

    def on_train_epoch_end(self):

        # Aggregate metrics from each batch

        avg_kld_loss = torch.stack([x["training_kld_loss"] for x in self.train_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["training_recon_loss_c"] for x in self.train_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["training_recon_loss_s"] for x in self.train_outputs]).mean()
        avg_recon_loss = torch.stack([x["training_recon_loss"] for x in self.train_outputs]).mean()
        avg_loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()

        self.log_dict({
            "training_kld_loss": avg_kld_loss,
            "training_recon_loss": avg_recon_loss,
            "training_recon_loss_c": avg_recon_loss_c,
            "training_recon_loss_s": avg_recon_loss_s,
            "training_loss": avg_loss
          },
          # on_epoch=False,
          on_epoch=True,
          prog_bar=True,
          logger=True,
        )

        self.train_outputs.clear()
        # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#logging
        # self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
    def on_validation_epoch_start(self):
        self.model.set_mode("testing")
        

    # def on_training_epoch_start(self):
    #     
    #     if self.current_epoch < 1:
    #         self.batch_size = 32
    #         self.precision = 32
    #         self.trainer.precision = 32
    #         self.trainer.train_dataloader.batch_size = self.batch_size  # Update batch size
    #     else:
    #         print(f"Chaning batch size (to {self.precision_to_set}) and batch_size (to {self.batch_size_to_set})")
    #         self.batch_size = self.batch_size_to_set
    #         self.precision = self.precision_to_set
    #         self.trainer.precision = self.precision
    #         self.trainer.train_dataloader.batch_size = self.batch_size  # Update batch size

    
    def _shared_eval_step(self, batch, batch_idx):

        '''
        The validation and testing steps are similar, only the names of the logged quantities differ.
        The common part is performed here.
        '''

        s_t = batch["s_t"]        
        time_avg_s = batch["time_avg_s"]        
        mse_mesh_to_tmp_mean = batch["d_content"]        
        mse_mesh_to_pop_mean = batch["d_style"]
        
        bottleneck, time_avg_s_hat, shat_t = self(s_t)

        # content
        recon_loss_c = self.rec_loss(time_avg_s, time_avg_s_hat)
        recon_loss_s = self.rec_loss(s_t, shat_t)
        recon_loss = recon_loss_c + self.w_s * recon_loss_s

        if self._is_variational():
            
            bottleneck = self.model.decoder._partition_z(bottleneck["mu"], bottleneck["log_var"])
            
            self.mu_c = bottleneck["mu_c"]
            self.log_var_c = bottleneck["log_var_c"]
            self.mu_s = bottleneck["mu_s"]
            self.log_var_s = bottleneck["log_var_s"]
            
            kld_loss_c = self.KL_div(self.mu_c, self.log_var_c)
            kld_loss_s = self.KL_div(self.mu_s, self.log_var_s)
            kld_loss = kld_loss_c + kld_loss_s
            loss = recon_loss + self._effective_w_kl() * kld_loss
        else:
            loss = recon_loss
            kld_loss = kld_loss_c = kld_loss_s = torch.zeros_like(loss)

        recon_error = mse(s_t, shat_t)

        # N-dimensional
        rec_ratio_to_pop_mean_c = safe_mean_ratio(mse(time_avg_s, time_avg_s_hat), mse(time_avg_s))
        rec_ratio_to_time_mean = safe_mean_ratio(recon_error, mse_mesh_to_tmp_mean)

        # Only computable when template_mesh was provided to the dataset
        if mse_mesh_to_pop_mean is not None:
            rec_ratio_to_pop_mean = safe_mean_ratio(recon_error, mse_mesh_to_pop_mean)
        else:
            rec_ratio_to_pop_mean = torch.tensor(float("nan"))
               
        return loss,\
               recon_loss, recon_loss_c, recon_loss_s,\
               kld_loss_c, kld_loss_s, kld_loss,\
               rec_ratio_to_time_mean,\
               rec_ratio_to_pop_mean,\
               rec_ratio_to_pop_mean_c

    
    def validation_step(self, batch, batch_idx):

        # loss_dict = self._shared_eval_step(batch, batch_idx)
        # loss_dict = { "val_"+k: v for k, v in loss_dict.items() }

        loss, recon_loss, recon_loss_c, recon_loss_s, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean, rec_ratio_to_pop_mean_c = self._shared_eval_step(batch, batch_idx)

        loss_dict = {
          "val_kld_loss": kld_loss,
          "val_recon_loss": recon_loss,
          "val_recon_loss_c": recon_loss_c,
          "val_recon_loss_s": recon_loss_s,
          "val_loss": loss,
          "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "val_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }
        
        self.val_outputs.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        self.log_dict(loss_dict)
        return loss_dict

    
    def on_validation_epoch_end(self):

        #TODO: iterate over keys of the elements of `outputs`

        avg_kld_loss = torch.stack([x["val_kld_loss"] for x in self.val_outputs]).mean()
        avg_recon_loss = torch.stack([x["val_recon_loss"] for x in self.val_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["val_recon_loss_c"] for x in self.val_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["val_recon_loss_s"] for x in self.val_outputs]).mean()
        avg_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["val_rec_ratio_to_time_mean"] for x in self.val_outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["val_rec_ratio_to_pop_mean"] for x in self.val_outputs]).mean()
        rec_ratio_to_pop_mean_c = torch.stack([x["val_rec_ratio_to_pop_mean_c"] for x in self.val_outputs]).mean()
        

        self.log_dict({
            "val_kld_loss": avg_kld_loss, 
            "val_recon_loss": avg_recon_loss,
            "val_recon_loss_c": avg_recon_loss_c,
            "val_recon_loss_s": avg_recon_loss_s,
            "val_loss": avg_loss,
            "val_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
            "val_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
            "val_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
          },
          on_epoch=True,
          prog_bar=True,
          logger=True
        )

        self.val_outputs.clear()


    def on_test_epoch_start(self):
        self.model.set_mode("testing")
        

    def test_step(self, batch, batch_idx):
                
        loss, recon_loss, recon_loss_c, recon_loss_s, kld_loss_c, kld_loss_s, kld_loss, rec_ratio_to_time_mean, rec_ratio_to_pop_mean, rec_ratio_to_pop_mean_c = self._shared_eval_step(batch, batch_idx)

        loss_dict = {
          "test_kld_loss": kld_loss, 
          "test_recon_loss": recon_loss,
          "test_recon_loss_c": recon_loss_c,
          "test_recon_loss_s": recon_loss_s,
          "test_loss": loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "test_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }

        self.test_outputs.append({k: v.detach().cpu() for k, v in loss_dict.items()})
        self.log_dict(loss_dict)
        return loss_dict

    def on_test_epoch_end(self):

       # outputs =self.outputs
        avg_kld_loss = torch.stack([x["test_kld_loss"] for x in self.test_outputs]).mean()
        avg_recon_loss = torch.stack([x["test_recon_loss"] for x in self.test_outputs]).mean()
        avg_recon_loss_c = torch.stack([x["test_recon_loss_c"] for x in self.test_outputs]).mean()
        avg_recon_loss_s = torch.stack([x["test_recon_loss_s"] for x in self.test_outputs]).mean()
        avg_loss = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        rec_ratio_to_time_mean = torch.stack([x["test_rec_ratio_to_time_mean"] for x in self.test_outputs]).mean()
        rec_ratio_to_pop_mean = torch.stack([x["test_rec_ratio_to_pop_mean"] for x in self.test_outputs]).mean()
        rec_ratio_to_pop_mean_c = torch.stack([x["test_rec_ratio_to_pop_mean_c"] for x in self.test_outputs]).mean()
        
        loss_dict = {
          "test_kld_loss": avg_kld_loss, 
          "test_recon_loss": avg_recon_loss,
          "test_recon_loss_c": avg_recon_loss_c,
          "test_recon_loss_s": avg_recon_loss_s,
          "test_loss": avg_loss,
          "test_rec_ratio_to_time_mean": rec_ratio_to_time_mean,
          "test_rec_ratio_to_pop_mean": rec_ratio_to_pop_mean,
          "test_rec_ratio_to_pop_mean_c": rec_ratio_to_pop_mean_c
        }

        self.log_dict(loss_dict)        
        self.test_outputs.clear()

        
    def predict_step(self, batch, batch_idx):

        s_t = batch["s_t"]        
        time_avg_s = batch["time_avg_s"]        
        mse_mesh_to_tmp_mean = batch["d_content"]        
        mse_mesh_to_pop_mean = batch["d_style"]
        
        bottleneck, time_avg_s_hat, s_hat_t = self(s_t)


        ### IMAGES OF TEMPORAL AVERAGE
        if self.additional_params.dataset.preprocessing.center_around_mean:
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s[0].cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"temporal_avg_mesh_{batch_idx}_orig.png")
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s_hat[0].cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"temporal_avg_mesh_{batch_idx}_rec.png")
        else:
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s[0], self.mesh_template.faces,
                                                       f"temporal_avg_mesh_{batch_idx}_orig.png")
            SyntheticMeshPopulation.render_mesh_as_png(time_avg_s_hat[0], self.mesh_template.faces,
                                                       f"temporal_avg_mesh_{batch_idx}_rec.png")

        merge_pngs_horizontally(f"temporal_avg_mesh_{batch_idx}_orig.png", f"temporal_avg_mesh_{batch_idx}_rec.png", f"temporal_avg_mesh_{batch_idx}.png")

        self.logger.experiment.log_artifact(
            local_path=f"temporal_avg_mesh_{batch_idx}.png",
            artifact_path="images", run_id=self.logger.run_id
        )

        ### ANIMATIONS OF MOVING MESH
        if self.additional_params.dataset.preprocessing.center_around_mean:
            SyntheticMeshPopulation._generate_gif(s_t.cpu()+self.mesh_template.vertices, self.mesh_template.faces, f"moving_mesh_{batch_idx}_orig.gif")
            SyntheticMeshPopulation._generate_gif(s_hat_t.cpu() + self.mesh_template.vertices, self.mesh_template.faces, f"moving_mesh_{batch_idx}_rec.gif")
        else:
            SyntheticMeshPopulation._generate_gif(s_t, self.mesh_template.faces, f"moving_mesh_{batch_idx}_orig.gif")
            SyntheticMeshPopulation._generate_gif(s_hat_t, self.mesh_template.faces, f"moving_mesh_{batch_idx}_rec.gif")

        merge_gifs_horizontally(f"moving_mesh_{batch_idx}_orig.gif", f"moving_mesh_{batch_idx}_rec.gif", f"moving_mesh_{batch_idx}.gif")
        self.logger.experiment.log_artifact(
            local_path=f"moving_mesh_{batch_idx}.gif",
            artifact_path="animations", run_id=self.logger.run_id
        )
        
        return 1 # to prevent warning messages


    # TODO: Select optimizer from menu (dict)

    def configure_optimizers(self):

        algorithm = self.optimizer_params.algorithm
        algorithm = torch.optim.__dict__[algorithm]
        parameters = vars(self.optimizer_params.parameters)
        optimizer = algorithm(self.model.parameters(), **parameters)
        return optimizer


def merge_pngs_horizontally(png1, png2, output_png):
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    #Read the two images
    image1 = Image.open(png1)
    image2 = Image.open(png2)
    #resize, first image
    image1_size = image1.size
    # image2_size = image2.size
    new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
    new_image.paste(image1,(0,0))
    new_image.paste(image2,(image1_size[0],0))
    new_image.save(output_png, "PNG")

    
def merge_gifs_horizontally(gif_file1, gif_file2, output_file):

    #Create reader object for the gif
    gif1 = imageio.get_reader(gif_file1)
    gif2 = imageio.get_reader(gif_file2)

    #Create writer object
    new_gif = imageio.get_writer(output_file)

    for frame_number in range(gif1.get_length()):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()
