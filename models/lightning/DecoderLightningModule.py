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


    ########### PREDICTING
    def predict_step(self, batch, batch_idx):

        s_avg, s_t, z = self._unpack_data_from_batch(batch)
        z = {"mu": z, "log_var": None}
        s_avg_hat, shat_t = self(z)

        ### IMAGES OF TEMPORAL AVERAGE
        _s_avg = s_avg[0].cpu()
        _s_avg_hat = s_avg_hat[0].cpu() + self.model.template_mesh.v
        if self.params.dataset.preprocessing.center_around_mean:
                _s_avg += self.model.template_mesh.v
                _s_avg_hat += self.model.template_mesh.v
        SyntheticMeshPopulation.render_mesh_as_png(_s_avg,
                                                       self.model.template_mesh.f,
                                                       f"temporal_avg_mesh_{batch_idx}_orig.png")
        SyntheticMeshPopulation.render_mesh_as_png(_s_avg_hat.cpu() + self.model.template_mesh.v,
                                                       self.model.template_mesh.f,
                                                       f"temporal_avg_mesh_{batch_idx}_rec.png")

        merge_pngs_horizontally(
            f"temporal_avg_mesh_{batch_idx}_orig.png", 
            f"temporal_avg_mesh_{batch_idx}_rec.png",
            f"temporal_avg_mesh_{batch_idx}.png"
        )

        self.logger.experiment.log_artifact(
            local_path=f"temporal_avg_mesh_{batch_idx}.png",
            artifact_path="images", run_id=self.logger.run_id
        )

        ### ANIMATIONS OF MOVING MESH
        if self.params.dataset.preprocessing.center_around_mean:
            s_t = s_t.cpu() + self.model.template_mesh.v
            s_hat_t = s_hat_t.cpu() + self.model.template_mesh.v
            SyntheticMeshPopulation._generate_gif(s_t, self.model.template_mesh.f, f"moving_mesh_{batch_idx}_orig.gif")
            SyntheticMeshPopulation._generate_gif(s_hat_t, self.model.template_mesh.f, f"moving_mesh_{batch_idx}_rec.gif")

        merge_gifs_horizontally(
            f"moving_mesh_{batch_idx}_orig.gif", 
            f"moving_mesh_{batch_idx}_rec.gif",
            f"moving_mesh_{batch_idx}.gif"
        )

        self.logger.experiment.log_artifact(
            local_path=f"moving_mesh_{batch_idx}.gif",
            artifact_path="animations", run_id=self.logger.run_id
        )

        return 0  # to prevent warning messages


def merge_pngs_horizontally(png1, png2, output_png):
    # https://www.tutorialspoint.com/python_pillow/Python_pillow_merging_images.htm
    # Read the two images
    image1 = Image.open(png1)
    image2 = Image.open(png2)
    # resize, first image
    image1_size = image1.size
    # image2_size = image2.size
    new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    new_image.save(output_png, "PNG")


def merge_gifs_horizontally(gif_file1, gif_file2, output_file):
    # Create reader object for the gif
    gif1 = imageio.get_reader(gif_file1)
    gif2 = imageio.get_reader(gif_file2)

    # Create writer object
    new_gif = imageio.get_writer(output_file)

    for frame_number in range(gif1.get_length()):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()
