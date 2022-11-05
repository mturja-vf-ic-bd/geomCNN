import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
import torch

from src.models.unet import UNet


class UNetModel(pl.LightningModule):
    def __init__(self, in_channels, depth, learning_rate=1e-3):
        super(UNetModel, self).__init__()
        self.save_hyperparameters()
        self.model = UNet(in_channels=in_channels, num_classes=in_channels, depth=depth, merge_mode='concat')

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def common_step(self, batch, batch_idx, mode="train"):
        x, dem, y = batch
        output = self.model(x)
        loss = nn.MSELoss()(output, x)
        return loss

    def log_scalars(self, scalar_name, scalar_value):
        self.log(scalar_name, scalar_value, on_step=True)

    def compute_total_loss(self, loss_dict):
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        self.log_scalars(f"train/train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        self.log_scalars(f"validation/valid_loss", valid_loss)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        x, dem, y = batch
        output = self.model(x)
        plot_mat(x[0], output[0])
        return x, output

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.001
        )


def plot_mat(input, output):
    n_feat = input.shape[0]
    plt.figure(figsize=(10, 5 * n_feat))
    c = 0
    for i in range(n_feat):
        plt.subplot(n_feat, 2, c+1)
        plt.imshow(input[i].detach().cpu().numpy())
        plt.subplot(n_feat, 2, c+2)
        plt.imshow(output[i].detach().cpu().numpy())
        c += 2
    plt.tight_layout()
    plt.show()