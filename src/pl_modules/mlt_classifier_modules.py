from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics


class mltImageClassifier(pl.LightningModule):
    def __init__(self,
                 backbone,
                 criterion,
                 learning_rate=1e-3,
                 metrics=["acc"],
                 base_task="group",
                 device="cuda"):
        super(mltImageClassifier, self).__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.criterion = criterion
        self.tasks = backbone.tasks
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        self.base_task = base_task
        self.best_metric = 0
        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy().to(device)
                val_metric = torchmetrics.Accuracy().to(device)
            elif m == "auc":
                train_metric = torchmetrics.AUROC(pos_label=1).to(device)
                val_metric = torchmetrics.AUROC(pos_label=1).to(device)
            elif m == "precision":
                train_metric = torchmetrics.Precision().to(device)
                val_metric = torchmetrics.Precision().to(device)
            elif m == "recall":
                train_metric = torchmetrics.Recall().to(device)
                val_metric = torchmetrics.Recall().to(device)
            elif m == "auc":
                train_metric = torchmetrics.AUC().to(device)
                val_metric = torchmetrics.AUC().to(device)
            self.train_metrics[m] = train_metric
            self.val_metrics[m] = val_metric

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def common_step(self, batch, batch_idx, mode="train"):
        x, dem, y = batch
        output = self.backbone(x, dem)
        loss_dict = {}
        for i, task in enumerate(self.tasks):
            y_hat = output[task]
            y_true = y[task]
            if y_true.dtype == torch.float64:
                y_true = y_true.float()
            loss_dict[task] = self.criterion[task](y_hat, y_true.float())
            if task == self.base_task:
                y_hat = nn.Sigmoid()(y_hat)
                if mode == "train":
                    for m in self.metric_names:
                        self.train_metrics[m].update(y_hat, y[task])
                elif mode == "valid":
                    for m in self.metric_names:
                        self.val_metrics[m].update(y_hat, y[task])
        return loss_dict

    def training_epoch_end(self, outputs):
        # update and log
        print_msg = f"\n[Epoch {self.current_epoch}] Training metrics:"
        for m in self.metric_names:
            val = self.train_metrics[m].compute()
            print_msg += f" {m}: {val}, "
            self.log(f"train/{m}", val)
            self.train_metrics[m].reset()
        print(print_msg)

    def validation_epoch_end(self, outputs):
        # update and log
        print_msg = f"\n[Epoch {self.current_epoch}] Validation metrics:"
        auc = 0
        for m in self.metric_names:
            val = self.val_metrics[m].compute()
            if m == "auc":
                auc = val
            self.log(f"validation/{m}", val)
            print_msg += f" {m}: {val}, "
            self.val_metrics[m].reset()
        if auc > self.best_metric:
            self.best_metric = auc
        self.log(f"validation/best_auc", self.best_metric)
        print_msg += f" best auc: {self.best_metric}"
        print(print_msg)

    def log_scalars(self, scalar_name, scalar_value):
        self.log(scalar_name, scalar_value, on_step=True)

    def compute_total_loss(self, loss_dict):
        loss = 0
        for k, v in loss_dict.items():
            loss += v
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        total_loss = self.compute_total_loss(train_loss)
        # for k, v in train_loss.items():
        #     self.log_scalars(f'train/{k}_loss', v)
        # self.log_scalars(f"train/train_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        total_loss = self.compute_total_loss(valid_loss)
        for k, v in valid_loss.items():
            self.log_scalars(f'validation/{k}_loss', v)
        self.log_scalars(f"validation/valid_loss", total_loss)
        return total_loss

    def predict_step(self, batch, batch_idx):
        x, dem, y = batch
        output = self.backbone(x, dem)
        output['group'] = nn.Sigmoid()(output['group'])
        sorted, indices = torch.sort(output['group'], dim=-1, descending=False)
        y_true = y['group'][indices]
        auc = torchmetrics.AUC()
        auc.update(sorted, y_true)
        a = auc.compute()
        y_pred = torch.where(sorted > 0.3028, 1, 0)
        print(f"pred = {torch.stack([y_pred, y_true, sorted], dim=-1)}")
        prec = y_true[y_pred == 1].sum() / y_pred.sum()
        rec = y_pred[y_true == 1].sum() / y_true.sum()
        ppv = prec
        npv = (1 - y_true[y_pred == 0]).sum() / (1 - y_pred).sum()
        print(f"Precision = {prec:.4f}, Recall = {rec:.4f}, PPV = {ppv:.4f}, NPV = {npv:.4f}, AUC = {a}")
        return output

    def test_step(self, batch, batch_idx):
        test_loss = self.common_step(batch, batch_idx, mode="test")
        return self.compute_total_loss(test_loss)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default="efficientnet-b0")
        parser.add_argument('--pretrained', action='store_true', default=False)
        parser.add_argument('--task_list', nargs="+", type=str, help="list of task names")
        parser.add_argument("--loss_list", nargs="+", type=str, help="losses for each task")
        return parser
