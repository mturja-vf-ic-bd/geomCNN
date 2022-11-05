import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn
from monai.networks.blocks import Convolution
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

from src.models.cnn_model import SimpleCNN
from src.models.mtl_model import MLTModel2
from src.pl_modules.classifier_modules import ImageClassifier
from src.pl_modules.mlt_classifier_modules import mltImageClassifier
from src.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBarBase


def weight_reset(m):
    if isinstance(m, torch.nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def cli_main():
    # pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_workers', type=int, default=1)
    parser.add_argument('--backbone_name', type=str, default="simple_cnn")
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--write_dir')
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--exp_name')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--task_list", nargs="+", help="Task list")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    # task_list = ["group", "ADOS_severity_score_lookup", "ADOS_social_affect_restricted_repetitive_behavior_total"]
    task_list = ["group"]

    # -----------
    # Data
    # -----------
    # data_module = GeomCnnDataModule(batch_size=args.batch_size, num_workers=args.data_workers)
    data_module_generator = GeomCnnDataModuleKFold(
        batch_size=args.batch_size,
        num_workers=args.data_workers,
        n_splits=args.n_folds,
        task_names=task_list
    )
    data_modules = data_module_generator.get_folds()

    # ------------
    # model
    # ------------

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not args.from_ckpt:
        if args.backbone_name == "eff_bn":
            backbone = EfficientNetBN(
                model_name="efficientnet-b0",
                in_channels=args.in_channels,
                pretrained=False,
                num_classes=2
            )
        elif args.backbone_name == "densenet":
            backbone = DenseNet(
                spatial_dims=2,
                in_channels=args.in_channels,
                out_channels=2
            )
        elif args.backbone_name == "mtl":
            backbone = MLTModel2(tasks=task_list, in_channels=args.in_channels, hidden_dim=64, dropout=args.dropout,
                                out_dim={"group": 1})
        else:
            backbone = SimpleCNN(in_channels=6)
        model = mltImageClassifier(backbone, learning_rate=args.learning_rate,
                                   criterion={
                                       "group": torch.nn.BCEWithLogitsLoss().to(device)},
                                   # criterion={
                                   #     "group": torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 4.0]).to(device)),
                                   #     "ADOS_severity_score_lookup": torch.nn.MSELoss(),
                                   #     "ADOS_social_affect_restricted_repetitive_behavior_total": torch.nn.MSELoss()},
                                   device=device,
                                   metrics=["acc", "precision", "recall", "auc"])
        for i in range(args.n_folds):
            # logger
            logger = TensorBoardLogger(
                save_dir=os.path.join(args.write_dir, "logs", args.backbone_name, "fold_" + str(i)),
                name=args.exp_name)
            checkpointer = ModelCheckpoint(monitor='validation/auc',
                                           save_top_k=1,
                                           every_n_train_steps=1,
                                           verbose=False,
                                           mode="max",
                                           auto_insert_metric_name=False,
                                           filename='epoch-{epoch:02d}-auc-{validation/auc:.3f}',
                                           dirpath=os.path.join(args.write_dir, "logs", args.backbone_name,
                                                                "fold_" + str(i), args.exp_name, "checkpoints"))
            # ------------
            # training
            # ------------
            trainer = pl.Trainer(max_epochs=args.max_epochs,
                                 gpus=args.gpus,
                                 log_every_n_steps=5,
                                 num_sanity_val_steps=0,
                                 logger=logger,
                                 callbacks=[checkpointer])
            trainer.fit(model,
                        datamodule=data_modules[i])
            model.apply(weight_reset)
            break
    else:
        # or call with pretrained model
        trainer = pl.Trainer(max_epochs=1,
                             gpus=args.gpus)
        model = mltImageClassifier.load_from_checkpoint(args.ckpt)
        trainer.predict(model, dataloaders=data_modules[0].predict_dataloader())


if __name__ == '__main__':
    cli_main()
