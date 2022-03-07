import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

from src.models.cnn_model import SimpleCNN
from src.pl_modules.classifier_modules import ImageClassifier
from src.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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
    parser.add_argument('--data_workers', type=int, default=1)
    parser.add_argument('--backbone_name', type=str, default="simple_cnn")
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--write_dir')
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--exp_name')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
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
    else:
        backbone = SimpleCNN()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ImageClassifier(backbone, learning_rate=args.learning_rate,
                            criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 5.0])),
                            device=device,
                            metrics=["acc", "precision", "recall"])

    # -----------
    # Data
    # -----------
    # data_module = GeomCnnDataModule(batch_size=args.batch_size, num_workers=args.data_workers)
    data_module_generator = GeomCnnDataModuleKFold(
        batch_size=args.batch_size,
        num_workers=args.data_workers,
        n_splits=args.n_folds
    )
    data_modules = data_module_generator.get_folds()

    for i in range(args.n_folds):
        # logger
        logger = TensorBoardLogger(save_dir=os.path.join(args.write_dir, "logs", args.backbone_name, "fold_" + str(i)),
                                   name=args.exp_name)
        # early stopping
        es = EarlyStopping(monitor='validation/valid_loss',
                           patience=30)
        checkpointer = ModelCheckpoint(monitor='validation/valid_loss',
                                      save_top_k=5,
                                      verbose=True,
                                      save_last=True,
                                      dirpath=os.path.join(args.write_dir, "logs", args.backbone_name,
                                                           "fold_" + str(i), "checkpoints"))
        # ------------
        # training
        # ------------
        trainer = pl.Trainer(max_epochs=args.max_epochs,
                             gpus=args.gpus,
                             log_every_n_steps=5,
                             num_sanity_val_steps=0,
                             logger=logger,
                             callbacks=[es, checkpointer])
        trainer.fit(model,
                    datamodule=data_modules[i])


if __name__ == '__main__':
    cli_main()
