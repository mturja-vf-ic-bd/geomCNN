import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn
from monai.networks.blocks import Convolution
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

from src.pl_modules.classifier_modules import ImageClassifier
from src.pl_modules.unet_modules import UNetModel
from src.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBarBase


def weight_reset(m):
    if isinstance(m, torch.nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--write_dir')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default="unet")
    parser.add_argument('--ckpt', type=str)
    parser.add_argument("--from_ckpt", dest='from_ckpt', default=False, action='store_true')
    parser.add_argument("--mode", default="train", type=str)
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
        n_splits=args.n_folds,
        num_workers=1,
    )
    data_modules = data_module_generator.get_folds()

    # ------------
    # model
    # ------------

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.from_ckpt:
        print(f"Loading from checkpoint")
        model = UNetModel.load_from_checkpoint(args.ckpt)
    else:
        model = UNetModel(in_channels=args.in_channels, depth=args.depth).to(device)

    if args.mode == "train":
        for i in range(args.n_folds):
            # logger
            logger = TensorBoardLogger(
                save_dir=os.path.join(args.write_dir, "logs", "unet", "fold_" + str(i)),
                name=args.exp_name)
            checkpointer = ModelCheckpoint(monitor='validation/valid_loss',
                                           save_top_k=1,
                                           every_n_train_steps=10,
                                           verbose=False,
                                           mode="min",
                                           auto_insert_metric_name=False,
                                           filename='epoch-{epoch:02d}-loss-{validation/valid_loss:.3f}',
                                           dirpath=os.path.join(args.write_dir, "logs", "unet",
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
        input, output = trainer.predict(model, dataloaders=data_modules[0].predict_dataloader())
        return input, output


if __name__ == '__main__':
    cli_main()
