import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

from src.models.cnn_model import SimpleCNN
from src.models.mtl_model import MLTModel
from src.pl_modules.classifier_modules import ImageClassifier
from src.pl_modules.mlt_classifier_modules import mltImageClassifier
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
    parser.add_argument('--n_folds', type=int, default=4)
    parser.add_argument('--exp_name')
    parser.add_argument("--task_list", nargs="+", help="Task list")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file for prediction")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageClassifier.add_model_specific_args(parser)
    args = parser.parse_args()
    task_list = ["group", "Gender", "ADOS_restricted_repetitive_behavior_total",
                                   "ADOS_severity_score_lookup",
                                   "ADOS_social_affect_restricted_repetitive_behavior_total"]

    # ------------
    # model
    # ------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = mltImageClassifier.load_from_checkpoint(args.ckpt, device=device).to(device)

    # -----------
    # Data
    # -----------
    # data_module = GeomCnnDataModule(batch_size=args.batch_size, num_workers=args.data_workers)
    data_module_generator = GeomCnnDataModuleKFold(
        batch_size=args.batch_size,
        num_workers=1,
        n_splits=args.n_folds,
        task_names=task_list
    )
    data_modules = data_module_generator.get_folds()
    trainer = pl.Trainer()
    output = []
    for i in range(args.n_folds):
        # logger
        output.append(trainer.predict(model, data_modules[i])[0])
    return output


if __name__ == '__main__':
    output = cli_main()
