from typing import Optional
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
from sklearn.model_selection import StratifiedKFold
# from imblearn.over_sampling import SMOTE


from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)

from src.data_utils.utils import get_image_files_3
from src.data_utils.CustomDataset import GeomCnnDataset, GeomCnnDatasetDF
from sklearn.model_selection import train_test_split


class GeomCnnDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = -1,
                 val_frac: float = 0.5,
                 num_workers=4,
                 data_tuple=None):
        super(GeomCnnDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.train_transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                RandRotate(range_x=np.pi / 24, prob=0.5, keep_size=True),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureType(),
            ]
        )
        self.val_transforms = Compose(
            [LoadImage(image_only=True), AddChannel(), EnsureType()]
        )
        self.test_transform = Compose(
                [LoadImage(image_only=True), AddChannel(), EnsureType()]
            )
        self.data_tuple = data_tuple
        self.save_hyperparameters()
        self.setup()

    def setup(self, stage: Optional[str] = None):
        print("Setting up data loaders ...")
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            train_df = get_image_files_3("TRAIN_DATA_DIR")
            length = len(train_df)
            if self.batch_size == -1:
                self.batch_size = length
            if self.data_tuple is None:
                print("Not implemented")
                # train_x, val_x, \
                # train_y, val_y = \
                #     train_test_split(train_files, train_labels,
                #                      test_size=self.val_frac, shuffle=True,
                #                      stratify=train_labels, random_state=42)
                # self.train_ds = GeomCnnDataset(train_x, train_y, self.train_transforms)
                # self.val_ds = GeomCnnDataset(val_x, val_y, self.val_transforms)
                # logging.info(f"Training samples: {len(train_y)}, Validation samples: {len(val_y)}")
            else:
                self.train_ds = GeomCnnDataset(self.data_tuple[0], self.data_tuple[1], self.data_tuple[2], self.train_transforms)
                self.val_ds = GeomCnnDataset(self.data_tuple[3], self.data_tuple[4], self.data_tuple[5], self.val_transforms)
                logging.info(f"Training samples: {len(self.data_tuple[0])}, Validation samples: {len(self.data_tuple[3])}")

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "predict"):
            self.val_ds = GeomCnnDataset(self.data_tuple[3], self.data_tuple[4], self.data_tuple[5], self.val_transforms)
        print("Finished loading !!!")

    def train_dataloader(self):
        size_class_0 = (self.train_ds.labels["group"] == 0).sum()
        size_class_1 = (self.train_ds.labels["group"] == 1).sum()
        print(f"Class 0: {size_class_0}, Class 1: {size_class_1}")
        if size_class_0 > size_class_1:
            class_weights = [1.0, size_class_0 / size_class_1]
        else:
            class_weights = [size_class_1 / size_class_0, 1.0]
        sample_weight = [0] * len(self.train_ds)
        for idx, (data, dem, label) in enumerate(self.train_ds):
            sample_weight[idx] = class_weights[label['group']]
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        return DataLoader(self.train_ds, self.batch_size, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size)


class GeomCnnDataModuleKFold:
    def __init__(self,
                 batch_size,
                 num_workers,
                 n_splits=2,
                 task_names=["group"]):
        super(GeomCnnDataModuleKFold, self).__init__()
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.num_workers = num_workers
        self.task_names = task_names
        self.datamodules = self.split_data()

    def convert_to_numpy(self, df):
        y_task = {}
        for task in self.task_names:
            y_task[task] = df[task].values
        X = df["FILEPATHS"].values.tolist()
        dem = df[["V06 demographics,Age_at_visit_start", "V12 demographics,Age_at_visit_start", "Gender"]].to_numpy()
        return X, dem, y_task

    def split_data(self):
        df = get_image_files_3("TRAIN_DATA_DIR")
        skf = StratifiedKFold(n_splits=self.n_splits, random_state=1, shuffle=True)
        datamodule_list = []
        for train_index, val_index in skf.split(df["FILEPATHS"], df["group"]):
            train_df = df.iloc[train_index]
            trainX, trainDem, trainY = self.convert_to_numpy(train_df)
            print(f"{trainY['group']}")
            # sm = SMOTE(random_state=1, sampling_strategy="minority")
            # trainX, trainY = sm.fit_resample(trainX, trainY["group"])
            valid_df = df.iloc[val_index]
            validX, validDem, validY = self.convert_to_numpy(valid_df)
            print(f"Validation counts: ASD-:{len(validY['group']) - sum(validY['group'])}, ASD+:{sum(validY['group'])}")
            datamodule_list.append(GeomCnnDataModule(
                batch_size=self.batch_size,
                data_tuple=(trainX, trainDem, trainY, validX, validDem, validY),
                num_workers=self.num_workers))
        return datamodule_list

    def get_folds(self):
        return self.datamodules

