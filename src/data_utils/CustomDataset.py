import torch
import numpy as np


class GeomCnnDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, demographics, labels, transforms):
        super(GeomCnnDataset, self).__init__()
        self.image_files = image_files
        self.demographics = demographics
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        y = {}
        for k, v in self.labels.items():
            y[k] = v[idx]
        return np.concatenate(self.transforms(self.image_files[idx]), axis=0), self.demographics[idx], y


class GeomCnnDatasetDF(torch.utils.data.Dataset):
    def __init__(self, df, transforms):
        super(GeomCnnDatasetDF, self).__init__()
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return np.concatenate(self.transforms(list(self.df["FILEPATHS"].iloc[idx])), axis=0), \
               self.df.iloc[idx]