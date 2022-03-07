import unittest
import torch
import numpy as np

from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage, ScaleIntensity, RandRotate, RandFlip, RandZoom, EnsureType,
)
from monai.data import PILReader

from src.data_utils.CustomDataset import GeomCnnDataset
from src.data_utils.utils import get_image_files


class TestGeomCnnDataset(unittest.TestCase):
    def test_dataset_length(self):
        _transforms = Compose(
            [
                LoadImage(image_only=True),
                ScaleIntensity(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                EnsureType(),
            ]
        )

        image_files, labels = get_image_files()
        _dataset = GeomCnnDataset(
            image_files=image_files,
            labels=labels,
            transforms=_transforms
        )

        self.assertEqual(len(_dataset), 203,
                         "Unexpected number of samples")

    def test_dataset_shape(self):
        _transforms = Compose(
            [
                LoadImage(image_only=True),
                AddChannel(),
                RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
                RandFlip(spatial_axis=0, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                ScaleIntensity()
            ]
        )
        image_files, labels = get_image_files()
        _dataset = GeomCnnDataset(
            image_files=image_files,
            labels=labels,
            transforms=_transforms
        )
        batch_size = 10
        data_loader = torch.utils.data.DataLoader(_dataset,
                                                  batch_size=batch_size)
        for images, labels in data_loader:
            self.assertEqual(images.shape, (batch_size, 4, 512, 512),
                             "Image size doesn't match")
            self.assertEqual(labels.shape, (batch_size, ))
            break
