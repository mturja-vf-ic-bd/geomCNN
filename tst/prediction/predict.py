import unittest

from src.data_utils.GeomCnnDataset import GeomCnnDataModule
from src.prediction.predict import predict
from src.data_utils.utils import get_test_dataloader


class TestPrediction(unittest.TestCase):
    def test_prediction_shape(self):
        checkpoint = "/Users/mturja/PycharmProjects/geomCNN/src/checkpoints/sa_vs_thickness_epoch=16.ckpt"
        data_loader = get_test_dataloader()
        output = predict(data_loader, checkpoint)
        print(output)

